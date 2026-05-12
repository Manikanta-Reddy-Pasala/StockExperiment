# Aarti Industries Ltd. (AARTIIND)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 486.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 218 |
| ALERT1 | 147 |
| ALERT2 | 146 |
| ALERT2_SKIP | 71 |
| ALERT3 | 424 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 176 |
| PARTIAL | 27 |
| TARGET_HIT | 19 |
| STOP_HIT | 166 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 212 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 81 / 131
- **Target hits / Stop hits / Partials:** 19 / 166 / 27
- **Avg / median % per leg:** 1.18% / -0.57%
- **Sum % (uncompounded):** 249.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 92 | 29 | 31.5% | 16 | 72 | 4 | 1.39% | 128.1% |
| BUY @ 2nd Alert (retest1) | 10 | 8 | 80.0% | 0 | 6 | 4 | 3.18% | 31.8% |
| BUY @ 3rd Alert (retest2) | 82 | 21 | 25.6% | 16 | 66 | 0 | 1.17% | 96.3% |
| SELL (all) | 120 | 52 | 43.3% | 3 | 94 | 23 | 1.01% | 121.1% |
| SELL @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 5.04% | 30.2% |
| SELL @ 3rd Alert (retest2) | 114 | 46 | 40.4% | 3 | 91 | 20 | 0.80% | 90.8% |
| retest1 (combined) | 16 | 14 | 87.5% | 0 | 9 | 7 | 3.88% | 62.0% |
| retest2 (combined) | 196 | 67 | 34.2% | 19 | 157 | 20 | 0.95% | 187.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 13:15:00 | 503.05 | 499.82 | 499.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 14:15:00 | 505.45 | 500.94 | 500.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 14:15:00 | 512.10 | 512.13 | 508.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-24 14:45:00 | 512.30 | 512.13 | 508.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 509.30 | 511.48 | 508.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 09:30:00 | 508.10 | 511.48 | 508.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 10:15:00 | 506.75 | 510.54 | 508.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 11:00:00 | 506.75 | 510.54 | 508.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 11:15:00 | 504.40 | 509.31 | 508.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 12:00:00 | 504.40 | 509.31 | 508.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2023-05-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 14:15:00 | 506.10 | 507.78 | 507.82 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 12:15:00 | 509.15 | 507.86 | 507.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 13:15:00 | 509.70 | 508.23 | 507.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 10:15:00 | 511.55 | 512.67 | 511.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 10:15:00 | 511.55 | 512.67 | 511.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 10:15:00 | 511.55 | 512.67 | 511.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 11:00:00 | 511.55 | 512.67 | 511.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 11:15:00 | 511.85 | 512.50 | 511.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 13:30:00 | 512.90 | 512.16 | 511.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 14:00:00 | 512.60 | 512.16 | 511.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-31 12:15:00 | 510.00 | 512.24 | 511.86 | SL hit (close<static) qty=1.00 sl=510.20 alert=retest2 |

### Cycle 4 — SELL (started 2023-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 13:15:00 | 511.80 | 514.74 | 515.02 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 10:15:00 | 518.15 | 515.11 | 515.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 12:15:00 | 519.25 | 516.31 | 515.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 11:15:00 | 516.00 | 518.96 | 517.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 11:15:00 | 516.00 | 518.96 | 517.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 516.00 | 518.96 | 517.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:00:00 | 516.00 | 518.96 | 517.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 515.00 | 518.16 | 517.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 13:00:00 | 515.00 | 518.16 | 517.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2023-06-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 14:15:00 | 512.80 | 516.59 | 516.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 10:15:00 | 510.00 | 514.43 | 515.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 09:15:00 | 508.50 | 506.32 | 508.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 09:15:00 | 508.50 | 506.32 | 508.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 508.50 | 506.32 | 508.80 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 15:15:00 | 512.50 | 509.98 | 509.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 09:15:00 | 516.85 | 511.35 | 510.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-16 14:15:00 | 519.85 | 521.63 | 519.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 14:15:00 | 519.85 | 521.63 | 519.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 14:15:00 | 519.85 | 521.63 | 519.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-16 15:00:00 | 519.85 | 521.63 | 519.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 15:15:00 | 518.85 | 521.07 | 519.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 09:45:00 | 520.15 | 521.16 | 519.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 10:30:00 | 520.35 | 520.33 | 519.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 11:30:00 | 520.40 | 520.45 | 519.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 14:30:00 | 520.00 | 519.92 | 519.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 519.20 | 519.98 | 519.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 09:30:00 | 517.00 | 519.98 | 519.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 10:15:00 | 518.20 | 519.63 | 519.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 10:30:00 | 518.00 | 519.63 | 519.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-06-20 11:15:00 | 516.90 | 519.08 | 519.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2023-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 11:15:00 | 516.90 | 519.08 | 519.33 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 09:15:00 | 523.70 | 519.81 | 519.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 10:15:00 | 524.10 | 520.67 | 519.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 11:15:00 | 529.45 | 530.81 | 526.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-22 12:00:00 | 529.45 | 530.81 | 526.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 12:15:00 | 525.15 | 529.68 | 526.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 13:00:00 | 525.15 | 529.68 | 526.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 13:15:00 | 526.55 | 529.05 | 526.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 14:15:00 | 524.40 | 529.05 | 526.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 14:15:00 | 524.90 | 528.22 | 526.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 14:30:00 | 525.80 | 528.22 | 526.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 15:15:00 | 524.85 | 527.55 | 526.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 09:15:00 | 516.05 | 527.55 | 526.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2023-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 09:15:00 | 510.80 | 524.20 | 524.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 10:15:00 | 508.35 | 521.03 | 523.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 14:15:00 | 507.30 | 507.07 | 511.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 15:00:00 | 507.30 | 507.07 | 511.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 514.80 | 508.68 | 511.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:00:00 | 514.80 | 508.68 | 511.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 512.40 | 509.43 | 511.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 11:15:00 | 514.85 | 509.43 | 511.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 12:15:00 | 508.20 | 509.52 | 511.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-28 12:15:00 | 507.10 | 508.60 | 510.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-28 13:15:00 | 507.25 | 508.51 | 509.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-30 09:45:00 | 506.40 | 507.37 | 508.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 09:30:00 | 507.00 | 505.39 | 506.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 10:15:00 | 507.00 | 505.71 | 506.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-03 11:00:00 | 507.00 | 505.71 | 506.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 11:15:00 | 505.40 | 505.65 | 506.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 14:00:00 | 504.65 | 505.42 | 506.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 14:45:00 | 504.85 | 505.08 | 506.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-04 09:15:00 | 504.20 | 505.07 | 506.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 09:15:00 | 481.75 | 487.72 | 493.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 09:15:00 | 481.89 | 487.72 | 493.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 09:15:00 | 481.08 | 487.72 | 493.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 09:15:00 | 481.65 | 487.72 | 493.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-07-06 15:15:00 | 486.95 | 486.28 | 489.91 | SL hit (close>ema200) qty=0.50 sl=486.28 alert=retest2 |

### Cycle 11 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 462.25 | 453.97 | 453.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 11:15:00 | 465.90 | 457.71 | 455.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 14:15:00 | 466.25 | 467.96 | 466.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 14:15:00 | 466.25 | 467.96 | 466.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 14:15:00 | 466.25 | 467.96 | 466.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 14:45:00 | 466.30 | 467.96 | 466.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 15:15:00 | 466.05 | 467.57 | 466.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 09:15:00 | 463.55 | 467.57 | 466.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 465.40 | 467.14 | 466.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 13:30:00 | 471.15 | 467.25 | 466.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-24 12:45:00 | 468.60 | 467.99 | 467.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-25 09:15:00 | 462.50 | 466.05 | 466.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 09:15:00 | 462.50 | 466.05 | 466.52 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 11:15:00 | 465.65 | 464.95 | 464.90 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-07-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 13:15:00 | 462.30 | 464.66 | 464.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-27 14:15:00 | 460.45 | 463.82 | 464.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 09:15:00 | 464.35 | 461.84 | 462.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 09:15:00 | 464.35 | 461.84 | 462.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 464.35 | 461.84 | 462.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 10:00:00 | 464.35 | 461.84 | 462.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 10:15:00 | 463.55 | 462.18 | 462.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 10:30:00 | 465.40 | 462.18 | 462.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2023-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 12:15:00 | 467.70 | 463.81 | 463.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 09:15:00 | 473.75 | 467.23 | 465.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 11:15:00 | 474.75 | 475.27 | 471.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-02 12:00:00 | 474.75 | 475.27 | 471.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 468.30 | 473.51 | 471.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 14:00:00 | 468.30 | 473.51 | 471.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 471.45 | 473.10 | 471.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 09:15:00 | 473.20 | 472.68 | 471.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 11:15:00 | 473.95 | 472.32 | 471.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 15:00:00 | 473.50 | 472.17 | 471.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-04 13:45:00 | 473.05 | 473.86 | 472.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 14:15:00 | 472.75 | 473.64 | 472.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-04 15:00:00 | 472.75 | 473.64 | 472.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 15:15:00 | 473.00 | 473.51 | 472.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 09:15:00 | 474.70 | 473.51 | 472.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 472.50 | 473.31 | 472.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 10:00:00 | 472.50 | 473.31 | 472.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 10:15:00 | 475.05 | 473.66 | 473.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-07 11:15:00 | 476.55 | 473.66 | 473.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-07 14:45:00 | 475.45 | 475.07 | 474.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-08 09:15:00 | 467.90 | 473.42 | 473.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2023-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 09:15:00 | 467.90 | 473.42 | 473.44 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 09:15:00 | 478.10 | 473.60 | 473.24 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 11:15:00 | 470.15 | 472.74 | 472.90 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 15:15:00 | 474.45 | 472.95 | 472.91 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-08-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 09:15:00 | 468.85 | 472.13 | 472.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-10 13:15:00 | 463.00 | 468.09 | 470.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 11:15:00 | 462.10 | 459.86 | 462.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-14 12:00:00 | 462.10 | 459.86 | 462.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 12:15:00 | 460.70 | 460.03 | 462.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 09:15:00 | 458.95 | 461.49 | 462.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 11:15:00 | 460.45 | 462.09 | 462.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 11:45:00 | 459.80 | 461.58 | 462.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 12:45:00 | 458.90 | 460.97 | 461.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 12:15:00 | 452.20 | 451.31 | 454.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 12:30:00 | 454.15 | 451.31 | 454.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 14:15:00 | 455.00 | 452.33 | 454.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 15:00:00 | 455.00 | 452.33 | 454.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 15:15:00 | 453.65 | 452.60 | 454.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 09:15:00 | 455.70 | 452.60 | 454.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 09:15:00 | 455.25 | 453.13 | 454.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 09:45:00 | 455.75 | 453.13 | 454.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 10:15:00 | 454.65 | 453.43 | 454.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-22 12:00:00 | 453.25 | 453.40 | 454.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-23 10:15:00 | 456.40 | 453.85 | 453.97 | SL hit (close>static) qty=1.00 sl=455.50 alert=retest2 |

### Cycle 21 — BUY (started 2023-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 11:15:00 | 457.70 | 454.62 | 454.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 12:15:00 | 460.35 | 455.76 | 454.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 14:15:00 | 458.80 | 460.63 | 458.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 14:15:00 | 458.80 | 460.63 | 458.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 14:15:00 | 458.80 | 460.63 | 458.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 15:00:00 | 458.80 | 460.63 | 458.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 459.50 | 460.40 | 458.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 09:15:00 | 461.70 | 460.40 | 458.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 459.10 | 460.14 | 458.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-28 11:00:00 | 462.10 | 459.48 | 458.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-29 09:15:00 | 466.20 | 459.83 | 459.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-05 09:15:00 | 508.31 | 501.29 | 495.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 507.05 | 515.34 | 515.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 14:15:00 | 497.85 | 505.91 | 510.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 11:15:00 | 518.50 | 506.66 | 509.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 11:15:00 | 518.50 | 506.66 | 509.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 11:15:00 | 518.50 | 506.66 | 509.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 12:00:00 | 518.50 | 506.66 | 509.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 12:15:00 | 518.35 | 508.99 | 509.90 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 13:15:00 | 528.45 | 512.89 | 511.59 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 14:15:00 | 512.55 | 516.33 | 516.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 09:15:00 | 511.40 | 514.77 | 516.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-18 11:15:00 | 515.40 | 514.38 | 515.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 11:15:00 | 515.40 | 514.38 | 515.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 11:15:00 | 515.40 | 514.38 | 515.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-18 12:00:00 | 515.40 | 514.38 | 515.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 12:15:00 | 515.60 | 514.63 | 515.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-18 13:00:00 | 515.60 | 514.63 | 515.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 13:15:00 | 515.25 | 514.75 | 515.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-18 13:30:00 | 514.70 | 514.75 | 515.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 14:15:00 | 511.15 | 514.03 | 515.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-20 09:30:00 | 510.60 | 513.06 | 514.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 14:15:00 | 485.07 | 491.55 | 494.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-09-29 11:15:00 | 491.00 | 490.72 | 493.18 | SL hit (close>ema200) qty=0.50 sl=490.72 alert=retest2 |

### Cycle 25 — BUY (started 2023-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 11:15:00 | 484.80 | 480.95 | 480.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 12:15:00 | 486.40 | 484.09 | 482.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 10:15:00 | 485.85 | 486.05 | 484.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 10:15:00 | 485.85 | 486.05 | 484.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 10:15:00 | 485.85 | 486.05 | 484.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 10:45:00 | 484.95 | 486.05 | 484.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 14:15:00 | 486.20 | 486.54 | 485.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 14:45:00 | 485.70 | 486.54 | 485.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 15:15:00 | 487.00 | 486.64 | 485.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 09:15:00 | 486.60 | 486.64 | 485.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 488.75 | 487.06 | 485.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 10:15:00 | 490.70 | 487.06 | 485.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 11:00:00 | 489.10 | 487.47 | 485.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 10:30:00 | 489.25 | 490.00 | 488.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 15:00:00 | 488.95 | 489.60 | 488.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 15:15:00 | 488.10 | 489.30 | 488.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 09:15:00 | 488.75 | 489.30 | 488.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 491.00 | 489.64 | 488.82 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-10-18 10:15:00 | 484.50 | 488.61 | 488.43 | SL hit (close<static) qty=1.00 sl=484.65 alert=retest2 |

### Cycle 26 — SELL (started 2023-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 11:15:00 | 484.90 | 487.87 | 488.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 481.10 | 485.04 | 486.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 449.00 | 445.41 | 451.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-27 09:45:00 | 448.00 | 445.41 | 451.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 452.70 | 446.87 | 451.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 11:00:00 | 452.70 | 446.87 | 451.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 11:15:00 | 451.65 | 447.83 | 451.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 12:15:00 | 450.75 | 447.83 | 451.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 12:45:00 | 449.95 | 448.21 | 451.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-27 14:15:00 | 453.15 | 449.68 | 451.67 | SL hit (close>static) qty=1.00 sl=453.00 alert=retest2 |

### Cycle 27 — BUY (started 2023-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 12:15:00 | 454.95 | 452.79 | 452.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 13:15:00 | 457.40 | 453.72 | 453.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 11:15:00 | 455.50 | 455.89 | 454.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-31 12:00:00 | 455.50 | 455.89 | 454.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 13:15:00 | 456.45 | 455.96 | 454.85 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-11-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 10:15:00 | 451.00 | 454.10 | 454.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 11:15:00 | 449.95 | 453.27 | 453.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 09:15:00 | 453.75 | 451.31 | 452.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 09:15:00 | 453.75 | 451.31 | 452.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 453.75 | 451.31 | 452.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 09:45:00 | 454.10 | 451.31 | 452.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 10:15:00 | 452.75 | 451.60 | 452.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:15:00 | 453.70 | 451.60 | 452.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 454.45 | 452.17 | 452.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 12:00:00 | 454.45 | 452.17 | 452.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2023-11-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 12:15:00 | 456.60 | 453.05 | 453.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 09:15:00 | 461.80 | 456.12 | 454.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 15:15:00 | 459.70 | 460.03 | 457.65 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 09:15:00 | 492.00 | 460.03 | 457.65 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 11:15:00 | 516.60 | 507.67 | 497.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-11-09 09:15:00 | 510.90 | 511.30 | 503.69 | SL hit (close<ema200) qty=0.50 sl=511.30 alert=retest1 |

### Cycle 30 — SELL (started 2023-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 14:15:00 | 516.05 | 522.24 | 522.96 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-11-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 09:15:00 | 528.80 | 522.76 | 522.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 11:15:00 | 530.00 | 524.96 | 523.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 14:15:00 | 531.45 | 531.56 | 528.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-24 15:00:00 | 531.45 | 531.56 | 528.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 529.10 | 530.94 | 528.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 09:45:00 | 528.75 | 530.94 | 528.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 530.15 | 530.78 | 528.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-28 12:45:00 | 532.50 | 530.92 | 529.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-29 09:30:00 | 532.20 | 531.16 | 529.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-29 10:45:00 | 531.60 | 532.94 | 530.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-06 09:15:00 | 585.75 | 568.01 | 562.87 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2023-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 13:15:00 | 566.00 | 570.49 | 571.10 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 11:15:00 | 572.30 | 571.31 | 571.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 09:15:00 | 575.00 | 572.82 | 572.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 12:15:00 | 570.55 | 573.14 | 572.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 12:15:00 | 570.55 | 573.14 | 572.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 12:15:00 | 570.55 | 573.14 | 572.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 12:45:00 | 569.60 | 573.14 | 572.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 13:15:00 | 572.20 | 572.95 | 572.46 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 14:15:00 | 567.55 | 571.87 | 572.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 11:15:00 | 564.40 | 568.62 | 570.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 13:15:00 | 569.40 | 568.73 | 570.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-13 14:00:00 | 569.40 | 568.73 | 570.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 14:15:00 | 570.70 | 569.12 | 570.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 14:45:00 | 571.40 | 569.12 | 570.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 15:15:00 | 569.85 | 569.27 | 570.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:15:00 | 572.85 | 569.27 | 570.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 569.40 | 569.29 | 570.01 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-12-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 11:15:00 | 574.45 | 570.60 | 570.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 586.90 | 575.46 | 573.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 12:15:00 | 600.40 | 606.45 | 601.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 12:15:00 | 600.40 | 606.45 | 601.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 600.40 | 606.45 | 601.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:00:00 | 600.40 | 606.45 | 601.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 585.20 | 602.20 | 600.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:30:00 | 591.80 | 602.20 | 600.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 573.20 | 596.40 | 597.90 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 11:15:00 | 600.90 | 594.68 | 594.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 09:15:00 | 613.15 | 599.94 | 597.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 13:15:00 | 639.25 | 641.01 | 632.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-28 13:30:00 | 639.60 | 641.01 | 632.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 651.00 | 651.23 | 647.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:30:00 | 649.50 | 651.23 | 647.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 652.40 | 653.82 | 650.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 09:30:00 | 646.00 | 653.82 | 650.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 10:15:00 | 655.30 | 654.12 | 651.14 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-01-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 11:15:00 | 644.50 | 650.81 | 651.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-05 10:15:00 | 636.60 | 644.74 | 647.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 12:15:00 | 606.30 | 605.81 | 616.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-09 13:00:00 | 606.30 | 605.81 | 616.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 615.80 | 606.10 | 608.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 10:00:00 | 615.80 | 606.10 | 608.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 10:15:00 | 614.25 | 607.73 | 609.47 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-01-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 12:15:00 | 615.75 | 610.64 | 610.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-11 13:15:00 | 617.70 | 612.05 | 611.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-12 13:15:00 | 618.15 | 619.57 | 616.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-12 14:00:00 | 618.15 | 619.57 | 616.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 14:15:00 | 616.65 | 618.98 | 616.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-12 15:00:00 | 616.65 | 618.98 | 616.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 15:15:00 | 615.00 | 618.19 | 616.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 09:15:00 | 613.05 | 618.19 | 616.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 605.65 | 615.68 | 615.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 10:00:00 | 605.65 | 615.68 | 615.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 10:15:00 | 611.55 | 614.85 | 614.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 11:15:00 | 603.80 | 610.22 | 612.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 09:15:00 | 611.60 | 595.44 | 600.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 09:15:00 | 611.60 | 595.44 | 600.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 611.60 | 595.44 | 600.16 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-01-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 11:15:00 | 617.50 | 603.89 | 603.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 09:15:00 | 647.75 | 616.24 | 609.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 09:15:00 | 666.00 | 668.36 | 654.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-23 10:00:00 | 666.00 | 668.36 | 654.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 11:15:00 | 653.50 | 664.55 | 655.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 12:00:00 | 653.50 | 664.55 | 655.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 12:15:00 | 649.20 | 661.48 | 654.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 13:00:00 | 649.20 | 661.48 | 654.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 13:15:00 | 645.70 | 658.33 | 653.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 13:45:00 | 644.80 | 658.33 | 653.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 10:15:00 | 645.85 | 650.78 | 651.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 11:15:00 | 640.00 | 648.62 | 650.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-29 10:15:00 | 639.00 | 638.66 | 642.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-29 11:00:00 | 639.00 | 638.66 | 642.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 11:15:00 | 641.75 | 639.28 | 642.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 11:30:00 | 642.45 | 639.28 | 642.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 12:15:00 | 642.75 | 639.97 | 642.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 12:30:00 | 642.60 | 639.97 | 642.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 13:15:00 | 642.75 | 640.53 | 642.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 14:00:00 | 642.75 | 640.53 | 642.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 14:15:00 | 642.25 | 640.87 | 642.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-30 10:45:00 | 637.20 | 640.49 | 641.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-30 14:30:00 | 635.25 | 640.03 | 641.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-31 10:15:00 | 647.35 | 640.59 | 641.09 | SL hit (close>static) qty=1.00 sl=646.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 11:15:00 | 651.80 | 642.83 | 642.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 12:15:00 | 653.35 | 644.94 | 643.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 09:15:00 | 649.00 | 651.16 | 647.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-01 10:00:00 | 649.00 | 651.16 | 647.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 10:15:00 | 646.75 | 650.28 | 647.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 11:00:00 | 646.75 | 650.28 | 647.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 643.60 | 648.94 | 646.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:00:00 | 643.60 | 648.94 | 646.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 12:15:00 | 644.85 | 648.13 | 646.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:30:00 | 643.40 | 648.13 | 646.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 14:15:00 | 643.60 | 646.49 | 646.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 15:15:00 | 642.90 | 646.49 | 646.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 15:15:00 | 642.90 | 645.77 | 645.81 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 09:15:00 | 647.90 | 646.20 | 646.00 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 15:15:00 | 644.20 | 646.05 | 646.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 11:15:00 | 642.40 | 644.65 | 645.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-05 12:15:00 | 648.80 | 645.48 | 645.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 12:15:00 | 648.80 | 645.48 | 645.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 12:15:00 | 648.80 | 645.48 | 645.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-05 13:00:00 | 648.80 | 645.48 | 645.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-05 13:15:00 | 650.00 | 646.38 | 646.09 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 15:15:00 | 642.65 | 645.42 | 645.68 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 09:15:00 | 662.25 | 648.78 | 647.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 12:15:00 | 665.80 | 660.45 | 655.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 09:15:00 | 658.90 | 663.15 | 659.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 09:15:00 | 658.90 | 663.15 | 659.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 658.90 | 663.15 | 659.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 10:00:00 | 658.90 | 663.15 | 659.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 648.80 | 660.28 | 658.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 11:00:00 | 648.80 | 660.28 | 658.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 11:15:00 | 653.20 | 658.86 | 657.67 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-02-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 13:15:00 | 638.20 | 653.18 | 655.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 14:15:00 | 632.75 | 649.09 | 653.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 09:15:00 | 659.85 | 649.12 | 652.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 659.85 | 649.12 | 652.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 659.85 | 649.12 | 652.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-09 10:00:00 | 659.85 | 649.12 | 652.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 647.75 | 648.84 | 651.92 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2024-02-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 13:15:00 | 660.50 | 653.85 | 653.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-09 15:15:00 | 663.05 | 656.98 | 655.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 14:15:00 | 671.05 | 672.49 | 665.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-12 15:00:00 | 671.05 | 672.49 | 665.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 659.00 | 670.18 | 665.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 09:30:00 | 657.85 | 670.18 | 665.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 10:15:00 | 666.60 | 669.47 | 665.57 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2024-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 09:15:00 | 658.55 | 663.20 | 663.53 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 15:15:00 | 669.00 | 663.67 | 663.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 684.30 | 667.80 | 665.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 09:15:00 | 691.15 | 692.86 | 685.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 09:15:00 | 691.15 | 692.86 | 685.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 691.15 | 692.86 | 685.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 09:30:00 | 684.45 | 692.86 | 685.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 13:15:00 | 689.80 | 691.25 | 686.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 09:15:00 | 696.95 | 690.42 | 687.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 15:15:00 | 686.00 | 694.52 | 693.68 | SL hit (close<static) qty=1.00 sl=686.55 alert=retest2 |

### Cycle 54 — SELL (started 2024-02-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 09:15:00 | 686.35 | 692.89 | 693.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-23 13:15:00 | 676.20 | 680.98 | 685.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-26 12:15:00 | 677.15 | 676.52 | 680.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-26 13:00:00 | 677.15 | 676.52 | 680.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 673.05 | 674.71 | 678.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 10:45:00 | 671.90 | 674.03 | 677.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 13:30:00 | 672.20 | 672.71 | 676.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 09:30:00 | 670.30 | 670.54 | 674.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 10:15:00 | 671.80 | 658.52 | 660.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-01 10:15:00 | 675.20 | 661.86 | 661.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2024-03-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 10:15:00 | 675.20 | 661.86 | 661.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 11:15:00 | 677.25 | 664.94 | 663.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 09:15:00 | 668.95 | 671.84 | 668.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 09:15:00 | 668.95 | 671.84 | 668.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 668.95 | 671.84 | 668.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 09:45:00 | 668.45 | 671.84 | 668.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 10:15:00 | 668.80 | 671.24 | 668.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 11:00:00 | 668.80 | 671.24 | 668.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 11:15:00 | 665.20 | 670.03 | 668.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 12:00:00 | 665.20 | 670.03 | 668.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 12:15:00 | 662.90 | 668.60 | 667.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 13:00:00 | 662.90 | 668.60 | 667.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2024-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 13:15:00 | 662.05 | 667.29 | 667.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 646.60 | 660.19 | 663.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 13:15:00 | 652.60 | 652.25 | 657.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-06 14:00:00 | 652.60 | 652.25 | 657.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 662.85 | 654.70 | 657.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 10:00:00 | 662.85 | 654.70 | 657.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 10:15:00 | 660.50 | 655.86 | 657.89 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 14:15:00 | 665.25 | 660.00 | 659.39 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 11:15:00 | 655.35 | 658.55 | 658.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 10:15:00 | 653.65 | 656.09 | 657.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 638.25 | 631.72 | 640.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 09:15:00 | 638.25 | 631.72 | 640.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 638.25 | 631.72 | 640.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 10:00:00 | 638.25 | 631.72 | 640.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 641.45 | 633.67 | 640.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 10:45:00 | 644.65 | 633.67 | 640.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 644.50 | 635.83 | 640.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 12:00:00 | 644.50 | 635.83 | 640.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 12:15:00 | 642.85 | 637.24 | 640.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 12:30:00 | 640.20 | 637.24 | 640.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 13:15:00 | 643.50 | 638.49 | 641.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 09:15:00 | 639.35 | 641.01 | 641.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 12:15:00 | 640.10 | 637.63 | 638.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 13:15:00 | 641.30 | 638.40 | 638.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 14:00:00 | 641.10 | 638.94 | 638.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-18 14:15:00 | 645.15 | 640.18 | 639.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2024-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 14:15:00 | 645.15 | 640.18 | 639.51 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 11:15:00 | 635.55 | 638.80 | 639.08 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-19 13:15:00 | 640.80 | 639.39 | 639.31 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 14:15:00 | 635.15 | 638.55 | 638.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 15:15:00 | 634.10 | 637.66 | 638.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 12:15:00 | 640.05 | 637.40 | 638.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 12:15:00 | 640.05 | 637.40 | 638.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 640.05 | 637.40 | 638.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 13:00:00 | 640.05 | 637.40 | 638.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 13:15:00 | 636.40 | 637.20 | 637.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-20 14:45:00 | 634.90 | 636.84 | 637.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 09:15:00 | 644.15 | 637.91 | 637.98 | SL hit (close>static) qty=1.00 sl=640.30 alert=retest2 |

### Cycle 63 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 648.50 | 640.03 | 638.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 11:15:00 | 651.10 | 647.68 | 644.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 11:15:00 | 655.00 | 655.85 | 652.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-27 12:00:00 | 655.00 | 655.85 | 652.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 658.85 | 656.62 | 653.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 15:00:00 | 658.85 | 656.62 | 653.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 654.45 | 656.47 | 654.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 10:00:00 | 654.45 | 656.47 | 654.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 10:15:00 | 658.40 | 656.86 | 654.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 10:30:00 | 656.70 | 656.86 | 654.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 685.05 | 684.81 | 679.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 09:30:00 | 680.55 | 684.81 | 679.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 12:15:00 | 693.00 | 690.99 | 687.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-05 10:15:00 | 696.15 | 690.69 | 688.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-05 13:15:00 | 694.00 | 692.69 | 689.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 10:15:00 | 694.35 | 693.98 | 691.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-18 14:15:00 | 722.95 | 739.99 | 740.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-04-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 14:15:00 | 722.95 | 739.99 | 740.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 09:15:00 | 714.40 | 733.28 | 737.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-23 09:15:00 | 738.65 | 726.57 | 728.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 09:15:00 | 738.65 | 726.57 | 728.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 09:15:00 | 738.65 | 726.57 | 728.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 10:00:00 | 738.65 | 726.57 | 728.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2024-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 10:15:00 | 743.00 | 729.86 | 729.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 09:15:00 | 749.65 | 740.14 | 735.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 15:15:00 | 745.25 | 746.64 | 741.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 09:15:00 | 745.25 | 746.64 | 741.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 741.00 | 745.51 | 741.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 09:45:00 | 744.10 | 745.51 | 741.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 740.75 | 744.56 | 741.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:45:00 | 738.95 | 744.56 | 741.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 11:15:00 | 743.90 | 744.43 | 741.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 12:30:00 | 745.80 | 744.65 | 742.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-30 12:15:00 | 749.10 | 751.88 | 752.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 12:15:00 | 749.10 | 751.88 | 752.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 14:15:00 | 742.30 | 749.51 | 751.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 09:15:00 | 746.55 | 742.59 | 745.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 09:15:00 | 746.55 | 742.59 | 745.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 09:15:00 | 746.55 | 742.59 | 745.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-03 09:30:00 | 754.50 | 742.59 | 745.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 10:15:00 | 742.15 | 742.50 | 745.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 11:15:00 | 738.55 | 742.50 | 745.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 12:45:00 | 737.00 | 740.67 | 743.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-06 12:15:00 | 746.35 | 743.87 | 743.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2024-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 12:15:00 | 746.35 | 743.87 | 743.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-06 14:15:00 | 750.30 | 745.68 | 744.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 09:15:00 | 738.35 | 744.37 | 744.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 09:15:00 | 738.35 | 744.37 | 744.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 738.35 | 744.37 | 744.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 10:00:00 | 738.35 | 744.37 | 744.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2024-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 10:15:00 | 721.40 | 739.78 | 742.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 11:15:00 | 715.40 | 734.90 | 739.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 15:15:00 | 675.00 | 672.09 | 684.62 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 09:30:00 | 659.90 | 669.11 | 682.13 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 14:15:00 | 669.35 | 667.21 | 676.83 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 15:00:00 | 668.45 | 667.46 | 676.07 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 657.90 | 665.79 | 673.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 11:45:00 | 655.45 | 662.37 | 670.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 14:00:00 | 655.35 | 659.78 | 668.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 09:15:00 | 635.88 | 652.85 | 662.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 09:15:00 | 635.03 | 652.85 | 662.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 10:15:00 | 626.90 | 647.52 | 659.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 10:15:00 | 622.68 | 647.52 | 659.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 10:15:00 | 622.58 | 647.52 | 659.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-16 10:15:00 | 632.00 | 630.83 | 642.94 | SL hit (close>ema200) qty=0.50 sl=630.83 alert=retest1 |

### Cycle 69 — BUY (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 09:15:00 | 638.55 | 629.69 | 629.01 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 14:15:00 | 620.85 | 629.17 | 629.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 616.90 | 625.37 | 627.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 09:15:00 | 626.35 | 620.70 | 623.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 09:15:00 | 626.35 | 620.70 | 623.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 626.35 | 620.70 | 623.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:00:00 | 626.35 | 620.70 | 623.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 627.40 | 622.04 | 623.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:00:00 | 627.40 | 622.04 | 623.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 624.50 | 622.88 | 623.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:45:00 | 624.30 | 622.88 | 623.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 626.00 | 623.51 | 623.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:00:00 | 626.00 | 623.51 | 623.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2024-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 14:15:00 | 629.35 | 624.68 | 624.46 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 10:15:00 | 618.80 | 624.95 | 625.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 11:15:00 | 616.25 | 623.21 | 624.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 10:15:00 | 617.00 | 616.85 | 620.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 11:00:00 | 617.00 | 616.85 | 620.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 625.50 | 616.97 | 618.72 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 626.20 | 620.35 | 620.04 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 588.95 | 615.80 | 618.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 581.15 | 608.87 | 615.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 615.00 | 605.10 | 610.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 615.00 | 605.10 | 610.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 615.00 | 605.10 | 610.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 615.00 | 605.10 | 610.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 620.00 | 608.08 | 611.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:45:00 | 619.85 | 608.08 | 611.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 626.80 | 611.82 | 612.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:00:00 | 626.80 | 611.82 | 612.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 626.90 | 614.84 | 613.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 14:15:00 | 628.50 | 619.43 | 616.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 10:15:00 | 633.80 | 634.40 | 628.59 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 11:15:00 | 636.45 | 634.40 | 628.59 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 14:30:00 | 636.30 | 634.04 | 630.22 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 09:15:00 | 638.45 | 634.23 | 630.66 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 14:15:00 | 668.27 | 650.63 | 641.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 14:15:00 | 668.12 | 650.63 | 641.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 14:15:00 | 670.37 | 650.63 | 641.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-06-11 14:15:00 | 657.75 | 659.82 | 651.54 | SL hit (close<ema200) qty=0.50 sl=659.82 alert=retest1 |

### Cycle 76 — SELL (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 12:15:00 | 697.25 | 702.41 | 702.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 13:15:00 | 695.20 | 700.97 | 701.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 15:15:00 | 689.35 | 687.72 | 691.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 09:15:00 | 693.90 | 687.72 | 691.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 690.50 | 688.28 | 691.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:30:00 | 694.90 | 688.28 | 691.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 689.05 | 688.43 | 691.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 12:15:00 | 685.75 | 688.37 | 691.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 13:45:00 | 687.10 | 688.25 | 690.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 15:00:00 | 685.60 | 687.72 | 690.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 09:15:00 | 693.75 | 688.82 | 690.18 | SL hit (close>static) qty=1.00 sl=692.15 alert=retest2 |

### Cycle 77 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 701.95 | 693.00 | 691.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 13:15:00 | 704.00 | 696.28 | 693.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 15:15:00 | 708.95 | 708.95 | 703.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 09:45:00 | 707.10 | 709.30 | 704.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 712.90 | 712.93 | 709.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:45:00 | 708.00 | 712.93 | 709.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 708.90 | 719.32 | 717.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:30:00 | 710.35 | 719.32 | 717.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 707.40 | 716.93 | 716.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 707.40 | 716.93 | 716.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 706.75 | 714.90 | 715.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 688.10 | 702.90 | 707.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 14:15:00 | 700.20 | 698.64 | 703.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-10 14:30:00 | 700.80 | 698.64 | 703.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 698.75 | 698.88 | 702.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:30:00 | 703.15 | 698.88 | 702.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 701.50 | 698.27 | 700.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 15:00:00 | 701.50 | 698.27 | 700.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 702.50 | 699.11 | 700.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 704.55 | 699.11 | 700.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 704.45 | 700.18 | 701.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:30:00 | 707.20 | 700.18 | 701.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 707.65 | 701.67 | 701.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:45:00 | 709.40 | 701.67 | 701.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 11:15:00 | 703.20 | 701.98 | 701.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 13:15:00 | 710.05 | 704.23 | 702.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 14:15:00 | 708.95 | 709.35 | 706.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 15:00:00 | 708.95 | 709.35 | 706.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 709.25 | 709.39 | 707.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:00:00 | 709.25 | 709.39 | 707.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 708.50 | 709.21 | 707.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:30:00 | 707.00 | 709.21 | 707.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 709.00 | 709.17 | 707.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:15:00 | 707.85 | 709.17 | 707.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 708.45 | 709.03 | 707.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:30:00 | 708.30 | 709.03 | 707.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 706.65 | 708.55 | 707.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:00:00 | 706.65 | 708.55 | 707.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 704.80 | 707.80 | 707.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:45:00 | 704.45 | 707.80 | 707.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 702.90 | 706.82 | 706.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 701.40 | 706.82 | 706.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 697.05 | 704.87 | 705.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 686.45 | 699.14 | 702.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 682.10 | 681.74 | 689.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:45:00 | 682.85 | 681.74 | 689.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 689.70 | 683.69 | 689.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:45:00 | 689.45 | 683.69 | 689.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 689.90 | 684.93 | 689.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 691.35 | 684.93 | 689.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 696.85 | 687.31 | 689.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 696.85 | 687.31 | 689.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 695.70 | 688.99 | 690.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 692.85 | 688.99 | 690.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 695.00 | 689.59 | 690.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:00:00 | 695.00 | 689.59 | 690.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 693.40 | 690.35 | 690.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 661.05 | 690.35 | 690.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 15:15:00 | 685.15 | 686.89 | 688.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 14:15:00 | 692.10 | 689.79 | 689.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2024-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 14:15:00 | 692.10 | 689.79 | 689.54 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 09:15:00 | 685.55 | 689.22 | 689.34 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 15:15:00 | 693.00 | 689.76 | 689.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 704.55 | 692.72 | 690.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 713.05 | 713.86 | 708.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 10:00:00 | 713.05 | 713.86 | 708.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 742.20 | 743.56 | 738.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:30:00 | 738.35 | 743.56 | 738.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 742.90 | 743.12 | 739.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 10:30:00 | 745.80 | 743.78 | 739.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 13:00:00 | 744.00 | 743.55 | 740.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 711.40 | 734.28 | 737.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 711.40 | 734.28 | 737.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 707.00 | 725.51 | 732.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 731.50 | 718.68 | 726.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 731.50 | 718.68 | 726.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 731.50 | 718.68 | 726.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 731.50 | 718.68 | 726.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 724.45 | 719.83 | 725.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 15:00:00 | 717.30 | 723.16 | 726.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 10:15:00 | 741.20 | 727.45 | 727.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 10:15:00 | 741.20 | 727.45 | 727.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 12:15:00 | 746.25 | 733.37 | 730.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 748.10 | 750.19 | 743.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 748.10 | 750.19 | 743.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 748.50 | 749.76 | 746.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 13:45:00 | 744.70 | 749.76 | 746.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 747.75 | 749.36 | 746.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:30:00 | 745.40 | 749.36 | 746.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 742.50 | 747.77 | 746.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:00:00 | 742.50 | 747.77 | 746.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 743.30 | 746.87 | 745.87 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 12:15:00 | 741.45 | 744.91 | 745.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 14:15:00 | 734.70 | 742.46 | 743.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 09:15:00 | 613.00 | 607.12 | 624.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-19 09:45:00 | 612.00 | 607.12 | 624.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 12:15:00 | 624.25 | 613.77 | 623.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 13:00:00 | 624.25 | 613.77 | 623.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 13:15:00 | 620.15 | 615.05 | 623.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 14:30:00 | 618.80 | 615.65 | 622.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 09:15:00 | 617.50 | 616.32 | 622.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 09:15:00 | 625.80 | 621.76 | 622.51 | SL hit (close>static) qty=1.00 sl=624.50 alert=retest2 |

### Cycle 87 — BUY (started 2024-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 11:15:00 | 627.45 | 623.58 | 623.25 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 12:15:00 | 620.65 | 622.90 | 623.14 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 15:15:00 | 624.55 | 623.34 | 623.29 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 09:15:00 | 621.65 | 623.00 | 623.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 13:15:00 | 618.70 | 621.39 | 622.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 625.05 | 621.91 | 622.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 625.05 | 621.91 | 622.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 625.05 | 621.91 | 622.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:45:00 | 627.00 | 621.91 | 622.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 623.10 | 622.15 | 622.36 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2024-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 11:15:00 | 626.35 | 622.99 | 622.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 12:15:00 | 627.00 | 623.79 | 623.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 14:15:00 | 641.00 | 642.44 | 637.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 15:00:00 | 641.00 | 642.44 | 637.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 633.65 | 640.29 | 637.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:00:00 | 633.65 | 640.29 | 637.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 629.35 | 638.10 | 636.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 11:00:00 | 629.35 | 638.10 | 636.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 12:15:00 | 627.70 | 634.36 | 635.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 624.05 | 632.30 | 634.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 09:15:00 | 632.00 | 629.53 | 630.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 09:15:00 | 632.00 | 629.53 | 630.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 632.00 | 629.53 | 630.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 10:00:00 | 632.00 | 629.53 | 630.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 630.10 | 629.65 | 630.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 11:30:00 | 628.65 | 629.53 | 630.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 10:15:00 | 627.80 | 628.28 | 629.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 09:15:00 | 597.22 | 623.79 | 626.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 09:15:00 | 596.41 | 623.79 | 626.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-05 09:15:00 | 619.35 | 615.64 | 620.08 | SL hit (close>ema200) qty=0.50 sl=615.64 alert=retest2 |

### Cycle 93 — BUY (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 09:15:00 | 569.15 | 566.49 | 566.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 11:15:00 | 574.60 | 569.07 | 567.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 581.20 | 584.06 | 579.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 581.20 | 584.06 | 579.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 581.20 | 584.06 | 579.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 581.20 | 584.06 | 579.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 581.35 | 584.65 | 582.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 580.95 | 584.65 | 582.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 581.70 | 584.06 | 582.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:00:00 | 581.70 | 584.06 | 582.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 581.60 | 583.57 | 582.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:15:00 | 581.05 | 583.57 | 582.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 578.20 | 582.49 | 581.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:00:00 | 578.20 | 582.49 | 581.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 13:15:00 | 576.50 | 581.29 | 581.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:45:00 | 576.20 | 581.29 | 581.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 582.00 | 581.77 | 581.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 09:15:00 | 588.30 | 581.77 | 581.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 13:30:00 | 585.00 | 585.06 | 583.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 584.80 | 584.58 | 583.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 10:15:00 | 581.00 | 584.24 | 583.68 | SL hit (close<static) qty=1.00 sl=581.45 alert=retest2 |

### Cycle 94 — SELL (started 2024-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 13:15:00 | 580.65 | 583.44 | 583.46 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 09:15:00 | 586.70 | 583.91 | 583.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 11:15:00 | 590.00 | 585.32 | 584.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 579.00 | 585.18 | 584.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 579.00 | 585.18 | 584.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 579.00 | 585.18 | 584.84 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 577.20 | 583.58 | 584.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 570.65 | 581.00 | 582.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 538.10 | 537.98 | 549.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:45:00 | 537.45 | 537.98 | 549.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 528.45 | 529.27 | 531.42 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2024-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 15:15:00 | 533.50 | 532.28 | 532.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 09:15:00 | 536.30 | 533.08 | 532.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 534.90 | 536.29 | 534.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 09:15:00 | 534.90 | 536.29 | 534.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 534.90 | 536.29 | 534.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:00:00 | 534.90 | 536.29 | 534.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 533.40 | 535.71 | 534.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:45:00 | 533.20 | 535.71 | 534.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 531.55 | 534.88 | 534.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:30:00 | 530.60 | 534.88 | 534.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 13:15:00 | 534.10 | 534.65 | 534.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 13:30:00 | 533.90 | 534.65 | 534.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 533.85 | 534.49 | 534.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 15:00:00 | 533.85 | 534.49 | 534.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 535.00 | 534.59 | 534.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:15:00 | 532.60 | 534.59 | 534.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 527.15 | 533.11 | 533.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 522.10 | 528.17 | 530.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 523.00 | 519.97 | 524.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 13:15:00 | 523.00 | 519.97 | 524.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 523.00 | 519.97 | 524.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:00:00 | 523.00 | 519.97 | 524.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 523.00 | 521.26 | 524.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 520.95 | 521.26 | 524.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 519.80 | 520.97 | 523.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:45:00 | 515.95 | 519.47 | 522.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 490.15 | 499.56 | 508.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 10:15:00 | 497.55 | 496.67 | 504.69 | SL hit (close>ema200) qty=0.50 sl=496.67 alert=retest2 |

### Cycle 99 — BUY (started 2024-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 15:15:00 | 505.95 | 502.56 | 502.50 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 491.95 | 500.44 | 501.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 11:15:00 | 490.70 | 497.13 | 499.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 09:15:00 | 492.75 | 492.32 | 495.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 10:15:00 | 495.00 | 492.32 | 495.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 496.85 | 493.23 | 496.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:00:00 | 496.85 | 493.23 | 496.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 503.50 | 495.28 | 496.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 12:00:00 | 503.50 | 495.28 | 496.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 12:15:00 | 504.60 | 497.14 | 497.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 13:00:00 | 504.60 | 497.14 | 497.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 508.90 | 499.50 | 498.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 14:15:00 | 511.45 | 501.89 | 499.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 511.25 | 513.80 | 509.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 14:00:00 | 511.25 | 513.80 | 509.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 512.55 | 513.39 | 510.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 507.00 | 513.39 | 510.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 511.80 | 513.07 | 510.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:30:00 | 511.20 | 513.07 | 510.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 510.65 | 512.30 | 510.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:30:00 | 510.80 | 512.30 | 510.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 506.70 | 511.18 | 510.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 13:00:00 | 506.70 | 511.18 | 510.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 505.90 | 510.13 | 509.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 13:30:00 | 506.35 | 510.13 | 509.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 509.60 | 513.37 | 511.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 509.60 | 513.37 | 511.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 500.65 | 510.82 | 510.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 500.65 | 510.82 | 510.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 503.80 | 509.42 | 510.04 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 511.00 | 508.71 | 508.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 516.85 | 510.76 | 509.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 11:15:00 | 516.35 | 518.42 | 515.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 12:00:00 | 516.35 | 518.42 | 515.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 516.15 | 517.96 | 515.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:45:00 | 515.80 | 517.96 | 515.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 516.65 | 517.70 | 515.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:45:00 | 516.50 | 517.70 | 515.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 515.55 | 517.27 | 515.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:30:00 | 516.00 | 517.27 | 515.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 515.00 | 516.82 | 515.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 510.90 | 516.82 | 515.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 509.35 | 515.32 | 514.93 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 503.95 | 513.05 | 513.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 487.75 | 507.99 | 511.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 10:15:00 | 448.20 | 447.51 | 465.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 11:00:00 | 448.20 | 447.51 | 465.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 438.95 | 434.64 | 437.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 438.95 | 434.64 | 437.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 440.25 | 435.76 | 437.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 441.00 | 435.76 | 437.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 437.65 | 437.05 | 437.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 13:45:00 | 440.00 | 437.05 | 437.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 436.00 | 436.84 | 437.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 15:15:00 | 431.15 | 436.84 | 437.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 09:45:00 | 435.65 | 430.71 | 431.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 10:15:00 | 441.10 | 432.79 | 432.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 441.10 | 432.79 | 432.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 447.00 | 438.32 | 435.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 452.95 | 453.02 | 448.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 11:00:00 | 452.95 | 453.02 | 448.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 450.15 | 452.14 | 449.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:30:00 | 448.55 | 452.14 | 449.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 448.00 | 451.32 | 449.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:30:00 | 447.65 | 451.32 | 449.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 449.50 | 450.95 | 449.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:15:00 | 450.50 | 450.95 | 449.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 447.60 | 450.28 | 448.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:15:00 | 448.80 | 450.28 | 448.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 447.40 | 449.71 | 448.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 14:15:00 | 450.10 | 448.81 | 448.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 15:15:00 | 447.05 | 448.30 | 448.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 15:15:00 | 447.05 | 448.30 | 448.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 09:15:00 | 441.00 | 446.84 | 447.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 447.10 | 445.35 | 446.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 09:15:00 | 447.10 | 445.35 | 446.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 447.10 | 445.35 | 446.29 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2024-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 09:15:00 | 451.05 | 446.95 | 446.79 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 09:15:00 | 444.55 | 446.68 | 446.78 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 11:15:00 | 448.30 | 446.84 | 446.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 12:15:00 | 450.30 | 447.54 | 447.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 09:15:00 | 446.45 | 449.95 | 449.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 09:15:00 | 446.45 | 449.95 | 449.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 446.45 | 449.95 | 449.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 446.45 | 449.95 | 449.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 451.45 | 450.25 | 449.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 11:30:00 | 451.65 | 450.89 | 449.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 452.50 | 449.69 | 449.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 15:15:00 | 448.50 | 449.69 | 449.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 15:15:00 | 448.50 | 449.69 | 449.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 14:15:00 | 446.95 | 448.78 | 449.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 437.75 | 436.62 | 440.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 15:00:00 | 437.75 | 436.62 | 440.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 437.25 | 436.89 | 439.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:15:00 | 435.25 | 436.89 | 439.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:15:00 | 436.25 | 437.73 | 438.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:00:00 | 436.55 | 437.49 | 438.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 413.49 | 423.39 | 428.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 414.44 | 423.39 | 428.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 414.72 | 423.39 | 428.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 418.15 | 417.14 | 421.70 | SL hit (close>ema200) qty=0.50 sl=417.14 alert=retest2 |

### Cycle 111 — BUY (started 2024-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 15:15:00 | 415.50 | 414.09 | 413.94 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 10:15:00 | 409.60 | 413.01 | 413.46 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 417.85 | 413.61 | 413.36 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 14:15:00 | 411.25 | 413.25 | 413.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 12:15:00 | 410.20 | 411.89 | 412.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 10:15:00 | 411.10 | 410.22 | 411.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 10:15:00 | 411.10 | 410.22 | 411.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 411.10 | 410.22 | 411.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:00:00 | 411.10 | 410.22 | 411.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 410.60 | 410.30 | 411.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 15:15:00 | 410.00 | 410.85 | 411.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 09:15:00 | 413.00 | 411.14 | 411.37 | SL hit (close>static) qty=1.00 sl=411.50 alert=retest2 |

### Cycle 115 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 415.35 | 411.99 | 411.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 11:15:00 | 416.25 | 412.84 | 412.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 410.40 | 414.05 | 413.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 410.40 | 414.05 | 413.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 410.40 | 414.05 | 413.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:00:00 | 410.40 | 414.05 | 413.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 410.55 | 413.35 | 413.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:30:00 | 410.40 | 413.35 | 413.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 11:15:00 | 409.45 | 412.57 | 412.67 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 14:15:00 | 415.10 | 412.79 | 412.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 417.10 | 414.05 | 413.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 416.10 | 418.52 | 416.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 416.10 | 418.52 | 416.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 416.10 | 418.52 | 416.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 416.40 | 418.52 | 416.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 408.25 | 416.47 | 415.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 408.25 | 416.47 | 415.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 408.35 | 414.84 | 415.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 403.60 | 411.56 | 413.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 411.00 | 409.27 | 411.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 411.00 | 409.27 | 411.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 411.00 | 409.27 | 411.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 413.65 | 409.27 | 411.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 407.85 | 408.99 | 411.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:30:00 | 412.75 | 408.99 | 411.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 402.50 | 399.81 | 403.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:30:00 | 402.75 | 399.81 | 403.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 398.65 | 399.58 | 402.92 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2025-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 13:15:00 | 415.05 | 405.48 | 405.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-10 12:15:00 | 425.80 | 410.92 | 408.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 13:15:00 | 414.20 | 419.98 | 415.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-13 13:15:00 | 414.20 | 419.98 | 415.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 13:15:00 | 414.20 | 419.98 | 415.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 14:00:00 | 414.20 | 419.98 | 415.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 14:15:00 | 410.95 | 418.17 | 415.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 15:00:00 | 410.95 | 418.17 | 415.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 412.95 | 415.46 | 414.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-14 10:30:00 | 413.00 | 415.46 | 414.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 409.65 | 414.30 | 414.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-14 12:00:00 | 409.65 | 414.30 | 414.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 12:15:00 | 405.30 | 412.50 | 413.36 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 424.50 | 415.40 | 414.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 14:15:00 | 428.65 | 422.14 | 418.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 13:15:00 | 452.70 | 453.66 | 447.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 14:00:00 | 452.70 | 453.66 | 447.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 438.90 | 450.64 | 447.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 10:00:00 | 438.90 | 450.64 | 447.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 431.60 | 446.83 | 445.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:00:00 | 431.60 | 446.83 | 445.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 430.85 | 443.63 | 444.59 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 447.70 | 443.89 | 443.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 14:15:00 | 448.80 | 445.20 | 444.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 438.55 | 444.26 | 444.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 438.55 | 444.26 | 444.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 438.55 | 444.26 | 444.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 438.55 | 444.26 | 444.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 10:15:00 | 441.20 | 443.65 | 443.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 423.40 | 437.62 | 440.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 422.70 | 416.24 | 422.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 422.70 | 416.24 | 422.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 422.70 | 416.24 | 422.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 422.70 | 416.24 | 422.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 425.20 | 418.03 | 422.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 425.20 | 418.03 | 422.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 425.25 | 419.48 | 422.85 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 437.10 | 426.59 | 425.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 440.95 | 435.93 | 431.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 452.00 | 452.05 | 445.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 452.00 | 452.05 | 445.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 452.00 | 452.05 | 445.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:45:00 | 465.55 | 454.85 | 447.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 10:30:00 | 461.05 | 454.32 | 450.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 09:30:00 | 461.55 | 456.97 | 453.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 11:00:00 | 461.30 | 457.84 | 454.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 463.50 | 468.38 | 464.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:45:00 | 462.20 | 468.38 | 464.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 469.30 | 468.57 | 465.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:30:00 | 470.85 | 468.41 | 465.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 455.65 | 464.33 | 464.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 455.65 | 464.33 | 464.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 454.10 | 461.39 | 463.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 441.60 | 440.97 | 448.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:00:00 | 441.60 | 440.97 | 448.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 446.95 | 442.54 | 447.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 446.05 | 442.54 | 447.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 440.95 | 442.22 | 447.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 09:15:00 | 438.65 | 442.31 | 446.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 10:00:00 | 439.10 | 441.67 | 445.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:00:00 | 439.00 | 440.28 | 443.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 416.72 | 430.41 | 437.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 417.14 | 430.41 | 437.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 417.05 | 430.41 | 437.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 12:15:00 | 417.20 | 416.32 | 423.78 | SL hit (close>ema200) qty=0.50 sl=416.32 alert=retest2 |

### Cycle 127 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 423.25 | 417.89 | 417.21 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 12:15:00 | 416.25 | 418.34 | 418.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 13:15:00 | 413.90 | 417.45 | 417.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 408.35 | 408.25 | 411.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 09:45:00 | 409.50 | 408.25 | 411.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 385.30 | 379.84 | 383.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 385.30 | 379.84 | 383.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 388.60 | 381.59 | 384.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 388.60 | 381.59 | 384.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 385.35 | 383.39 | 384.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:00:00 | 385.35 | 383.39 | 384.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 381.75 | 383.06 | 384.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 15:15:00 | 381.20 | 383.06 | 384.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 392.40 | 384.63 | 384.75 | SL hit (close>static) qty=1.00 sl=386.00 alert=retest2 |

### Cycle 129 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 395.50 | 386.81 | 385.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 399.70 | 389.38 | 387.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 405.10 | 408.52 | 403.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 405.10 | 408.52 | 403.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 407.95 | 407.27 | 404.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 11:00:00 | 412.60 | 408.34 | 405.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 13:15:00 | 396.40 | 405.08 | 404.34 | SL hit (close<static) qty=1.00 sl=402.55 alert=retest2 |

### Cycle 130 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 391.65 | 402.39 | 403.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 389.95 | 399.90 | 401.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 14:15:00 | 391.55 | 390.40 | 393.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 15:00:00 | 391.55 | 390.40 | 393.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 390.95 | 390.83 | 393.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 13:00:00 | 386.70 | 389.41 | 392.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:00:00 | 385.80 | 388.69 | 391.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 12:45:00 | 386.55 | 387.22 | 387.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 13:30:00 | 387.00 | 387.41 | 388.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 388.95 | 387.72 | 388.09 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 396.20 | 389.61 | 388.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 396.20 | 389.61 | 388.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 10:15:00 | 401.45 | 391.98 | 390.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 399.00 | 399.29 | 396.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 09:15:00 | 401.60 | 399.29 | 396.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 403.20 | 406.92 | 404.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 403.20 | 406.92 | 404.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 405.10 | 406.56 | 404.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 402.05 | 406.56 | 404.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 402.20 | 405.69 | 404.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 402.50 | 405.69 | 404.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 401.30 | 404.81 | 404.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:30:00 | 400.70 | 404.81 | 404.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 14:15:00 | 397.95 | 403.04 | 403.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 09:15:00 | 395.95 | 400.67 | 402.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 396.10 | 392.47 | 394.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 396.10 | 392.47 | 394.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 396.10 | 392.47 | 394.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 396.10 | 392.47 | 394.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 395.10 | 392.99 | 394.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 403.50 | 392.99 | 394.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 400.05 | 395.52 | 395.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:30:00 | 401.40 | 395.52 | 395.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 398.35 | 396.09 | 396.08 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 12:15:00 | 393.15 | 395.50 | 395.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 13:15:00 | 391.85 | 394.77 | 395.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 395.80 | 393.58 | 394.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 395.80 | 393.58 | 394.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 395.80 | 393.58 | 394.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:45:00 | 396.40 | 393.58 | 394.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 390.75 | 393.02 | 394.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:30:00 | 389.10 | 392.69 | 394.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 12:45:00 | 389.60 | 392.05 | 393.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 14:15:00 | 390.25 | 391.77 | 393.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 385.85 | 391.69 | 393.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 386.65 | 390.68 | 392.44 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-02 12:15:00 | 397.60 | 392.88 | 393.07 | SL hit (close>static) qty=1.00 sl=396.85 alert=retest2 |

### Cycle 135 — BUY (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 13:15:00 | 397.75 | 393.85 | 393.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 400.55 | 395.19 | 394.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 12:15:00 | 393.60 | 396.79 | 395.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 12:15:00 | 393.60 | 396.79 | 395.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 393.60 | 396.79 | 395.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 13:00:00 | 393.60 | 396.79 | 395.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 13:15:00 | 396.15 | 396.66 | 395.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 15:00:00 | 398.25 | 396.98 | 395.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 379.50 | 393.95 | 394.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 379.50 | 393.95 | 394.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 378.10 | 388.79 | 392.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 360.10 | 359.74 | 368.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 360.10 | 359.74 | 368.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 369.05 | 360.80 | 362.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:45:00 | 368.20 | 360.80 | 362.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 370.65 | 362.77 | 363.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 11:00:00 | 370.65 | 362.77 | 363.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 376.35 | 365.49 | 364.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 380.35 | 368.46 | 366.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 399.80 | 400.96 | 396.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 15:00:00 | 399.80 | 400.96 | 396.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 429.40 | 439.03 | 434.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 429.40 | 439.03 | 434.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 424.20 | 436.06 | 433.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 424.20 | 436.06 | 433.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2025-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 14:15:00 | 429.00 | 432.05 | 432.40 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 15:15:00 | 434.60 | 431.96 | 431.83 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 12:15:00 | 429.40 | 431.76 | 431.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 09:15:00 | 427.85 | 430.42 | 431.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 432.00 | 425.82 | 427.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 432.00 | 425.82 | 427.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 432.00 | 425.82 | 427.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 432.00 | 425.82 | 427.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 429.80 | 426.62 | 427.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 427.05 | 426.62 | 427.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 14:00:00 | 427.90 | 426.42 | 427.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 438.95 | 428.82 | 428.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 438.95 | 428.82 | 428.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 10:15:00 | 442.20 | 431.50 | 429.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 11:15:00 | 443.75 | 444.83 | 439.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 11:45:00 | 443.30 | 444.83 | 439.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 437.25 | 442.71 | 439.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:00:00 | 437.25 | 442.71 | 439.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 436.20 | 441.41 | 438.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:45:00 | 435.60 | 441.41 | 438.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 433.40 | 439.80 | 438.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 443.30 | 439.80 | 438.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 449.90 | 456.69 | 451.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 449.90 | 456.69 | 451.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 444.00 | 454.15 | 451.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 458.10 | 454.15 | 451.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 12:15:00 | 453.55 | 455.27 | 452.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 13:00:00 | 453.55 | 455.27 | 452.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 449.10 | 454.03 | 452.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 14:00:00 | 449.10 | 454.03 | 452.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 451.55 | 453.54 | 452.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 461.00 | 453.19 | 452.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 10:15:00 | 473.20 | 476.77 | 477.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 10:15:00 | 473.20 | 476.77 | 477.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 466.50 | 474.72 | 476.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 11:15:00 | 471.00 | 470.45 | 472.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 12:00:00 | 471.00 | 470.45 | 472.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 470.00 | 470.36 | 472.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 14:15:00 | 466.40 | 470.12 | 472.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 10:00:00 | 467.60 | 469.39 | 471.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 12:15:00 | 468.65 | 469.25 | 470.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:00:00 | 468.45 | 469.05 | 470.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 469.40 | 469.12 | 469.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:30:00 | 471.95 | 469.12 | 469.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 468.00 | 468.70 | 469.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:45:00 | 468.05 | 468.70 | 469.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 464.95 | 467.96 | 469.07 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-27 12:15:00 | 475.05 | 470.50 | 470.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 475.05 | 470.50 | 470.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 477.60 | 472.48 | 471.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 15:15:00 | 477.35 | 478.15 | 475.67 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 09:15:00 | 480.10 | 478.15 | 475.67 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 10:45:00 | 480.95 | 478.84 | 476.42 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 476.25 | 478.20 | 476.73 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-29 13:15:00 | 476.25 | 478.20 | 476.73 | SL hit (close<ema400) qty=1.00 sl=476.73 alert=retest1 |

### Cycle 144 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 471.00 | 475.93 | 476.14 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 476.40 | 472.90 | 472.68 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 470.90 | 472.61 | 472.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 15:15:00 | 470.00 | 472.09 | 472.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 473.10 | 471.95 | 472.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 10:15:00 | 473.10 | 471.95 | 472.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 473.10 | 471.95 | 472.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:30:00 | 473.60 | 471.95 | 472.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 474.50 | 472.46 | 472.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:30:00 | 474.20 | 472.46 | 472.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 12:15:00 | 477.20 | 473.41 | 472.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 13:15:00 | 479.85 | 474.70 | 473.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 15:15:00 | 480.75 | 481.16 | 478.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 09:15:00 | 478.00 | 481.16 | 478.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 479.60 | 480.85 | 478.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:30:00 | 479.55 | 480.85 | 478.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 479.45 | 480.57 | 478.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:45:00 | 479.80 | 480.57 | 478.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 477.00 | 479.85 | 478.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:00:00 | 477.00 | 479.85 | 478.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 477.95 | 479.47 | 478.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 480.80 | 478.30 | 478.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 475.20 | 482.75 | 482.51 | SL hit (close<static) qty=1.00 sl=476.10 alert=retest2 |

### Cycle 148 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 478.55 | 481.91 | 482.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 469.60 | 476.95 | 479.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 459.05 | 458.78 | 464.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 459.05 | 458.78 | 464.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 466.75 | 461.71 | 464.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 467.40 | 461.71 | 464.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 465.50 | 462.46 | 464.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 462.05 | 462.46 | 464.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 438.95 | 447.21 | 452.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 443.60 | 441.97 | 447.69 | SL hit (close>ema200) qty=0.50 sl=441.97 alert=retest2 |

### Cycle 149 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 459.95 | 447.02 | 446.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 468.00 | 456.60 | 452.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 14:15:00 | 479.30 | 479.60 | 475.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 14:45:00 | 479.00 | 479.60 | 475.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 477.95 | 479.11 | 476.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 477.95 | 479.11 | 476.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 479.05 | 479.10 | 476.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:30:00 | 477.55 | 479.10 | 476.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 484.90 | 480.06 | 477.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 10:15:00 | 485.35 | 483.42 | 481.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 11:15:00 | 475.00 | 479.95 | 480.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 475.00 | 479.95 | 480.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 472.50 | 476.60 | 478.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 446.95 | 445.04 | 449.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:00:00 | 446.95 | 445.04 | 449.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 447.35 | 444.79 | 447.28 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 451.85 | 448.88 | 448.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 455.10 | 450.66 | 449.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 11:15:00 | 454.70 | 455.32 | 453.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 12:00:00 | 454.70 | 455.32 | 453.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 448.20 | 454.13 | 453.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 448.20 | 454.13 | 453.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 446.25 | 452.55 | 452.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 444.10 | 447.81 | 450.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 426.35 | 424.41 | 430.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 15:00:00 | 426.35 | 424.41 | 430.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 430.75 | 426.19 | 430.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:45:00 | 430.15 | 426.19 | 430.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 426.75 | 426.30 | 429.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:30:00 | 430.00 | 426.30 | 429.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 432.40 | 427.90 | 430.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:00:00 | 432.40 | 427.90 | 430.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 432.60 | 428.84 | 430.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:45:00 | 432.60 | 428.84 | 430.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 427.50 | 429.68 | 430.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:30:00 | 428.85 | 429.68 | 430.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 430.95 | 429.24 | 429.90 | EMA400 retest candle locked (from downside) |

### Cycle 153 — BUY (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 10:15:00 | 431.45 | 430.31 | 430.27 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 14:15:00 | 428.90 | 430.01 | 430.15 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 09:15:00 | 439.10 | 431.60 | 430.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 11:15:00 | 442.05 | 435.14 | 432.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 445.30 | 445.57 | 441.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 15:00:00 | 445.30 | 445.57 | 441.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 428.20 | 442.00 | 440.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 428.20 | 442.00 | 440.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 421.15 | 437.83 | 438.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 417.20 | 426.56 | 432.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 13:15:00 | 385.30 | 384.64 | 391.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 13:45:00 | 385.55 | 384.64 | 391.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 376.25 | 373.94 | 375.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:45:00 | 376.65 | 373.94 | 375.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 372.05 | 373.56 | 375.56 | EMA400 retest candle locked (from downside) |

### Cycle 157 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 384.90 | 377.20 | 376.63 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 14:15:00 | 376.35 | 378.47 | 378.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 15:15:00 | 375.45 | 377.87 | 378.27 | Break + close below crossover candle low |

### Cycle 159 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 381.60 | 378.61 | 378.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 13:15:00 | 383.30 | 380.50 | 379.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 391.35 | 391.66 | 388.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 12:15:00 | 387.10 | 390.23 | 388.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 387.10 | 390.23 | 388.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:45:00 | 387.85 | 390.23 | 388.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 384.75 | 389.13 | 388.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:45:00 | 384.60 | 389.13 | 388.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 381.90 | 386.75 | 387.26 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 12:15:00 | 389.10 | 386.43 | 386.21 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 382.50 | 385.73 | 386.09 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 14:15:00 | 387.00 | 386.25 | 386.25 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 384.10 | 385.82 | 386.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 381.05 | 384.87 | 385.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 380.10 | 379.26 | 381.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:30:00 | 379.25 | 379.26 | 381.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 380.55 | 379.52 | 381.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 379.00 | 379.52 | 381.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:30:00 | 379.00 | 379.22 | 381.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 383.20 | 378.58 | 379.12 | SL hit (close>static) qty=1.00 sl=381.90 alert=retest2 |

### Cycle 165 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 383.40 | 379.55 | 379.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 385.45 | 380.73 | 380.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 379.90 | 381.12 | 380.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 379.90 | 381.12 | 380.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 379.90 | 381.12 | 380.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 379.90 | 381.12 | 380.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 383.60 | 381.61 | 380.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 386.20 | 381.71 | 380.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 12:45:00 | 385.00 | 383.52 | 382.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 15:15:00 | 380.10 | 382.00 | 382.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 380.10 | 382.00 | 382.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 377.85 | 380.89 | 381.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 388.75 | 380.61 | 380.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 388.75 | 380.61 | 380.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 388.75 | 380.61 | 380.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 388.75 | 380.61 | 380.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 388.45 | 382.18 | 381.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 393.00 | 388.88 | 386.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 12:15:00 | 389.40 | 390.24 | 388.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 13:00:00 | 389.40 | 390.24 | 388.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 388.50 | 389.88 | 388.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 15:00:00 | 388.50 | 389.88 | 388.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 390.00 | 389.90 | 388.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 392.55 | 389.90 | 388.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 15:15:00 | 389.30 | 391.22 | 391.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 15:15:00 | 389.30 | 391.22 | 391.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 13:15:00 | 387.55 | 389.66 | 390.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 390.00 | 389.59 | 390.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 390.00 | 389.59 | 390.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 390.00 | 389.59 | 390.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 390.00 | 389.59 | 390.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 391.20 | 389.91 | 390.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 391.20 | 389.91 | 390.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 391.45 | 390.22 | 390.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:45:00 | 390.55 | 390.22 | 390.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 390.10 | 390.20 | 390.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:15:00 | 389.30 | 390.20 | 390.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 395.60 | 391.32 | 390.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 395.60 | 391.32 | 390.81 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 388.90 | 390.35 | 390.46 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 392.10 | 390.67 | 390.58 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 11:15:00 | 389.70 | 390.54 | 390.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 388.60 | 390.15 | 390.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 14:15:00 | 391.00 | 390.21 | 390.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 14:15:00 | 391.00 | 390.21 | 390.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 391.00 | 390.21 | 390.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 391.00 | 390.21 | 390.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 391.30 | 390.43 | 390.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 391.50 | 390.43 | 390.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 394.25 | 391.20 | 390.78 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 15:15:00 | 389.75 | 391.86 | 391.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 386.15 | 390.12 | 391.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 390.05 | 389.87 | 390.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 13:00:00 | 390.05 | 389.87 | 390.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 371.55 | 374.06 | 376.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 371.55 | 374.06 | 376.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 376.00 | 374.11 | 375.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 376.80 | 374.11 | 375.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 374.65 | 374.22 | 375.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 376.60 | 374.22 | 375.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 374.10 | 374.20 | 375.15 | EMA400 retest candle locked (from downside) |

### Cycle 175 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 378.25 | 375.82 | 375.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 15:15:00 | 380.00 | 377.17 | 376.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 379.25 | 379.65 | 378.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 379.25 | 379.65 | 378.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 379.25 | 379.65 | 378.42 | EMA400 retest candle locked (from upside) |

### Cycle 176 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 10:15:00 | 377.80 | 378.07 | 378.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 15:15:00 | 376.85 | 377.69 | 377.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 13:15:00 | 377.50 | 377.37 | 377.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 13:15:00 | 377.50 | 377.37 | 377.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 377.50 | 377.37 | 377.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:30:00 | 380.15 | 377.37 | 377.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 376.40 | 377.18 | 377.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 376.40 | 377.18 | 377.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 376.50 | 376.08 | 376.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:30:00 | 376.90 | 376.08 | 376.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 376.50 | 376.16 | 376.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:30:00 | 376.75 | 376.16 | 376.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 376.55 | 376.24 | 376.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:30:00 | 376.50 | 376.24 | 376.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 376.50 | 376.29 | 376.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:30:00 | 376.60 | 376.29 | 376.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 380.10 | 377.05 | 377.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:45:00 | 380.15 | 377.05 | 377.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 15:15:00 | 381.20 | 377.88 | 377.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 09:15:00 | 382.10 | 379.63 | 378.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 14:15:00 | 380.90 | 381.19 | 380.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 15:00:00 | 380.90 | 381.19 | 380.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 380.80 | 381.11 | 380.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 380.05 | 380.58 | 379.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 377.85 | 380.03 | 379.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:15:00 | 376.85 | 380.03 | 379.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 376.90 | 379.41 | 379.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 15:15:00 | 374.05 | 377.35 | 378.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 375.60 | 374.61 | 376.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 375.60 | 374.61 | 376.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 375.60 | 374.61 | 376.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:45:00 | 374.35 | 374.61 | 376.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 376.70 | 375.03 | 376.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:30:00 | 377.55 | 375.03 | 376.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 376.60 | 375.34 | 376.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 12:15:00 | 376.35 | 375.34 | 376.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:15:00 | 376.50 | 375.62 | 376.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 14:30:00 | 375.05 | 375.96 | 376.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 15:00:00 | 376.35 | 375.96 | 376.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 376.75 | 376.12 | 376.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 376.90 | 376.12 | 376.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 375.10 | 374.09 | 375.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 375.10 | 374.09 | 375.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 375.05 | 374.29 | 375.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:15:00 | 374.65 | 374.29 | 375.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 372.15 | 373.86 | 374.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 379.45 | 374.98 | 374.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 379.45 | 374.98 | 374.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 380.80 | 376.74 | 375.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 10:15:00 | 376.80 | 378.21 | 377.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 10:15:00 | 376.80 | 378.21 | 377.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 376.80 | 378.21 | 377.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:00:00 | 376.80 | 378.21 | 377.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 377.50 | 378.07 | 377.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 12:15:00 | 378.60 | 378.07 | 377.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 12:45:00 | 378.80 | 378.21 | 377.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:15:00 | 378.00 | 377.93 | 377.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 15:00:00 | 381.00 | 378.55 | 377.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 380.55 | 379.80 | 378.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:30:00 | 380.30 | 379.80 | 378.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 377.25 | 379.99 | 379.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:45:00 | 377.30 | 379.99 | 379.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 377.75 | 379.54 | 379.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:15:00 | 377.55 | 379.54 | 379.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-28 15:15:00 | 377.55 | 379.15 | 379.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — SELL (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 15:15:00 | 377.55 | 379.15 | 379.28 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 385.45 | 380.41 | 379.84 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 380.15 | 382.07 | 382.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 379.45 | 381.34 | 381.76 | Break + close below crossover candle low |

### Cycle 183 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 388.20 | 382.64 | 382.27 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 380.45 | 384.66 | 384.87 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 13:15:00 | 388.20 | 385.33 | 385.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 09:15:00 | 416.10 | 392.88 | 388.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 14:15:00 | 396.25 | 400.46 | 394.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 14:15:00 | 396.25 | 400.46 | 394.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 396.25 | 400.46 | 394.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 396.25 | 400.46 | 394.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 385.05 | 397.38 | 394.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 10:45:00 | 397.60 | 396.90 | 394.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 11:30:00 | 397.40 | 397.32 | 394.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 12:00:00 | 399.00 | 397.32 | 394.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 386.80 | 393.33 | 393.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 386.80 | 393.33 | 393.64 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 394.15 | 392.02 | 391.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 14:15:00 | 397.45 | 393.66 | 392.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 395.50 | 396.59 | 394.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 14:00:00 | 395.50 | 396.59 | 394.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 393.30 | 395.93 | 394.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 393.30 | 395.93 | 394.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 393.15 | 395.38 | 394.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 391.70 | 395.38 | 394.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 392.00 | 393.75 | 393.93 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 396.00 | 393.87 | 393.67 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 390.50 | 393.99 | 394.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 15:15:00 | 389.15 | 390.92 | 392.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 381.35 | 380.51 | 382.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 14:00:00 | 381.35 | 380.51 | 382.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 380.55 | 380.10 | 381.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:15:00 | 381.15 | 380.10 | 381.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 381.25 | 380.33 | 381.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 12:30:00 | 380.05 | 380.11 | 381.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:45:00 | 380.15 | 379.11 | 380.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:45:00 | 380.10 | 379.08 | 379.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 10:45:00 | 380.00 | 380.24 | 380.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 11:15:00 | 382.20 | 380.63 | 380.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 382.20 | 380.63 | 380.45 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 378.90 | 380.71 | 380.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 375.85 | 379.41 | 380.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 361.25 | 353.77 | 356.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 361.25 | 353.77 | 356.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 361.25 | 353.77 | 356.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 361.95 | 353.77 | 356.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 357.85 | 354.58 | 357.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:45:00 | 353.90 | 354.37 | 356.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:45:00 | 355.85 | 354.93 | 355.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 360.40 | 356.38 | 356.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 360.40 | 356.38 | 356.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 372.40 | 360.13 | 358.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 11:15:00 | 372.75 | 373.06 | 368.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 12:00:00 | 372.75 | 373.06 | 368.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 366.50 | 371.50 | 369.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 366.50 | 371.50 | 369.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 365.70 | 370.34 | 368.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 365.60 | 370.34 | 368.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 364.90 | 368.07 | 368.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 363.50 | 366.68 | 367.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 367.30 | 366.37 | 367.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 367.30 | 366.37 | 367.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 367.30 | 366.37 | 367.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 367.30 | 366.37 | 367.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 365.55 | 366.20 | 367.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 13:15:00 | 363.80 | 366.20 | 367.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 09:30:00 | 365.20 | 365.12 | 366.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 369.35 | 366.73 | 366.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 369.35 | 366.73 | 366.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 373.35 | 368.43 | 367.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 381.10 | 382.28 | 377.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 383.70 | 382.28 | 377.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 379.35 | 380.98 | 378.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 379.35 | 380.98 | 378.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 376.35 | 379.82 | 378.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 381.15 | 379.82 | 378.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 380.50 | 379.95 | 378.78 | EMA400 retest candle locked (from upside) |

### Cycle 196 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 376.10 | 378.18 | 378.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 375.05 | 377.22 | 377.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 377.45 | 376.36 | 377.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 14:15:00 | 377.45 | 376.36 | 377.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 377.45 | 376.36 | 377.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 377.45 | 376.36 | 377.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 374.00 | 375.89 | 376.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 372.50 | 375.89 | 376.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:45:00 | 372.45 | 375.02 | 376.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 12:45:00 | 373.55 | 372.49 | 373.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 373.80 | 373.15 | 373.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 375.35 | 372.53 | 372.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 14:00:00 | 375.35 | 372.53 | 372.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 375.85 | 373.20 | 373.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 14:45:00 | 375.75 | 373.20 | 373.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-01 15:15:00 | 375.30 | 373.62 | 373.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — BUY (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 15:15:00 | 375.30 | 373.62 | 373.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 376.80 | 374.42 | 373.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 374.95 | 374.95 | 374.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 10:00:00 | 374.95 | 374.95 | 374.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 374.20 | 374.80 | 374.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 374.20 | 374.80 | 374.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 375.35 | 374.91 | 374.33 | EMA400 retest candle locked (from upside) |

### Cycle 198 — SELL (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 14:15:00 | 372.95 | 373.98 | 374.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 370.40 | 373.04 | 373.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 12:15:00 | 372.15 | 371.73 | 372.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 12:15:00 | 372.15 | 371.73 | 372.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 372.15 | 371.73 | 372.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:00:00 | 372.15 | 371.73 | 372.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 373.60 | 372.10 | 372.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:00:00 | 373.60 | 372.10 | 372.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 375.15 | 372.71 | 373.02 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 374.60 | 373.46 | 373.33 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 371.00 | 373.62 | 373.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 368.85 | 372.00 | 372.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 12:15:00 | 359.70 | 358.24 | 361.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 12:15:00 | 359.70 | 358.24 | 361.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 359.70 | 358.24 | 361.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:45:00 | 360.75 | 358.24 | 361.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 360.90 | 358.77 | 361.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:30:00 | 361.60 | 358.77 | 361.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 361.95 | 359.41 | 361.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:45:00 | 362.30 | 359.41 | 361.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 363.00 | 360.13 | 361.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:45:00 | 364.10 | 360.65 | 361.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 363.35 | 361.19 | 361.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 363.35 | 361.19 | 361.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 364.95 | 362.46 | 362.18 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 09:15:00 | 358.95 | 361.77 | 361.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 10:15:00 | 357.60 | 360.93 | 361.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 14:15:00 | 353.50 | 353.36 | 356.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 15:00:00 | 353.50 | 353.36 | 356.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 344.30 | 342.89 | 346.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 345.85 | 342.89 | 346.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 346.10 | 343.60 | 345.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:45:00 | 346.95 | 343.60 | 345.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 348.00 | 344.48 | 345.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:15:00 | 347.90 | 344.48 | 345.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 353.50 | 346.28 | 346.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:00:00 | 353.50 | 346.28 | 346.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — BUY (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 10:15:00 | 354.30 | 347.89 | 347.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 13:15:00 | 356.60 | 351.26 | 348.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 15:15:00 | 351.00 | 351.43 | 349.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 15:15:00 | 351.00 | 351.43 | 349.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 351.00 | 351.43 | 349.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 350.40 | 351.43 | 349.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 352.20 | 351.59 | 349.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 11:45:00 | 354.10 | 351.69 | 350.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 09:15:00 | 355.30 | 350.90 | 350.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 12:30:00 | 353.90 | 352.49 | 351.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 14:45:00 | 353.60 | 353.21 | 351.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 350.45 | 353.26 | 352.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 350.45 | 353.26 | 352.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 356.20 | 353.85 | 352.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 357.05 | 354.75 | 353.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 360.90 | 358.29 | 355.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-03 09:15:00 | 389.51 | 379.90 | 372.36 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 204 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 450.45 | 461.76 | 461.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 446.75 | 455.36 | 458.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 454.30 | 452.29 | 455.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 13:00:00 | 454.30 | 452.29 | 455.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 454.10 | 452.65 | 455.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:45:00 | 459.45 | 452.65 | 455.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 453.15 | 452.75 | 455.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 453.15 | 452.75 | 455.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 450.85 | 452.18 | 454.19 | EMA400 retest candle locked (from downside) |

### Cycle 205 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 463.60 | 455.38 | 454.84 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 452.50 | 457.17 | 457.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 09:15:00 | 450.25 | 455.78 | 456.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 14:15:00 | 446.70 | 446.31 | 449.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 15:00:00 | 446.70 | 446.31 | 449.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 444.10 | 445.87 | 449.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 442.15 | 445.87 | 449.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 15:15:00 | 449.95 | 445.37 | 446.79 | SL hit (close>static) qty=1.00 sl=449.55 alert=retest2 |

### Cycle 207 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 453.20 | 448.21 | 447.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 12:15:00 | 455.45 | 449.66 | 448.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 454.00 | 454.64 | 452.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 454.00 | 454.64 | 452.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 454.00 | 454.64 | 452.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 454.00 | 454.64 | 452.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 452.15 | 454.14 | 452.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:45:00 | 450.35 | 454.14 | 452.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 450.60 | 453.43 | 452.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:00:00 | 450.60 | 453.43 | 452.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 446.65 | 451.09 | 451.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 441.80 | 448.49 | 450.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 422.40 | 415.51 | 422.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 14:15:00 | 422.40 | 415.51 | 422.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 422.40 | 415.51 | 422.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 422.40 | 415.51 | 422.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 419.00 | 416.20 | 422.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 423.10 | 417.23 | 422.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 420.40 | 417.50 | 420.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 420.40 | 417.50 | 420.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 420.00 | 418.00 | 420.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 401.55 | 418.00 | 420.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 11:00:00 | 418.40 | 412.85 | 414.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 13:15:00 | 424.50 | 416.98 | 416.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 424.50 | 416.98 | 416.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 427.90 | 419.16 | 417.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 432.00 | 441.52 | 435.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 432.00 | 441.52 | 435.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 432.00 | 441.52 | 435.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 432.00 | 441.52 | 435.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 438.30 | 440.88 | 435.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 11:15:00 | 439.50 | 440.88 | 435.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 428.75 | 437.28 | 434.63 | SL hit (close<static) qty=1.00 sl=431.50 alert=retest2 |

### Cycle 210 — SELL (started 2026-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 14:15:00 | 420.60 | 432.49 | 432.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 413.90 | 426.84 | 430.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 425.65 | 423.52 | 426.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 425.65 | 423.52 | 426.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 425.65 | 423.52 | 426.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:30:00 | 428.60 | 423.52 | 426.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 424.70 | 423.99 | 426.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 426.65 | 423.99 | 426.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 423.15 | 423.83 | 426.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 419.85 | 423.83 | 426.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 429.95 | 425.10 | 425.37 | SL hit (close>static) qty=1.00 sl=428.90 alert=retest2 |

### Cycle 211 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 430.75 | 426.23 | 425.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 435.20 | 428.02 | 426.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 422.70 | 429.18 | 427.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 422.70 | 429.18 | 427.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 422.70 | 429.18 | 427.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:15:00 | 418.50 | 429.18 | 427.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 418.75 | 427.09 | 427.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 418.30 | 425.33 | 426.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 11:15:00 | 418.70 | 418.09 | 421.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 12:00:00 | 418.70 | 418.09 | 421.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 420.45 | 418.56 | 421.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:45:00 | 419.30 | 418.56 | 421.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 427.35 | 420.32 | 421.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:00:00 | 427.35 | 420.32 | 421.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 427.05 | 421.67 | 422.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 427.05 | 421.67 | 422.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — BUY (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 15:15:00 | 427.80 | 422.89 | 422.70 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 410.65 | 420.44 | 421.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 407.45 | 417.84 | 420.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 409.20 | 408.65 | 413.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:00:00 | 409.20 | 408.65 | 413.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 418.00 | 411.03 | 413.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 418.00 | 411.03 | 413.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 417.50 | 412.33 | 413.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 418.50 | 412.33 | 413.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 215 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 429.20 | 417.60 | 416.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 434.10 | 420.90 | 417.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 415.60 | 426.24 | 422.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 415.60 | 426.24 | 422.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 415.60 | 426.24 | 422.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 415.60 | 426.24 | 422.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 418.25 | 424.64 | 422.42 | EMA400 retest candle locked (from upside) |

### Cycle 216 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 417.85 | 421.05 | 421.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 413.75 | 419.17 | 420.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 413.95 | 407.35 | 411.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 413.95 | 407.35 | 411.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 413.95 | 407.35 | 411.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 413.95 | 407.35 | 411.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 411.65 | 408.21 | 411.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:30:00 | 410.40 | 411.21 | 412.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 401.90 | 411.15 | 412.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 424.00 | 407.63 | 406.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 217 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 424.00 | 407.63 | 406.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 425.85 | 411.28 | 408.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 417.25 | 418.98 | 414.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 417.25 | 418.98 | 414.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 424.00 | 425.97 | 422.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 15:00:00 | 424.00 | 425.97 | 422.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 424.75 | 425.41 | 422.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 428.90 | 425.41 | 422.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:15:00 | 430.00 | 426.39 | 424.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 428.80 | 425.94 | 424.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 13:15:00 | 428.85 | 425.35 | 424.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 14:15:00 | 471.79 | 459.09 | 452.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 218 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 481.60 | 501.94 | 502.81 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-16 12:00:00 | 508.25 | 2023-05-22 13:15:00 | 503.05 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2023-05-16 14:15:00 | 508.00 | 2023-05-22 13:15:00 | 503.05 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2023-05-30 13:30:00 | 512.90 | 2023-05-31 12:15:00 | 510.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2023-05-30 14:00:00 | 512.60 | 2023-05-31 12:15:00 | 510.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2023-05-31 14:30:00 | 512.85 | 2023-06-06 13:15:00 | 511.80 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2023-05-31 15:15:00 | 513.95 | 2023-06-06 13:15:00 | 511.80 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2023-06-01 09:15:00 | 519.45 | 2023-06-06 13:15:00 | 511.80 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2023-06-01 12:15:00 | 514.30 | 2023-06-06 13:15:00 | 511.80 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-06-02 12:00:00 | 514.55 | 2023-06-06 13:15:00 | 511.80 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2023-06-02 13:30:00 | 514.00 | 2023-06-06 13:15:00 | 511.80 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2023-06-05 09:15:00 | 516.15 | 2023-06-06 13:15:00 | 511.80 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2023-06-06 11:45:00 | 514.00 | 2023-06-06 13:15:00 | 511.80 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2023-06-06 12:15:00 | 514.00 | 2023-06-06 13:15:00 | 511.80 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2023-06-19 09:45:00 | 520.15 | 2023-06-20 11:15:00 | 516.90 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2023-06-19 10:30:00 | 520.35 | 2023-06-20 11:15:00 | 516.90 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2023-06-19 11:30:00 | 520.40 | 2023-06-20 11:15:00 | 516.90 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2023-06-19 14:30:00 | 520.00 | 2023-06-20 11:15:00 | 516.90 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2023-06-28 12:15:00 | 507.10 | 2023-07-06 09:15:00 | 481.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-28 13:15:00 | 507.25 | 2023-07-06 09:15:00 | 481.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-30 09:45:00 | 506.40 | 2023-07-06 09:15:00 | 481.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-03 09:30:00 | 507.00 | 2023-07-06 09:15:00 | 481.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-28 12:15:00 | 507.10 | 2023-07-06 15:15:00 | 486.95 | STOP_HIT | 0.50 | 3.97% |
| SELL | retest2 | 2023-06-28 13:15:00 | 507.25 | 2023-07-06 15:15:00 | 486.95 | STOP_HIT | 0.50 | 4.00% |
| SELL | retest2 | 2023-06-30 09:45:00 | 506.40 | 2023-07-06 15:15:00 | 486.95 | STOP_HIT | 0.50 | 3.84% |
| SELL | retest2 | 2023-07-03 09:30:00 | 507.00 | 2023-07-06 15:15:00 | 486.95 | STOP_HIT | 0.50 | 3.95% |
| SELL | retest2 | 2023-07-03 14:00:00 | 504.65 | 2023-07-07 11:15:00 | 479.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-03 14:45:00 | 504.85 | 2023-07-07 11:15:00 | 479.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-04 09:15:00 | 504.20 | 2023-07-07 11:15:00 | 478.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-03 14:00:00 | 504.65 | 2023-07-10 12:15:00 | 454.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-07-03 14:45:00 | 504.85 | 2023-07-10 12:15:00 | 454.37 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-07-04 09:15:00 | 504.20 | 2023-07-10 14:15:00 | 453.78 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2023-07-21 13:30:00 | 471.15 | 2023-07-25 09:15:00 | 462.50 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2023-07-24 12:45:00 | 468.60 | 2023-07-25 09:15:00 | 462.50 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2023-08-03 09:15:00 | 473.20 | 2023-08-08 09:15:00 | 467.90 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2023-08-03 11:15:00 | 473.95 | 2023-08-08 09:15:00 | 467.90 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2023-08-03 15:00:00 | 473.50 | 2023-08-08 09:15:00 | 467.90 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2023-08-04 13:45:00 | 473.05 | 2023-08-08 09:15:00 | 467.90 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2023-08-07 11:15:00 | 476.55 | 2023-08-08 09:15:00 | 467.90 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2023-08-07 14:45:00 | 475.45 | 2023-08-08 09:15:00 | 467.90 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2023-08-16 09:15:00 | 458.95 | 2023-08-23 10:15:00 | 456.40 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2023-08-17 11:15:00 | 460.45 | 2023-08-23 11:15:00 | 457.70 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2023-08-17 11:45:00 | 459.80 | 2023-08-23 11:15:00 | 457.70 | STOP_HIT | 1.00 | 0.46% |
| SELL | retest2 | 2023-08-17 12:45:00 | 458.90 | 2023-08-23 11:15:00 | 457.70 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2023-08-22 12:00:00 | 453.25 | 2023-08-23 11:15:00 | 457.70 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2023-08-28 11:00:00 | 462.10 | 2023-09-05 09:15:00 | 508.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-29 09:15:00 | 466.20 | 2023-09-05 09:15:00 | 512.82 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-09-20 09:30:00 | 510.60 | 2023-09-28 14:15:00 | 485.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-20 09:30:00 | 510.60 | 2023-09-29 11:15:00 | 491.00 | STOP_HIT | 0.50 | 3.84% |
| BUY | retest2 | 2023-10-16 10:15:00 | 490.70 | 2023-10-18 10:15:00 | 484.50 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2023-10-16 11:00:00 | 489.10 | 2023-10-18 10:15:00 | 484.50 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2023-10-17 10:30:00 | 489.25 | 2023-10-18 10:15:00 | 484.50 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2023-10-17 15:00:00 | 488.95 | 2023-10-18 10:15:00 | 484.50 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2023-10-27 12:15:00 | 450.75 | 2023-10-27 14:15:00 | 453.15 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2023-10-27 12:45:00 | 449.95 | 2023-10-27 14:15:00 | 453.15 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest1 | 2023-11-06 09:15:00 | 492.00 | 2023-11-08 11:15:00 | 516.60 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-11-06 09:15:00 | 492.00 | 2023-11-09 09:15:00 | 510.90 | STOP_HIT | 0.50 | 3.84% |
| BUY | retest2 | 2023-11-12 18:15:00 | 517.85 | 2023-11-20 14:15:00 | 516.05 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2023-11-28 12:45:00 | 532.50 | 2023-12-06 09:15:00 | 585.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-29 09:30:00 | 532.20 | 2023-12-06 09:15:00 | 585.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-29 10:45:00 | 531.60 | 2023-12-06 09:15:00 | 584.76 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-01-30 10:45:00 | 637.20 | 2024-01-31 10:15:00 | 647.35 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-01-30 14:30:00 | 635.25 | 2024-01-31 10:15:00 | 647.35 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-01-31 10:30:00 | 639.85 | 2024-01-31 11:15:00 | 651.80 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-02-20 09:15:00 | 696.95 | 2024-02-21 15:15:00 | 686.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-02-27 10:45:00 | 671.90 | 2024-03-01 10:15:00 | 675.20 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2024-02-27 13:30:00 | 672.20 | 2024-03-01 10:15:00 | 675.20 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2024-02-28 09:30:00 | 670.30 | 2024-03-01 10:15:00 | 675.20 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-03-01 10:15:00 | 671.80 | 2024-03-01 10:15:00 | 675.20 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-03-15 09:15:00 | 639.35 | 2024-03-18 14:15:00 | 645.15 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-03-18 12:15:00 | 640.10 | 2024-03-18 14:15:00 | 645.15 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-03-18 13:15:00 | 641.30 | 2024-03-18 14:15:00 | 645.15 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-03-18 14:00:00 | 641.10 | 2024-03-18 14:15:00 | 645.15 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-03-20 14:45:00 | 634.90 | 2024-03-21 09:15:00 | 644.15 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-04-05 10:15:00 | 696.15 | 2024-04-18 14:15:00 | 722.95 | STOP_HIT | 1.00 | 3.85% |
| BUY | retest2 | 2024-04-05 13:15:00 | 694.00 | 2024-04-18 14:15:00 | 722.95 | STOP_HIT | 1.00 | 4.17% |
| BUY | retest2 | 2024-04-08 10:15:00 | 694.35 | 2024-04-18 14:15:00 | 722.95 | STOP_HIT | 1.00 | 4.12% |
| BUY | retest2 | 2024-04-25 12:30:00 | 745.80 | 2024-04-30 12:15:00 | 749.10 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2024-05-03 11:15:00 | 738.55 | 2024-05-06 12:15:00 | 746.35 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-05-03 12:45:00 | 737.00 | 2024-05-06 12:15:00 | 746.35 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest1 | 2024-05-13 09:30:00 | 659.90 | 2024-05-15 09:15:00 | 635.88 | PARTIAL | 0.50 | 3.64% |
| SELL | retest1 | 2024-05-13 14:15:00 | 669.35 | 2024-05-15 09:15:00 | 635.03 | PARTIAL | 0.50 | 5.13% |
| SELL | retest1 | 2024-05-13 15:00:00 | 668.45 | 2024-05-15 10:15:00 | 626.90 | PARTIAL | 0.50 | 6.22% |
| SELL | retest2 | 2024-05-14 11:45:00 | 655.45 | 2024-05-15 10:15:00 | 622.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-14 14:00:00 | 655.35 | 2024-05-15 10:15:00 | 622.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-05-13 09:30:00 | 659.90 | 2024-05-16 10:15:00 | 632.00 | STOP_HIT | 0.50 | 4.23% |
| SELL | retest1 | 2024-05-13 14:15:00 | 669.35 | 2024-05-16 10:15:00 | 632.00 | STOP_HIT | 0.50 | 5.58% |
| SELL | retest1 | 2024-05-13 15:00:00 | 668.45 | 2024-05-16 10:15:00 | 632.00 | STOP_HIT | 0.50 | 5.45% |
| SELL | retest2 | 2024-05-14 11:45:00 | 655.45 | 2024-05-16 10:15:00 | 632.00 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2024-05-14 14:00:00 | 655.35 | 2024-05-16 10:15:00 | 632.00 | STOP_HIT | 0.50 | 3.56% |
| BUY | retest1 | 2024-06-07 11:15:00 | 636.45 | 2024-06-10 14:15:00 | 668.27 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-06-07 14:30:00 | 636.30 | 2024-06-10 14:15:00 | 668.12 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-06-10 09:15:00 | 638.45 | 2024-06-10 14:15:00 | 670.37 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-06-07 11:15:00 | 636.45 | 2024-06-11 14:15:00 | 657.75 | STOP_HIT | 0.50 | 3.35% |
| BUY | retest1 | 2024-06-07 14:30:00 | 636.30 | 2024-06-11 14:15:00 | 657.75 | STOP_HIT | 0.50 | 3.37% |
| BUY | retest1 | 2024-06-10 09:15:00 | 638.45 | 2024-06-11 14:15:00 | 657.75 | STOP_HIT | 0.50 | 3.02% |
| BUY | retest2 | 2024-06-20 11:15:00 | 700.80 | 2024-06-25 12:15:00 | 697.25 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-06-24 10:00:00 | 699.80 | 2024-06-25 12:15:00 | 697.25 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-06-25 12:15:00 | 699.20 | 2024-06-25 12:15:00 | 697.25 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-06-28 12:15:00 | 685.75 | 2024-07-01 09:15:00 | 693.75 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-06-28 13:45:00 | 687.10 | 2024-07-01 09:15:00 | 693.75 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-06-28 15:00:00 | 685.60 | 2024-07-01 09:15:00 | 693.75 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-07-23 12:15:00 | 661.05 | 2024-07-24 14:15:00 | 692.10 | STOP_HIT | 1.00 | -4.70% |
| SELL | retest2 | 2024-07-23 15:15:00 | 685.15 | 2024-07-24 14:15:00 | 692.10 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-08-02 10:30:00 | 745.80 | 2024-08-05 10:15:00 | 711.40 | STOP_HIT | 1.00 | -4.61% |
| BUY | retest2 | 2024-08-02 13:00:00 | 744.00 | 2024-08-05 10:15:00 | 711.40 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2024-08-06 15:00:00 | 717.30 | 2024-08-07 10:15:00 | 741.20 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2024-08-19 14:30:00 | 618.80 | 2024-08-21 09:15:00 | 625.80 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-08-20 09:15:00 | 617.50 | 2024-08-21 09:15:00 | 625.80 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-09-02 11:30:00 | 628.65 | 2024-09-04 09:15:00 | 597.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-03 10:15:00 | 627.80 | 2024-09-04 09:15:00 | 596.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-02 11:30:00 | 628.65 | 2024-09-05 09:15:00 | 619.35 | STOP_HIT | 0.50 | 1.48% |
| SELL | retest2 | 2024-09-03 10:15:00 | 627.80 | 2024-09-05 09:15:00 | 619.35 | STOP_HIT | 0.50 | 1.35% |
| BUY | retest2 | 2024-09-27 09:15:00 | 588.30 | 2024-09-30 10:15:00 | 581.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-09-27 13:30:00 | 585.00 | 2024-09-30 10:15:00 | 581.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-09-30 09:15:00 | 584.80 | 2024-09-30 10:15:00 | 581.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-09-30 12:00:00 | 584.75 | 2024-09-30 13:15:00 | 580.65 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-10-21 11:45:00 | 515.95 | 2024-10-22 14:15:00 | 490.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 11:45:00 | 515.95 | 2024-10-23 10:15:00 | 497.55 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2024-11-19 15:15:00 | 431.15 | 2024-11-25 10:15:00 | 441.10 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2024-11-25 09:45:00 | 435.65 | 2024-11-25 10:15:00 | 441.10 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-11-29 14:15:00 | 450.10 | 2024-11-29 15:15:00 | 447.05 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-12-09 11:30:00 | 451.65 | 2024-12-10 15:15:00 | 448.50 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-12-10 09:15:00 | 452.50 | 2024-12-10 15:15:00 | 448.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-12-16 10:15:00 | 435.25 | 2024-12-19 09:15:00 | 413.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 09:15:00 | 436.25 | 2024-12-19 09:15:00 | 414.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 10:00:00 | 436.55 | 2024-12-19 09:15:00 | 414.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 10:15:00 | 435.25 | 2024-12-20 10:15:00 | 418.15 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2024-12-17 09:15:00 | 436.25 | 2024-12-20 10:15:00 | 418.15 | STOP_HIT | 0.50 | 4.15% |
| SELL | retest2 | 2024-12-17 10:00:00 | 436.55 | 2024-12-20 10:15:00 | 418.15 | STOP_HIT | 0.50 | 4.21% |
| SELL | retest2 | 2024-12-31 15:15:00 | 410.00 | 2025-01-01 09:15:00 | 413.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-02-03 10:45:00 | 465.55 | 2025-02-10 09:15:00 | 455.65 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-02-04 10:30:00 | 461.05 | 2025-02-10 09:15:00 | 455.65 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-02-05 09:30:00 | 461.55 | 2025-02-10 09:15:00 | 455.65 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-02-05 11:00:00 | 461.30 | 2025-02-10 09:15:00 | 455.65 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-02-07 11:30:00 | 470.85 | 2025-02-10 09:15:00 | 455.65 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2025-02-13 09:15:00 | 438.65 | 2025-02-14 10:15:00 | 416.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 10:00:00 | 439.10 | 2025-02-14 10:15:00 | 417.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:00:00 | 439.00 | 2025-02-14 10:15:00 | 417.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 09:15:00 | 438.65 | 2025-02-17 12:15:00 | 417.20 | STOP_HIT | 0.50 | 4.89% |
| SELL | retest2 | 2025-02-13 10:00:00 | 439.10 | 2025-02-17 12:15:00 | 417.20 | STOP_HIT | 0.50 | 4.99% |
| SELL | retest2 | 2025-02-13 13:00:00 | 439.00 | 2025-02-17 12:15:00 | 417.20 | STOP_HIT | 0.50 | 4.97% |
| SELL | retest2 | 2025-03-04 15:15:00 | 381.20 | 2025-03-05 09:15:00 | 392.40 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2025-03-10 11:00:00 | 412.60 | 2025-03-10 13:15:00 | 396.40 | STOP_HIT | 1.00 | -3.93% |
| SELL | retest2 | 2025-03-13 13:00:00 | 386.70 | 2025-03-19 09:15:00 | 396.20 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-03-13 14:00:00 | 385.80 | 2025-03-19 09:15:00 | 396.20 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-03-18 12:45:00 | 386.55 | 2025-03-19 09:15:00 | 396.20 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-03-18 13:30:00 | 387.00 | 2025-03-19 09:15:00 | 396.20 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2025-04-01 11:30:00 | 389.10 | 2025-04-02 12:15:00 | 397.60 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-04-01 12:45:00 | 389.60 | 2025-04-02 12:15:00 | 397.60 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-04-01 14:15:00 | 390.25 | 2025-04-02 12:15:00 | 397.60 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-04-02 09:15:00 | 385.85 | 2025-04-02 12:15:00 | 397.60 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2025-04-03 15:00:00 | 398.25 | 2025-04-04 09:15:00 | 379.50 | STOP_HIT | 1.00 | -4.71% |
| SELL | retest2 | 2025-05-02 11:15:00 | 427.05 | 2025-05-05 09:15:00 | 438.95 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-05-02 14:00:00 | 427.90 | 2025-05-05 09:15:00 | 438.95 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-05-12 09:15:00 | 461.00 | 2025-05-21 10:15:00 | 473.20 | STOP_HIT | 1.00 | 2.65% |
| SELL | retest2 | 2025-05-22 14:15:00 | 466.40 | 2025-05-27 12:15:00 | 475.05 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-05-23 10:00:00 | 467.60 | 2025-05-27 12:15:00 | 475.05 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-05-23 12:15:00 | 468.65 | 2025-05-27 12:15:00 | 475.05 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-05-26 11:00:00 | 468.45 | 2025-05-27 12:15:00 | 475.05 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest1 | 2025-05-29 09:15:00 | 480.10 | 2025-05-29 13:15:00 | 476.25 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest1 | 2025-05-29 10:45:00 | 480.95 | 2025-05-29 13:15:00 | 476.25 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-06-09 09:15:00 | 480.80 | 2025-06-11 13:15:00 | 475.20 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-06-17 11:15:00 | 462.05 | 2025-06-19 12:15:00 | 438.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:15:00 | 462.05 | 2025-06-20 10:15:00 | 443.60 | STOP_HIT | 0.50 | 3.99% |
| BUY | retest2 | 2025-07-03 10:15:00 | 485.35 | 2025-07-04 11:15:00 | 475.00 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-08-29 12:15:00 | 379.00 | 2025-09-02 09:15:00 | 383.20 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-08-29 13:30:00 | 379.00 | 2025-09-02 09:15:00 | 383.20 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-09-03 09:15:00 | 386.20 | 2025-09-04 15:15:00 | 380.10 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-09-03 12:45:00 | 385.00 | 2025-09-04 15:15:00 | 380.10 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-09-11 09:15:00 | 392.55 | 2025-09-12 15:15:00 | 389.30 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-09-16 13:15:00 | 389.30 | 2025-09-17 09:15:00 | 395.60 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-10-16 12:15:00 | 376.35 | 2025-10-23 09:15:00 | 379.45 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-10-16 13:15:00 | 376.50 | 2025-10-23 09:15:00 | 379.45 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-10-16 14:30:00 | 375.05 | 2025-10-23 09:15:00 | 379.45 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-10-16 15:00:00 | 376.35 | 2025-10-23 09:15:00 | 379.45 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-10-24 12:15:00 | 378.60 | 2025-10-28 15:15:00 | 377.55 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-10-24 12:45:00 | 378.80 | 2025-10-28 15:15:00 | 377.55 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-10-24 14:15:00 | 378.00 | 2025-10-28 15:15:00 | 377.55 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-10-24 15:00:00 | 381.00 | 2025-10-28 15:15:00 | 377.55 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-11-10 10:45:00 | 397.60 | 2025-11-11 09:15:00 | 386.80 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2025-11-10 11:30:00 | 397.40 | 2025-11-11 09:15:00 | 386.80 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-11-10 12:00:00 | 399.00 | 2025-11-11 09:15:00 | 386.80 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2025-11-26 12:30:00 | 380.05 | 2025-11-28 11:15:00 | 382.20 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-11-27 10:45:00 | 380.15 | 2025-11-28 11:15:00 | 382.20 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-11-27 13:45:00 | 380.10 | 2025-11-28 11:15:00 | 382.20 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-11-28 10:45:00 | 380.00 | 2025-11-28 11:15:00 | 382.20 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-12-10 11:45:00 | 353.90 | 2025-12-12 09:15:00 | 360.40 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-12-11 14:45:00 | 355.85 | 2025-12-12 09:15:00 | 360.40 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-12-18 13:15:00 | 363.80 | 2025-12-19 14:15:00 | 369.35 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-12-19 09:30:00 | 365.20 | 2025-12-19 14:15:00 | 369.35 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-12-30 09:15:00 | 372.50 | 2026-01-01 15:15:00 | 375.30 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-12-30 10:45:00 | 372.45 | 2026-01-01 15:15:00 | 375.30 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-31 12:45:00 | 373.55 | 2026-01-01 15:15:00 | 375.30 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2026-01-01 09:15:00 | 373.80 | 2026-01-01 15:15:00 | 375.30 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2026-01-27 11:45:00 | 354.10 | 2026-02-03 09:15:00 | 389.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-28 09:15:00 | 355.30 | 2026-02-03 09:15:00 | 390.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-28 12:30:00 | 353.90 | 2026-02-03 09:15:00 | 389.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-28 14:45:00 | 353.60 | 2026-02-03 09:15:00 | 388.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-30 09:30:00 | 357.05 | 2026-02-03 09:15:00 | 392.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-30 10:30:00 | 360.90 | 2026-02-03 09:15:00 | 396.99 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 442.15 | 2026-02-24 15:15:00 | 449.95 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-03-09 09:15:00 | 401.55 | 2026-03-10 13:15:00 | 424.50 | STOP_HIT | 1.00 | -5.72% |
| SELL | retest2 | 2026-03-10 11:00:00 | 418.40 | 2026-03-10 13:15:00 | 424.50 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-03-13 11:15:00 | 439.50 | 2026-03-13 12:15:00 | 428.75 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2026-03-17 11:15:00 | 419.85 | 2026-03-18 10:15:00 | 429.95 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2026-04-01 14:30:00 | 410.40 | 2026-04-08 09:15:00 | 424.00 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2026-04-02 09:15:00 | 401.90 | 2026-04-08 09:15:00 | 424.00 | STOP_HIT | 1.00 | -5.50% |
| BUY | retest2 | 2026-04-13 10:15:00 | 428.90 | 2026-04-22 14:15:00 | 471.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 13:15:00 | 430.00 | 2026-04-22 14:15:00 | 471.68 | TARGET_HIT | 1.00 | 9.69% |
| BUY | retest2 | 2026-04-15 09:15:00 | 428.80 | 2026-04-22 14:15:00 | 471.74 | TARGET_HIT | 1.00 | 10.01% |
| BUY | retest2 | 2026-04-15 13:15:00 | 428.85 | 2026-04-23 09:15:00 | 473.00 | TARGET_HIT | 1.00 | 10.29% |
| BUY | retest2 | 2026-04-24 14:30:00 | 473.50 | 2026-05-04 09:15:00 | 520.85 | TARGET_HIT | 1.00 | 10.00% |
