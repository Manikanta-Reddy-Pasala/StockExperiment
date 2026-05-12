# Minda Corporation Ltd. (MINDACORP)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 537.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 79 |
| ALERT1 | 52 |
| ALERT2 | 52 |
| ALERT2_SKIP | 24 |
| ALERT3 | 150 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 57 |
| PARTIAL | 10 |
| TARGET_HIT | 2 |
| STOP_HIT | 57 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 69 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 46
- **Target hits / Stop hits / Partials:** 2 / 57 / 10
- **Avg / median % per leg:** 0.46% / -0.84%
- **Sum % (uncompounded):** 31.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 1 | 3.2% | 0 | 31 | 0 | -1.75% | -54.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 31 | 1 | 3.2% | 0 | 31 | 0 | -1.75% | -54.3% |
| SELL (all) | 38 | 22 | 57.9% | 2 | 26 | 10 | 2.27% | 86.3% |
| SELL @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 2 | 1 | 3.63% | 10.9% |
| SELL @ 3rd Alert (retest2) | 35 | 19 | 54.3% | 2 | 24 | 9 | 2.15% | 75.4% |
| retest1 (combined) | 3 | 3 | 100.0% | 0 | 2 | 1 | 3.63% | 10.9% |
| retest2 (combined) | 66 | 20 | 30.3% | 2 | 55 | 9 | 0.32% | 21.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 485.00 | 478.74 | 478.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 487.10 | 482.31 | 480.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 12:15:00 | 503.95 | 503.98 | 499.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 12:45:00 | 502.75 | 503.98 | 499.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 514.20 | 505.75 | 501.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:15:00 | 517.70 | 505.75 | 501.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 500.50 | 508.33 | 507.82 | SL hit (close<static) qty=1.00 sl=501.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 500.25 | 506.71 | 507.13 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 508.00 | 505.88 | 505.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 12:15:00 | 510.55 | 506.82 | 506.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 538.10 | 538.59 | 529.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 09:30:00 | 539.00 | 538.59 | 529.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 526.75 | 537.00 | 533.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:00:00 | 526.75 | 537.00 | 533.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 525.70 | 534.74 | 532.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 525.50 | 534.74 | 532.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 528.35 | 531.67 | 531.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 528.35 | 531.67 | 531.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 526.80 | 530.70 | 531.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 11:15:00 | 524.40 | 529.44 | 530.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 514.50 | 513.93 | 517.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 10:15:00 | 515.60 | 513.93 | 517.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 515.85 | 514.32 | 517.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:30:00 | 515.95 | 514.32 | 517.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 515.60 | 514.57 | 517.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:30:00 | 517.40 | 514.57 | 517.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 515.75 | 514.96 | 516.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 515.75 | 514.96 | 516.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 518.10 | 515.59 | 516.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:15:00 | 531.65 | 515.59 | 516.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 536.00 | 519.67 | 518.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 550.00 | 535.13 | 528.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 10:15:00 | 570.70 | 571.80 | 564.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 10:30:00 | 570.15 | 571.80 | 564.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 567.30 | 571.62 | 567.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 567.30 | 571.62 | 567.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 565.75 | 570.45 | 567.63 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 552.45 | 563.68 | 564.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 546.10 | 557.60 | 561.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 543.15 | 533.56 | 538.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 543.15 | 533.56 | 538.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 543.15 | 533.56 | 538.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:45:00 | 542.00 | 533.56 | 538.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 541.60 | 535.17 | 538.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:15:00 | 535.70 | 538.25 | 538.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 13:15:00 | 508.92 | 522.37 | 530.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 512.00 | 507.27 | 511.64 | SL hit (close>ema200) qty=0.50 sl=507.27 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 518.70 | 512.24 | 511.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 09:15:00 | 522.90 | 515.43 | 513.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 512.25 | 518.81 | 516.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 09:15:00 | 512.25 | 518.81 | 516.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 512.25 | 518.81 | 516.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:30:00 | 512.00 | 518.81 | 516.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 521.00 | 519.25 | 517.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 11:15:00 | 521.50 | 519.25 | 517.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 14:30:00 | 521.20 | 519.88 | 518.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 524.95 | 520.08 | 518.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 10:00:00 | 524.80 | 521.03 | 519.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 519.30 | 521.21 | 519.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:00:00 | 519.30 | 521.21 | 519.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 520.10 | 520.99 | 519.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:45:00 | 518.05 | 520.99 | 519.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 519.05 | 520.60 | 519.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:30:00 | 518.60 | 520.60 | 519.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 519.75 | 520.43 | 519.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 519.60 | 520.43 | 519.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 518.50 | 520.05 | 519.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-03 15:15:00 | 516.05 | 519.12 | 519.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 15:15:00 | 516.05 | 519.12 | 519.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 09:15:00 | 514.95 | 518.28 | 518.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 10:15:00 | 521.40 | 518.91 | 519.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 10:15:00 | 521.40 | 518.91 | 519.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 521.40 | 518.91 | 519.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:45:00 | 521.75 | 518.91 | 519.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 518.25 | 518.77 | 519.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:15:00 | 522.00 | 518.77 | 519.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 12:15:00 | 522.70 | 519.56 | 519.44 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 514.40 | 518.73 | 519.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 10:15:00 | 512.35 | 517.45 | 518.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 11:15:00 | 519.25 | 517.81 | 518.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 11:15:00 | 519.25 | 517.81 | 518.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 519.25 | 517.81 | 518.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:45:00 | 521.75 | 517.81 | 518.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 516.50 | 517.55 | 518.39 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 13:15:00 | 526.50 | 519.34 | 519.13 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 516.05 | 519.36 | 519.36 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 13:15:00 | 523.30 | 519.79 | 519.50 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 13:15:00 | 518.10 | 519.37 | 519.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 14:15:00 | 517.20 | 518.93 | 519.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 09:15:00 | 519.95 | 518.80 | 519.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 519.95 | 518.80 | 519.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 519.95 | 518.80 | 519.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:00:00 | 511.00 | 516.72 | 517.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 13:15:00 | 517.35 | 514.42 | 514.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 517.35 | 514.42 | 514.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 518.85 | 516.54 | 515.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 517.35 | 518.20 | 516.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 517.35 | 518.20 | 516.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 517.35 | 518.20 | 516.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 517.20 | 518.20 | 516.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 516.40 | 517.84 | 516.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:30:00 | 515.50 | 517.84 | 516.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 521.05 | 518.48 | 517.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:30:00 | 515.85 | 518.48 | 517.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 531.20 | 531.73 | 527.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 527.50 | 531.73 | 527.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 529.00 | 530.68 | 528.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 524.35 | 530.68 | 528.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 526.95 | 529.93 | 528.34 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 12:15:00 | 520.95 | 526.48 | 527.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 13:15:00 | 519.50 | 525.08 | 526.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 11:15:00 | 518.00 | 517.68 | 520.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 11:15:00 | 518.00 | 517.68 | 520.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 518.00 | 517.68 | 520.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 509.00 | 514.08 | 516.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 09:45:00 | 510.80 | 511.20 | 513.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 15:00:00 | 509.35 | 512.48 | 513.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 512.15 | 510.29 | 510.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 511.40 | 510.70 | 511.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:45:00 | 511.60 | 510.70 | 511.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 13:15:00 | 483.55 | 494.56 | 500.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 13:15:00 | 485.26 | 494.56 | 500.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 13:15:00 | 483.88 | 494.56 | 500.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 13:15:00 | 486.54 | 494.56 | 500.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 495.70 | 491.80 | 495.54 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-04 13:15:00 | 495.70 | 491.80 | 495.54 | SL hit (close>ema200) qty=0.50 sl=491.80 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 469.10 | 462.60 | 462.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 12:15:00 | 474.00 | 464.88 | 463.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 10:15:00 | 513.35 | 514.30 | 508.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 11:00:00 | 513.35 | 514.30 | 508.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 506.85 | 511.96 | 508.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:00:00 | 506.85 | 511.96 | 508.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 507.20 | 511.01 | 508.33 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 12:15:00 | 503.75 | 507.13 | 507.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 14:15:00 | 499.50 | 504.89 | 506.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 11:15:00 | 508.65 | 504.45 | 505.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 11:15:00 | 508.65 | 504.45 | 505.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 508.65 | 504.45 | 505.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:00:00 | 508.65 | 504.45 | 505.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 508.40 | 505.24 | 505.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:45:00 | 508.35 | 505.24 | 505.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 13:15:00 | 509.30 | 506.05 | 505.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 09:15:00 | 514.40 | 508.23 | 507.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 12:15:00 | 509.85 | 509.99 | 508.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 12:15:00 | 509.85 | 509.99 | 508.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 509.85 | 509.99 | 508.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:30:00 | 507.90 | 509.99 | 508.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 506.55 | 509.30 | 508.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:00:00 | 506.55 | 509.30 | 508.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 511.80 | 509.80 | 508.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:30:00 | 512.25 | 509.80 | 508.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 508.50 | 510.01 | 508.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 504.50 | 510.01 | 508.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 507.25 | 509.46 | 508.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 506.60 | 509.46 | 508.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 505.75 | 508.44 | 508.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:00:00 | 505.75 | 508.44 | 508.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 505.30 | 507.82 | 508.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 498.15 | 505.88 | 507.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 504.90 | 504.42 | 506.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 09:15:00 | 504.90 | 504.42 | 506.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 504.90 | 504.42 | 506.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:00:00 | 504.90 | 504.42 | 506.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 506.50 | 504.84 | 506.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 506.50 | 504.84 | 506.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 503.75 | 504.62 | 506.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:15:00 | 502.50 | 504.62 | 506.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:45:00 | 502.40 | 504.24 | 505.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 13:30:00 | 502.65 | 503.93 | 505.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 14:00:00 | 502.70 | 503.93 | 505.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 497.35 | 501.66 | 503.94 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-01 11:15:00 | 506.70 | 502.11 | 502.63 | SL hit (close>static) qty=1.00 sl=506.65 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 507.35 | 503.71 | 503.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 512.45 | 506.40 | 504.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 508.00 | 508.60 | 506.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 15:00:00 | 508.00 | 508.60 | 506.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 506.20 | 508.12 | 506.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 509.70 | 508.12 | 506.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 10:00:00 | 508.75 | 508.25 | 506.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 12:30:00 | 508.60 | 507.81 | 506.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 509.05 | 507.33 | 506.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 508.55 | 508.97 | 508.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:45:00 | 508.10 | 508.97 | 508.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 504.35 | 508.04 | 507.71 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 504.35 | 508.04 | 507.71 | SL hit (close<static) qty=1.00 sl=505.15 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 505.00 | 507.43 | 507.47 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 515.15 | 506.85 | 506.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 11:15:00 | 524.05 | 510.99 | 508.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 11:15:00 | 515.90 | 517.82 | 514.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 12:00:00 | 515.90 | 517.82 | 514.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 518.90 | 517.30 | 515.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:30:00 | 520.65 | 517.10 | 515.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 511.30 | 515.45 | 514.71 | SL hit (close<static) qty=1.00 sl=514.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 13:15:00 | 510.65 | 513.77 | 514.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 14:15:00 | 509.75 | 512.97 | 513.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 508.35 | 507.91 | 509.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 508.35 | 507.91 | 509.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 508.35 | 507.91 | 509.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 509.55 | 507.91 | 509.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 510.40 | 508.41 | 509.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:00:00 | 510.40 | 508.41 | 509.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 511.10 | 508.95 | 510.05 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 10:15:00 | 513.35 | 510.94 | 510.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 515.90 | 512.26 | 511.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 513.55 | 515.16 | 513.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 09:15:00 | 513.55 | 515.16 | 513.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 513.55 | 515.16 | 513.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:30:00 | 520.60 | 517.38 | 514.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 533.60 | 535.91 | 536.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 533.60 | 535.91 | 536.03 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 15:15:00 | 538.00 | 536.33 | 536.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 09:15:00 | 577.65 | 544.59 | 539.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 575.00 | 577.35 | 567.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-25 15:00:00 | 575.00 | 577.35 | 567.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 577.55 | 577.17 | 568.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 10:00:00 | 586.75 | 575.49 | 571.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 13:15:00 | 568.95 | 571.30 | 571.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 13:15:00 | 568.95 | 571.30 | 571.41 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 576.35 | 571.55 | 571.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 579.00 | 573.04 | 571.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 584.00 | 585.72 | 581.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:30:00 | 586.40 | 585.72 | 581.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 578.15 | 584.21 | 580.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 578.15 | 584.21 | 580.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 580.50 | 583.47 | 580.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:30:00 | 582.20 | 581.40 | 580.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 15:15:00 | 580.80 | 585.00 | 585.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 15:15:00 | 580.80 | 585.00 | 585.35 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 586.20 | 585.57 | 585.57 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 584.25 | 585.31 | 585.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 14:15:00 | 582.45 | 584.25 | 584.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 587.10 | 584.38 | 584.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 587.10 | 584.38 | 584.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 587.10 | 584.38 | 584.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 587.50 | 584.38 | 584.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 587.50 | 585.01 | 585.06 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 587.75 | 585.55 | 585.30 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 580.05 | 585.51 | 585.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 565.00 | 581.41 | 583.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 565.60 | 563.10 | 569.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 15:00:00 | 565.60 | 563.10 | 569.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 565.10 | 562.70 | 567.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 565.10 | 562.70 | 567.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 567.00 | 563.56 | 567.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 568.65 | 563.56 | 567.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 565.30 | 563.91 | 567.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 14:00:00 | 565.00 | 564.12 | 567.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 15:15:00 | 570.40 | 565.88 | 567.52 | SL hit (close>static) qty=1.00 sl=567.60 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 584.00 | 569.50 | 569.02 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 15:15:00 | 566.55 | 572.81 | 573.60 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 576.50 | 574.34 | 574.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 12:15:00 | 577.70 | 575.01 | 574.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 578.00 | 578.10 | 576.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 14:15:00 | 578.00 | 578.10 | 576.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 578.00 | 578.10 | 576.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 578.00 | 578.10 | 576.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 576.85 | 577.85 | 576.43 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 566.55 | 574.27 | 575.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 564.25 | 569.42 | 572.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 15:15:00 | 565.00 | 564.72 | 567.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 09:15:00 | 567.50 | 564.72 | 567.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 569.95 | 565.77 | 567.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 572.25 | 565.77 | 567.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 568.45 | 566.31 | 567.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:30:00 | 569.90 | 566.31 | 567.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 568.90 | 566.82 | 567.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 568.40 | 566.82 | 567.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 566.80 | 566.82 | 567.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:30:00 | 567.60 | 566.82 | 567.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 566.70 | 566.67 | 567.45 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 568.85 | 567.84 | 567.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 14:15:00 | 575.90 | 569.45 | 568.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 565.50 | 569.60 | 568.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 565.50 | 569.60 | 568.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 565.50 | 569.60 | 568.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 565.50 | 569.60 | 568.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 570.60 | 569.80 | 568.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 566.20 | 569.80 | 568.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 568.30 | 569.50 | 568.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:00:00 | 568.30 | 569.50 | 568.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 568.00 | 569.20 | 568.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 14:15:00 | 569.65 | 569.10 | 568.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 15:15:00 | 570.90 | 568.88 | 568.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 11:45:00 | 575.50 | 571.93 | 570.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 13:45:00 | 569.90 | 571.84 | 570.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 563.15 | 570.10 | 569.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 563.15 | 570.10 | 569.96 | SL hit (close<static) qty=1.00 sl=567.05 alert=retest2 |

### Cycle 40 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 564.00 | 568.88 | 569.42 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 578.45 | 570.79 | 570.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 12:15:00 | 583.50 | 574.93 | 572.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 590.45 | 591.07 | 585.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 11:15:00 | 586.75 | 590.07 | 585.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 586.75 | 590.07 | 585.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:45:00 | 587.60 | 590.07 | 585.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 590.15 | 590.08 | 586.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 610.90 | 584.68 | 584.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 579.65 | 583.68 | 584.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 579.65 | 583.68 | 584.14 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 590.70 | 585.59 | 584.92 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 15:15:00 | 581.00 | 584.42 | 584.57 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 588.85 | 585.20 | 584.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 602.85 | 591.18 | 588.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 605.35 | 607.62 | 603.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 605.35 | 607.62 | 603.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 605.00 | 607.09 | 603.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 607.10 | 607.09 | 603.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 602.25 | 606.12 | 603.09 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 594.60 | 602.41 | 603.44 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 608.85 | 602.35 | 602.26 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 601.00 | 602.80 | 602.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 591.75 | 600.59 | 601.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 11:15:00 | 599.15 | 598.42 | 600.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 12:00:00 | 599.15 | 598.42 | 600.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 600.45 | 598.82 | 600.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 13:00:00 | 600.45 | 598.82 | 600.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 599.00 | 598.86 | 600.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 13:30:00 | 600.45 | 598.86 | 600.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 595.50 | 598.19 | 599.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:30:00 | 599.55 | 598.19 | 599.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 591.50 | 590.96 | 593.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:30:00 | 590.70 | 590.96 | 593.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 593.35 | 591.44 | 593.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:45:00 | 595.35 | 591.44 | 593.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 594.40 | 592.03 | 593.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 601.90 | 592.03 | 593.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 605.75 | 594.77 | 594.47 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 588.05 | 593.52 | 594.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 12:15:00 | 585.60 | 591.19 | 593.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 587.80 | 583.64 | 586.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 587.80 | 583.64 | 586.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 587.80 | 583.64 | 586.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 587.80 | 583.64 | 586.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 586.60 | 584.24 | 586.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:45:00 | 582.90 | 583.86 | 586.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 13:15:00 | 589.35 | 584.92 | 584.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 13:15:00 | 589.35 | 584.92 | 584.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 14:15:00 | 601.50 | 588.24 | 586.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 12:15:00 | 591.65 | 592.02 | 589.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 12:45:00 | 591.05 | 592.02 | 589.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 588.35 | 591.29 | 589.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:00:00 | 588.35 | 591.29 | 589.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 589.05 | 590.84 | 589.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 10:30:00 | 591.95 | 590.12 | 589.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 11:15:00 | 585.50 | 589.19 | 588.93 | SL hit (close<static) qty=1.00 sl=587.35 alert=retest2 |

### Cycle 52 — SELL (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 09:15:00 | 580.20 | 588.03 | 588.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 565.30 | 578.27 | 581.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 576.80 | 575.23 | 579.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:15:00 | 579.00 | 575.23 | 579.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 582.95 | 576.78 | 579.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:00:00 | 582.95 | 576.78 | 579.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 581.75 | 577.77 | 579.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:15:00 | 581.00 | 577.77 | 579.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 594.60 | 581.65 | 581.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 10:15:00 | 598.15 | 589.85 | 587.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 14:15:00 | 598.60 | 598.68 | 594.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 14:45:00 | 598.50 | 598.68 | 594.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 594.85 | 598.83 | 596.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 594.85 | 598.83 | 596.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 595.85 | 598.23 | 596.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:15:00 | 593.00 | 598.23 | 596.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 593.00 | 597.19 | 596.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 591.80 | 597.19 | 596.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 586.45 | 595.04 | 595.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 14:15:00 | 581.95 | 587.70 | 591.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 591.50 | 588.03 | 590.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 591.50 | 588.03 | 590.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 591.50 | 588.03 | 590.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 592.35 | 588.03 | 590.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 593.70 | 589.16 | 591.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:45:00 | 593.25 | 589.16 | 591.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 593.10 | 589.95 | 591.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:15:00 | 592.05 | 589.95 | 591.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 595.25 | 591.67 | 591.82 | SL hit (close>static) qty=1.00 sl=595.15 alert=retest2 |

### Cycle 55 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 596.45 | 592.62 | 592.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 10:15:00 | 601.00 | 595.49 | 593.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 600.35 | 601.16 | 598.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:00:00 | 600.35 | 601.16 | 598.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 599.25 | 600.78 | 598.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 599.10 | 600.78 | 598.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 596.50 | 599.92 | 598.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 596.50 | 599.92 | 598.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 597.30 | 599.40 | 598.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:45:00 | 597.05 | 599.40 | 598.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 597.25 | 598.97 | 598.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 596.15 | 598.97 | 598.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 596.00 | 598.37 | 597.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:30:00 | 596.05 | 597.90 | 597.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 10:15:00 | 592.85 | 596.89 | 597.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 12:15:00 | 591.00 | 595.14 | 596.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 15:15:00 | 585.00 | 584.09 | 588.32 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 09:45:00 | 573.65 | 581.79 | 586.89 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 572.90 | 568.50 | 572.47 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 572.90 | 568.50 | 572.47 | SL hit (close>ema400) qty=1.00 sl=572.47 alert=retest1 |

### Cycle 57 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 577.20 | 573.47 | 573.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 13:15:00 | 581.15 | 577.29 | 575.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 603.00 | 603.10 | 595.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 603.00 | 603.10 | 595.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 597.95 | 601.28 | 596.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:30:00 | 596.85 | 601.28 | 596.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 598.25 | 600.02 | 597.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 597.65 | 600.02 | 597.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 599.20 | 599.86 | 597.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 600.00 | 599.86 | 597.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 603.75 | 600.64 | 598.32 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 591.10 | 596.57 | 597.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 588.35 | 594.15 | 596.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 579.85 | 576.75 | 582.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 580.00 | 576.75 | 582.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 578.80 | 577.64 | 581.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:00:00 | 576.00 | 577.31 | 581.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 575.20 | 575.69 | 578.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 547.20 | 562.05 | 567.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 546.44 | 562.05 | 567.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-21 10:15:00 | 518.40 | 532.16 | 542.76 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 563.40 | 540.93 | 538.86 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 533.40 | 540.51 | 540.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 529.20 | 536.91 | 538.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 539.70 | 537.47 | 538.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 539.70 | 537.47 | 538.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 539.70 | 537.47 | 538.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 539.70 | 537.47 | 538.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 540.35 | 538.04 | 539.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 542.30 | 538.04 | 539.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 537.80 | 537.31 | 538.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 537.80 | 537.31 | 538.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 540.90 | 538.03 | 538.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:30:00 | 539.75 | 538.03 | 538.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 540.95 | 538.61 | 538.93 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 543.30 | 539.79 | 539.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 544.60 | 540.75 | 539.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 563.50 | 563.91 | 556.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 563.50 | 563.91 | 556.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 563.50 | 563.91 | 556.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 563.50 | 563.91 | 556.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 545.00 | 559.33 | 556.85 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 547.80 | 555.37 | 555.38 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 560.40 | 555.39 | 555.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 573.90 | 559.10 | 556.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 578.05 | 578.67 | 573.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 578.05 | 578.67 | 573.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 584.90 | 589.83 | 585.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:45:00 | 583.75 | 589.83 | 585.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 584.00 | 588.66 | 585.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:30:00 | 585.95 | 587.93 | 585.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 10:15:00 | 586.75 | 587.93 | 585.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 11:45:00 | 585.80 | 587.60 | 585.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 14:45:00 | 585.85 | 586.38 | 585.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 582.15 | 585.53 | 584.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 598.70 | 585.53 | 584.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 595.35 | 589.07 | 587.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:30:00 | 593.95 | 593.96 | 591.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 596.05 | 592.73 | 592.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 599.30 | 594.05 | 592.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 12:15:00 | 602.65 | 595.79 | 593.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 10:15:00 | 582.05 | 591.92 | 593.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 10:15:00 | 582.05 | 591.92 | 593.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 11:15:00 | 578.05 | 589.14 | 591.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 595.00 | 588.59 | 590.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 14:15:00 | 595.00 | 588.59 | 590.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 595.00 | 588.59 | 590.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 595.00 | 588.59 | 590.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 595.00 | 589.87 | 590.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 588.00 | 589.87 | 590.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 15:15:00 | 583.00 | 577.49 | 576.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 15:15:00 | 583.00 | 577.49 | 576.99 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 573.15 | 576.72 | 576.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 12:15:00 | 572.10 | 575.79 | 576.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 15:15:00 | 574.80 | 574.33 | 575.44 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:15:00 | 561.55 | 574.33 | 575.44 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 533.47 | 557.70 | 565.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 529.20 | 528.84 | 538.75 | SL hit (close>ema200) qty=0.50 sl=528.84 alert=retest1 |

### Cycle 67 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 489.05 | 485.36 | 484.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 499.65 | 488.80 | 486.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 496.90 | 500.83 | 495.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 496.90 | 500.83 | 495.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 496.90 | 500.83 | 495.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:00:00 | 501.00 | 500.86 | 495.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 489.15 | 499.86 | 499.26 | SL hit (close<static) qty=1.00 sl=494.35 alert=retest2 |

### Cycle 68 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 488.30 | 497.55 | 498.26 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 522.30 | 500.82 | 498.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 15:15:00 | 525.85 | 505.82 | 500.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 519.30 | 525.07 | 515.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:45:00 | 520.05 | 525.07 | 515.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 515.80 | 523.22 | 515.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 513.15 | 523.22 | 515.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 515.60 | 521.69 | 515.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:45:00 | 515.80 | 521.69 | 515.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 515.65 | 520.48 | 515.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:30:00 | 516.80 | 520.09 | 515.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 14:45:00 | 542.00 | 520.29 | 516.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 504.25 | 518.64 | 516.23 | SL hit (close<static) qty=1.00 sl=513.05 alert=retest2 |

### Cycle 70 — SELL (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 11:15:00 | 504.85 | 513.18 | 514.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 492.10 | 506.60 | 509.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 501.65 | 501.04 | 505.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 14:00:00 | 501.65 | 501.04 | 505.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 498.40 | 500.21 | 503.75 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2026-04-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 15:15:00 | 508.30 | 505.54 | 505.24 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 13:15:00 | 501.60 | 504.74 | 504.95 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 15:15:00 | 507.35 | 505.43 | 505.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 525.70 | 509.48 | 507.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 522.20 | 524.39 | 517.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:45:00 | 520.05 | 524.39 | 517.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 518.15 | 522.55 | 518.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 12:45:00 | 518.50 | 522.55 | 518.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 517.50 | 521.54 | 518.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:30:00 | 517.45 | 521.54 | 518.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 516.70 | 520.57 | 518.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:45:00 | 514.85 | 520.57 | 518.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 514.35 | 517.72 | 517.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:45:00 | 516.20 | 517.72 | 517.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 516.60 | 517.49 | 517.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:45:00 | 514.00 | 517.49 | 517.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 12:15:00 | 516.90 | 517.38 | 517.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:45:00 | 514.70 | 517.38 | 517.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2026-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 13:15:00 | 515.45 | 516.99 | 517.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-10 15:15:00 | 513.00 | 515.71 | 516.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 15:15:00 | 513.95 | 512.44 | 513.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-15 09:15:00 | 516.30 | 512.44 | 513.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 518.55 | 513.66 | 514.40 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 519.50 | 515.60 | 515.20 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 09:15:00 | 513.95 | 515.07 | 515.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 12:15:00 | 505.05 | 512.97 | 514.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 519.45 | 511.11 | 512.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 09:15:00 | 519.45 | 511.11 | 512.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 519.45 | 511.11 | 512.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 10:00:00 | 519.45 | 511.11 | 512.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 10:15:00 | 527.35 | 514.36 | 513.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 11:15:00 | 537.25 | 518.93 | 516.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 10:15:00 | 542.30 | 542.36 | 537.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-22 11:00:00 | 542.30 | 542.36 | 537.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 539.80 | 542.25 | 539.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:00:00 | 539.80 | 542.25 | 539.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 538.80 | 541.56 | 539.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:30:00 | 538.10 | 541.56 | 539.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 543.00 | 541.85 | 539.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:30:00 | 538.90 | 541.85 | 539.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 540.45 | 541.57 | 539.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 12:45:00 | 540.70 | 541.57 | 539.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 540.30 | 541.13 | 540.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:45:00 | 539.65 | 541.13 | 540.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 538.95 | 540.70 | 539.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 533.65 | 540.70 | 539.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 528.75 | 538.31 | 538.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 09:15:00 | 526.25 | 530.12 | 533.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 14:15:00 | 527.35 | 526.48 | 530.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 15:00:00 | 527.35 | 526.48 | 530.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 516.60 | 520.75 | 524.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:30:00 | 514.55 | 519.70 | 523.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 514.40 | 520.64 | 522.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:15:00 | 514.80 | 519.75 | 522.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 11:00:00 | 515.50 | 518.90 | 521.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 520.85 | 518.62 | 520.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 15:00:00 | 520.85 | 518.62 | 520.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 520.00 | 518.90 | 520.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 525.85 | 518.90 | 520.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 524.25 | 519.97 | 520.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:45:00 | 519.95 | 519.93 | 520.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 519.80 | 520.04 | 520.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 14:00:00 | 514.45 | 518.93 | 520.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 12:00:00 | 518.15 | 519.16 | 519.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 519.50 | 519.23 | 519.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:00:00 | 519.50 | 519.23 | 519.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 516.00 | 518.58 | 519.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-06 10:15:00 | 522.00 | 519.90 | 519.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 522.00 | 519.90 | 519.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 525.75 | 521.43 | 520.52 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-19 10:15:00 | 517.70 | 2025-05-20 13:15:00 | 500.50 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2025-06-18 15:15:00 | 535.70 | 2025-06-19 13:15:00 | 508.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 15:15:00 | 535.70 | 2025-06-24 09:15:00 | 512.00 | STOP_HIT | 0.50 | 4.42% |
| BUY | retest2 | 2025-07-01 11:15:00 | 521.50 | 2025-07-03 15:15:00 | 516.05 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-07-01 14:30:00 | 521.20 | 2025-07-03 15:15:00 | 516.05 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-02 09:15:00 | 524.95 | 2025-07-03 15:15:00 | 516.05 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-07-02 10:00:00 | 524.80 | 2025-07-03 15:15:00 | 516.05 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-07-11 10:00:00 | 511.00 | 2025-07-14 13:15:00 | 517.35 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-07-25 09:30:00 | 509.00 | 2025-08-01 13:15:00 | 483.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-28 09:45:00 | 510.80 | 2025-08-01 13:15:00 | 485.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-28 15:00:00 | 509.35 | 2025-08-01 13:15:00 | 483.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 10:15:00 | 512.15 | 2025-08-01 13:15:00 | 486.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 09:30:00 | 509.00 | 2025-08-04 13:15:00 | 495.70 | STOP_HIT | 0.50 | 2.61% |
| SELL | retest2 | 2025-07-28 09:45:00 | 510.80 | 2025-08-04 13:15:00 | 495.70 | STOP_HIT | 0.50 | 2.96% |
| SELL | retest2 | 2025-07-28 15:00:00 | 509.35 | 2025-08-04 13:15:00 | 495.70 | STOP_HIT | 0.50 | 2.68% |
| SELL | retest2 | 2025-07-30 10:15:00 | 512.15 | 2025-08-04 13:15:00 | 495.70 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2025-08-05 09:15:00 | 491.40 | 2025-08-06 10:15:00 | 466.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 09:15:00 | 491.40 | 2025-08-11 14:15:00 | 460.15 | STOP_HIT | 0.50 | 6.36% |
| SELL | retest2 | 2025-08-28 12:15:00 | 502.50 | 2025-09-01 11:15:00 | 506.70 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-08-28 12:45:00 | 502.40 | 2025-09-01 11:15:00 | 506.70 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-08-28 13:30:00 | 502.65 | 2025-09-01 11:15:00 | 506.70 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-08-28 14:00:00 | 502.70 | 2025-09-01 11:15:00 | 506.70 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-09-03 09:15:00 | 509.70 | 2025-09-04 14:15:00 | 504.35 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-09-03 10:00:00 | 508.75 | 2025-09-04 14:15:00 | 504.35 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-09-03 12:30:00 | 508.60 | 2025-09-04 14:15:00 | 504.35 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-09-04 09:15:00 | 509.05 | 2025-09-04 14:15:00 | 504.35 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-09-10 09:30:00 | 520.65 | 2025-09-10 11:15:00 | 511.30 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-09-17 10:30:00 | 520.60 | 2025-09-23 14:15:00 | 533.60 | STOP_HIT | 1.00 | 2.50% |
| BUY | retest2 | 2025-09-29 10:00:00 | 586.75 | 2025-09-30 13:15:00 | 568.95 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2025-10-07 09:30:00 | 582.20 | 2025-10-08 15:15:00 | 580.80 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-10-15 14:00:00 | 565.00 | 2025-10-15 15:15:00 | 570.40 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-10-30 14:15:00 | 569.65 | 2025-10-31 14:15:00 | 563.15 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-10-30 15:15:00 | 570.90 | 2025-10-31 14:15:00 | 563.15 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-10-31 11:45:00 | 575.50 | 2025-10-31 14:15:00 | 563.15 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-10-31 13:45:00 | 569.90 | 2025-10-31 14:15:00 | 563.15 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-11-07 09:15:00 | 610.90 | 2025-11-07 09:15:00 | 579.65 | STOP_HIT | 1.00 | -5.12% |
| SELL | retest2 | 2025-12-01 11:45:00 | 582.90 | 2025-12-02 13:15:00 | 589.35 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-04 10:30:00 | 591.95 | 2025-12-04 11:15:00 | 585.50 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-12-19 12:15:00 | 592.05 | 2025-12-19 13:15:00 | 595.25 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2025-12-29 09:45:00 | 573.65 | 2025-12-31 09:15:00 | 572.90 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2026-01-13 11:00:00 | 576.00 | 2026-01-19 09:15:00 | 547.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:15:00 | 575.20 | 2026-01-19 09:15:00 | 546.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 11:00:00 | 576.00 | 2026-01-21 10:15:00 | 518.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 09:15:00 | 575.20 | 2026-01-21 10:15:00 | 517.68 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-09 09:30:00 | 585.95 | 2026-02-16 10:15:00 | 582.05 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2026-02-09 10:15:00 | 586.75 | 2026-02-16 10:15:00 | 582.05 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-02-09 11:45:00 | 585.80 | 2026-02-16 10:15:00 | 582.05 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-02-09 14:45:00 | 585.85 | 2026-02-16 10:15:00 | 582.05 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2026-02-10 09:15:00 | 598.70 | 2026-02-16 10:15:00 | 582.05 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2026-02-11 09:15:00 | 595.35 | 2026-02-16 10:15:00 | 582.05 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-02-12 09:30:00 | 593.95 | 2026-02-16 10:15:00 | 582.05 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2026-02-13 09:15:00 | 596.05 | 2026-02-16 10:15:00 | 582.05 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2026-02-13 12:15:00 | 602.65 | 2026-02-16 10:15:00 | 582.05 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2026-02-17 09:15:00 | 588.00 | 2026-02-25 15:15:00 | 583.00 | STOP_HIT | 1.00 | 0.85% |
| SELL | retest1 | 2026-02-27 09:15:00 | 561.55 | 2026-03-02 09:15:00 | 533.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-02-27 09:15:00 | 561.55 | 2026-03-05 09:15:00 | 529.20 | STOP_HIT | 0.50 | 5.76% |
| SELL | retest2 | 2026-03-06 09:30:00 | 525.05 | 2026-03-09 09:15:00 | 498.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 09:30:00 | 525.05 | 2026-03-10 09:15:00 | 509.20 | STOP_HIT | 0.50 | 3.02% |
| BUY | retest2 | 2026-03-19 11:00:00 | 501.00 | 2026-03-23 09:15:00 | 489.15 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2026-03-27 13:30:00 | 516.80 | 2026-03-30 09:15:00 | 504.25 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2026-03-27 14:45:00 | 542.00 | 2026-03-30 09:15:00 | 504.25 | STOP_HIT | 1.00 | -6.96% |
| SELL | retest2 | 2026-04-29 10:30:00 | 514.55 | 2026-05-06 10:15:00 | 522.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-04-30 09:15:00 | 514.40 | 2026-05-06 10:15:00 | 522.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-04-30 10:15:00 | 514.80 | 2026-05-06 10:15:00 | 522.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-04-30 11:00:00 | 515.50 | 2026-05-06 10:15:00 | 522.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-05-04 11:45:00 | 519.95 | 2026-05-06 10:15:00 | 522.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2026-05-04 13:15:00 | 519.80 | 2026-05-06 10:15:00 | 522.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-05-04 14:00:00 | 514.45 | 2026-05-06 10:15:00 | 522.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-05-05 12:00:00 | 518.15 | 2026-05-06 10:15:00 | 522.00 | STOP_HIT | 1.00 | -0.74% |
