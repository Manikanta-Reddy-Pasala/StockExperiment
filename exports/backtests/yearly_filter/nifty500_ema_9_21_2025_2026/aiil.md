# Authum Investment & Infrastructure Ltd. (AIIL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-11 15:15:00 (1983 bars)
- **Last close:** 504.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 71 |
| ALERT1 | 43 |
| ALERT2 | 42 |
| ALERT2_SKIP | 23 |
| ALERT3 | 109 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 52 |
| PARTIAL | 9 |
| TARGET_HIT | 10 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 63 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 38
- **Target hits / Stop hits / Partials:** 10 / 44 / 9
- **Avg / median % per leg:** 0.99% / -1.17%
- **Sum % (uncompounded):** 62.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 10 | 34.5% | 6 | 22 | 1 | 1.24% | 35.9% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.09% | 6.3% |
| BUY @ 3rd Alert (retest2) | 26 | 8 | 30.8% | 6 | 20 | 0 | 1.14% | 29.6% |
| SELL (all) | 34 | 15 | 44.1% | 4 | 22 | 8 | 0.78% | 26.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 34 | 15 | 44.1% | 4 | 22 | 8 | 0.78% | 26.5% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.09% | 6.3% |
| retest2 (combined) | 60 | 23 | 38.3% | 10 | 42 | 8 | 0.94% | 56.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 373.64 | 358.46 | 356.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 12:15:00 | 393.60 | 375.84 | 368.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 10:15:00 | 423.00 | 426.45 | 416.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 11:00:00 | 423.00 | 426.45 | 416.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 440.34 | 443.12 | 437.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 440.34 | 443.12 | 437.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 432.86 | 441.07 | 437.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 432.86 | 441.07 | 437.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 441.36 | 441.13 | 437.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 15:00:00 | 444.60 | 441.74 | 438.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 443.56 | 441.59 | 438.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 15:15:00 | 442.20 | 439.53 | 438.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 15:00:00 | 442.40 | 443.46 | 442.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 446.56 | 443.46 | 442.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 457.56 | 444.83 | 444.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-03 09:15:00 | 489.06 | 479.45 | 472.52 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 495.34 | 500.13 | 500.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 490.84 | 497.99 | 499.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 492.30 | 489.58 | 492.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 492.30 | 489.58 | 492.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 497.42 | 491.15 | 492.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 497.42 | 491.15 | 492.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 498.80 | 492.68 | 493.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 498.52 | 492.68 | 493.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 501.12 | 494.37 | 494.18 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 490.84 | 493.56 | 493.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 14:15:00 | 487.40 | 491.68 | 492.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 501.50 | 492.99 | 493.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 501.50 | 492.99 | 493.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 501.50 | 492.99 | 493.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:45:00 | 511.90 | 492.99 | 493.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 496.72 | 493.73 | 493.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 14:15:00 | 506.38 | 498.79 | 496.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 15:15:00 | 518.00 | 518.62 | 514.39 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:15:00 | 522.80 | 518.62 | 514.39 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 531.06 | 537.78 | 532.65 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-26 10:15:00 | 531.06 | 537.78 | 532.65 | SL hit (close<ema400) qty=1.00 sl=532.65 alert=retest1 |

### Cycle 6 — SELL (started 2025-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 14:15:00 | 525.88 | 529.59 | 529.83 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 530.76 | 530.08 | 530.03 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 14:15:00 | 524.98 | 529.89 | 530.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 10:15:00 | 516.88 | 525.33 | 527.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 14:15:00 | 525.78 | 522.33 | 525.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 14:15:00 | 525.78 | 522.33 | 525.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 525.78 | 522.33 | 525.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 525.78 | 522.33 | 525.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 521.10 | 522.08 | 524.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 516.96 | 522.08 | 524.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 09:15:00 | 491.11 | 503.12 | 511.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-02 14:15:00 | 501.96 | 501.83 | 507.83 | SL hit (close>ema200) qty=0.50 sl=501.83 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 12:15:00 | 534.40 | 512.80 | 510.91 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 522.02 | 531.66 | 532.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 12:15:00 | 514.74 | 526.63 | 529.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 14:15:00 | 526.78 | 525.53 | 528.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 14:15:00 | 526.78 | 525.53 | 528.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 526.78 | 525.53 | 528.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:30:00 | 528.00 | 525.53 | 528.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 531.40 | 526.62 | 528.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:45:00 | 526.20 | 527.12 | 528.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 550.52 | 529.02 | 528.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 550.52 | 529.02 | 528.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 558.94 | 544.55 | 536.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 539.52 | 544.96 | 539.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 539.52 | 544.96 | 539.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 539.52 | 544.96 | 539.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:00:00 | 539.52 | 544.96 | 539.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 532.52 | 542.48 | 538.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:30:00 | 530.22 | 542.48 | 538.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 540.00 | 541.98 | 538.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 12:15:00 | 542.28 | 541.98 | 538.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 15:15:00 | 542.40 | 540.88 | 538.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 13:15:00 | 554.34 | 571.22 | 571.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 13:15:00 | 554.34 | 571.22 | 571.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 510.84 | 556.46 | 564.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 14:15:00 | 551.00 | 539.94 | 551.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 14:15:00 | 551.00 | 539.94 | 551.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 551.00 | 539.94 | 551.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 15:00:00 | 551.00 | 539.94 | 551.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 542.00 | 540.35 | 550.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 538.68 | 540.35 | 550.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 10:00:00 | 539.78 | 540.24 | 549.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:15:00 | 537.50 | 540.61 | 548.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 12:15:00 | 540.24 | 538.37 | 542.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 543.08 | 539.31 | 542.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:30:00 | 545.06 | 539.31 | 542.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 553.30 | 542.11 | 543.86 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-29 13:15:00 | 553.30 | 542.11 | 543.86 | SL hit (close>static) qty=1.00 sl=551.38 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 565.00 | 546.69 | 545.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 576.92 | 555.66 | 550.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 561.78 | 564.72 | 558.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 561.78 | 564.72 | 558.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 561.78 | 564.72 | 558.79 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 553.64 | 557.22 | 557.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 15:15:00 | 550.00 | 554.87 | 556.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 555.22 | 554.06 | 555.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 11:15:00 | 555.22 | 554.06 | 555.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 555.22 | 554.06 | 555.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 555.22 | 554.06 | 555.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 555.76 | 554.40 | 555.49 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 560.56 | 556.59 | 556.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 14:15:00 | 575.64 | 563.43 | 560.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 573.64 | 574.29 | 569.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 13:15:00 | 565.00 | 571.94 | 569.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 565.00 | 571.94 | 569.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:00:00 | 565.00 | 571.94 | 569.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 572.44 | 572.04 | 569.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:30:00 | 567.30 | 572.04 | 569.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 577.86 | 572.97 | 570.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:30:00 | 571.52 | 572.97 | 570.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 571.28 | 574.39 | 572.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 14:00:00 | 571.28 | 574.39 | 572.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 570.08 | 573.53 | 572.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 570.08 | 573.53 | 572.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 566.44 | 572.11 | 571.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 570.36 | 572.11 | 571.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 563.80 | 570.45 | 570.84 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 573.00 | 570.31 | 570.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 12:15:00 | 577.18 | 573.49 | 571.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 15:15:00 | 572.94 | 574.28 | 572.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 15:15:00 | 572.94 | 574.28 | 572.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 572.94 | 574.28 | 572.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 577.98 | 574.28 | 572.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 10:15:00 | 577.76 | 588.65 | 583.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 10:15:00 | 570.98 | 585.11 | 582.29 | SL hit (close<static) qty=1.00 sl=572.30 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 551.84 | 578.46 | 579.52 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 601.40 | 581.79 | 580.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 603.80 | 586.19 | 582.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 591.54 | 593.17 | 587.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 15:00:00 | 591.54 | 593.17 | 587.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 582.08 | 592.10 | 589.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:00:00 | 582.08 | 592.10 | 589.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 585.98 | 590.87 | 589.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 14:30:00 | 587.66 | 589.02 | 588.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 09:15:00 | 580.00 | 586.60 | 587.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 09:15:00 | 580.00 | 586.60 | 587.40 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 12:15:00 | 592.68 | 588.59 | 588.16 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 13:15:00 | 582.68 | 587.41 | 587.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 571.80 | 584.29 | 586.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 582.62 | 581.79 | 584.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 10:00:00 | 582.62 | 581.79 | 584.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 586.18 | 582.67 | 584.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 585.00 | 582.67 | 584.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 584.22 | 582.98 | 584.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:30:00 | 576.80 | 582.21 | 584.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 14:45:00 | 581.02 | 582.36 | 583.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 15:15:00 | 577.00 | 582.36 | 583.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 589.82 | 584.36 | 584.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 11:15:00 | 589.82 | 584.36 | 584.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 12:15:00 | 592.96 | 586.08 | 585.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 15:15:00 | 602.00 | 604.33 | 597.98 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-26 10:00:00 | 615.88 | 606.64 | 599.60 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 646.67 | 635.74 | 624.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 614.00 | 637.28 | 631.99 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 614.00 | 637.28 | 631.99 | SL hit (close<ema200) qty=0.50 sl=637.28 alert=retest1 |

### Cycle 24 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 628.26 | 637.81 | 638.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 614.98 | 633.25 | 636.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 12:15:00 | 618.42 | 617.14 | 624.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 12:45:00 | 618.38 | 617.14 | 624.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 617.98 | 617.21 | 622.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:15:00 | 608.40 | 616.78 | 620.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 625.52 | 620.78 | 620.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 625.52 | 620.78 | 620.25 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 15:15:00 | 618.40 | 621.22 | 621.38 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 625.94 | 622.16 | 621.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 630.40 | 625.65 | 623.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 11:15:00 | 653.00 | 653.47 | 647.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 11:45:00 | 652.92 | 653.47 | 647.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 638.30 | 650.44 | 648.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 638.30 | 650.44 | 648.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 640.28 | 648.41 | 647.57 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 11:15:00 | 636.58 | 646.05 | 646.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 633.02 | 643.44 | 645.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 14:15:00 | 636.28 | 635.85 | 639.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 14:15:00 | 636.28 | 635.85 | 639.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 636.28 | 635.85 | 639.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 15:15:00 | 626.00 | 635.85 | 639.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 594.70 | 599.88 | 604.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 618.72 | 601.57 | 603.69 | SL hit (close>ema200) qty=0.50 sl=601.57 alert=retest2 |

### Cycle 29 — BUY (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 11:15:00 | 612.98 | 606.41 | 605.68 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 15:15:00 | 602.88 | 605.16 | 605.36 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 09:15:00 | 618.00 | 607.73 | 606.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 09:15:00 | 619.68 | 614.52 | 610.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 14:15:00 | 616.56 | 616.71 | 613.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 15:15:00 | 619.84 | 616.71 | 613.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 631.58 | 634.48 | 628.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:00:00 | 631.58 | 634.48 | 628.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 653.34 | 657.16 | 654.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 652.88 | 657.16 | 654.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 651.38 | 656.01 | 654.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:45:00 | 651.08 | 656.01 | 654.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 635.56 | 650.19 | 651.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 622.78 | 636.93 | 643.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 645.66 | 632.99 | 638.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 645.66 | 632.99 | 638.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 645.66 | 632.99 | 638.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 645.66 | 632.99 | 638.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 646.98 | 635.79 | 639.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 646.98 | 635.79 | 639.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 651.06 | 642.86 | 641.84 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 641.92 | 643.06 | 643.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 14:15:00 | 633.10 | 641.07 | 642.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 638.88 | 634.54 | 636.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 638.88 | 634.54 | 636.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 638.88 | 634.54 | 636.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 638.32 | 634.54 | 636.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 637.60 | 635.15 | 636.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 09:15:00 | 632.76 | 635.15 | 636.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 11:15:00 | 601.12 | 607.68 | 612.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 12:15:00 | 601.70 | 600.83 | 605.73 | SL hit (close>ema200) qty=0.50 sl=600.83 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 558.02 | 551.53 | 551.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 15:15:00 | 562.20 | 553.66 | 552.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 562.00 | 565.97 | 561.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 09:45:00 | 562.60 | 565.97 | 561.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 564.80 | 565.74 | 561.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:15:00 | 561.58 | 565.74 | 561.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 562.40 | 565.07 | 561.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:30:00 | 561.66 | 565.07 | 561.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 561.30 | 564.31 | 561.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 561.30 | 564.31 | 561.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 561.64 | 563.78 | 561.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:30:00 | 561.40 | 563.78 | 561.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 554.66 | 561.96 | 560.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 554.66 | 561.96 | 560.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 554.60 | 560.48 | 560.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 549.34 | 560.48 | 560.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 09:15:00 | 544.02 | 557.19 | 558.81 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 12:15:00 | 560.48 | 558.23 | 558.06 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 09:15:00 | 550.88 | 557.17 | 557.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 10:15:00 | 548.46 | 555.43 | 556.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 12:15:00 | 542.78 | 541.91 | 545.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 13:00:00 | 542.78 | 541.91 | 545.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 534.28 | 534.90 | 539.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 526.48 | 539.79 | 540.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 10:15:00 | 541.00 | 531.81 | 533.87 | SL hit (close>static) qty=1.00 sl=540.72 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 543.48 | 535.79 | 535.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 13:15:00 | 548.22 | 540.29 | 538.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 536.80 | 541.62 | 539.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 536.80 | 541.62 | 539.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 536.80 | 541.62 | 539.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 536.80 | 541.62 | 539.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 541.14 | 541.52 | 539.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 11:15:00 | 542.64 | 541.52 | 539.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:15:00 | 542.66 | 541.44 | 539.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 528.12 | 538.47 | 539.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 09:15:00 | 528.12 | 538.47 | 539.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 522.60 | 528.50 | 532.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 521.54 | 521.01 | 523.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 521.54 | 521.01 | 523.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 531.02 | 522.60 | 523.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 531.02 | 522.60 | 523.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 530.54 | 524.19 | 524.02 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 521.72 | 524.66 | 524.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 520.16 | 523.76 | 524.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 516.78 | 509.37 | 512.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 516.78 | 509.37 | 512.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 516.78 | 509.37 | 512.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 517.98 | 509.37 | 512.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 516.92 | 510.88 | 512.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 516.92 | 510.88 | 512.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 519.80 | 514.01 | 513.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 529.40 | 520.22 | 517.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 13:15:00 | 512.64 | 521.25 | 518.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 13:15:00 | 512.64 | 521.25 | 518.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 512.64 | 521.25 | 518.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:00:00 | 512.64 | 521.25 | 518.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 522.04 | 521.41 | 519.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:30:00 | 521.70 | 521.41 | 519.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 515.56 | 520.17 | 518.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 516.04 | 520.17 | 518.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 521.18 | 520.37 | 519.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 515.84 | 520.37 | 519.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 517.80 | 519.86 | 519.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:00:00 | 517.80 | 519.86 | 519.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 521.54 | 520.20 | 519.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:30:00 | 519.90 | 520.20 | 519.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 550.56 | 555.95 | 548.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 549.64 | 555.95 | 548.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 543.18 | 553.39 | 548.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:45:00 | 541.30 | 553.39 | 548.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 545.00 | 551.72 | 547.77 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 15:15:00 | 540.00 | 545.08 | 545.66 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 551.84 | 546.43 | 546.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 11:15:00 | 553.28 | 548.65 | 547.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 573.72 | 575.27 | 567.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:45:00 | 573.00 | 575.27 | 567.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 578.46 | 580.12 | 577.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 578.48 | 580.12 | 577.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 576.18 | 579.33 | 577.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 578.74 | 579.33 | 577.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 579.72 | 579.41 | 577.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 10:30:00 | 582.06 | 579.34 | 578.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 13:15:00 | 581.90 | 579.39 | 578.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 571.56 | 576.99 | 577.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 571.56 | 576.99 | 577.50 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 11:15:00 | 579.32 | 577.94 | 577.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 13:15:00 | 586.80 | 580.04 | 578.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 15:15:00 | 624.20 | 625.19 | 615.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 09:15:00 | 619.48 | 625.19 | 615.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 613.62 | 622.88 | 614.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 613.62 | 622.88 | 614.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 615.40 | 621.38 | 614.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:00:00 | 619.20 | 616.13 | 614.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:30:00 | 619.14 | 616.63 | 614.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:15:00 | 619.12 | 616.63 | 614.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:45:00 | 619.16 | 617.18 | 615.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 617.98 | 624.86 | 621.89 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-07 13:15:00 | 618.50 | 620.26 | 620.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2026-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 13:15:00 | 618.50 | 620.26 | 620.31 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 09:15:00 | 624.98 | 620.70 | 620.47 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 616.92 | 620.05 | 620.22 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 15:15:00 | 622.00 | 620.32 | 620.19 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 617.00 | 619.66 | 619.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 612.24 | 616.51 | 618.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 618.08 | 614.46 | 616.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 618.08 | 614.46 | 616.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 618.08 | 614.46 | 616.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 13:15:00 | 609.78 | 614.09 | 615.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 620.00 | 617.10 | 617.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 620.00 | 617.10 | 617.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 646.30 | 622.94 | 619.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 15:15:00 | 645.00 | 653.02 | 639.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 636.00 | 657.44 | 650.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 636.00 | 657.44 | 650.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:30:00 | 634.40 | 657.44 | 650.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 630.30 | 652.01 | 648.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:00:00 | 630.30 | 652.01 | 648.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 12:15:00 | 628.00 | 643.59 | 645.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 617.40 | 634.66 | 640.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 519.00 | 514.43 | 532.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 15:00:00 | 519.00 | 514.43 | 532.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 508.90 | 504.42 | 515.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 499.60 | 511.39 | 515.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 500.90 | 507.71 | 511.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 495.30 | 506.81 | 509.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:15:00 | 501.30 | 500.75 | 505.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 506.65 | 501.58 | 504.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:00:00 | 506.65 | 501.58 | 504.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 510.60 | 503.38 | 504.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:30:00 | 508.90 | 503.38 | 504.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 526.60 | 508.60 | 506.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 526.60 | 508.60 | 506.86 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 13:15:00 | 504.20 | 512.66 | 513.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 15:15:00 | 502.20 | 509.18 | 511.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 501.40 | 497.98 | 502.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 11:15:00 | 501.40 | 497.98 | 502.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 501.40 | 497.98 | 502.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:45:00 | 501.15 | 497.98 | 502.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 505.25 | 499.43 | 502.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:45:00 | 505.35 | 499.43 | 502.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 500.60 | 499.66 | 502.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 506.50 | 499.66 | 502.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 505.80 | 500.89 | 502.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 505.80 | 500.89 | 502.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 504.00 | 501.51 | 502.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 496.95 | 501.51 | 502.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 497.40 | 500.69 | 502.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 484.00 | 498.86 | 500.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 494.50 | 493.83 | 495.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 13:15:00 | 500.85 | 497.22 | 496.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 13:15:00 | 500.85 | 497.22 | 496.86 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 487.15 | 494.78 | 495.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 10:15:00 | 484.95 | 488.97 | 491.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 500.70 | 488.85 | 490.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 14:15:00 | 500.70 | 488.85 | 490.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 500.70 | 488.85 | 490.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:45:00 | 500.40 | 488.85 | 490.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 501.65 | 491.41 | 491.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 501.75 | 491.41 | 491.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 503.80 | 493.89 | 492.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 513.20 | 500.17 | 497.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 09:15:00 | 504.00 | 506.23 | 502.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 504.00 | 506.23 | 502.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 504.00 | 506.23 | 502.57 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 13:15:00 | 499.75 | 502.36 | 502.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 497.45 | 500.98 | 501.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 15:15:00 | 500.00 | 498.14 | 499.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 15:15:00 | 500.00 | 498.14 | 499.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 500.00 | 498.14 | 499.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:45:00 | 493.90 | 496.80 | 498.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:30:00 | 493.60 | 495.58 | 497.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 14:45:00 | 494.50 | 494.81 | 497.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 15:15:00 | 490.00 | 494.81 | 497.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 11:15:00 | 469.20 | 476.97 | 483.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 11:15:00 | 468.92 | 476.97 | 483.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 11:15:00 | 469.77 | 476.97 | 483.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 11:15:00 | 465.50 | 476.97 | 483.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 444.51 | 459.05 | 471.58 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 478.80 | 437.38 | 432.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 484.90 | 453.74 | 441.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 09:15:00 | 452.10 | 462.80 | 449.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 09:45:00 | 455.50 | 462.80 | 449.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 454.60 | 459.61 | 450.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:30:00 | 455.00 | 459.61 | 450.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 452.25 | 456.75 | 451.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 452.25 | 456.75 | 451.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 452.10 | 455.82 | 451.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 434.55 | 455.82 | 451.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 436.90 | 452.03 | 449.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 434.75 | 452.03 | 449.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 11:15:00 | 441.20 | 448.13 | 448.40 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 14:15:00 | 450.30 | 448.76 | 448.62 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 15:15:00 | 445.50 | 448.11 | 448.33 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 09:15:00 | 473.15 | 453.12 | 450.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 490.70 | 478.22 | 470.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 12:15:00 | 480.05 | 481.05 | 473.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 12:45:00 | 476.65 | 481.05 | 473.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 484.40 | 482.74 | 476.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 13:00:00 | 494.80 | 483.69 | 480.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 489.65 | 482.69 | 482.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 11:15:00 | 488.45 | 483.58 | 482.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 15:15:00 | 473.40 | 497.70 | 500.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 473.40 | 497.70 | 500.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 444.45 | 487.05 | 495.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 11:15:00 | 422.10 | 421.32 | 432.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 12:00:00 | 422.10 | 421.32 | 432.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 432.65 | 419.49 | 423.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:30:00 | 433.00 | 419.49 | 423.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 437.85 | 425.73 | 425.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 12:15:00 | 439.50 | 428.48 | 426.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 433.45 | 434.78 | 430.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 433.45 | 434.78 | 430.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 433.45 | 434.78 | 430.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 433.45 | 434.78 | 430.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 431.55 | 434.34 | 432.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 431.55 | 434.34 | 432.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 429.50 | 433.37 | 431.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 441.45 | 433.37 | 431.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 423.60 | 437.16 | 435.88 | SL hit (close<static) qty=1.00 sl=428.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 427.70 | 433.77 | 434.47 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 444.00 | 435.81 | 434.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 13:15:00 | 446.80 | 440.62 | 437.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 12:15:00 | 444.80 | 444.83 | 441.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 12:45:00 | 444.80 | 444.83 | 441.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 464.50 | 468.03 | 464.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 11:00:00 | 464.50 | 468.03 | 464.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 466.55 | 467.74 | 464.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 13:15:00 | 470.50 | 467.67 | 464.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-28 10:15:00 | 517.55 | 495.89 | 490.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 484.00 | 491.73 | 492.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 473.00 | 487.99 | 490.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 463.30 | 457.60 | 464.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 463.30 | 457.60 | 464.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 463.30 | 457.60 | 464.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 463.30 | 457.60 | 464.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 467.90 | 459.66 | 464.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:00:00 | 467.90 | 459.66 | 464.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 466.15 | 460.96 | 464.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:30:00 | 466.40 | 460.96 | 464.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 467.95 | 463.22 | 465.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:45:00 | 467.10 | 463.22 | 465.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 473.00 | 466.72 | 466.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 485.20 | 470.41 | 468.19 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 15:00:00 | 444.60 | 2025-06-03 09:15:00 | 489.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-22 09:15:00 | 443.56 | 2025-06-03 09:15:00 | 487.92 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-22 15:15:00 | 442.20 | 2025-06-03 09:15:00 | 486.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-26 15:00:00 | 442.40 | 2025-06-03 09:15:00 | 486.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-28 09:15:00 | 457.56 | 2025-06-04 12:15:00 | 503.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2025-06-24 09:15:00 | 522.80 | 2025-06-26 10:15:00 | 531.06 | STOP_HIT | 1.00 | 1.58% |
| SELL | retest2 | 2025-07-01 09:15:00 | 516.96 | 2025-07-02 09:15:00 | 491.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-01 09:15:00 | 516.96 | 2025-07-02 14:15:00 | 501.96 | STOP_HIT | 0.50 | 2.90% |
| SELL | retest2 | 2025-07-14 11:45:00 | 526.20 | 2025-07-15 09:15:00 | 550.52 | STOP_HIT | 1.00 | -4.62% |
| BUY | retest2 | 2025-07-16 12:15:00 | 542.28 | 2025-07-24 13:15:00 | 554.34 | STOP_HIT | 1.00 | 2.22% |
| BUY | retest2 | 2025-07-16 15:15:00 | 542.40 | 2025-07-24 13:15:00 | 554.34 | STOP_HIT | 1.00 | 2.20% |
| SELL | retest2 | 2025-07-28 09:15:00 | 538.68 | 2025-07-29 13:15:00 | 553.30 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-07-28 10:00:00 | 539.78 | 2025-07-29 13:15:00 | 553.30 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-07-28 12:15:00 | 537.50 | 2025-07-29 13:15:00 | 553.30 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2025-07-29 12:15:00 | 540.24 | 2025-07-29 13:15:00 | 553.30 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-08-13 09:15:00 | 577.98 | 2025-08-14 10:15:00 | 570.98 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-08-14 10:15:00 | 577.76 | 2025-08-14 10:15:00 | 570.98 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-08-19 14:30:00 | 587.66 | 2025-08-20 09:15:00 | 580.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-08-21 12:30:00 | 576.80 | 2025-08-22 11:15:00 | 589.82 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-08-21 14:45:00 | 581.02 | 2025-08-22 11:15:00 | 589.82 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-08-21 15:15:00 | 577.00 | 2025-08-22 11:15:00 | 589.82 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest1 | 2025-08-26 10:00:00 | 615.88 | 2025-08-29 09:15:00 | 646.67 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-08-26 10:00:00 | 615.88 | 2025-09-01 09:15:00 | 614.00 | STOP_HIT | 0.50 | -0.31% |
| BUY | retest2 | 2025-09-01 14:45:00 | 644.16 | 2025-09-04 10:15:00 | 628.26 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-09-03 09:15:00 | 646.78 | 2025-09-04 10:15:00 | 628.26 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-09-04 09:15:00 | 638.68 | 2025-09-04 10:15:00 | 628.26 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-09-08 15:15:00 | 608.40 | 2025-09-10 09:15:00 | 625.52 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-09-19 15:15:00 | 626.00 | 2025-09-26 13:15:00 | 594.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 15:15:00 | 626.00 | 2025-09-29 09:15:00 | 618.72 | STOP_HIT | 0.50 | 1.16% |
| SELL | retest2 | 2025-10-23 09:15:00 | 632.76 | 2025-10-28 11:15:00 | 601.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-23 09:15:00 | 632.76 | 2025-10-29 12:15:00 | 601.70 | STOP_HIT | 0.50 | 4.91% |
| SELL | retest2 | 2025-11-25 09:15:00 | 526.48 | 2025-11-26 10:15:00 | 541.00 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-11-28 11:15:00 | 542.64 | 2025-12-01 09:15:00 | 528.12 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-11-28 13:15:00 | 542.66 | 2025-12-01 09:15:00 | 528.12 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-12-29 10:30:00 | 582.06 | 2025-12-30 09:15:00 | 571.56 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-12-29 13:15:00 | 581.90 | 2025-12-30 09:15:00 | 571.56 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2026-01-05 11:00:00 | 619.20 | 2026-01-07 13:15:00 | 618.50 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2026-01-05 11:30:00 | 619.14 | 2026-01-07 13:15:00 | 618.50 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2026-01-05 12:15:00 | 619.12 | 2026-01-07 13:15:00 | 618.50 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2026-01-05 12:45:00 | 619.16 | 2026-01-07 13:15:00 | 618.50 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2026-01-12 13:15:00 | 609.78 | 2026-01-12 15:15:00 | 620.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-01-30 09:15:00 | 499.60 | 2026-02-03 09:15:00 | 526.60 | STOP_HIT | 1.00 | -5.40% |
| SELL | retest2 | 2026-02-01 09:15:00 | 500.90 | 2026-02-03 09:15:00 | 526.60 | STOP_HIT | 1.00 | -5.13% |
| SELL | retest2 | 2026-02-01 12:15:00 | 495.30 | 2026-02-03 09:15:00 | 526.60 | STOP_HIT | 1.00 | -6.32% |
| SELL | retest2 | 2026-02-02 10:15:00 | 501.30 | 2026-02-03 09:15:00 | 526.60 | STOP_HIT | 1.00 | -5.05% |
| SELL | retest2 | 2026-02-11 09:15:00 | 484.00 | 2026-02-12 13:15:00 | 500.85 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2026-02-12 09:15:00 | 494.50 | 2026-02-12 13:15:00 | 500.85 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-02-25 11:45:00 | 493.90 | 2026-03-02 11:15:00 | 469.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 12:30:00 | 493.60 | 2026-03-02 11:15:00 | 468.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 14:45:00 | 494.50 | 2026-03-02 11:15:00 | 469.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 15:15:00 | 490.00 | 2026-03-02 11:15:00 | 465.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:45:00 | 493.90 | 2026-03-04 09:15:00 | 444.51 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 12:30:00 | 493.60 | 2026-03-04 09:15:00 | 444.24 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 14:45:00 | 494.50 | 2026-03-04 09:15:00 | 445.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 15:15:00 | 490.00 | 2026-03-04 09:15:00 | 441.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-06 10:45:00 | 432.85 | 2026-03-09 09:15:00 | 411.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 10:45:00 | 432.85 | 2026-03-10 09:15:00 | 437.75 | STOP_HIT | 0.50 | -1.13% |
| SELL | retest2 | 2026-03-10 10:00:00 | 437.75 | 2026-03-10 10:15:00 | 465.20 | STOP_HIT | 1.00 | -6.27% |
| BUY | retest2 | 2026-03-20 13:00:00 | 494.80 | 2026-03-27 15:15:00 | 473.40 | STOP_HIT | 1.00 | -4.32% |
| BUY | retest2 | 2026-03-24 09:15:00 | 489.65 | 2026-03-27 15:15:00 | 473.40 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2026-03-24 11:15:00 | 488.45 | 2026-03-27 15:15:00 | 473.40 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2026-04-10 09:15:00 | 441.45 | 2026-04-13 09:15:00 | 423.60 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest2 | 2026-04-22 13:15:00 | 470.50 | 2026-04-28 10:15:00 | 517.55 | TARGET_HIT | 1.00 | 10.00% |
