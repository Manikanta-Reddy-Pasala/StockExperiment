# Syrma SGS Technology Ltd. (SYRMA)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1100.55
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 67 |
| ALERT1 | 45 |
| ALERT2 | 45 |
| ALERT2_SKIP | 26 |
| ALERT3 | 103 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 65 |
| PARTIAL | 17 |
| TARGET_HIT | 4 |
| STOP_HIT | 64 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 85 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 44
- **Target hits / Stop hits / Partials:** 4 / 64 / 17
- **Avg / median % per leg:** 1.14% / -0.19%
- **Sum % (uncompounded):** 96.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 7 | 22.6% | 3 | 28 | 0 | -0.46% | -14.4% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.38% | -10.1% |
| BUY @ 3rd Alert (retest2) | 28 | 7 | 25.0% | 3 | 25 | 0 | -0.15% | -4.3% |
| SELL (all) | 54 | 34 | 63.0% | 1 | 36 | 17 | 2.05% | 111.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 54 | 34 | 63.0% | 1 | 36 | 17 | 2.05% | 111.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.38% | -10.1% |
| retest2 (combined) | 82 | 41 | 50.0% | 4 | 61 | 17 | 1.30% | 106.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 517.35 | 485.27 | 482.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 536.60 | 495.53 | 487.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 541.40 | 549.50 | 532.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 541.40 | 549.50 | 532.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 541.40 | 549.50 | 532.29 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 524.90 | 530.48 | 530.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 14:15:00 | 523.90 | 529.16 | 530.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 09:15:00 | 527.60 | 527.54 | 529.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 527.60 | 527.54 | 529.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 527.60 | 527.54 | 529.23 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 11:15:00 | 532.30 | 528.56 | 528.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 11:15:00 | 539.45 | 534.12 | 531.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 14:15:00 | 537.95 | 539.25 | 536.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 15:15:00 | 534.90 | 539.25 | 536.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 534.90 | 538.38 | 536.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 540.30 | 538.38 | 536.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 10:30:00 | 539.60 | 539.96 | 537.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:30:00 | 542.35 | 546.35 | 544.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 11:00:00 | 538.25 | 544.73 | 544.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 11:15:00 | 537.35 | 543.25 | 543.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 11:15:00 | 537.35 | 543.25 | 543.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 531.95 | 539.91 | 541.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 14:15:00 | 540.30 | 539.99 | 541.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-28 15:00:00 | 540.30 | 539.99 | 541.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 543.20 | 540.79 | 541.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 13:45:00 | 537.40 | 539.65 | 540.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 536.30 | 539.27 | 540.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:15:00 | 536.75 | 539.02 | 540.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 13:45:00 | 537.95 | 538.64 | 539.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 538.20 | 538.56 | 539.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:30:00 | 539.75 | 538.56 | 539.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 541.00 | 539.04 | 539.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 561.25 | 539.04 | 539.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 553.90 | 542.02 | 540.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 09:15:00 | 553.90 | 542.02 | 540.96 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 15:15:00 | 538.00 | 540.82 | 541.05 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 545.65 | 541.79 | 541.47 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 12:15:00 | 537.85 | 540.96 | 541.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 533.60 | 539.45 | 540.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 14:15:00 | 536.75 | 536.71 | 538.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 15:00:00 | 536.75 | 536.71 | 538.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 537.60 | 536.53 | 538.06 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 15:15:00 | 540.00 | 538.70 | 538.62 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 12:15:00 | 538.00 | 538.57 | 538.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 14:15:00 | 535.00 | 537.78 | 538.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 537.65 | 537.31 | 537.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 537.65 | 537.31 | 537.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 537.65 | 537.31 | 537.90 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 11:15:00 | 545.40 | 538.88 | 538.51 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 15:15:00 | 538.15 | 540.60 | 540.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 09:15:00 | 537.05 | 539.89 | 540.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 14:15:00 | 536.30 | 536.18 | 538.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 14:15:00 | 536.30 | 536.18 | 538.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 536.30 | 536.18 | 538.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 536.30 | 536.18 | 538.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 538.85 | 536.72 | 538.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 536.50 | 536.72 | 538.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 534.95 | 536.36 | 537.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 10:15:00 | 531.95 | 536.36 | 537.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:45:00 | 524.50 | 521.60 | 524.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 10:15:00 | 532.95 | 525.55 | 524.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 532.95 | 525.55 | 524.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 11:15:00 | 536.15 | 527.67 | 525.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 13:15:00 | 526.95 | 527.84 | 526.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 13:15:00 | 526.95 | 527.84 | 526.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 526.95 | 527.84 | 526.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:45:00 | 524.00 | 527.84 | 526.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 526.10 | 527.49 | 526.25 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 520.35 | 525.53 | 525.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 11:15:00 | 517.15 | 523.85 | 524.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 513.40 | 505.69 | 508.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 513.40 | 505.69 | 508.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 513.40 | 505.69 | 508.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:15:00 | 515.95 | 505.69 | 508.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 519.50 | 508.46 | 509.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:45:00 | 519.70 | 508.46 | 509.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 521.25 | 512.13 | 511.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 13:15:00 | 523.85 | 514.47 | 512.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 533.00 | 535.60 | 528.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:00:00 | 533.00 | 535.60 | 528.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 613.35 | 608.22 | 601.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 14:45:00 | 619.15 | 612.66 | 606.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 625.05 | 614.34 | 610.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-10 09:15:00 | 681.07 | 645.00 | 631.34 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 691.10 | 700.73 | 701.01 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 11:15:00 | 705.40 | 701.67 | 701.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 13:15:00 | 712.70 | 704.25 | 702.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 701.90 | 705.18 | 703.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 701.90 | 705.18 | 703.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 701.90 | 705.18 | 703.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 708.05 | 705.18 | 703.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 11:00:00 | 710.75 | 706.30 | 704.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 11:30:00 | 715.75 | 707.65 | 705.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 13:15:00 | 702.45 | 705.05 | 705.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 702.45 | 705.05 | 705.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 14:15:00 | 694.00 | 702.84 | 704.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 711.65 | 703.67 | 704.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 711.65 | 703.67 | 704.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 711.65 | 703.67 | 704.42 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 712.80 | 705.49 | 705.18 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 696.05 | 703.60 | 704.35 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 15:15:00 | 707.25 | 704.89 | 704.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-25 09:15:00 | 716.25 | 707.16 | 705.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 745.60 | 752.74 | 746.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 745.60 | 752.74 | 746.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 745.60 | 752.74 | 746.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 770.80 | 755.13 | 750.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 10:15:00 | 745.35 | 752.81 | 753.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 10:15:00 | 745.35 | 752.81 | 753.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 14:15:00 | 727.90 | 742.70 | 748.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 12:15:00 | 719.90 | 719.40 | 728.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 712.35 | 718.44 | 724.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 712.35 | 718.44 | 724.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 719.35 | 718.44 | 724.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 720.55 | 714.64 | 719.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:45:00 | 706.20 | 715.45 | 718.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 10:45:00 | 711.00 | 711.14 | 715.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 13:15:00 | 710.65 | 711.44 | 714.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:45:00 | 711.85 | 713.03 | 714.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 13:15:00 | 670.89 | 695.79 | 705.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 13:15:00 | 675.45 | 695.79 | 705.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 13:15:00 | 675.12 | 695.79 | 705.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 13:15:00 | 676.26 | 695.79 | 705.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 692.40 | 688.64 | 699.12 | SL hit (close>ema200) qty=0.50 sl=688.64 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 714.15 | 699.26 | 698.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 715.70 | 702.55 | 700.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 714.85 | 720.06 | 714.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 714.85 | 720.06 | 714.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 714.85 | 720.06 | 714.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 715.55 | 720.06 | 714.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 732.20 | 722.49 | 716.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:45:00 | 735.80 | 724.01 | 719.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:45:00 | 738.35 | 727.61 | 721.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 730.00 | 741.69 | 742.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 14:15:00 | 730.00 | 741.69 | 742.19 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 10:15:00 | 753.95 | 742.37 | 741.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 11:15:00 | 763.15 | 746.53 | 743.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-01 10:15:00 | 740.00 | 750.02 | 747.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 10:15:00 | 740.00 | 750.02 | 747.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 740.00 | 750.02 | 747.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 740.00 | 750.02 | 747.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 729.50 | 745.92 | 745.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:45:00 | 727.50 | 745.92 | 745.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 12:15:00 | 740.05 | 744.74 | 745.00 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 758.65 | 747.73 | 746.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 793.10 | 763.52 | 756.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 13:15:00 | 839.20 | 840.45 | 818.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 13:45:00 | 837.55 | 840.45 | 818.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 853.75 | 856.60 | 842.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 10:00:00 | 864.00 | 854.77 | 851.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 13:15:00 | 838.85 | 849.43 | 850.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 838.85 | 849.43 | 850.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 14:15:00 | 835.00 | 846.54 | 848.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 814.50 | 813.86 | 824.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 12:15:00 | 815.55 | 815.29 | 822.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 815.55 | 815.29 | 822.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:30:00 | 829.20 | 815.29 | 822.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 808.80 | 814.23 | 819.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 10:30:00 | 804.60 | 812.79 | 818.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:15:00 | 805.45 | 812.79 | 818.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 12:15:00 | 804.95 | 811.45 | 817.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 13:00:00 | 804.25 | 803.33 | 808.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 810.45 | 804.96 | 808.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:45:00 | 814.85 | 804.96 | 808.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 810.00 | 805.97 | 808.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:30:00 | 812.20 | 807.71 | 809.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 816.00 | 809.37 | 809.70 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-19 11:15:00 | 813.95 | 810.29 | 810.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 11:15:00 | 813.95 | 810.29 | 810.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 14:15:00 | 824.30 | 813.76 | 811.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 12:15:00 | 821.70 | 822.18 | 817.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 12:45:00 | 821.40 | 822.18 | 817.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 814.90 | 820.35 | 817.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 15:00:00 | 814.90 | 820.35 | 817.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 819.00 | 820.08 | 817.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 825.50 | 820.08 | 817.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 15:15:00 | 820.00 | 821.95 | 820.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 09:30:00 | 826.00 | 825.49 | 822.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 09:45:00 | 822.80 | 840.24 | 838.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 10:15:00 | 826.25 | 837.44 | 837.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 826.25 | 837.44 | 837.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 813.35 | 832.62 | 835.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 786.50 | 780.81 | 793.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 09:30:00 | 790.45 | 780.81 | 793.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 796.55 | 785.11 | 793.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:30:00 | 822.85 | 785.11 | 793.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 808.95 | 789.88 | 795.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:45:00 | 811.25 | 789.88 | 795.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 810.70 | 794.04 | 796.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 810.70 | 794.04 | 796.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 808.70 | 799.82 | 798.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 830.70 | 813.01 | 806.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 817.05 | 817.28 | 810.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:45:00 | 817.30 | 817.28 | 810.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 812.45 | 816.60 | 811.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:45:00 | 813.15 | 816.60 | 811.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 816.20 | 816.52 | 812.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 829.15 | 817.02 | 812.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 13:15:00 | 823.45 | 832.18 | 832.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 13:15:00 | 823.45 | 832.18 | 832.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 14:15:00 | 820.05 | 829.75 | 831.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 833.50 | 829.50 | 831.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 833.50 | 829.50 | 831.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 833.50 | 829.50 | 831.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 837.95 | 829.50 | 831.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 838.95 | 831.39 | 831.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 838.95 | 831.39 | 831.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 848.00 | 834.71 | 833.37 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 829.40 | 834.20 | 834.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 828.10 | 831.84 | 833.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 11:15:00 | 796.65 | 796.42 | 804.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 11:45:00 | 800.70 | 796.42 | 804.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 796.50 | 793.78 | 800.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 804.55 | 793.78 | 800.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 796.55 | 784.13 | 787.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:00:00 | 796.55 | 784.13 | 787.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 796.00 | 786.50 | 788.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 778.90 | 786.50 | 788.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 768.30 | 772.54 | 778.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 763.65 | 772.30 | 775.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 791.90 | 765.76 | 768.46 | SL hit (close>static) qty=1.00 sl=788.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 804.70 | 773.55 | 771.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 811.50 | 794.69 | 784.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 10:15:00 | 810.10 | 810.82 | 800.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 10:30:00 | 807.00 | 810.82 | 800.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 813.50 | 816.94 | 810.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 12:30:00 | 818.85 | 817.24 | 811.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 14:45:00 | 816.40 | 817.11 | 812.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 797.10 | 812.45 | 810.98 | SL hit (close<static) qty=1.00 sl=810.25 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 798.05 | 809.57 | 809.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 790.05 | 805.67 | 808.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 09:15:00 | 802.75 | 792.65 | 796.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 802.75 | 792.65 | 796.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 802.75 | 792.65 | 796.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 12:00:00 | 789.95 | 793.10 | 796.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 13:30:00 | 790.70 | 793.13 | 795.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 14:30:00 | 790.85 | 792.20 | 795.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 799.95 | 793.69 | 793.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 799.95 | 793.69 | 793.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 816.95 | 807.37 | 801.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 14:15:00 | 827.80 | 831.98 | 818.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 15:00:00 | 827.80 | 831.98 | 818.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 880.45 | 893.84 | 888.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 880.45 | 893.84 | 888.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 887.65 | 892.60 | 888.36 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 875.30 | 885.75 | 886.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 15:15:00 | 873.65 | 883.33 | 885.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 12:15:00 | 877.65 | 877.19 | 881.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 12:15:00 | 877.65 | 877.19 | 881.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 877.65 | 877.19 | 881.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:30:00 | 879.35 | 877.19 | 881.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 876.60 | 877.08 | 880.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:30:00 | 880.20 | 877.08 | 880.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 876.85 | 877.03 | 880.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:00:00 | 876.85 | 877.03 | 880.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 873.65 | 875.57 | 879.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 10:45:00 | 867.60 | 873.61 | 877.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 12:00:00 | 869.00 | 872.69 | 877.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:15:00 | 868.05 | 872.06 | 876.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:45:00 | 865.55 | 869.11 | 873.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 12:15:00 | 825.55 | 840.42 | 852.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 12:15:00 | 824.65 | 840.42 | 852.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 13:15:00 | 824.22 | 837.02 | 850.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 14:15:00 | 822.27 | 833.39 | 847.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 10:15:00 | 836.85 | 830.70 | 842.29 | SL hit (close>ema200) qty=0.50 sl=830.70 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 856.75 | 843.36 | 842.57 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 821.60 | 840.09 | 842.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 12:15:00 | 818.95 | 835.87 | 840.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 840.10 | 830.44 | 835.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 840.10 | 830.44 | 835.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 840.10 | 830.44 | 835.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:45:00 | 841.35 | 830.44 | 835.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 837.85 | 831.92 | 835.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:15:00 | 836.00 | 831.92 | 835.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 820.00 | 829.53 | 834.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:00:00 | 817.20 | 825.45 | 831.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 776.34 | 786.25 | 794.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-08 12:15:00 | 735.48 | 747.79 | 764.43 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 736.50 | 721.07 | 719.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 13:15:00 | 738.45 | 729.72 | 724.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 725.80 | 728.94 | 725.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 15:15:00 | 725.80 | 728.94 | 725.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 725.80 | 728.94 | 725.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 717.95 | 728.94 | 725.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 727.00 | 728.55 | 725.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 719.70 | 728.55 | 725.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 728.90 | 728.62 | 725.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 13:30:00 | 732.65 | 728.93 | 726.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:45:00 | 731.45 | 729.87 | 727.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 735.35 | 730.33 | 727.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 13:30:00 | 732.35 | 732.32 | 729.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 734.75 | 733.06 | 730.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 721.60 | 731.40 | 731.07 | SL hit (close<static) qty=1.00 sl=725.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 718.30 | 728.78 | 729.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 11:15:00 | 715.00 | 726.02 | 728.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 726.60 | 720.13 | 723.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 09:15:00 | 726.60 | 720.13 | 723.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 726.60 | 720.13 | 723.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:30:00 | 735.50 | 720.13 | 723.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 729.40 | 721.99 | 724.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:15:00 | 732.25 | 721.99 | 724.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 734.00 | 727.15 | 726.42 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 15:15:00 | 722.80 | 725.55 | 725.78 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 727.65 | 726.22 | 726.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 732.70 | 728.31 | 727.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 724.15 | 728.16 | 727.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 724.15 | 728.16 | 727.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 724.15 | 728.16 | 727.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:00:00 | 724.15 | 728.16 | 727.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 723.60 | 727.25 | 726.99 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 11:15:00 | 718.15 | 725.43 | 726.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 12:15:00 | 715.85 | 723.51 | 725.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 724.80 | 722.62 | 724.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 724.80 | 722.62 | 724.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 724.80 | 722.62 | 724.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 724.80 | 722.62 | 724.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 735.10 | 725.11 | 725.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:45:00 | 735.95 | 725.11 | 725.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 738.85 | 727.86 | 726.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 13:15:00 | 740.90 | 732.40 | 728.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 749.00 | 749.39 | 743.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 749.00 | 749.39 | 743.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 748.50 | 749.78 | 745.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:45:00 | 749.30 | 749.78 | 745.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 755.30 | 750.88 | 746.44 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 738.00 | 745.32 | 746.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 735.65 | 743.39 | 745.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 706.55 | 702.81 | 710.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 706.55 | 702.81 | 710.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 707.10 | 703.89 | 709.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 712.20 | 703.89 | 709.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 702.30 | 703.71 | 708.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:15:00 | 709.75 | 703.71 | 708.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 710.00 | 704.97 | 708.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:30:00 | 708.80 | 704.97 | 708.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 705.00 | 704.98 | 708.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 702.50 | 704.98 | 708.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:15:00 | 703.10 | 705.36 | 708.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 714.00 | 706.73 | 708.25 | SL hit (close>static) qty=1.00 sl=710.20 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 680.15 | 663.69 | 663.40 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 660.45 | 663.73 | 664.00 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 686.95 | 663.48 | 662.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 695.75 | 673.63 | 667.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 764.00 | 769.37 | 753.10 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 09:15:00 | 791.30 | 769.37 | 753.10 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 11:30:00 | 777.85 | 772.39 | 758.98 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 753.50 | 768.61 | 758.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-02 12:15:00 | 753.50 | 768.61 | 758.48 | SL hit (close<ema400) qty=1.00 sl=758.48 alert=retest1 |

### Cycle 52 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 861.10 | 875.02 | 875.64 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 09:15:00 | 886.05 | 875.03 | 874.76 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 13:15:00 | 865.65 | 872.98 | 873.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 10:15:00 | 863.05 | 868.27 | 871.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 12:15:00 | 863.15 | 862.28 | 865.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 14:15:00 | 867.00 | 863.52 | 865.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 867.00 | 863.52 | 865.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:45:00 | 867.05 | 863.52 | 865.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 868.25 | 864.47 | 865.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 870.10 | 864.47 | 865.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 862.75 | 864.53 | 865.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:30:00 | 860.10 | 863.06 | 864.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:15:00 | 817.10 | 835.25 | 841.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 15:15:00 | 831.05 | 829.52 | 836.40 | SL hit (close>ema200) qty=0.50 sl=829.52 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 854.30 | 838.37 | 837.85 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 14:15:00 | 833.00 | 837.56 | 837.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 15:15:00 | 831.80 | 836.41 | 837.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 752.80 | 748.75 | 770.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 752.80 | 748.75 | 770.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 767.10 | 751.31 | 763.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 767.10 | 751.31 | 763.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 771.00 | 755.25 | 764.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 765.90 | 755.25 | 764.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:30:00 | 764.15 | 758.67 | 764.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 12:15:00 | 765.65 | 760.16 | 764.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:45:00 | 763.75 | 763.24 | 765.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 727.60 | 752.03 | 759.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 725.94 | 752.03 | 759.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 727.37 | 752.03 | 759.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 725.56 | 752.03 | 759.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 747.25 | 732.34 | 742.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 747.25 | 732.34 | 742.30 | SL hit (close>ema200) qty=0.50 sl=732.34 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 758.90 | 747.52 | 746.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 762.05 | 750.43 | 748.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 755.50 | 760.22 | 755.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 755.50 | 760.22 | 755.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 755.50 | 760.22 | 755.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 755.50 | 760.22 | 755.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 760.00 | 760.17 | 755.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 735.70 | 760.17 | 755.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 739.00 | 755.94 | 754.16 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 11:15:00 | 748.60 | 752.63 | 752.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 728.15 | 747.36 | 750.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 729.35 | 723.51 | 731.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 729.35 | 723.51 | 731.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 729.35 | 723.51 | 731.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 731.45 | 723.51 | 731.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 743.25 | 727.82 | 731.85 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 747.15 | 734.93 | 734.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 756.80 | 739.31 | 736.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 770.65 | 774.75 | 762.50 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 10:15:00 | 774.05 | 774.75 | 762.50 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 765.80 | 771.60 | 763.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:00:00 | 765.80 | 771.60 | 763.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 756.80 | 768.64 | 763.33 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-19 13:15:00 | 756.80 | 768.64 | 763.33 | SL hit (close<ema400) qty=1.00 sl=763.33 alert=retest1 |

### Cycle 60 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 742.10 | 768.30 | 768.97 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 787.40 | 767.32 | 765.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 826.70 | 787.45 | 775.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 11:15:00 | 811.70 | 813.13 | 800.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 11:45:00 | 812.20 | 813.13 | 800.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 795.60 | 810.61 | 804.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:00:00 | 795.60 | 810.61 | 804.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 787.00 | 805.89 | 802.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:00:00 | 787.00 | 805.89 | 802.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 787.15 | 799.39 | 800.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 780.60 | 795.63 | 798.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 806.55 | 791.25 | 795.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 806.55 | 791.25 | 795.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 806.55 | 791.25 | 795.03 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 804.90 | 798.56 | 797.78 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 768.55 | 792.32 | 795.11 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 10:15:00 | 803.70 | 794.23 | 793.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 810.30 | 798.89 | 795.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 808.55 | 809.14 | 804.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 14:30:00 | 808.60 | 809.14 | 804.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 863.90 | 864.54 | 853.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 878.65 | 863.16 | 857.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 13:15:00 | 966.52 | 932.75 | 904.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 976.45 | 988.74 | 988.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 09:15:00 | 965.10 | 970.98 | 975.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 10:15:00 | 976.25 | 972.04 | 975.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 10:15:00 | 976.25 | 972.04 | 975.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 976.25 | 972.04 | 975.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:45:00 | 977.00 | 972.04 | 975.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 972.20 | 972.07 | 975.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 957.60 | 976.02 | 976.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 11:15:00 | 986.75 | 970.41 | 969.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 986.75 | 970.41 | 969.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 12:15:00 | 994.00 | 975.13 | 971.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 10:15:00 | 1041.80 | 1043.12 | 1022.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 10:30:00 | 1042.70 | 1043.12 | 1022.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-26 09:15:00 | 540.30 | 2025-05-28 11:15:00 | 537.35 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-05-26 10:30:00 | 539.60 | 2025-05-28 11:15:00 | 537.35 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-05-28 09:30:00 | 542.35 | 2025-05-28 11:15:00 | 537.35 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-05-28 11:00:00 | 538.25 | 2025-05-28 11:15:00 | 537.35 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-05-29 13:45:00 | 537.40 | 2025-06-02 09:15:00 | 553.90 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-05-30 09:15:00 | 536.30 | 2025-06-02 09:15:00 | 553.90 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2025-05-30 10:15:00 | 536.75 | 2025-06-02 09:15:00 | 553.90 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2025-05-30 13:45:00 | 537.95 | 2025-06-02 09:15:00 | 553.90 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-06-12 10:15:00 | 531.95 | 2025-06-18 10:15:00 | 532.95 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-06-17 09:45:00 | 524.50 | 2025-06-18 10:15:00 | 532.95 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-07-07 14:45:00 | 619.15 | 2025-07-10 09:15:00 | 681.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-09 09:15:00 | 625.05 | 2025-07-14 09:15:00 | 687.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-22 10:15:00 | 708.05 | 2025-07-23 13:15:00 | 702.45 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-07-22 11:00:00 | 710.75 | 2025-07-23 13:15:00 | 702.45 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-07-22 11:30:00 | 715.75 | 2025-07-23 13:15:00 | 702.45 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-08-01 09:15:00 | 770.80 | 2025-08-04 10:15:00 | 745.35 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2025-08-08 14:45:00 | 706.20 | 2025-08-12 13:15:00 | 670.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-11 10:45:00 | 711.00 | 2025-08-12 13:15:00 | 675.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-11 13:15:00 | 710.65 | 2025-08-12 13:15:00 | 675.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-12 09:45:00 | 711.85 | 2025-08-12 13:15:00 | 676.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-08 14:45:00 | 706.20 | 2025-08-13 09:15:00 | 692.40 | STOP_HIT | 0.50 | 1.95% |
| SELL | retest2 | 2025-08-11 10:45:00 | 711.00 | 2025-08-13 09:15:00 | 692.40 | STOP_HIT | 0.50 | 2.62% |
| SELL | retest2 | 2025-08-11 13:15:00 | 710.65 | 2025-08-13 09:15:00 | 692.40 | STOP_HIT | 0.50 | 2.57% |
| SELL | retest2 | 2025-08-12 09:45:00 | 711.85 | 2025-08-13 09:15:00 | 692.40 | STOP_HIT | 0.50 | 2.73% |
| SELL | retest2 | 2025-08-14 09:30:00 | 692.85 | 2025-08-18 09:15:00 | 714.15 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-08-14 14:45:00 | 694.05 | 2025-08-18 09:15:00 | 714.15 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-08-21 09:45:00 | 735.80 | 2025-08-26 14:15:00 | 730.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-08-21 10:45:00 | 738.35 | 2025-08-26 14:15:00 | 730.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-09-11 10:00:00 | 864.00 | 2025-09-11 13:15:00 | 838.85 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-09-17 10:30:00 | 804.60 | 2025-09-19 11:15:00 | 813.95 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-09-17 11:15:00 | 805.45 | 2025-09-19 11:15:00 | 813.95 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-09-17 12:15:00 | 804.95 | 2025-09-19 11:15:00 | 813.95 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-09-18 13:00:00 | 804.25 | 2025-09-19 11:15:00 | 813.95 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-09-23 09:15:00 | 825.50 | 2025-09-26 10:15:00 | 826.25 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-09-23 15:15:00 | 820.00 | 2025-09-26 10:15:00 | 826.25 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2025-09-24 09:30:00 | 826.00 | 2025-09-26 10:15:00 | 826.25 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-09-26 09:45:00 | 822.80 | 2025-09-26 10:15:00 | 826.25 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2025-10-07 09:15:00 | 829.15 | 2025-10-09 13:15:00 | 823.45 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-10-27 09:15:00 | 763.65 | 2025-10-28 09:15:00 | 791.90 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2025-10-31 12:30:00 | 818.85 | 2025-11-03 09:15:00 | 797.10 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2025-10-31 14:45:00 | 816.40 | 2025-11-03 09:15:00 | 797.10 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-11-06 12:00:00 | 789.95 | 2025-11-10 09:15:00 | 799.95 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-11-06 13:30:00 | 790.70 | 2025-11-10 09:15:00 | 799.95 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-11-06 14:30:00 | 790.85 | 2025-11-10 09:15:00 | 799.95 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-11-20 10:45:00 | 867.60 | 2025-11-24 12:15:00 | 825.55 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2025-11-20 12:00:00 | 869.00 | 2025-11-24 12:15:00 | 824.65 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-11-20 13:15:00 | 868.05 | 2025-11-24 13:15:00 | 824.22 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2025-11-21 09:45:00 | 865.55 | 2025-11-24 14:15:00 | 822.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 10:45:00 | 867.60 | 2025-11-25 10:15:00 | 836.85 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2025-11-20 12:00:00 | 869.00 | 2025-11-25 10:15:00 | 836.85 | STOP_HIT | 0.50 | 3.70% |
| SELL | retest2 | 2025-11-20 13:15:00 | 868.05 | 2025-11-25 10:15:00 | 836.85 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2025-11-21 09:45:00 | 865.55 | 2025-11-25 10:15:00 | 836.85 | STOP_HIT | 0.50 | 3.32% |
| SELL | retest2 | 2025-11-26 11:30:00 | 834.00 | 2025-11-26 13:15:00 | 848.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-12-01 14:00:00 | 817.20 | 2025-12-05 09:15:00 | 776.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 14:00:00 | 817.20 | 2025-12-08 12:15:00 | 735.48 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-12-23 13:30:00 | 732.65 | 2025-12-29 09:15:00 | 721.60 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-12-23 14:45:00 | 731.45 | 2025-12-29 09:15:00 | 721.60 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-12-24 10:15:00 | 735.35 | 2025-12-29 09:15:00 | 721.60 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-12-24 13:30:00 | 732.35 | 2025-12-29 09:15:00 | 721.60 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-01-14 14:15:00 | 702.50 | 2026-01-16 09:15:00 | 714.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-01-14 15:15:00 | 703.10 | 2026-01-16 09:15:00 | 714.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-01-16 12:45:00 | 702.50 | 2026-01-20 09:15:00 | 667.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 701.15 | 2026-01-20 09:15:00 | 666.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 10:30:00 | 698.20 | 2026-01-20 09:15:00 | 663.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 12:45:00 | 702.50 | 2026-01-22 09:15:00 | 655.40 | STOP_HIT | 0.50 | 6.70% |
| SELL | retest2 | 2026-01-19 09:15:00 | 701.15 | 2026-01-22 09:15:00 | 655.40 | STOP_HIT | 0.50 | 6.52% |
| SELL | retest2 | 2026-01-19 10:30:00 | 698.20 | 2026-01-22 09:15:00 | 655.40 | STOP_HIT | 0.50 | 6.13% |
| BUY | retest1 | 2026-02-02 09:15:00 | 791.30 | 2026-02-02 12:15:00 | 753.50 | STOP_HIT | 1.00 | -4.78% |
| BUY | retest1 | 2026-02-02 11:30:00 | 777.85 | 2026-02-02 12:15:00 | 753.50 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2026-02-11 14:30:00 | 892.60 | 2026-02-13 09:15:00 | 861.10 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2026-02-19 11:30:00 | 860.10 | 2026-02-24 11:15:00 | 817.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 11:30:00 | 860.10 | 2026-02-24 15:15:00 | 831.05 | STOP_HIT | 0.50 | 3.38% |
| SELL | retest2 | 2026-03-06 09:15:00 | 765.90 | 2026-03-09 09:15:00 | 727.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 10:30:00 | 764.15 | 2026-03-09 09:15:00 | 725.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 12:15:00 | 765.65 | 2026-03-09 09:15:00 | 727.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:45:00 | 763.75 | 2026-03-09 09:15:00 | 725.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 09:15:00 | 765.90 | 2026-03-10 09:15:00 | 747.25 | STOP_HIT | 0.50 | 2.44% |
| SELL | retest2 | 2026-03-06 10:30:00 | 764.15 | 2026-03-10 09:15:00 | 747.25 | STOP_HIT | 0.50 | 2.21% |
| SELL | retest2 | 2026-03-06 12:15:00 | 765.65 | 2026-03-10 09:15:00 | 747.25 | STOP_HIT | 0.50 | 2.40% |
| SELL | retest2 | 2026-03-06 14:45:00 | 763.75 | 2026-03-10 09:15:00 | 747.25 | STOP_HIT | 0.50 | 2.16% |
| BUY | retest1 | 2026-03-19 10:15:00 | 774.05 | 2026-03-19 13:15:00 | 756.80 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-03-20 09:15:00 | 777.35 | 2026-03-23 09:15:00 | 747.80 | STOP_HIT | 1.00 | -3.80% |
| BUY | retest2 | 2026-03-23 09:30:00 | 759.90 | 2026-03-23 10:15:00 | 742.10 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2026-04-15 09:15:00 | 878.65 | 2026-04-16 13:15:00 | 966.52 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 09:15:00 | 957.60 | 2026-05-04 11:15:00 | 986.75 | STOP_HIT | 1.00 | -3.04% |
