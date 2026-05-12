# Gallantt Ispat Ltd. (GALLANTT)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 866.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 81 |
| ALERT1 | 52 |
| ALERT2 | 50 |
| ALERT2_SKIP | 24 |
| ALERT3 | 121 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 52 |
| PARTIAL | 2 |
| TARGET_HIT | 8 |
| STOP_HIT | 46 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 36
- **Target hits / Stop hits / Partials:** 8 / 44 / 2
- **Avg / median % per leg:** 0.75% / -1.18%
- **Sum % (uncompounded):** 40.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 14 | 66.7% | 8 | 13 | 0 | 3.79% | 79.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 14 | 66.7% | 8 | 13 | 0 | 3.79% | 79.5% |
| SELL (all) | 33 | 4 | 12.1% | 0 | 31 | 2 | -1.18% | -39.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 4 | 12.1% | 0 | 31 | 2 | -1.18% | -39.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 54 | 18 | 33.3% | 8 | 44 | 2 | 0.75% | 40.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 458.60 | 437.33 | 434.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 463.70 | 446.18 | 439.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 462.60 | 463.56 | 453.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:45:00 | 460.65 | 463.56 | 453.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 467.50 | 472.30 | 466.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:45:00 | 468.40 | 472.30 | 466.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 464.05 | 470.65 | 466.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:45:00 | 463.10 | 470.65 | 466.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 464.05 | 469.33 | 466.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 15:15:00 | 466.90 | 469.33 | 466.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 11:15:00 | 467.70 | 467.59 | 466.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:00:00 | 470.00 | 469.18 | 467.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 12:45:00 | 467.40 | 472.71 | 472.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 13:15:00 | 470.05 | 472.18 | 472.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 13:15:00 | 470.05 | 472.18 | 472.43 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 479.00 | 473.69 | 473.08 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 11:15:00 | 470.65 | 472.80 | 472.80 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 15:15:00 | 473.10 | 472.80 | 472.79 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 472.10 | 472.66 | 472.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 10:15:00 | 470.70 | 472.27 | 472.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 11:15:00 | 472.55 | 472.33 | 472.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 11:15:00 | 472.55 | 472.33 | 472.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 472.55 | 472.33 | 472.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:45:00 | 474.90 | 472.33 | 472.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 475.35 | 472.93 | 472.80 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 11:15:00 | 471.25 | 472.89 | 472.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 13:15:00 | 467.55 | 471.53 | 472.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 09:15:00 | 458.80 | 458.30 | 463.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-28 10:00:00 | 458.80 | 458.30 | 463.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 459.00 | 459.00 | 462.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:30:00 | 461.05 | 459.00 | 462.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 459.50 | 459.12 | 461.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:15:00 | 456.45 | 459.12 | 461.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 457.00 | 458.70 | 461.10 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 14:15:00 | 465.15 | 462.43 | 462.10 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 448.85 | 460.96 | 462.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 444.75 | 450.99 | 455.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 10:15:00 | 441.45 | 437.59 | 440.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 10:15:00 | 441.45 | 437.59 | 440.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 441.45 | 437.59 | 440.91 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 09:15:00 | 450.75 | 443.46 | 442.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 14:15:00 | 464.10 | 449.56 | 446.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 15:15:00 | 465.00 | 465.20 | 458.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 09:15:00 | 466.65 | 465.20 | 458.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 458.00 | 464.79 | 462.19 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 12:15:00 | 450.60 | 458.76 | 459.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 14:15:00 | 449.80 | 455.77 | 458.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 455.50 | 453.61 | 455.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:00:00 | 455.50 | 453.61 | 455.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 455.20 | 453.93 | 455.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 14:15:00 | 454.60 | 453.93 | 455.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:15:00 | 453.35 | 454.11 | 455.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 15:15:00 | 459.40 | 449.13 | 450.12 | SL hit (close>static) qty=1.00 sl=456.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 11:15:00 | 460.45 | 451.77 | 451.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-19 13:15:00 | 462.65 | 455.41 | 452.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 15:15:00 | 544.80 | 546.52 | 532.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 09:15:00 | 540.10 | 546.52 | 532.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 540.05 | 542.34 | 535.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:00:00 | 540.05 | 542.34 | 535.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 537.45 | 541.36 | 535.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:30:00 | 539.85 | 541.36 | 535.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 536.80 | 540.45 | 536.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 541.40 | 540.45 | 536.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:00:00 | 539.05 | 540.17 | 536.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 10:15:00 | 533.50 | 538.84 | 536.03 | SL hit (close<static) qty=1.00 sl=535.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 12:15:00 | 525.50 | 534.27 | 534.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 12:15:00 | 523.55 | 528.22 | 530.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 13:15:00 | 529.15 | 528.41 | 530.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 14:00:00 | 529.15 | 528.41 | 530.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 525.10 | 527.99 | 529.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 521.60 | 526.47 | 528.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 14:15:00 | 523.00 | 525.88 | 527.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 550.00 | 530.07 | 529.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 550.00 | 530.07 | 529.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 11:15:00 | 555.65 | 537.58 | 532.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 10:15:00 | 549.60 | 549.62 | 542.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 10:45:00 | 550.65 | 549.62 | 542.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 542.95 | 546.80 | 542.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:30:00 | 540.70 | 546.80 | 542.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 546.60 | 546.76 | 542.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 15:15:00 | 549.85 | 546.76 | 542.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 15:15:00 | 540.00 | 548.11 | 546.53 | SL hit (close<static) qty=1.00 sl=540.25 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 12:15:00 | 541.75 | 545.34 | 545.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 13:15:00 | 540.90 | 544.45 | 545.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 547.10 | 544.98 | 545.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 14:15:00 | 547.10 | 544.98 | 545.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 547.10 | 544.98 | 545.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 547.10 | 544.98 | 545.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 549.90 | 545.97 | 545.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 10:15:00 | 553.00 | 547.38 | 546.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 13:15:00 | 579.00 | 579.09 | 565.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 14:00:00 | 579.00 | 579.09 | 565.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 579.55 | 583.22 | 579.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:00:00 | 579.55 | 583.22 | 579.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 575.95 | 581.76 | 578.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:00:00 | 575.95 | 581.76 | 578.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 575.60 | 580.53 | 578.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 575.60 | 580.53 | 578.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 579.85 | 580.19 | 578.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 574.50 | 580.19 | 578.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 581.90 | 580.53 | 579.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 11:30:00 | 586.00 | 581.13 | 579.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 13:00:00 | 583.35 | 581.57 | 579.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 13:45:00 | 583.00 | 581.51 | 579.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 15:00:00 | 584.50 | 582.10 | 580.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 582.85 | 582.69 | 580.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 576.95 | 582.69 | 580.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 587.70 | 588.15 | 585.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 14:45:00 | 592.00 | 589.16 | 586.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:45:00 | 591.35 | 588.66 | 586.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 593.85 | 586.25 | 586.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-21 10:15:00 | 641.69 | 598.88 | 592.11 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 09:15:00 | 758.15 | 771.54 | 772.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 750.50 | 762.70 | 766.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 768.30 | 763.82 | 767.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 768.30 | 763.82 | 767.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 768.30 | 763.82 | 767.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:45:00 | 766.60 | 763.82 | 767.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 768.85 | 764.83 | 767.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:30:00 | 775.05 | 764.83 | 767.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 768.35 | 766.02 | 767.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 13:00:00 | 768.35 | 766.02 | 767.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 770.30 | 766.87 | 767.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:00:00 | 770.30 | 766.87 | 767.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 775.50 | 768.60 | 768.39 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 09:15:00 | 757.00 | 767.61 | 768.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 10:15:00 | 750.55 | 764.20 | 766.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 10:15:00 | 633.45 | 628.53 | 644.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 11:00:00 | 633.45 | 628.53 | 644.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 636.85 | 629.95 | 642.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:45:00 | 638.70 | 629.95 | 642.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 636.35 | 632.61 | 641.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:30:00 | 640.75 | 632.61 | 641.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 639.85 | 633.48 | 639.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 639.85 | 633.48 | 639.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 639.70 | 634.73 | 639.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:15:00 | 632.40 | 634.73 | 639.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 15:15:00 | 600.78 | 609.79 | 617.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 11:15:00 | 585.80 | 582.80 | 589.91 | SL hit (close>ema200) qty=0.50 sl=582.80 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 616.15 | 592.94 | 592.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 09:15:00 | 638.65 | 619.00 | 613.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 15:15:00 | 725.95 | 728.75 | 710.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 09:15:00 | 721.05 | 728.75 | 710.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 709.85 | 722.05 | 711.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:00:00 | 709.85 | 722.05 | 711.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 714.10 | 720.46 | 712.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:45:00 | 709.35 | 720.46 | 712.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 708.45 | 717.17 | 711.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:45:00 | 708.25 | 717.17 | 711.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 710.00 | 715.74 | 711.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 706.25 | 715.74 | 711.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 711.20 | 714.83 | 711.70 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 677.75 | 705.24 | 707.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 09:15:00 | 676.40 | 692.70 | 700.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 15:15:00 | 681.90 | 681.90 | 690.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 09:15:00 | 679.55 | 681.90 | 690.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 662.70 | 668.02 | 673.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:15:00 | 660.60 | 664.23 | 667.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 15:00:00 | 660.70 | 663.53 | 666.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 685.00 | 667.58 | 668.07 | SL hit (close>static) qty=1.00 sl=674.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 10:15:00 | 681.55 | 670.37 | 669.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 12:15:00 | 689.75 | 676.57 | 672.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 671.00 | 677.92 | 673.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 14:15:00 | 671.00 | 677.92 | 673.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 671.00 | 677.92 | 673.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 674.40 | 677.92 | 673.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 669.65 | 676.26 | 673.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 666.80 | 676.26 | 673.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 660.80 | 671.21 | 671.59 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 10:15:00 | 675.10 | 670.45 | 670.42 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 668.25 | 670.01 | 670.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 663.85 | 668.95 | 669.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 657.90 | 657.48 | 661.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 657.90 | 657.48 | 661.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 657.90 | 657.48 | 661.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 662.85 | 657.48 | 661.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 668.35 | 658.79 | 660.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 672.35 | 658.79 | 660.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 660.20 | 659.07 | 660.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 656.50 | 659.07 | 660.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:30:00 | 658.50 | 654.76 | 656.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:30:00 | 656.60 | 655.57 | 656.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 12:15:00 | 664.35 | 658.63 | 657.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 664.35 | 658.63 | 657.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 667.85 | 660.48 | 658.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 667.35 | 671.68 | 666.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 10:15:00 | 667.35 | 671.68 | 666.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 667.35 | 671.68 | 666.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:30:00 | 670.10 | 671.68 | 666.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 669.40 | 671.22 | 667.09 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 09:15:00 | 652.55 | 665.76 | 665.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 15:15:00 | 651.05 | 658.43 | 661.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 658.85 | 658.52 | 661.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:00:00 | 658.85 | 658.52 | 661.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 663.75 | 656.01 | 658.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 664.60 | 656.01 | 658.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 667.85 | 658.38 | 658.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:30:00 | 667.35 | 658.38 | 658.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 665.50 | 659.80 | 659.55 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 12:15:00 | 657.95 | 660.39 | 660.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 13:15:00 | 656.10 | 659.53 | 660.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 12:15:00 | 657.30 | 655.92 | 657.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 12:15:00 | 657.30 | 655.92 | 657.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 657.30 | 655.92 | 657.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:00:00 | 657.30 | 655.92 | 657.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 657.30 | 656.20 | 657.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:45:00 | 656.90 | 656.20 | 657.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 655.40 | 656.04 | 657.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:30:00 | 652.95 | 654.82 | 656.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 11:30:00 | 652.45 | 651.88 | 653.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 14:15:00 | 661.30 | 653.43 | 653.70 | SL hit (close>static) qty=1.00 sl=658.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 656.00 | 653.94 | 653.91 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 651.15 | 655.04 | 655.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 14:15:00 | 638.10 | 651.15 | 653.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 526.45 | 522.91 | 538.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 09:45:00 | 526.60 | 522.91 | 538.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 537.20 | 525.77 | 538.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 537.20 | 525.77 | 538.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 530.85 | 526.78 | 537.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 532.60 | 526.78 | 537.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 536.60 | 528.75 | 537.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:00:00 | 536.60 | 528.75 | 537.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 533.75 | 529.75 | 537.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:30:00 | 532.85 | 529.75 | 537.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 533.10 | 531.51 | 536.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 532.95 | 531.51 | 536.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 539.30 | 533.07 | 536.92 | SL hit (close>static) qty=1.00 sl=538.50 alert=retest2 |

### Cycle 33 — BUY (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 10:15:00 | 542.25 | 529.51 | 529.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 12:15:00 | 544.55 | 534.13 | 531.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 11:15:00 | 552.90 | 552.99 | 543.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 12:00:00 | 552.90 | 552.99 | 543.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 561.80 | 553.21 | 545.90 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 531.20 | 541.82 | 542.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 523.40 | 535.93 | 539.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 534.65 | 531.25 | 535.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 11:15:00 | 534.65 | 531.25 | 535.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 534.65 | 531.25 | 535.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:45:00 | 539.00 | 531.25 | 535.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 537.65 | 532.53 | 535.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 537.65 | 532.53 | 535.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 545.00 | 535.03 | 536.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:00:00 | 545.00 | 535.03 | 536.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 554.95 | 539.01 | 538.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 621.00 | 556.93 | 546.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 647.50 | 656.34 | 629.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 10:00:00 | 647.50 | 656.34 | 629.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 624.15 | 649.90 | 629.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:00:00 | 624.15 | 649.90 | 629.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 623.60 | 644.64 | 628.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:30:00 | 620.50 | 644.64 | 628.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 598.80 | 618.64 | 621.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 593.65 | 606.16 | 613.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 597.30 | 597.13 | 604.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 15:00:00 | 597.30 | 597.13 | 604.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 588.60 | 595.21 | 602.25 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 09:15:00 | 636.30 | 607.41 | 605.09 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 609.15 | 615.05 | 615.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 601.25 | 612.29 | 614.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 593.95 | 593.25 | 599.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 15:00:00 | 593.95 | 593.25 | 599.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 591.70 | 592.37 | 597.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:30:00 | 597.10 | 592.37 | 597.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 591.55 | 590.40 | 593.92 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 15:15:00 | 595.90 | 593.38 | 593.36 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 13:15:00 | 592.25 | 593.28 | 593.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 590.50 | 592.72 | 593.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 592.85 | 592.41 | 592.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 592.85 | 592.41 | 592.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 592.85 | 592.41 | 592.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:45:00 | 592.75 | 592.41 | 592.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 588.55 | 591.64 | 592.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:30:00 | 589.95 | 591.64 | 592.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 590.50 | 591.41 | 592.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 590.50 | 591.41 | 592.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 592.35 | 591.60 | 592.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:45:00 | 591.50 | 591.60 | 592.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 590.80 | 591.44 | 592.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:30:00 | 590.10 | 591.51 | 592.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 590.00 | 591.51 | 592.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:30:00 | 590.55 | 591.71 | 592.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 10:15:00 | 590.40 | 591.71 | 592.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 590.75 | 591.52 | 591.99 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-02 15:15:00 | 593.75 | 592.21 | 592.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 15:15:00 | 593.75 | 592.21 | 592.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 09:15:00 | 595.90 | 592.95 | 592.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 11:15:00 | 592.05 | 592.78 | 592.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 11:15:00 | 592.05 | 592.78 | 592.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 592.05 | 592.78 | 592.53 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 15:15:00 | 590.80 | 592.12 | 592.29 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 595.10 | 592.71 | 592.55 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 12:15:00 | 590.00 | 592.20 | 592.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 587.85 | 591.33 | 591.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 15:15:00 | 592.30 | 591.31 | 591.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 15:15:00 | 592.30 | 591.31 | 591.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 592.30 | 591.31 | 591.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:30:00 | 589.05 | 591.12 | 591.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 10:15:00 | 588.55 | 591.12 | 591.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 12:30:00 | 589.10 | 590.80 | 591.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 597.45 | 591.76 | 591.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 09:15:00 | 597.45 | 591.76 | 591.58 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 592.30 | 596.08 | 596.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 13:15:00 | 591.85 | 595.24 | 595.91 | Break + close below crossover candle low |

### Cycle 47 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 604.20 | 596.79 | 596.48 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 11:15:00 | 595.45 | 596.32 | 596.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 13:15:00 | 592.80 | 595.44 | 595.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 15:15:00 | 592.00 | 591.35 | 592.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:15:00 | 591.95 | 591.35 | 592.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 592.20 | 591.52 | 592.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 592.20 | 591.52 | 592.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 596.05 | 592.43 | 593.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:45:00 | 596.80 | 592.43 | 593.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 11:15:00 | 599.60 | 593.86 | 593.74 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 555.40 | 587.03 | 590.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 526.80 | 574.98 | 585.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 524.25 | 519.20 | 530.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 14:00:00 | 524.25 | 519.20 | 530.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 524.85 | 520.90 | 528.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 15:15:00 | 516.85 | 521.24 | 525.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 12:15:00 | 526.60 | 523.10 | 522.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 12:15:00 | 526.60 | 523.10 | 522.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 15:15:00 | 530.00 | 525.21 | 523.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 09:15:00 | 521.80 | 524.53 | 523.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 521.80 | 524.53 | 523.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 521.80 | 524.53 | 523.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 520.20 | 524.53 | 523.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 521.00 | 523.82 | 523.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 521.00 | 523.82 | 523.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 521.05 | 523.55 | 523.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:00:00 | 521.05 | 523.55 | 523.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 522.10 | 523.26 | 523.24 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 516.55 | 521.92 | 522.63 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 10:15:00 | 528.45 | 522.89 | 522.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 11:15:00 | 539.20 | 526.16 | 524.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 15:15:00 | 528.05 | 529.29 | 526.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 15:15:00 | 528.05 | 529.29 | 526.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 528.05 | 529.29 | 526.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 523.50 | 529.29 | 526.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 524.55 | 528.34 | 526.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 11:30:00 | 529.50 | 527.48 | 526.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 09:15:00 | 535.35 | 527.99 | 527.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 12:15:00 | 560.05 | 567.68 | 568.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-01-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 12:15:00 | 560.05 | 567.68 | 568.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 11:15:00 | 558.00 | 563.14 | 565.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 561.15 | 560.20 | 563.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 561.15 | 560.20 | 563.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 555.15 | 552.89 | 557.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 15:00:00 | 555.15 | 552.89 | 557.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 565.00 | 555.32 | 557.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 15:15:00 | 550.20 | 556.64 | 557.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 11:15:00 | 562.30 | 558.60 | 558.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 11:15:00 | 562.30 | 558.60 | 558.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 13:15:00 | 567.30 | 560.54 | 559.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 10:15:00 | 563.05 | 564.58 | 561.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 11:00:00 | 563.05 | 564.58 | 561.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 560.55 | 563.77 | 561.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:45:00 | 562.45 | 563.77 | 561.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 556.75 | 562.37 | 561.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 13:00:00 | 556.75 | 562.37 | 561.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 552.10 | 560.31 | 560.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 15:15:00 | 550.00 | 556.98 | 558.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 551.20 | 548.30 | 552.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 551.20 | 548.30 | 552.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 551.20 | 548.30 | 552.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 552.35 | 548.30 | 552.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 535.35 | 536.73 | 541.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 527.55 | 536.73 | 541.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 10:15:00 | 501.17 | 513.44 | 517.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-02 14:15:00 | 516.80 | 513.45 | 516.00 | SL hit (close>ema200) qty=0.50 sl=513.45 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 531.75 | 518.19 | 517.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 540.00 | 522.55 | 519.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 591.00 | 591.07 | 576.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 554.20 | 583.61 | 580.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 554.20 | 583.61 | 580.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 554.20 | 583.61 | 580.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 557.70 | 578.42 | 578.31 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 11:15:00 | 561.20 | 574.98 | 576.76 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 11:15:00 | 590.00 | 577.57 | 576.60 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 576.05 | 577.31 | 577.39 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 13:15:00 | 580.00 | 577.85 | 577.63 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 15:15:00 | 575.00 | 577.45 | 577.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 568.15 | 575.59 | 576.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 552.40 | 550.72 | 558.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:45:00 | 554.05 | 550.72 | 558.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 569.20 | 553.36 | 555.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:45:00 | 566.85 | 553.36 | 555.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 571.80 | 557.04 | 556.86 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 10:15:00 | 554.85 | 565.31 | 566.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 548.05 | 556.97 | 561.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 552.00 | 551.12 | 556.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 14:15:00 | 552.00 | 551.12 | 556.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 552.00 | 551.12 | 556.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 552.00 | 551.12 | 556.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 560.00 | 553.07 | 556.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 560.00 | 553.07 | 556.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 558.10 | 554.08 | 556.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:30:00 | 560.70 | 554.08 | 556.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 558.00 | 556.16 | 556.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:45:00 | 558.15 | 556.16 | 556.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 559.40 | 556.81 | 557.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 15:00:00 | 559.40 | 556.81 | 557.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 558.45 | 557.14 | 557.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:15:00 | 563.95 | 557.14 | 557.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 564.05 | 558.52 | 557.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 11:15:00 | 577.45 | 562.81 | 559.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 11:15:00 | 574.60 | 575.05 | 569.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 11:30:00 | 574.95 | 575.05 | 569.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 581.10 | 576.26 | 571.88 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 551.20 | 567.89 | 569.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 14:15:00 | 539.00 | 553.75 | 561.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 12:15:00 | 548.15 | 546.32 | 553.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 13:15:00 | 546.70 | 546.32 | 553.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 541.00 | 545.49 | 551.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:45:00 | 534.00 | 543.56 | 549.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:15:00 | 539.15 | 535.69 | 536.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 10:15:00 | 565.00 | 536.29 | 535.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 10:15:00 | 565.00 | 536.29 | 535.28 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 539.20 | 541.20 | 541.40 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 542.70 | 540.98 | 540.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 548.10 | 542.41 | 541.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 09:15:00 | 542.95 | 543.09 | 542.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 10:00:00 | 542.95 | 543.09 | 542.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 548.20 | 549.54 | 546.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 557.30 | 547.60 | 546.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 13:15:00 | 554.25 | 552.71 | 549.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 15:15:00 | 560.00 | 552.72 | 550.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 541.85 | 551.71 | 550.36 | SL hit (close<static) qty=1.00 sl=544.40 alert=retest2 |

### Cycle 70 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 538.30 | 549.03 | 549.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 532.05 | 543.87 | 546.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 540.00 | 539.36 | 542.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 540.00 | 539.36 | 542.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 540.00 | 539.36 | 542.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 540.00 | 539.36 | 542.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 544.00 | 540.28 | 542.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 14:30:00 | 537.50 | 539.07 | 541.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 552.20 | 541.52 | 542.42 | SL hit (close>static) qty=1.00 sl=545.45 alert=retest2 |

### Cycle 71 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 547.75 | 543.91 | 543.42 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 15:15:00 | 538.05 | 542.93 | 543.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 09:15:00 | 534.20 | 541.18 | 542.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 14:15:00 | 551.95 | 540.10 | 540.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 14:15:00 | 551.95 | 540.10 | 540.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 551.95 | 540.10 | 540.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 551.95 | 540.10 | 540.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 531.50 | 538.38 | 540.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 554.85 | 538.38 | 540.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 544.20 | 539.54 | 540.49 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 13:15:00 | 548.50 | 542.10 | 541.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 09:15:00 | 563.85 | 548.56 | 544.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 549.00 | 557.56 | 552.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 549.00 | 557.56 | 552.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 549.00 | 557.56 | 552.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:15:00 | 546.25 | 557.56 | 552.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 547.85 | 555.61 | 552.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 553.15 | 555.36 | 552.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-07 09:15:00 | 608.47 | 576.84 | 565.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 838.35 | 856.11 | 858.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 827.65 | 845.27 | 852.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 844.85 | 835.55 | 842.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 844.85 | 835.55 | 842.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 844.85 | 835.55 | 842.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 844.85 | 835.55 | 842.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 861.25 | 840.69 | 844.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 861.00 | 840.69 | 844.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 872.95 | 850.82 | 848.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 896.80 | 868.88 | 858.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 864.25 | 873.24 | 864.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 13:15:00 | 864.25 | 873.24 | 864.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 864.25 | 873.24 | 864.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 864.25 | 873.24 | 864.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 855.60 | 869.71 | 863.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 855.60 | 869.71 | 863.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 850.00 | 865.77 | 862.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:15:00 | 858.40 | 864.00 | 862.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 13:15:00 | 857.40 | 860.70 | 860.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 857.40 | 860.70 | 860.99 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 14:15:00 | 865.30 | 861.62 | 861.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 15:15:00 | 871.00 | 863.49 | 862.26 | Break + close above crossover candle high |

### Cycle 78 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 841.00 | 858.99 | 860.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 840.75 | 855.35 | 858.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 858.15 | 853.79 | 856.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 14:15:00 | 858.15 | 853.79 | 856.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 858.15 | 853.79 | 856.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 850.60 | 853.79 | 856.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 858.95 | 854.82 | 856.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 849.85 | 854.82 | 856.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 851.75 | 854.21 | 856.31 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 873.40 | 858.05 | 857.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 886.95 | 873.63 | 866.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 14:15:00 | 877.35 | 877.41 | 870.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 14:15:00 | 877.35 | 877.41 | 870.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 877.35 | 877.41 | 870.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:30:00 | 872.15 | 877.41 | 870.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 844.95 | 871.33 | 868.91 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 10:15:00 | 835.25 | 864.12 | 865.85 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 13:15:00 | 865.50 | 861.04 | 860.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 881.00 | 866.20 | 863.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 870.65 | 871.72 | 867.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 15:00:00 | 870.65 | 871.72 | 867.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 866.00 | 870.58 | 867.50 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 15:15:00 | 466.90 | 2025-05-21 13:15:00 | 470.05 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2025-05-16 11:15:00 | 467.70 | 2025-05-21 13:15:00 | 470.05 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2025-05-19 10:00:00 | 470.00 | 2025-05-21 13:15:00 | 470.05 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-05-21 12:45:00 | 467.40 | 2025-05-21 13:15:00 | 470.05 | STOP_HIT | 1.00 | 0.57% |
| SELL | retest2 | 2025-06-16 14:15:00 | 454.60 | 2025-06-18 15:15:00 | 459.40 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-06-17 10:15:00 | 453.35 | 2025-06-18 15:15:00 | 459.40 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-06-19 09:15:00 | 454.75 | 2025-06-19 11:15:00 | 460.45 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-06-19 09:45:00 | 454.25 | 2025-06-19 11:15:00 | 460.45 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-06-27 09:15:00 | 541.40 | 2025-06-27 10:15:00 | 533.50 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-06-27 10:00:00 | 539.05 | 2025-06-27 10:15:00 | 533.50 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-07-02 13:15:00 | 521.60 | 2025-07-03 09:15:00 | 550.00 | STOP_HIT | 1.00 | -5.44% |
| SELL | retest2 | 2025-07-02 14:15:00 | 523.00 | 2025-07-03 09:15:00 | 550.00 | STOP_HIT | 1.00 | -5.16% |
| BUY | retest2 | 2025-07-04 15:15:00 | 549.85 | 2025-07-07 15:15:00 | 540.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-07-15 11:30:00 | 586.00 | 2025-07-21 10:15:00 | 641.69 | TARGET_HIT | 1.00 | 9.50% |
| BUY | retest2 | 2025-07-15 13:00:00 | 583.35 | 2025-07-21 10:15:00 | 641.30 | TARGET_HIT | 1.00 | 9.93% |
| BUY | retest2 | 2025-07-15 13:45:00 | 583.00 | 2025-07-21 10:15:00 | 642.95 | TARGET_HIT | 1.00 | 10.28% |
| BUY | retest2 | 2025-07-15 15:00:00 | 584.50 | 2025-07-21 11:15:00 | 644.60 | TARGET_HIT | 1.00 | 10.28% |
| BUY | retest2 | 2025-07-17 14:45:00 | 592.00 | 2025-07-21 14:15:00 | 651.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-18 09:45:00 | 591.35 | 2025-07-21 14:15:00 | 650.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-21 09:15:00 | 593.85 | 2025-07-24 10:15:00 | 653.24 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-21 12:15:00 | 632.40 | 2025-08-26 15:15:00 | 600.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 12:15:00 | 632.40 | 2025-09-01 11:15:00 | 585.80 | STOP_HIT | 0.50 | 7.37% |
| SELL | retest2 | 2025-09-19 14:15:00 | 660.60 | 2025-09-22 09:15:00 | 685.00 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2025-09-19 15:00:00 | 660.70 | 2025-09-22 09:15:00 | 685.00 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2025-09-30 09:15:00 | 656.50 | 2025-10-01 12:15:00 | 664.35 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-10-01 09:30:00 | 658.50 | 2025-10-01 12:15:00 | 664.35 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-10-01 10:30:00 | 656.60 | 2025-10-01 12:15:00 | 664.35 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-10-14 10:30:00 | 652.95 | 2025-10-15 14:15:00 | 661.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-10-15 11:30:00 | 652.45 | 2025-10-15 14:15:00 | 661.30 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-10-29 09:15:00 | 532.95 | 2025-10-29 09:15:00 | 539.30 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-10-29 11:00:00 | 532.80 | 2025-11-03 09:15:00 | 539.85 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-10-29 11:30:00 | 532.00 | 2025-11-03 09:15:00 | 539.85 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-10-30 09:45:00 | 532.25 | 2025-11-03 09:15:00 | 539.85 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-12-01 14:30:00 | 590.10 | 2025-12-02 15:15:00 | 593.75 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-12-01 15:15:00 | 590.00 | 2025-12-02 15:15:00 | 593.75 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-12-02 09:30:00 | 590.55 | 2025-12-02 15:15:00 | 593.75 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-12-02 10:15:00 | 590.40 | 2025-12-02 15:15:00 | 593.75 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-12-05 09:30:00 | 589.05 | 2025-12-08 09:15:00 | 597.45 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-12-05 10:15:00 | 588.55 | 2025-12-08 09:15:00 | 597.45 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-12-05 12:30:00 | 589.10 | 2025-12-08 09:15:00 | 597.45 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-12-22 15:15:00 | 516.85 | 2025-12-24 12:15:00 | 526.60 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-12-30 11:30:00 | 529.50 | 2026-01-12 12:15:00 | 560.05 | STOP_HIT | 1.00 | 5.77% |
| BUY | retest2 | 2025-12-31 09:15:00 | 535.35 | 2026-01-12 12:15:00 | 560.05 | STOP_HIT | 1.00 | 4.61% |
| SELL | retest2 | 2026-01-16 15:15:00 | 550.20 | 2026-01-19 11:15:00 | 562.30 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2026-01-27 09:15:00 | 527.55 | 2026-02-02 10:15:00 | 501.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-27 09:15:00 | 527.55 | 2026-02-02 14:15:00 | 516.80 | STOP_HIT | 0.50 | 2.04% |
| SELL | retest2 | 2026-03-06 10:45:00 | 534.00 | 2026-03-12 10:15:00 | 565.00 | STOP_HIT | 1.00 | -5.81% |
| SELL | retest2 | 2026-03-11 10:15:00 | 539.15 | 2026-03-12 10:15:00 | 565.00 | STOP_HIT | 1.00 | -4.79% |
| BUY | retest2 | 2026-03-20 09:15:00 | 557.30 | 2026-03-23 09:15:00 | 541.85 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2026-03-20 13:15:00 | 554.25 | 2026-03-23 09:15:00 | 541.85 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2026-03-20 15:15:00 | 560.00 | 2026-03-23 09:15:00 | 541.85 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2026-03-24 14:30:00 | 537.50 | 2026-03-25 09:15:00 | 552.20 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2026-04-02 11:30:00 | 553.15 | 2026-04-07 09:15:00 | 608.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-29 10:15:00 | 858.40 | 2026-04-29 13:15:00 | 857.40 | STOP_HIT | 1.00 | -0.12% |
