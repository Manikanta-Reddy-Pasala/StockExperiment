# H.E.G. Ltd. (HEG)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 596.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 130 |
| ALERT1 | 93 |
| ALERT2 | 90 |
| ALERT2_SKIP | 48 |
| ALERT3 | 227 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 115 |
| PARTIAL | 14 |
| TARGET_HIT | 9 |
| STOP_HIT | 110 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 133 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 50 / 83
- **Target hits / Stop hits / Partials:** 9 / 110 / 14
- **Avg / median % per leg:** 0.45% / -0.75%
- **Sum % (uncompounded):** 59.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 16 | 31.4% | 9 | 42 | 0 | 0.57% | 29.3% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.47% | -6.9% |
| BUY @ 3rd Alert (retest2) | 49 | 16 | 32.7% | 9 | 40 | 0 | 0.74% | 36.2% |
| SELL (all) | 82 | 34 | 41.5% | 0 | 68 | 14 | 0.37% | 30.3% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.34% | -4.0% |
| SELL @ 3rd Alert (retest2) | 79 | 34 | 43.0% | 0 | 65 | 14 | 0.43% | 34.3% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.19% | -11.0% |
| retest2 (combined) | 128 | 50 | 39.1% | 9 | 105 | 14 | 0.55% | 70.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 480.04 | 477.33 | 476.97 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 11:15:00 | 476.23 | 477.46 | 477.54 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 13:15:00 | 478.40 | 477.73 | 477.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 482.00 | 478.61 | 478.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 475.54 | 478.70 | 478.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 12:15:00 | 475.54 | 478.70 | 478.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 475.54 | 478.70 | 478.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:00:00 | 475.54 | 478.70 | 478.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 13:15:00 | 474.59 | 477.88 | 477.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 15:15:00 | 472.01 | 476.33 | 477.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 09:15:00 | 480.16 | 477.10 | 477.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-17 09:15:00 | 480.16 | 477.10 | 477.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 480.16 | 477.10 | 477.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:45:00 | 480.94 | 477.10 | 477.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 480.05 | 477.69 | 477.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 11:15:00 | 477.19 | 477.69 | 477.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-17 13:15:00 | 483.78 | 478.05 | 477.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-05-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 13:15:00 | 483.78 | 478.05 | 477.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 11:15:00 | 496.05 | 483.43 | 480.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 09:15:00 | 496.59 | 523.88 | 514.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 09:15:00 | 496.59 | 523.88 | 514.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 496.59 | 523.88 | 514.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:00:00 | 496.59 | 523.88 | 514.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 492.80 | 517.67 | 512.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:30:00 | 488.01 | 517.67 | 512.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 12:15:00 | 485.69 | 506.20 | 507.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 14:15:00 | 445.79 | 490.73 | 500.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 09:15:00 | 460.94 | 454.34 | 463.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 09:15:00 | 460.94 | 454.34 | 463.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 460.94 | 454.34 | 463.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 09:30:00 | 463.30 | 454.34 | 463.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 459.58 | 455.39 | 462.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 11:15:00 | 458.23 | 455.39 | 462.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 11:45:00 | 458.21 | 455.47 | 462.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 10:00:00 | 458.63 | 455.10 | 459.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 15:15:00 | 458.00 | 450.46 | 451.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 456.20 | 452.82 | 452.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 456.20 | 452.82 | 452.56 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 426.44 | 448.21 | 450.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 402.51 | 439.07 | 446.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 14:15:00 | 413.40 | 409.19 | 421.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 15:00:00 | 413.40 | 409.19 | 421.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 426.49 | 413.29 | 420.87 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 15:15:00 | 427.80 | 424.26 | 424.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 430.19 | 425.45 | 424.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 436.20 | 436.25 | 433.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:30:00 | 438.93 | 436.78 | 433.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 443.82 | 445.49 | 442.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:45:00 | 443.58 | 445.49 | 442.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 442.33 | 444.86 | 442.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:00:00 | 442.33 | 444.86 | 442.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 442.56 | 444.40 | 442.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 13:00:00 | 443.20 | 444.16 | 442.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 14:00:00 | 443.18 | 443.96 | 442.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 444.40 | 443.44 | 442.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 11:15:00 | 439.86 | 443.34 | 442.98 | SL hit (close<static) qty=1.00 sl=440.88 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 12:15:00 | 439.10 | 442.49 | 442.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 09:15:00 | 438.21 | 440.29 | 441.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 09:15:00 | 439.61 | 439.14 | 440.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 439.61 | 439.14 | 440.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 439.61 | 439.14 | 440.14 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2024-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 12:15:00 | 455.66 | 443.53 | 441.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 15:15:00 | 463.00 | 451.57 | 446.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 15:15:00 | 462.20 | 462.94 | 455.93 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:15:00 | 466.47 | 462.94 | 455.93 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 12:15:00 | 472.13 | 462.42 | 457.38 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 453.00 | 460.94 | 458.72 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-24 09:15:00 | 453.00 | 460.94 | 458.72 | SL hit (close<ema400) qty=1.00 sl=458.72 alert=retest1 |

### Cycle 12 — SELL (started 2024-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 13:15:00 | 452.66 | 456.77 | 457.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 14:15:00 | 451.96 | 455.81 | 456.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 440.60 | 434.38 | 436.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 440.60 | 434.38 | 436.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 440.60 | 434.38 | 436.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:00:00 | 440.60 | 434.38 | 436.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 440.61 | 435.63 | 436.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:45:00 | 440.97 | 435.63 | 436.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 442.19 | 437.84 | 437.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 447.20 | 441.91 | 439.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 14:15:00 | 443.42 | 444.59 | 442.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 15:00:00 | 443.42 | 444.59 | 442.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 442.37 | 444.14 | 442.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 444.70 | 444.14 | 442.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 10:30:00 | 445.25 | 444.38 | 442.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 13:30:00 | 446.57 | 450.76 | 450.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 14:15:00 | 441.88 | 448.99 | 449.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 14:15:00 | 441.88 | 448.99 | 449.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 13:15:00 | 440.70 | 444.01 | 446.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 10:15:00 | 427.60 | 427.37 | 431.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-12 11:00:00 | 427.60 | 427.37 | 431.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 422.70 | 423.31 | 425.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:30:00 | 427.61 | 423.31 | 425.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 417.40 | 420.06 | 422.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 419.19 | 420.06 | 422.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 416.02 | 416.75 | 419.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 15:00:00 | 416.02 | 416.75 | 419.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 407.77 | 410.30 | 412.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 404.20 | 411.34 | 412.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 10:15:00 | 422.84 | 410.27 | 410.77 | SL hit (close>static) qty=1.00 sl=415.80 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 425.78 | 413.37 | 412.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 13:15:00 | 429.58 | 418.18 | 414.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 15:15:00 | 424.01 | 424.36 | 420.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 09:45:00 | 424.80 | 424.13 | 421.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 13:15:00 | 422.45 | 423.85 | 421.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 13:45:00 | 421.75 | 423.85 | 421.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 424.22 | 423.93 | 422.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 15:15:00 | 423.00 | 423.93 | 422.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 15:15:00 | 423.00 | 423.74 | 422.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 09:15:00 | 430.91 | 423.74 | 422.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 09:15:00 | 429.03 | 423.58 | 423.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 11:15:00 | 436.18 | 440.67 | 440.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 11:15:00 | 436.18 | 440.67 | 440.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 13:15:00 | 434.00 | 438.52 | 439.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 12:15:00 | 423.78 | 418.12 | 423.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 12:15:00 | 423.78 | 418.12 | 423.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 12:15:00 | 423.78 | 418.12 | 423.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 13:00:00 | 423.78 | 418.12 | 423.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 13:15:00 | 410.00 | 416.50 | 422.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:15:00 | 404.87 | 416.50 | 422.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 14:15:00 | 409.80 | 408.15 | 413.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 15:00:00 | 409.94 | 408.50 | 413.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 15:00:00 | 409.60 | 410.80 | 412.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 412.50 | 410.95 | 412.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 410.60 | 410.95 | 412.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:00:00 | 411.00 | 410.96 | 412.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 09:15:00 | 418.10 | 413.13 | 412.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 09:15:00 | 418.10 | 413.13 | 412.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 12:15:00 | 422.03 | 416.60 | 414.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 09:15:00 | 403.80 | 425.49 | 423.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 09:15:00 | 403.80 | 425.49 | 423.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 403.80 | 425.49 | 423.08 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 402.00 | 420.79 | 421.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-16 09:15:00 | 397.69 | 406.14 | 412.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 09:15:00 | 400.98 | 400.39 | 405.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-19 09:45:00 | 402.20 | 400.39 | 405.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 405.20 | 401.76 | 404.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 15:00:00 | 405.20 | 401.76 | 404.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 15:15:00 | 404.60 | 402.33 | 404.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:15:00 | 405.76 | 402.33 | 404.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 405.74 | 403.01 | 404.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 10:30:00 | 404.00 | 403.31 | 404.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 14:45:00 | 404.10 | 404.49 | 404.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 13:15:00 | 406.32 | 405.18 | 405.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 13:15:00 | 406.32 | 405.18 | 405.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 14:15:00 | 406.80 | 405.50 | 405.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 15:15:00 | 406.38 | 406.42 | 405.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 15:15:00 | 406.38 | 406.42 | 405.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 406.38 | 406.42 | 405.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:45:00 | 406.03 | 406.37 | 405.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 406.00 | 406.30 | 405.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:15:00 | 405.80 | 406.30 | 405.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 405.80 | 406.20 | 405.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:30:00 | 405.70 | 406.20 | 405.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 405.98 | 406.15 | 405.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:00:00 | 405.98 | 406.15 | 405.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 405.88 | 406.10 | 405.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:00:00 | 405.88 | 406.10 | 405.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 404.31 | 405.74 | 405.80 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 407.88 | 405.95 | 405.88 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 10:15:00 | 405.44 | 406.14 | 406.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 15:15:00 | 403.80 | 405.27 | 405.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 397.66 | 396.62 | 398.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 397.66 | 396.62 | 398.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 397.66 | 396.62 | 398.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:30:00 | 397.95 | 396.62 | 398.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 399.02 | 396.72 | 397.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:45:00 | 399.34 | 396.72 | 397.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 397.70 | 396.91 | 397.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 12:30:00 | 395.60 | 396.00 | 397.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 12:00:00 | 395.78 | 395.68 | 396.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 12:30:00 | 395.00 | 395.74 | 396.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 13:30:00 | 395.77 | 395.65 | 396.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 398.63 | 396.18 | 396.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-06 10:15:00 | 398.57 | 396.66 | 396.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 10:15:00 | 398.57 | 396.66 | 396.50 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 394.95 | 396.38 | 396.48 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 12:15:00 | 401.72 | 397.26 | 396.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 408.45 | 401.68 | 399.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 09:15:00 | 404.40 | 404.99 | 402.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 10:00:00 | 404.40 | 404.99 | 402.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 402.20 | 404.04 | 402.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 13:00:00 | 402.20 | 404.04 | 402.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 400.58 | 403.35 | 402.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 400.58 | 403.35 | 402.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 09:15:00 | 398.52 | 401.60 | 401.81 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 11:15:00 | 402.69 | 401.54 | 401.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 13:15:00 | 403.20 | 402.08 | 401.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 419.58 | 424.14 | 416.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 09:45:00 | 418.26 | 424.14 | 416.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 446.65 | 450.49 | 444.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 10:30:00 | 456.80 | 452.13 | 446.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 12:30:00 | 454.73 | 452.80 | 447.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 14:00:00 | 454.14 | 453.07 | 448.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 15:00:00 | 455.00 | 453.45 | 448.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 452.40 | 452.88 | 449.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 13:45:00 | 459.94 | 453.20 | 450.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 09:15:00 | 458.00 | 452.22 | 450.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 10:15:00 | 457.00 | 452.82 | 450.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 11:45:00 | 457.40 | 454.60 | 452.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-25 09:15:00 | 502.48 | 468.17 | 460.10 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 480.89 | 486.99 | 487.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 471.00 | 483.79 | 486.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 10:15:00 | 480.34 | 480.05 | 482.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 11:00:00 | 480.34 | 480.05 | 482.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 460.48 | 454.33 | 463.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:00:00 | 460.48 | 454.33 | 463.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 467.68 | 457.00 | 463.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:30:00 | 467.60 | 457.00 | 463.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 463.96 | 458.39 | 463.82 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 484.97 | 468.07 | 467.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 15:15:00 | 490.22 | 472.50 | 469.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 11:15:00 | 492.21 | 492.86 | 485.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 12:00:00 | 492.21 | 492.86 | 485.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 487.40 | 490.98 | 486.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:15:00 | 484.40 | 490.98 | 486.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 487.89 | 490.36 | 486.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:30:00 | 483.90 | 490.36 | 486.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 487.58 | 489.81 | 486.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:45:00 | 487.43 | 489.81 | 486.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 495.89 | 491.02 | 487.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 10:30:00 | 515.94 | 497.15 | 493.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 15:00:00 | 502.44 | 499.55 | 495.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 502.89 | 499.97 | 496.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 09:15:00 | 502.44 | 498.51 | 497.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 500.40 | 498.72 | 497.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 14:00:00 | 512.00 | 503.06 | 500.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 14:15:00 | 495.65 | 500.15 | 500.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 14:15:00 | 495.65 | 500.15 | 500.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 09:15:00 | 477.75 | 495.64 | 498.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 456.15 | 445.36 | 460.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:00:00 | 456.15 | 445.36 | 460.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 424.50 | 421.41 | 428.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 15:15:00 | 423.00 | 422.51 | 427.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:30:00 | 421.90 | 423.41 | 426.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 11:30:00 | 422.05 | 423.40 | 426.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 12:15:00 | 422.85 | 423.40 | 426.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 426.25 | 423.96 | 426.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:00:00 | 426.25 | 423.96 | 426.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 427.55 | 424.68 | 426.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 15:00:00 | 427.55 | 424.68 | 426.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 427.25 | 425.19 | 426.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:15:00 | 429.55 | 425.19 | 426.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 430.90 | 426.34 | 426.77 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-30 09:15:00 | 430.90 | 426.34 | 426.77 | SL hit (close>static) qty=1.00 sl=429.15 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 431.30 | 427.33 | 427.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 435.75 | 429.67 | 428.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 10:15:00 | 430.40 | 430.78 | 429.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 10:15:00 | 430.40 | 430.78 | 429.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 430.40 | 430.78 | 429.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 10:30:00 | 431.90 | 430.78 | 429.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 431.90 | 431.00 | 429.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 14:45:00 | 433.15 | 430.51 | 429.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 426.50 | 433.83 | 431.87 | SL hit (close<static) qty=1.00 sl=429.50 alert=retest2 |

### Cycle 32 — SELL (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 12:15:00 | 439.85 | 443.42 | 443.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 13:15:00 | 436.60 | 442.06 | 442.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 13:15:00 | 436.00 | 434.11 | 437.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 13:15:00 | 436.00 | 434.11 | 437.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 13:15:00 | 436.00 | 434.11 | 437.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 14:00:00 | 436.00 | 434.11 | 437.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 417.30 | 411.94 | 414.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 10:00:00 | 417.30 | 411.94 | 414.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 415.35 | 412.62 | 414.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 13:30:00 | 414.15 | 414.12 | 414.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 14:15:00 | 418.40 | 414.98 | 414.99 | SL hit (close>static) qty=1.00 sl=417.30 alert=retest2 |

### Cycle 33 — BUY (started 2024-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 15:15:00 | 417.35 | 415.45 | 415.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 423.70 | 417.10 | 415.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 12:15:00 | 417.95 | 418.43 | 416.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 13:00:00 | 417.95 | 418.43 | 416.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 412.60 | 417.27 | 416.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 14:00:00 | 412.60 | 417.27 | 416.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 410.30 | 415.87 | 416.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 409.25 | 414.55 | 415.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 11:15:00 | 411.55 | 408.91 | 411.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 11:15:00 | 411.55 | 408.91 | 411.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 411.55 | 408.91 | 411.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:45:00 | 411.75 | 408.91 | 411.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 408.85 | 408.90 | 410.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 14:30:00 | 407.90 | 408.83 | 410.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 417.00 | 410.42 | 411.00 | SL hit (close>static) qty=1.00 sl=411.75 alert=retest2 |

### Cycle 35 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 418.25 | 411.99 | 411.66 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 14:15:00 | 411.30 | 412.66 | 412.78 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 09:15:00 | 417.70 | 413.53 | 413.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 426.40 | 418.00 | 415.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 14:15:00 | 438.60 | 438.88 | 434.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-02 14:45:00 | 438.00 | 438.88 | 434.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 477.55 | 446.76 | 439.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 10:45:00 | 484.95 | 455.19 | 443.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-04 09:15:00 | 533.45 | 496.67 | 472.09 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 11:15:00 | 563.10 | 569.56 | 570.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 558.00 | 566.25 | 568.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 09:15:00 | 565.00 | 555.31 | 560.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 09:15:00 | 565.00 | 555.31 | 560.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 565.00 | 555.31 | 560.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:30:00 | 571.20 | 555.31 | 560.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 571.80 | 558.61 | 561.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:45:00 | 572.50 | 558.61 | 561.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2024-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 13:15:00 | 571.30 | 563.93 | 563.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 11:15:00 | 579.10 | 568.54 | 565.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 09:15:00 | 567.55 | 573.08 | 569.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 09:15:00 | 567.55 | 573.08 | 569.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 567.55 | 573.08 | 569.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:15:00 | 564.15 | 573.08 | 569.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 566.20 | 571.70 | 569.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:30:00 | 565.75 | 571.70 | 569.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 565.00 | 567.91 | 568.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 15:15:00 | 557.00 | 563.94 | 565.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 12:15:00 | 567.15 | 562.47 | 564.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 12:15:00 | 567.15 | 562.47 | 564.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 567.15 | 562.47 | 564.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 13:00:00 | 567.15 | 562.47 | 564.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 566.70 | 563.31 | 564.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 13:30:00 | 568.05 | 563.31 | 564.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 10:15:00 | 573.00 | 565.63 | 565.20 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 542.00 | 560.54 | 563.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 11:15:00 | 537.90 | 546.32 | 554.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 541.45 | 539.87 | 547.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 09:15:00 | 541.45 | 539.87 | 547.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 541.45 | 539.87 | 547.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 09:45:00 | 546.55 | 539.87 | 547.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 519.70 | 515.97 | 521.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:45:00 | 519.30 | 515.97 | 521.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 508.30 | 505.47 | 510.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 12:30:00 | 507.80 | 505.47 | 510.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 510.40 | 506.46 | 510.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 13:30:00 | 509.65 | 506.46 | 510.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 509.80 | 507.12 | 510.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:45:00 | 510.35 | 507.12 | 510.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 510.40 | 507.78 | 510.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:15:00 | 513.75 | 507.78 | 510.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 513.15 | 508.85 | 510.87 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 11:15:00 | 523.10 | 512.57 | 512.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 13:15:00 | 527.25 | 517.42 | 514.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 11:15:00 | 530.75 | 530.99 | 526.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 12:00:00 | 530.75 | 530.99 | 526.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 526.80 | 529.88 | 527.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 516.10 | 529.88 | 527.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 512.55 | 526.42 | 525.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 512.55 | 526.42 | 525.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 502.85 | 521.70 | 523.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 502.05 | 517.77 | 521.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 15:15:00 | 430.10 | 429.38 | 438.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 09:15:00 | 440.50 | 429.38 | 438.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 430.85 | 429.68 | 437.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 13:30:00 | 428.40 | 431.30 | 436.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 09:30:00 | 427.95 | 432.82 | 434.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 10:45:00 | 428.00 | 432.00 | 433.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 15:15:00 | 428.80 | 430.49 | 432.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 425.80 | 429.28 | 431.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:15:00 | 422.45 | 429.83 | 430.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 15:15:00 | 424.75 | 425.26 | 427.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 11:15:00 | 406.98 | 417.91 | 423.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 11:15:00 | 407.36 | 417.91 | 423.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 12:15:00 | 406.55 | 416.07 | 421.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 12:15:00 | 406.60 | 416.07 | 421.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-22 14:15:00 | 416.65 | 415.31 | 420.53 | SL hit (close>ema200) qty=0.50 sl=415.31 alert=retest2 |

### Cycle 45 — BUY (started 2025-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 10:15:00 | 376.70 | 376.03 | 375.94 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 13:15:00 | 373.55 | 375.78 | 375.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 369.15 | 374.59 | 375.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 13:15:00 | 366.20 | 366.15 | 369.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-10 14:00:00 | 366.20 | 366.15 | 369.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 354.50 | 363.59 | 367.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 13:45:00 | 350.00 | 357.78 | 363.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 15:00:00 | 351.05 | 356.43 | 362.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 09:15:00 | 340.95 | 355.54 | 361.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 11:00:00 | 347.55 | 346.77 | 351.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 333.50 | 338.68 | 343.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 14:15:00 | 339.20 | 338.79 | 343.44 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-14 14:15:00 | 339.20 | 338.79 | 343.44 | SL hit (close>ema200) qty=0.50 sl=338.79 alert=retest2 |

### Cycle 47 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 356.10 | 344.07 | 342.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 360.65 | 353.28 | 348.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 357.90 | 358.95 | 354.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 15:15:00 | 356.75 | 358.85 | 356.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 356.75 | 358.85 | 356.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 353.65 | 358.85 | 356.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 356.85 | 358.45 | 356.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:15:00 | 359.65 | 358.32 | 356.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 11:15:00 | 349.95 | 355.89 | 356.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 11:15:00 | 349.95 | 355.89 | 356.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 14:15:00 | 342.80 | 351.58 | 354.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 15:15:00 | 346.85 | 346.42 | 349.30 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-28 09:15:00 | 340.60 | 346.42 | 349.30 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-28 10:45:00 | 345.40 | 345.56 | 348.37 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-28 12:30:00 | 345.40 | 345.57 | 347.89 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 13:15:00 | 348.40 | 346.14 | 347.93 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-28 13:15:00 | 348.40 | 346.14 | 347.93 | SL hit (close>ema400) qty=1.00 sl=347.93 alert=retest1 |

### Cycle 49 — BUY (started 2025-02-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-28 15:15:00 | 357.00 | 350.46 | 349.72 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-03-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 09:15:00 | 343.80 | 349.13 | 349.18 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-03-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 15:15:00 | 350.20 | 348.86 | 348.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 09:15:00 | 362.50 | 351.59 | 350.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 12:15:00 | 390.85 | 391.72 | 385.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 12:45:00 | 389.35 | 391.72 | 385.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 389.00 | 390.71 | 386.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 401.25 | 390.71 | 386.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 404.60 | 393.49 | 388.43 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 10:15:00 | 396.65 | 400.00 | 400.34 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 13:15:00 | 405.45 | 401.10 | 400.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 10:15:00 | 411.00 | 404.92 | 402.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 14:15:00 | 432.35 | 432.46 | 427.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 14:30:00 | 431.20 | 432.46 | 427.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 490.70 | 444.12 | 433.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 13:15:00 | 497.05 | 463.34 | 446.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:45:00 | 502.75 | 479.84 | 460.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:30:00 | 510.95 | 486.76 | 483.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 10:00:00 | 498.25 | 494.33 | 488.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 477.40 | 493.25 | 491.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 477.40 | 493.25 | 491.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 478.85 | 490.37 | 490.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-04 11:15:00 | 476.45 | 487.58 | 489.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 476.45 | 487.58 | 489.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 474.05 | 484.88 | 487.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 454.85 | 454.34 | 466.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 454.85 | 454.34 | 466.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 458.40 | 456.05 | 465.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 456.70 | 456.89 | 465.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 15:15:00 | 456.50 | 459.53 | 464.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 09:30:00 | 455.75 | 447.51 | 453.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 10:15:00 | 466.25 | 456.85 | 455.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 10:15:00 | 466.25 | 456.85 | 455.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 477.15 | 460.91 | 457.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 478.95 | 481.68 | 476.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 15:00:00 | 478.95 | 481.68 | 476.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 476.10 | 480.56 | 476.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:15:00 | 477.15 | 480.56 | 476.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 477.20 | 479.89 | 476.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:30:00 | 476.05 | 479.89 | 476.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 10:15:00 | 481.85 | 480.28 | 476.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 09:15:00 | 485.70 | 480.69 | 478.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 10:15:00 | 484.30 | 481.15 | 478.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 14:00:00 | 484.65 | 482.41 | 480.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 470.70 | 479.29 | 479.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 09:15:00 | 470.70 | 479.29 | 479.32 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 12:15:00 | 480.30 | 477.85 | 477.65 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 15:15:00 | 475.00 | 477.39 | 477.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 461.40 | 474.20 | 476.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 479.05 | 467.96 | 470.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 479.05 | 467.96 | 470.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 479.05 | 467.96 | 470.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 479.05 | 467.96 | 470.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 482.20 | 470.81 | 471.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:45:00 | 481.70 | 470.81 | 471.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 11:15:00 | 484.60 | 473.57 | 472.83 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 15:15:00 | 471.00 | 474.34 | 474.68 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 11:15:00 | 475.65 | 474.94 | 474.86 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 465.35 | 473.01 | 474.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 464.10 | 471.22 | 473.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 10:15:00 | 465.25 | 463.86 | 467.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 11:00:00 | 465.25 | 463.86 | 467.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 465.85 | 464.26 | 467.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:30:00 | 465.15 | 464.26 | 467.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 466.45 | 464.70 | 467.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 13:00:00 | 466.45 | 464.70 | 467.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 468.90 | 465.54 | 467.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:00:00 | 468.90 | 465.54 | 467.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 466.65 | 465.76 | 467.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:30:00 | 468.50 | 465.76 | 467.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 467.50 | 466.11 | 467.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 462.25 | 466.11 | 467.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 456.80 | 464.25 | 466.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:30:00 | 454.80 | 461.56 | 464.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 12:45:00 | 454.95 | 459.67 | 463.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 15:15:00 | 432.06 | 442.95 | 448.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 15:15:00 | 432.20 | 442.95 | 448.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 455.65 | 433.21 | 437.71 | SL hit (close>ema200) qty=0.50 sl=433.21 alert=retest2 |

### Cycle 63 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 461.35 | 443.13 | 441.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 464.00 | 454.33 | 448.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 498.10 | 519.07 | 507.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 498.10 | 519.07 | 507.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 498.10 | 519.07 | 507.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 498.10 | 519.07 | 507.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 502.30 | 515.72 | 506.81 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 494.60 | 503.13 | 503.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 15:15:00 | 491.00 | 496.00 | 499.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 498.50 | 488.64 | 492.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 498.50 | 488.64 | 492.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 498.50 | 488.64 | 492.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 509.35 | 488.64 | 492.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 499.00 | 490.72 | 493.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 501.00 | 490.72 | 493.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 499.20 | 494.43 | 494.32 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 09:15:00 | 491.05 | 493.98 | 494.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 15:15:00 | 487.50 | 490.43 | 492.11 | Break + close below crossover candle low |

### Cycle 67 — BUY (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 09:15:00 | 507.55 | 493.86 | 493.51 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 15:15:00 | 492.00 | 495.29 | 495.64 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 502.00 | 496.37 | 496.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 11:15:00 | 517.15 | 500.53 | 497.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 12:15:00 | 520.00 | 520.88 | 515.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 13:00:00 | 520.00 | 520.88 | 515.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 514.65 | 520.34 | 517.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 514.65 | 520.34 | 517.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 517.95 | 519.86 | 517.94 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 10:15:00 | 511.70 | 516.05 | 516.62 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 533.95 | 518.43 | 517.43 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 517.45 | 524.76 | 525.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 14:15:00 | 514.00 | 522.61 | 524.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 10:15:00 | 521.45 | 521.21 | 523.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 10:15:00 | 521.45 | 521.21 | 523.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 521.45 | 521.21 | 523.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 521.45 | 521.21 | 523.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 503.95 | 502.16 | 506.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:30:00 | 509.70 | 502.16 | 506.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 504.15 | 502.71 | 505.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:00:00 | 500.20 | 502.33 | 505.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:30:00 | 500.20 | 501.92 | 504.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 13:45:00 | 500.10 | 501.71 | 504.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 15:15:00 | 498.00 | 502.16 | 504.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 498.00 | 501.32 | 503.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:30:00 | 502.15 | 501.34 | 503.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 500.45 | 501.16 | 503.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:30:00 | 503.20 | 501.16 | 503.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 495.80 | 499.26 | 501.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:30:00 | 498.10 | 499.26 | 501.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 501.30 | 499.67 | 501.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 501.30 | 499.67 | 501.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 497.00 | 499.13 | 501.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 503.50 | 499.13 | 501.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 499.60 | 499.23 | 501.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:30:00 | 503.90 | 499.23 | 501.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 493.00 | 488.83 | 490.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:45:00 | 494.05 | 488.83 | 490.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 493.10 | 489.69 | 491.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 15:00:00 | 493.10 | 489.69 | 491.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 491.85 | 490.12 | 491.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 499.30 | 490.12 | 491.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 499.45 | 491.99 | 491.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 499.45 | 491.99 | 491.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 11:15:00 | 512.65 | 500.55 | 497.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 13:15:00 | 510.30 | 511.02 | 506.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:45:00 | 510.80 | 511.02 | 506.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 502.45 | 511.00 | 509.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 502.45 | 511.00 | 509.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 503.95 | 509.59 | 508.94 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 11:15:00 | 502.75 | 508.23 | 508.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 10:15:00 | 495.85 | 502.61 | 504.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 495.15 | 492.71 | 495.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 14:15:00 | 495.15 | 492.71 | 495.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 495.15 | 492.71 | 495.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:45:00 | 495.10 | 492.71 | 495.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 494.50 | 493.34 | 495.71 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 10:15:00 | 499.00 | 496.41 | 496.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 09:15:00 | 506.65 | 500.19 | 498.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 15:15:00 | 502.90 | 502.92 | 500.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 09:15:00 | 495.15 | 502.92 | 500.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 499.55 | 502.24 | 500.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:30:00 | 497.15 | 502.24 | 500.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 518.50 | 505.49 | 502.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 13:30:00 | 527.25 | 513.03 | 506.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 14:15:00 | 527.40 | 513.03 | 506.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 534.70 | 540.49 | 541.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 534.70 | 540.49 | 541.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 531.35 | 535.32 | 537.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 519.95 | 517.65 | 524.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:30:00 | 518.45 | 517.65 | 524.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 515.55 | 516.53 | 520.63 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 528.30 | 522.49 | 521.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 09:15:00 | 591.85 | 542.18 | 532.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 570.50 | 575.35 | 555.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 15:00:00 | 570.50 | 575.35 | 555.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 557.60 | 570.92 | 556.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 557.60 | 570.92 | 556.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 556.05 | 567.94 | 556.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:45:00 | 555.70 | 567.94 | 556.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 552.35 | 564.83 | 556.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:00:00 | 552.35 | 564.83 | 556.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 554.35 | 562.73 | 556.12 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 531.70 | 549.40 | 551.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 525.15 | 544.55 | 548.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 520.00 | 518.26 | 523.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 520.00 | 518.26 | 523.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 521.85 | 518.98 | 523.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 516.15 | 518.98 | 523.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 13:15:00 | 514.90 | 506.15 | 505.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 13:15:00 | 514.90 | 506.15 | 505.34 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 12:15:00 | 505.00 | 507.80 | 508.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 13:15:00 | 502.40 | 506.72 | 507.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 507.00 | 505.39 | 506.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 507.00 | 505.39 | 506.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 507.00 | 505.39 | 506.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 13:15:00 | 501.10 | 504.48 | 505.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 15:15:00 | 476.05 | 481.83 | 486.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 472.30 | 467.82 | 472.84 | SL hit (close>ema200) qty=0.50 sl=467.82 alert=retest2 |

### Cycle 81 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 478.45 | 474.33 | 474.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 481.95 | 476.70 | 475.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 478.75 | 478.81 | 476.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 15:00:00 | 478.75 | 478.81 | 476.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 479.00 | 478.85 | 477.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 484.40 | 478.85 | 477.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 15:15:00 | 517.45 | 519.22 | 519.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 15:15:00 | 517.45 | 519.22 | 519.36 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 09:15:00 | 521.30 | 519.63 | 519.54 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 518.20 | 519.35 | 519.42 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 11:15:00 | 520.20 | 519.52 | 519.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 13:15:00 | 521.60 | 520.09 | 519.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 14:15:00 | 519.15 | 519.90 | 519.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 14:15:00 | 519.15 | 519.90 | 519.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 519.15 | 519.90 | 519.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 15:00:00 | 519.15 | 519.90 | 519.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 519.90 | 519.90 | 519.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:15:00 | 517.50 | 519.90 | 519.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 522.30 | 520.38 | 519.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 09:45:00 | 527.70 | 521.85 | 520.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 10:30:00 | 527.30 | 522.88 | 521.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 11:45:00 | 526.65 | 523.30 | 521.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 12:15:00 | 526.45 | 523.30 | 521.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 521.65 | 523.23 | 521.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 521.65 | 523.23 | 521.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 521.40 | 522.87 | 521.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 521.40 | 522.87 | 521.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 521.80 | 522.65 | 521.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 515.60 | 522.65 | 521.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 514.65 | 521.05 | 521.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 514.65 | 521.05 | 521.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 506.70 | 515.20 | 518.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 514.65 | 512.63 | 516.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 514.65 | 512.63 | 516.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 514.65 | 512.63 | 516.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 517.55 | 512.63 | 516.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 510.85 | 512.27 | 515.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 508.50 | 511.38 | 514.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 14:45:00 | 510.00 | 510.65 | 513.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:45:00 | 509.85 | 507.87 | 509.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 513.50 | 510.59 | 510.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 513.50 | 510.59 | 510.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 15:15:00 | 514.00 | 511.27 | 510.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 15:15:00 | 518.00 | 518.68 | 516.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 09:15:00 | 521.15 | 518.68 | 516.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 514.95 | 517.95 | 516.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:30:00 | 514.10 | 517.95 | 516.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 517.80 | 517.92 | 516.62 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 510.80 | 515.43 | 515.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 15:15:00 | 510.00 | 514.35 | 515.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 515.00 | 514.48 | 515.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 515.00 | 514.48 | 515.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 515.00 | 514.48 | 515.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:30:00 | 515.55 | 514.48 | 515.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 518.20 | 515.22 | 515.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:30:00 | 522.55 | 515.22 | 515.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 11:15:00 | 521.20 | 516.42 | 515.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 12:15:00 | 534.40 | 520.01 | 517.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 14:15:00 | 515.00 | 519.42 | 517.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 14:15:00 | 515.00 | 519.42 | 517.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 515.00 | 519.42 | 517.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:30:00 | 517.85 | 519.42 | 517.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 517.00 | 518.93 | 517.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 519.45 | 518.93 | 517.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 513.80 | 520.96 | 521.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 13:15:00 | 513.80 | 520.96 | 521.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 508.60 | 516.95 | 519.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 13:15:00 | 517.30 | 514.08 | 516.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 13:15:00 | 517.30 | 514.08 | 516.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 517.30 | 514.08 | 516.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:00:00 | 517.30 | 514.08 | 516.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 519.45 | 515.15 | 516.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:45:00 | 519.70 | 515.15 | 516.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 520.85 | 516.29 | 517.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 529.60 | 516.29 | 517.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 517.90 | 517.49 | 517.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:00:00 | 517.90 | 517.49 | 517.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 518.80 | 517.75 | 517.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:30:00 | 519.50 | 517.75 | 517.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 14:15:00 | 520.20 | 518.24 | 518.04 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 15:15:00 | 516.50 | 517.89 | 517.90 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 518.90 | 518.03 | 517.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 12:15:00 | 520.60 | 518.66 | 518.26 | Break + close above crossover candle high |

### Cycle 94 — SELL (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 14:15:00 | 512.20 | 517.46 | 517.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 15:15:00 | 509.60 | 515.89 | 517.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 11:15:00 | 520.00 | 516.32 | 516.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 11:15:00 | 520.00 | 516.32 | 516.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 520.00 | 516.32 | 516.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:00:00 | 520.00 | 516.32 | 516.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 520.10 | 517.08 | 517.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:30:00 | 521.00 | 517.08 | 517.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 521.85 | 518.03 | 517.64 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 514.55 | 517.66 | 517.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 14:15:00 | 512.50 | 516.63 | 517.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 512.55 | 510.43 | 512.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 512.55 | 510.43 | 512.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 512.55 | 510.43 | 512.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 512.35 | 510.43 | 512.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 513.40 | 511.03 | 512.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 513.40 | 511.03 | 512.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 512.50 | 511.32 | 512.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 10:15:00 | 511.65 | 511.32 | 512.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 13:15:00 | 517.50 | 514.24 | 513.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 13:15:00 | 517.50 | 514.24 | 513.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 523.95 | 516.22 | 514.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 13:15:00 | 516.85 | 517.65 | 516.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 13:15:00 | 516.85 | 517.65 | 516.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 516.85 | 517.65 | 516.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 516.50 | 517.65 | 516.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 515.90 | 517.30 | 516.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:45:00 | 516.45 | 517.30 | 516.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 515.50 | 516.94 | 516.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 518.60 | 516.94 | 516.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 520.65 | 518.36 | 517.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:30:00 | 517.70 | 518.36 | 517.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 585.85 | 586.25 | 582.04 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 567.85 | 579.50 | 580.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 564.65 | 576.53 | 579.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 09:15:00 | 566.75 | 539.69 | 547.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 566.75 | 539.69 | 547.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 566.75 | 539.69 | 547.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 566.75 | 539.69 | 547.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 565.50 | 544.85 | 549.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 565.50 | 544.85 | 549.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 539.50 | 546.47 | 549.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 14:45:00 | 535.10 | 543.90 | 547.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-17 09:15:00 | 508.34 | 516.02 | 520.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-18 11:15:00 | 517.70 | 511.94 | 515.18 | SL hit (close>ema200) qty=0.50 sl=511.94 alert=retest2 |

### Cycle 99 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 519.80 | 516.27 | 516.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 525.80 | 520.00 | 518.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 11:15:00 | 519.20 | 520.08 | 518.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 11:15:00 | 519.20 | 520.08 | 518.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 519.20 | 520.08 | 518.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:00:00 | 519.20 | 520.08 | 518.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 517.50 | 519.57 | 518.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:00:00 | 517.50 | 519.57 | 518.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 514.70 | 518.59 | 518.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 14:00:00 | 514.70 | 518.59 | 518.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 514.30 | 517.06 | 517.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 509.20 | 515.49 | 516.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 523.65 | 503.33 | 505.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 523.65 | 503.33 | 505.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 523.65 | 503.33 | 505.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:30:00 | 528.85 | 503.33 | 505.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 519.00 | 506.46 | 506.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 523.00 | 506.46 | 506.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 11:15:00 | 516.05 | 508.38 | 507.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 13:15:00 | 522.50 | 512.80 | 509.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 536.40 | 537.59 | 528.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:00:00 | 536.40 | 537.59 | 528.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 529.90 | 535.09 | 531.34 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 526.80 | 529.71 | 529.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 524.60 | 528.69 | 529.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 15:15:00 | 521.90 | 521.58 | 524.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 09:15:00 | 523.50 | 521.58 | 524.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 524.40 | 522.14 | 524.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:30:00 | 525.25 | 522.14 | 524.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 524.00 | 522.51 | 524.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:30:00 | 524.70 | 522.51 | 524.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 524.25 | 522.86 | 524.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:30:00 | 524.95 | 522.86 | 524.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 524.00 | 523.09 | 524.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 14:15:00 | 523.30 | 523.24 | 524.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 531.95 | 525.18 | 524.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 531.95 | 525.18 | 524.77 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 15:15:00 | 523.70 | 524.80 | 524.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 09:15:00 | 522.80 | 524.40 | 524.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 09:15:00 | 522.00 | 521.53 | 522.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 522.00 | 521.53 | 522.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 522.00 | 521.53 | 522.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:30:00 | 519.80 | 520.82 | 522.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 15:15:00 | 520.00 | 520.28 | 521.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 10:15:00 | 530.15 | 522.07 | 521.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 10:15:00 | 530.15 | 522.07 | 521.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 11:15:00 | 535.10 | 524.68 | 523.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 12:15:00 | 542.40 | 546.53 | 540.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-11 13:00:00 | 542.40 | 546.53 | 540.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 536.35 | 544.49 | 540.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:00:00 | 536.35 | 544.49 | 540.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 533.25 | 542.24 | 539.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:45:00 | 533.20 | 542.24 | 539.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 533.80 | 538.32 | 538.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:45:00 | 533.20 | 538.32 | 538.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 11:15:00 | 533.55 | 537.36 | 537.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 13:15:00 | 532.15 | 535.78 | 536.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 09:15:00 | 539.10 | 536.16 | 536.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 539.10 | 536.16 | 536.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 539.10 | 536.16 | 536.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 12:00:00 | 530.80 | 534.45 | 535.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 09:30:00 | 531.05 | 533.07 | 534.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 535.75 | 528.93 | 528.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 535.75 | 528.93 | 528.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 11:15:00 | 539.25 | 533.44 | 531.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 15:15:00 | 549.00 | 549.33 | 543.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 09:15:00 | 551.00 | 549.33 | 543.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 602.75 | 589.64 | 574.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 09:15:00 | 617.65 | 596.41 | 585.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 15:15:00 | 616.05 | 627.54 | 627.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 15:15:00 | 616.05 | 627.54 | 627.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 612.55 | 624.54 | 626.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 579.20 | 559.86 | 568.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 579.20 | 559.86 | 568.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 579.20 | 559.86 | 568.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 579.20 | 559.86 | 568.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 580.00 | 563.89 | 569.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:00:00 | 572.55 | 565.62 | 570.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 10:45:00 | 573.70 | 569.81 | 570.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 11:15:00 | 581.85 | 572.22 | 571.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 11:15:00 | 581.85 | 572.22 | 571.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 09:15:00 | 582.80 | 576.00 | 573.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 577.50 | 581.19 | 578.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 577.50 | 581.19 | 578.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 577.50 | 581.19 | 578.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:30:00 | 576.20 | 581.19 | 578.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 575.05 | 579.97 | 577.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:45:00 | 572.80 | 579.97 | 577.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 570.80 | 578.13 | 577.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 570.80 | 578.13 | 577.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 566.50 | 575.81 | 576.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 562.25 | 573.09 | 575.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 539.90 | 536.82 | 541.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 539.90 | 536.82 | 541.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 539.90 | 536.82 | 541.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 14:45:00 | 538.40 | 536.82 | 541.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 549.00 | 539.26 | 542.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 548.50 | 539.26 | 542.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 546.30 | 540.67 | 542.82 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 561.80 | 544.89 | 544.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 570.55 | 555.54 | 550.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 543.55 | 561.21 | 557.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 543.55 | 561.21 | 557.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 543.55 | 561.21 | 557.76 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 536.45 | 553.29 | 554.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 533.15 | 549.27 | 552.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 09:15:00 | 547.60 | 545.99 | 549.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-01 10:00:00 | 547.60 | 545.99 | 549.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 541.85 | 545.50 | 548.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 531.15 | 545.50 | 548.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 09:30:00 | 533.40 | 525.45 | 531.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 12:30:00 | 538.55 | 531.91 | 532.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 13:15:00 | 543.50 | 534.23 | 533.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 543.50 | 534.23 | 533.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 548.65 | 538.86 | 536.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 14:15:00 | 541.70 | 541.73 | 538.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 15:00:00 | 541.70 | 541.73 | 538.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 534.75 | 540.22 | 538.76 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 535.15 | 537.76 | 538.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 525.70 | 535.35 | 536.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 533.65 | 533.56 | 535.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 14:15:00 | 533.65 | 533.56 | 535.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 533.65 | 533.56 | 535.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 533.65 | 533.56 | 535.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 553.45 | 537.45 | 536.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 561.80 | 551.12 | 545.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 539.15 | 552.99 | 549.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 539.15 | 552.99 | 549.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 539.15 | 552.99 | 549.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 539.15 | 552.99 | 549.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 539.60 | 550.31 | 548.31 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 533.00 | 544.32 | 545.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 15:15:00 | 526.50 | 537.22 | 541.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 09:15:00 | 536.50 | 534.25 | 537.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 536.50 | 534.25 | 537.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 536.50 | 534.25 | 537.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:15:00 | 524.80 | 532.20 | 535.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:30:00 | 525.10 | 529.77 | 531.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 547.70 | 532.47 | 532.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 547.70 | 532.47 | 532.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 558.20 | 540.61 | 536.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 548.85 | 549.00 | 543.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 548.85 | 549.00 | 543.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 543.80 | 547.99 | 544.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 545.60 | 547.99 | 544.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 553.90 | 549.17 | 545.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 569.35 | 553.93 | 551.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 563.45 | 575.34 | 576.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 563.45 | 575.34 | 576.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 553.80 | 568.90 | 572.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 09:15:00 | 544.50 | 543.99 | 551.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 09:30:00 | 544.50 | 543.99 | 551.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 532.10 | 521.25 | 524.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:45:00 | 535.50 | 521.25 | 524.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 533.10 | 523.62 | 525.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:00:00 | 533.10 | 523.62 | 525.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 525.40 | 526.07 | 526.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 09:15:00 | 516.05 | 526.07 | 526.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 12:15:00 | 527.50 | 524.95 | 525.39 | SL hit (close>static) qty=1.00 sl=526.50 alert=retest2 |

### Cycle 119 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 511.10 | 502.19 | 501.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 512.45 | 505.55 | 502.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 497.00 | 505.58 | 503.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 497.00 | 505.58 | 503.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 497.00 | 505.58 | 503.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 497.00 | 505.58 | 503.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 497.35 | 503.93 | 503.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:00:00 | 501.00 | 503.35 | 503.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 12:15:00 | 496.25 | 501.93 | 502.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 496.25 | 501.93 | 502.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 491.00 | 499.74 | 501.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 502.80 | 497.54 | 499.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 502.80 | 497.54 | 499.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 502.80 | 497.54 | 499.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 503.00 | 497.54 | 499.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 501.75 | 498.39 | 499.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 502.00 | 498.39 | 499.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 499.95 | 499.80 | 500.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:30:00 | 497.50 | 499.49 | 500.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 472.62 | 490.36 | 495.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 486.10 | 482.87 | 488.70 | SL hit (close>ema200) qty=0.50 sl=482.87 alert=retest2 |

### Cycle 121 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 499.80 | 490.15 | 489.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 506.25 | 497.21 | 493.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 10:15:00 | 542.85 | 547.76 | 530.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-30 11:00:00 | 542.85 | 547.76 | 530.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 538.25 | 544.57 | 535.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 571.10 | 544.57 | 535.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:45:00 | 545.55 | 553.08 | 549.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:30:00 | 545.00 | 552.09 | 548.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 12:15:00 | 544.30 | 550.06 | 550.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 12:15:00 | 544.30 | 550.06 | 550.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 15:15:00 | 543.30 | 547.07 | 548.55 | Break + close below crossover candle low |

### Cycle 123 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 560.10 | 549.68 | 549.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 566.75 | 553.09 | 551.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 12:15:00 | 561.95 | 565.57 | 560.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:00:00 | 561.95 | 565.57 | 560.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 559.45 | 564.34 | 560.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 559.60 | 564.34 | 560.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 561.00 | 563.68 | 560.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 559.15 | 563.68 | 560.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 559.00 | 562.74 | 560.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:30:00 | 564.30 | 563.12 | 560.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 553.05 | 559.48 | 559.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-04-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 09:15:00 | 553.05 | 559.48 | 559.91 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 10:15:00 | 564.95 | 560.58 | 560.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 11:15:00 | 566.30 | 561.72 | 560.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 627.50 | 641.00 | 627.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 13:15:00 | 627.50 | 641.00 | 627.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 627.50 | 641.00 | 627.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 13:45:00 | 625.10 | 641.00 | 627.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 631.55 | 639.11 | 627.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 648.55 | 636.79 | 627.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 13:15:00 | 648.20 | 656.35 | 657.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 648.20 | 656.35 | 657.17 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 676.00 | 659.80 | 658.48 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 12:15:00 | 656.05 | 661.14 | 661.39 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 664.10 | 661.47 | 661.26 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 658.75 | 660.98 | 661.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 588.55 | 646.50 | 654.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 10:15:00 | 605.70 | 603.01 | 621.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 11:00:00 | 605.70 | 603.01 | 621.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 617.45 | 606.95 | 618.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 14:00:00 | 617.45 | 606.95 | 618.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 614.75 | 608.51 | 618.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:30:00 | 609.75 | 610.69 | 617.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-17 11:15:00 | 477.19 | 2024-05-17 13:15:00 | 483.78 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-05-28 11:15:00 | 458.23 | 2024-06-03 09:15:00 | 456.20 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2024-05-28 11:45:00 | 458.21 | 2024-06-03 09:15:00 | 456.20 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2024-05-29 10:00:00 | 458.63 | 2024-06-03 09:15:00 | 456.20 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest2 | 2024-05-31 15:15:00 | 458.00 | 2024-06-03 09:15:00 | 456.20 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2024-06-13 13:00:00 | 443.20 | 2024-06-14 11:15:00 | 439.86 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-06-13 14:00:00 | 443.18 | 2024-06-14 11:15:00 | 439.86 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-06-14 09:15:00 | 444.40 | 2024-06-14 11:15:00 | 439.86 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest1 | 2024-06-21 09:15:00 | 466.47 | 2024-06-24 09:15:00 | 453.00 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest1 | 2024-06-21 12:15:00 | 472.13 | 2024-06-24 09:15:00 | 453.00 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2024-07-03 09:15:00 | 444.70 | 2024-07-08 14:15:00 | 441.88 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-07-03 10:30:00 | 445.25 | 2024-07-08 14:15:00 | 441.88 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-07-08 13:30:00 | 446.57 | 2024-07-08 14:15:00 | 441.88 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-07-23 12:15:00 | 404.20 | 2024-07-24 10:15:00 | 422.84 | STOP_HIT | 1.00 | -4.61% |
| BUY | retest2 | 2024-07-29 09:15:00 | 430.91 | 2024-08-02 11:15:00 | 436.18 | STOP_HIT | 1.00 | 1.22% |
| BUY | retest2 | 2024-07-30 09:15:00 | 429.03 | 2024-08-02 11:15:00 | 436.18 | STOP_HIT | 1.00 | 1.67% |
| SELL | retest2 | 2024-08-06 14:15:00 | 404.87 | 2024-08-12 09:15:00 | 418.10 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2024-08-07 14:15:00 | 409.80 | 2024-08-12 09:15:00 | 418.10 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-08-07 15:00:00 | 409.94 | 2024-08-12 09:15:00 | 418.10 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-08-08 15:00:00 | 409.60 | 2024-08-12 09:15:00 | 418.10 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-08-09 10:15:00 | 410.60 | 2024-08-12 09:15:00 | 418.10 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-08-09 11:00:00 | 411.00 | 2024-08-12 09:15:00 | 418.10 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-08-20 10:30:00 | 404.00 | 2024-08-21 13:15:00 | 406.32 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2024-08-20 14:45:00 | 404.10 | 2024-08-21 13:15:00 | 406.32 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-09-04 12:30:00 | 395.60 | 2024-09-06 10:15:00 | 398.57 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-09-05 12:00:00 | 395.78 | 2024-09-06 10:15:00 | 398.57 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-09-05 12:30:00 | 395.00 | 2024-09-06 10:15:00 | 398.57 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-09-05 13:30:00 | 395.77 | 2024-09-06 10:15:00 | 398.57 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-09-20 10:30:00 | 456.80 | 2024-09-25 09:15:00 | 502.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-20 12:30:00 | 454.73 | 2024-09-25 09:15:00 | 500.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-20 14:00:00 | 454.14 | 2024-09-25 09:15:00 | 499.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-20 15:00:00 | 455.00 | 2024-09-25 09:15:00 | 500.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-23 13:45:00 | 459.94 | 2024-09-25 09:15:00 | 503.80 | TARGET_HIT | 1.00 | 9.54% |
| BUY | retest2 | 2024-09-24 09:15:00 | 458.00 | 2024-09-25 09:15:00 | 502.70 | TARGET_HIT | 1.00 | 9.76% |
| BUY | retest2 | 2024-09-24 10:15:00 | 457.00 | 2024-09-25 09:15:00 | 503.14 | TARGET_HIT | 1.00 | 10.10% |
| BUY | retest2 | 2024-09-24 11:45:00 | 457.40 | 2024-09-25 10:15:00 | 505.93 | TARGET_HIT | 1.00 | 10.61% |
| BUY | retest2 | 2024-10-15 10:30:00 | 515.94 | 2024-10-18 14:15:00 | 495.65 | STOP_HIT | 1.00 | -3.93% |
| BUY | retest2 | 2024-10-15 15:00:00 | 502.44 | 2024-10-18 14:15:00 | 495.65 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-10-16 09:15:00 | 502.89 | 2024-10-18 14:15:00 | 495.65 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-10-17 09:15:00 | 502.44 | 2024-10-18 14:15:00 | 495.65 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-10-17 14:00:00 | 512.00 | 2024-10-18 14:15:00 | 495.65 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2024-10-28 15:15:00 | 423.00 | 2024-10-30 09:15:00 | 430.90 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-10-29 10:30:00 | 421.90 | 2024-10-30 09:15:00 | 430.90 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-10-29 11:30:00 | 422.05 | 2024-10-30 09:15:00 | 430.90 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-10-29 12:15:00 | 422.85 | 2024-10-30 09:15:00 | 430.90 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-10-31 14:45:00 | 433.15 | 2024-11-04 09:15:00 | 426.50 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-11-05 09:45:00 | 433.15 | 2024-11-08 12:15:00 | 439.85 | STOP_HIT | 1.00 | 1.55% |
| BUY | retest2 | 2024-11-05 12:30:00 | 432.50 | 2024-11-08 12:15:00 | 439.85 | STOP_HIT | 1.00 | 1.70% |
| SELL | retest2 | 2024-11-18 13:30:00 | 414.15 | 2024-11-18 14:15:00 | 418.40 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-11-22 14:30:00 | 407.90 | 2024-11-25 09:15:00 | 417.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2024-12-03 10:45:00 | 484.95 | 2024-12-04 09:15:00 | 533.45 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-15 13:30:00 | 428.40 | 2025-01-22 11:15:00 | 406.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-17 09:30:00 | 427.95 | 2025-01-22 11:15:00 | 407.36 | PARTIAL | 0.50 | 4.81% |
| SELL | retest2 | 2025-01-17 10:45:00 | 428.00 | 2025-01-22 12:15:00 | 406.55 | PARTIAL | 0.50 | 5.01% |
| SELL | retest2 | 2025-01-17 15:15:00 | 428.80 | 2025-01-22 12:15:00 | 406.60 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2025-01-15 13:30:00 | 428.40 | 2025-01-22 14:15:00 | 416.65 | STOP_HIT | 0.50 | 2.74% |
| SELL | retest2 | 2025-01-17 09:30:00 | 427.95 | 2025-01-22 14:15:00 | 416.65 | STOP_HIT | 0.50 | 2.64% |
| SELL | retest2 | 2025-01-17 10:45:00 | 428.00 | 2025-01-22 14:15:00 | 416.65 | STOP_HIT | 0.50 | 2.65% |
| SELL | retest2 | 2025-01-17 15:15:00 | 428.80 | 2025-01-22 14:15:00 | 416.65 | STOP_HIT | 0.50 | 2.83% |
| SELL | retest2 | 2025-01-21 10:15:00 | 422.45 | 2025-01-24 13:15:00 | 403.51 | PARTIAL | 0.50 | 4.48% |
| SELL | retest2 | 2025-01-21 15:15:00 | 424.75 | 2025-01-27 09:15:00 | 401.33 | PARTIAL | 0.50 | 5.51% |
| SELL | retest2 | 2025-01-21 10:15:00 | 422.45 | 2025-01-27 12:15:00 | 409.00 | STOP_HIT | 0.50 | 3.18% |
| SELL | retest2 | 2025-01-21 15:15:00 | 424.75 | 2025-01-27 12:15:00 | 409.00 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2025-02-11 13:45:00 | 350.00 | 2025-02-14 13:15:00 | 333.50 | PARTIAL | 0.50 | 4.72% |
| SELL | retest2 | 2025-02-11 13:45:00 | 350.00 | 2025-02-14 14:15:00 | 339.20 | STOP_HIT | 0.50 | 3.09% |
| SELL | retest2 | 2025-02-11 15:00:00 | 351.05 | 2025-02-17 09:15:00 | 332.50 | PARTIAL | 0.50 | 5.28% |
| SELL | retest2 | 2025-02-11 15:00:00 | 351.05 | 2025-02-17 12:15:00 | 338.50 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2025-02-12 09:15:00 | 340.95 | 2025-02-19 09:15:00 | 356.10 | STOP_HIT | 1.00 | -4.44% |
| SELL | retest2 | 2025-02-13 11:00:00 | 347.55 | 2025-02-19 09:15:00 | 356.10 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-02-24 11:15:00 | 359.65 | 2025-02-25 11:15:00 | 349.95 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest1 | 2025-02-28 09:15:00 | 340.60 | 2025-02-28 13:15:00 | 348.40 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest1 | 2025-02-28 10:45:00 | 345.40 | 2025-02-28 13:15:00 | 348.40 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest1 | 2025-02-28 12:30:00 | 345.40 | 2025-02-28 13:15:00 | 348.40 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-03-25 13:15:00 | 497.05 | 2025-04-04 11:15:00 | 476.45 | STOP_HIT | 1.00 | -4.14% |
| BUY | retest2 | 2025-03-26 09:45:00 | 502.75 | 2025-04-04 11:15:00 | 476.45 | STOP_HIT | 1.00 | -5.23% |
| BUY | retest2 | 2025-04-02 12:30:00 | 510.95 | 2025-04-04 11:15:00 | 476.45 | STOP_HIT | 1.00 | -6.75% |
| BUY | retest2 | 2025-04-03 10:00:00 | 498.25 | 2025-04-04 11:15:00 | 476.45 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2025-04-08 10:30:00 | 456.70 | 2025-04-15 10:15:00 | 466.25 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-04-08 15:15:00 | 456.50 | 2025-04-15 10:15:00 | 466.25 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-04-11 09:30:00 | 455.75 | 2025-04-15 10:15:00 | 466.25 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-04-22 09:15:00 | 485.70 | 2025-04-23 09:15:00 | 470.70 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2025-04-22 10:15:00 | 484.30 | 2025-04-23 09:15:00 | 470.70 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2025-04-22 14:00:00 | 484.65 | 2025-04-23 09:15:00 | 470.70 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-05-06 11:30:00 | 454.80 | 2025-05-08 15:15:00 | 432.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 12:45:00 | 454.95 | 2025-05-08 15:15:00 | 432.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 11:30:00 | 454.80 | 2025-05-12 09:15:00 | 455.65 | STOP_HIT | 0.50 | -0.19% |
| SELL | retest2 | 2025-05-06 12:45:00 | 454.95 | 2025-05-12 09:15:00 | 455.65 | STOP_HIT | 0.50 | -0.15% |
| SELL | retest2 | 2025-06-17 12:00:00 | 500.20 | 2025-06-24 09:15:00 | 499.45 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-06-17 12:30:00 | 500.20 | 2025-06-24 09:15:00 | 499.45 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-06-17 13:45:00 | 500.10 | 2025-06-24 09:15:00 | 499.45 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-06-17 15:15:00 | 498.00 | 2025-06-24 09:15:00 | 499.45 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-07-14 13:30:00 | 527.25 | 2025-07-23 09:15:00 | 534.70 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2025-07-14 14:15:00 | 527.40 | 2025-07-23 09:15:00 | 534.70 | STOP_HIT | 1.00 | 1.38% |
| SELL | retest2 | 2025-08-08 09:15:00 | 516.15 | 2025-08-18 13:15:00 | 514.90 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-08-21 13:15:00 | 501.10 | 2025-08-26 15:15:00 | 476.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 13:15:00 | 501.10 | 2025-09-01 09:15:00 | 472.30 | STOP_HIT | 0.50 | 5.75% |
| BUY | retest2 | 2025-09-03 09:15:00 | 484.40 | 2025-09-22 15:15:00 | 517.45 | STOP_HIT | 1.00 | 6.82% |
| BUY | retest2 | 2025-09-25 09:45:00 | 527.70 | 2025-09-26 09:15:00 | 514.65 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-09-25 10:30:00 | 527.30 | 2025-09-26 09:15:00 | 514.65 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-09-25 11:45:00 | 526.65 | 2025-09-26 09:15:00 | 514.65 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-09-25 12:15:00 | 526.45 | 2025-09-26 09:15:00 | 514.65 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-09-29 11:30:00 | 508.50 | 2025-10-01 14:15:00 | 513.50 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-09-29 14:45:00 | 510.00 | 2025-10-01 14:15:00 | 513.50 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-10-01 10:45:00 | 509.85 | 2025-10-01 14:15:00 | 513.50 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-10-09 09:15:00 | 519.45 | 2025-10-10 13:15:00 | 513.80 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-10-23 10:15:00 | 511.65 | 2025-10-23 13:15:00 | 517.50 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-11-11 14:45:00 | 535.10 | 2025-11-17 09:15:00 | 508.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-11 14:45:00 | 535.10 | 2025-11-18 11:15:00 | 517.70 | STOP_HIT | 0.50 | 3.25% |
| SELL | retest2 | 2025-12-03 14:15:00 | 523.30 | 2025-12-04 09:15:00 | 531.95 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-12-08 10:30:00 | 519.80 | 2025-12-09 10:15:00 | 530.15 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-12-08 15:15:00 | 520.00 | 2025-12-09 10:15:00 | 530.15 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-12-16 12:00:00 | 530.80 | 2025-12-22 09:15:00 | 535.75 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-12-17 09:30:00 | 531.05 | 2025-12-22 09:15:00 | 535.75 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-12-31 09:15:00 | 617.65 | 2026-01-07 15:15:00 | 616.05 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-01-14 12:00:00 | 572.55 | 2026-01-16 11:15:00 | 581.85 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-01-16 10:45:00 | 573.70 | 2026-01-16 11:15:00 | 581.85 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-02-01 12:15:00 | 531.15 | 2026-02-03 13:15:00 | 543.50 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2026-02-03 09:30:00 | 533.40 | 2026-02-03 13:15:00 | 543.50 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-02-03 12:30:00 | 538.55 | 2026-02-03 13:15:00 | 543.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-02-13 15:15:00 | 524.80 | 2026-02-18 09:15:00 | 547.70 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2026-02-17 14:30:00 | 525.10 | 2026-02-18 09:15:00 | 547.70 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2026-02-24 09:15:00 | 569.35 | 2026-03-02 11:15:00 | 563.45 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2026-03-12 09:15:00 | 516.05 | 2026-03-12 12:15:00 | 527.50 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2026-03-12 14:45:00 | 521.25 | 2026-03-13 15:15:00 | 495.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 14:45:00 | 521.25 | 2026-03-16 14:15:00 | 497.50 | STOP_HIT | 0.50 | 4.56% |
| BUY | retest2 | 2026-03-19 12:00:00 | 501.00 | 2026-03-19 12:15:00 | 496.25 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-03-20 14:30:00 | 497.50 | 2026-03-23 10:15:00 | 472.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 14:30:00 | 497.50 | 2026-03-24 09:15:00 | 486.10 | STOP_HIT | 0.50 | 2.29% |
| BUY | retest2 | 2026-04-01 09:15:00 | 571.10 | 2026-04-07 12:15:00 | 544.30 | STOP_HIT | 1.00 | -4.69% |
| BUY | retest2 | 2026-04-02 11:45:00 | 545.55 | 2026-04-07 12:15:00 | 544.30 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2026-04-02 12:30:00 | 545.00 | 2026-04-07 12:15:00 | 544.30 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2026-04-10 09:30:00 | 564.30 | 2026-04-13 09:15:00 | 553.05 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2026-04-21 09:15:00 | 648.55 | 2026-04-24 13:15:00 | 648.20 | STOP_HIT | 1.00 | -0.05% |
