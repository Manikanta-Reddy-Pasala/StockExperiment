# Cartrade Tech Ltd. (CARTRADE)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 1949.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 223 |
| ALERT1 | 149 |
| ALERT2 | 145 |
| ALERT2_SKIP | 83 |
| ALERT3 | 415 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 158 |
| PARTIAL | 19 |
| TARGET_HIT | 19 |
| STOP_HIT | 144 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 182 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 68 / 114
- **Target hits / Stop hits / Partials:** 19 / 144 / 19
- **Avg / median % per leg:** 0.32% / -1.33%
- **Sum % (uncompounded):** 58.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 91 | 24 | 26.4% | 15 | 76 | 0 | 0.03% | 2.7% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 5 | 0 | -2.40% | -12.0% |
| BUY @ 3rd Alert (retest2) | 86 | 23 | 26.7% | 15 | 71 | 0 | 0.17% | 14.7% |
| SELL (all) | 91 | 44 | 48.4% | 4 | 68 | 19 | 0.62% | 56.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 91 | 44 | 48.4% | 4 | 68 | 19 | 0.62% | 56.3% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 5 | 0 | -2.40% | -12.0% |
| retest2 (combined) | 177 | 67 | 37.9% | 19 | 139 | 19 | 0.40% | 71.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 11:15:00 | 420.00 | 414.47 | 414.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 13:15:00 | 423.00 | 420.28 | 418.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-22 09:15:00 | 417.90 | 420.17 | 418.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 09:15:00 | 417.90 | 420.17 | 418.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 417.90 | 420.17 | 418.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-22 10:00:00 | 417.90 | 420.17 | 418.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 10:15:00 | 417.65 | 419.67 | 418.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-22 10:45:00 | 417.45 | 419.67 | 418.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 13:15:00 | 416.50 | 418.36 | 418.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-22 14:00:00 | 416.50 | 418.36 | 418.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2023-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-22 14:15:00 | 416.05 | 417.90 | 418.05 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 10:15:00 | 422.55 | 418.44 | 418.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 14:15:00 | 423.50 | 420.00 | 419.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 14:15:00 | 430.50 | 431.13 | 427.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-25 15:00:00 | 430.50 | 431.13 | 427.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 13:15:00 | 428.40 | 431.65 | 429.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 13:30:00 | 426.60 | 431.65 | 429.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 14:15:00 | 428.30 | 430.98 | 429.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 15:00:00 | 428.30 | 430.98 | 429.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 15:15:00 | 428.75 | 430.54 | 429.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-29 09:15:00 | 432.50 | 430.54 | 429.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-29 09:15:00 | 426.50 | 429.73 | 429.11 | SL hit (close<static) qty=1.00 sl=427.75 alert=retest2 |

### Cycle 4 — SELL (started 2023-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 11:15:00 | 423.90 | 428.01 | 428.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-29 12:15:00 | 419.40 | 426.29 | 427.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 09:15:00 | 419.10 | 418.75 | 421.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 09:15:00 | 419.10 | 418.75 | 421.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 419.10 | 418.75 | 421.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 09:30:00 | 421.00 | 418.75 | 421.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 422.75 | 419.37 | 420.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 09:30:00 | 422.85 | 419.37 | 420.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 10:15:00 | 423.55 | 420.21 | 420.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-01 12:15:00 | 419.95 | 420.57 | 420.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-02 10:00:00 | 421.95 | 419.05 | 419.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-02 12:00:00 | 421.00 | 419.75 | 420.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-02 13:15:00 | 421.00 | 420.36 | 420.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2023-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 13:15:00 | 421.00 | 420.36 | 420.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 09:15:00 | 453.45 | 427.32 | 423.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-09 15:15:00 | 525.00 | 526.65 | 513.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-12 09:15:00 | 515.05 | 526.65 | 513.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 518.55 | 525.03 | 514.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-13 09:30:00 | 529.50 | 517.87 | 515.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-14 13:15:00 | 510.50 | 515.45 | 515.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-14 13:15:00 | 510.50 | 515.45 | 515.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-14 14:15:00 | 505.05 | 513.37 | 514.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 10:15:00 | 488.60 | 487.29 | 490.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-20 11:00:00 | 488.60 | 487.29 | 490.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 13:15:00 | 488.10 | 486.90 | 489.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 14:00:00 | 488.10 | 486.90 | 489.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 14:15:00 | 489.25 | 487.37 | 489.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 14:30:00 | 489.05 | 487.37 | 489.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 15:15:00 | 489.95 | 487.89 | 489.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:15:00 | 489.85 | 487.89 | 489.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 488.50 | 488.01 | 489.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 10:30:00 | 486.70 | 487.66 | 489.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 11:00:00 | 486.25 | 487.66 | 489.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 11:45:00 | 486.55 | 487.81 | 489.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 12:30:00 | 487.00 | 487.66 | 488.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 485.55 | 485.86 | 487.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 09:30:00 | 486.80 | 485.86 | 487.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 14:15:00 | 478.65 | 478.44 | 481.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 14:30:00 | 480.65 | 478.44 | 481.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 481.00 | 478.72 | 480.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 09:30:00 | 482.00 | 478.72 | 480.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 479.95 | 478.97 | 480.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 11:30:00 | 475.00 | 478.49 | 480.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-27 09:15:00 | 485.25 | 478.61 | 479.50 | SL hit (close>static) qty=1.00 sl=481.00 alert=retest2 |

### Cycle 7 — BUY (started 2023-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 11:15:00 | 483.00 | 480.65 | 480.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 09:15:00 | 487.70 | 482.25 | 481.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 11:15:00 | 483.35 | 483.61 | 482.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-28 11:45:00 | 483.80 | 483.61 | 482.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 13:15:00 | 481.60 | 483.05 | 482.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 13:30:00 | 481.50 | 483.05 | 482.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 480.30 | 482.50 | 481.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 15:00:00 | 480.30 | 482.50 | 481.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 15:15:00 | 478.65 | 481.73 | 481.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 09:15:00 | 483.95 | 481.73 | 481.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 10:00:00 | 482.15 | 481.81 | 481.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 11:15:00 | 485.55 | 481.64 | 481.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-07 12:15:00 | 495.35 | 501.34 | 501.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2023-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 12:15:00 | 495.35 | 501.34 | 501.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 13:15:00 | 494.70 | 500.01 | 501.08 | Break + close below crossover candle low |

### Cycle 9 — BUY (started 2023-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 09:15:00 | 555.20 | 502.48 | 499.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 12:15:00 | 573.30 | 532.40 | 515.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 11:15:00 | 541.10 | 545.21 | 530.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-12 11:45:00 | 542.35 | 545.21 | 530.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 09:15:00 | 542.10 | 540.60 | 533.67 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-07-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 14:15:00 | 516.95 | 531.43 | 531.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 12:15:00 | 514.15 | 521.85 | 526.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-17 09:15:00 | 522.45 | 519.21 | 523.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 09:15:00 | 522.45 | 519.21 | 523.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 522.45 | 519.21 | 523.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:45:00 | 523.15 | 519.21 | 523.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 12:15:00 | 520.70 | 518.87 | 522.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 13:00:00 | 520.70 | 518.87 | 522.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 13:15:00 | 521.00 | 519.30 | 521.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 14:00:00 | 521.00 | 519.30 | 521.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 14:15:00 | 519.40 | 519.32 | 521.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 14:30:00 | 519.30 | 519.32 | 521.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 515.15 | 518.37 | 520.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-18 11:30:00 | 510.05 | 515.88 | 519.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-19 11:15:00 | 513.40 | 511.31 | 514.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-19 14:30:00 | 513.20 | 512.47 | 514.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-20 11:15:00 | 530.00 | 518.03 | 516.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2023-07-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 11:15:00 | 530.00 | 518.03 | 516.50 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 11:15:00 | 509.50 | 515.55 | 516.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 14:15:00 | 509.00 | 512.54 | 514.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 09:15:00 | 514.60 | 512.58 | 514.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 09:15:00 | 514.60 | 512.58 | 514.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 514.60 | 512.58 | 514.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 09:45:00 | 514.25 | 512.58 | 514.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 10:15:00 | 512.15 | 512.49 | 514.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 10:30:00 | 514.35 | 512.49 | 514.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 503.85 | 502.91 | 504.64 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-07-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 13:15:00 | 510.40 | 505.28 | 505.27 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-31 15:15:00 | 501.30 | 506.39 | 506.97 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 10:15:00 | 511.75 | 508.15 | 507.71 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 501.75 | 507.32 | 507.98 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 09:15:00 | 514.40 | 507.03 | 506.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 14:15:00 | 527.00 | 517.57 | 513.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 15:15:00 | 529.05 | 529.39 | 523.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 10:15:00 | 521.20 | 527.77 | 523.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 10:15:00 | 521.20 | 527.77 | 523.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 10:30:00 | 522.10 | 527.77 | 523.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 11:15:00 | 524.45 | 527.11 | 523.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-09 12:15:00 | 530.05 | 527.11 | 523.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-11 15:15:00 | 526.30 | 532.21 | 532.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2023-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 15:15:00 | 526.30 | 532.21 | 532.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 11:15:00 | 521.35 | 527.71 | 530.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-22 09:15:00 | 511.70 | 495.50 | 498.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 09:15:00 | 511.70 | 495.50 | 498.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 09:15:00 | 511.70 | 495.50 | 498.78 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 12:15:00 | 506.60 | 501.12 | 500.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 13:15:00 | 509.60 | 502.82 | 501.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 12:15:00 | 507.00 | 507.48 | 504.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 14:15:00 | 506.50 | 507.37 | 505.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 506.50 | 507.37 | 505.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 14:45:00 | 503.95 | 507.37 | 505.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 15:15:00 | 507.40 | 507.37 | 505.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 09:15:00 | 510.20 | 507.37 | 505.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-01 10:15:00 | 561.22 | 548.34 | 537.90 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2023-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 10:15:00 | 577.05 | 593.38 | 595.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 567.55 | 584.85 | 591.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 13:15:00 | 567.90 | 567.31 | 576.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 13:45:00 | 569.30 | 567.31 | 576.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 583.75 | 570.92 | 575.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:45:00 | 586.40 | 570.92 | 575.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 10:15:00 | 587.05 | 574.14 | 576.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 10:45:00 | 587.25 | 574.14 | 576.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2023-09-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 12:15:00 | 586.05 | 578.29 | 578.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 09:15:00 | 590.00 | 583.38 | 580.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 12:15:00 | 591.55 | 591.63 | 588.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 13:15:00 | 589.75 | 591.26 | 588.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 13:15:00 | 589.75 | 591.26 | 588.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 13:45:00 | 589.25 | 591.26 | 588.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 14:15:00 | 578.20 | 588.65 | 587.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 15:00:00 | 578.20 | 588.65 | 587.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 15:15:00 | 579.00 | 586.72 | 586.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 09:15:00 | 577.05 | 586.72 | 586.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2023-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 09:15:00 | 567.15 | 582.80 | 584.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 10:15:00 | 561.00 | 578.44 | 582.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 09:15:00 | 547.70 | 546.09 | 551.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-26 09:45:00 | 547.75 | 546.09 | 551.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 549.25 | 546.42 | 548.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 09:45:00 | 552.95 | 546.42 | 548.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 550.40 | 547.21 | 548.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 10:45:00 | 551.10 | 547.21 | 548.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 11:15:00 | 559.05 | 549.58 | 549.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 12:00:00 | 559.05 | 549.58 | 549.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2023-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 12:15:00 | 552.60 | 550.18 | 550.14 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 545.05 | 550.31 | 550.62 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 11:15:00 | 552.70 | 551.04 | 550.83 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 15:15:00 | 549.00 | 550.73 | 550.78 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 09:15:00 | 553.45 | 551.28 | 551.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 10:15:00 | 567.25 | 554.47 | 552.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 13:15:00 | 610.00 | 612.73 | 604.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-09 14:00:00 | 610.00 | 612.73 | 604.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 628.90 | 616.18 | 608.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-11 11:00:00 | 637.75 | 628.42 | 619.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 09:45:00 | 636.20 | 631.86 | 625.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 15:15:00 | 670.15 | 675.61 | 676.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2023-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 15:15:00 | 670.15 | 675.61 | 676.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 659.80 | 672.45 | 674.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 09:15:00 | 672.00 | 661.31 | 666.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 09:15:00 | 672.00 | 661.31 | 666.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 672.00 | 661.31 | 666.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 09:45:00 | 674.35 | 661.31 | 666.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 10:15:00 | 666.85 | 662.42 | 666.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 10:30:00 | 671.70 | 662.42 | 666.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 633.60 | 629.86 | 641.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 09:30:00 | 638.05 | 629.86 | 641.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 14:15:00 | 638.00 | 633.16 | 638.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 15:00:00 | 638.00 | 633.16 | 638.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 15:15:00 | 650.05 | 636.54 | 639.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-26 09:15:00 | 635.15 | 636.54 | 639.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 09:15:00 | 630.35 | 635.30 | 639.10 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-10-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-26 14:15:00 | 646.25 | 640.26 | 640.24 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-10-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 15:15:00 | 640.00 | 640.21 | 640.22 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 09:15:00 | 656.10 | 643.39 | 641.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 15:15:00 | 671.65 | 656.89 | 649.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 14:15:00 | 671.80 | 681.07 | 673.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 14:15:00 | 671.80 | 681.07 | 673.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 14:15:00 | 671.80 | 681.07 | 673.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 15:00:00 | 671.80 | 681.07 | 673.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 15:15:00 | 673.95 | 679.64 | 673.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-01 09:15:00 | 676.00 | 679.64 | 673.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-01 12:00:00 | 676.80 | 677.40 | 673.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-01 14:15:00 | 663.00 | 674.57 | 673.32 | SL hit (close<static) qty=1.00 sl=667.40 alert=retest2 |

### Cycle 32 — SELL (started 2023-11-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 15:15:00 | 660.00 | 671.65 | 672.11 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 09:15:00 | 682.55 | 671.60 | 671.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 12:15:00 | 686.85 | 677.65 | 674.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 11:15:00 | 703.40 | 704.35 | 699.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-08 12:00:00 | 703.40 | 704.35 | 699.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 702.15 | 704.15 | 701.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 09:45:00 | 701.50 | 704.15 | 701.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 716.95 | 706.71 | 702.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 10:45:00 | 699.00 | 706.71 | 702.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 10:15:00 | 835.65 | 853.00 | 847.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 11:00:00 | 835.65 | 853.00 | 847.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 11:15:00 | 833.30 | 849.06 | 846.15 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 13:15:00 | 823.80 | 841.09 | 842.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 14:15:00 | 821.65 | 837.20 | 840.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 803.00 | 791.62 | 802.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 09:15:00 | 803.00 | 791.62 | 802.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 803.00 | 791.62 | 802.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 10:00:00 | 803.00 | 791.62 | 802.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 10:15:00 | 795.05 | 792.31 | 801.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 12:30:00 | 789.55 | 791.89 | 800.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-24 10:15:00 | 819.05 | 797.48 | 799.31 | SL hit (close>static) qty=1.00 sl=810.00 alert=retest2 |

### Cycle 35 — BUY (started 2023-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 11:15:00 | 849.05 | 807.79 | 803.83 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 15:15:00 | 803.40 | 811.45 | 812.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-29 09:15:00 | 793.20 | 807.80 | 810.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-01 09:15:00 | 772.50 | 770.96 | 782.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-01 09:30:00 | 772.05 | 770.96 | 782.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 10:15:00 | 784.80 | 773.73 | 782.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-01 11:00:00 | 784.80 | 773.73 | 782.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 11:15:00 | 781.00 | 775.18 | 782.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-01 14:45:00 | 776.20 | 777.94 | 781.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-04 13:45:00 | 778.60 | 780.25 | 781.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 09:15:00 | 737.39 | 752.23 | 762.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 09:15:00 | 739.67 | 752.23 | 762.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-12-07 09:15:00 | 740.20 | 733.88 | 746.24 | SL hit (close>ema200) qty=0.50 sl=733.88 alert=retest2 |

### Cycle 37 — BUY (started 2023-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 12:15:00 | 774.55 | 742.84 | 739.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 15:15:00 | 780.50 | 766.34 | 761.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-14 15:15:00 | 774.25 | 774.48 | 768.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-15 09:15:00 | 766.15 | 774.48 | 768.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 09:15:00 | 763.50 | 772.29 | 768.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 09:30:00 | 763.30 | 772.29 | 768.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 10:15:00 | 759.00 | 769.63 | 767.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 10:45:00 | 759.60 | 769.63 | 767.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2023-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 12:15:00 | 760.20 | 766.00 | 766.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-18 09:15:00 | 756.55 | 763.44 | 764.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-18 10:15:00 | 767.80 | 764.31 | 765.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 10:15:00 | 767.80 | 764.31 | 765.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 10:15:00 | 767.80 | 764.31 | 765.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-18 11:00:00 | 767.80 | 764.31 | 765.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 11:15:00 | 768.80 | 765.21 | 765.38 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 12:15:00 | 772.45 | 766.66 | 766.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-19 09:15:00 | 778.75 | 769.92 | 767.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 14:15:00 | 766.70 | 774.14 | 771.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 14:15:00 | 766.70 | 774.14 | 771.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 14:15:00 | 766.70 | 774.14 | 771.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 15:00:00 | 766.70 | 774.14 | 771.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 15:15:00 | 766.00 | 772.51 | 770.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 09:15:00 | 759.45 | 772.51 | 770.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2023-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 10:15:00 | 757.00 | 767.36 | 768.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 11:15:00 | 753.50 | 764.59 | 767.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 726.30 | 724.27 | 736.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-22 09:45:00 | 727.60 | 724.27 | 736.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 14:15:00 | 731.85 | 726.37 | 733.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 15:00:00 | 731.85 | 726.37 | 733.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 15:15:00 | 732.00 | 727.50 | 733.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 09:15:00 | 722.00 | 727.50 | 733.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 09:15:00 | 730.75 | 728.15 | 732.87 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 09:15:00 | 743.00 | 733.66 | 733.48 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 13:15:00 | 724.35 | 732.64 | 733.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-28 09:15:00 | 718.20 | 727.85 | 730.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 10:15:00 | 715.85 | 715.47 | 721.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 12:15:00 | 715.80 | 716.03 | 720.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 12:15:00 | 715.80 | 716.03 | 720.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-29 13:30:00 | 714.05 | 715.43 | 719.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-01 10:15:00 | 723.55 | 717.58 | 719.34 | SL hit (close>static) qty=1.00 sl=720.80 alert=retest2 |

### Cycle 43 — BUY (started 2024-01-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 11:15:00 | 723.70 | 716.08 | 715.65 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 09:15:00 | 706.30 | 715.03 | 716.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 10:15:00 | 699.00 | 705.66 | 708.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 12:15:00 | 704.80 | 704.44 | 707.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-10 13:00:00 | 704.80 | 704.44 | 707.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 14:15:00 | 712.20 | 705.55 | 707.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-10 15:00:00 | 712.20 | 705.55 | 707.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 15:15:00 | 709.80 | 706.40 | 707.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 09:15:00 | 710.10 | 706.40 | 707.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 711.20 | 707.36 | 708.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 12:15:00 | 707.05 | 708.37 | 708.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-17 09:15:00 | 701.15 | 696.62 | 696.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-17 09:15:00 | 701.15 | 696.62 | 696.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-18 10:15:00 | 721.65 | 706.27 | 701.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-19 14:15:00 | 725.65 | 727.44 | 719.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-19 14:45:00 | 726.40 | 727.44 | 719.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 15:15:00 | 728.00 | 727.55 | 720.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 09:30:00 | 719.50 | 725.62 | 720.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 10:15:00 | 718.35 | 724.17 | 720.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 11:00:00 | 718.35 | 724.17 | 720.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 12:15:00 | 717.65 | 721.88 | 719.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 12:30:00 | 719.20 | 721.88 | 719.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 13:15:00 | 726.90 | 722.88 | 720.33 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2024-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 11:15:00 | 709.65 | 718.45 | 718.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 14:15:00 | 700.95 | 712.86 | 716.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 13:15:00 | 716.00 | 705.05 | 709.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 13:15:00 | 716.00 | 705.05 | 709.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 716.00 | 705.05 | 709.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:00:00 | 716.00 | 705.05 | 709.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 728.00 | 709.64 | 711.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 15:00:00 | 728.00 | 709.64 | 711.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 15:15:00 | 728.00 | 713.31 | 712.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 09:15:00 | 743.25 | 719.30 | 715.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-29 14:15:00 | 739.60 | 742.75 | 735.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-29 14:15:00 | 739.60 | 742.75 | 735.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 14:15:00 | 739.60 | 742.75 | 735.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-29 15:00:00 | 739.60 | 742.75 | 735.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 15:15:00 | 738.00 | 741.80 | 735.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 09:15:00 | 737.90 | 741.80 | 735.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 09:15:00 | 735.35 | 740.51 | 735.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 10:00:00 | 735.35 | 740.51 | 735.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 10:15:00 | 735.60 | 739.53 | 735.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 10:45:00 | 734.25 | 739.53 | 735.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 11:15:00 | 731.25 | 737.87 | 735.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 12:00:00 | 731.25 | 737.87 | 735.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 12:15:00 | 731.65 | 736.63 | 734.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 13:00:00 | 731.65 | 736.63 | 734.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2024-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 15:15:00 | 728.00 | 733.23 | 733.50 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 09:15:00 | 736.30 | 733.84 | 733.75 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 11:15:00 | 733.35 | 733.63 | 733.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-31 14:15:00 | 728.15 | 732.16 | 732.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-02 09:15:00 | 726.80 | 723.43 | 726.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 09:15:00 | 726.80 | 723.43 | 726.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 726.80 | 723.43 | 726.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-02 12:30:00 | 719.65 | 722.60 | 725.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-05 09:30:00 | 720.55 | 717.37 | 721.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 12:15:00 | 683.67 | 705.97 | 708.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 12:15:00 | 684.52 | 705.97 | 708.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-09 13:15:00 | 691.50 | 690.00 | 697.15 | SL hit (close>ema200) qty=0.50 sl=690.00 alert=retest2 |

### Cycle 51 — BUY (started 2024-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 15:15:00 | 698.55 | 694.53 | 694.24 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-02-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 11:15:00 | 691.85 | 693.97 | 694.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-14 12:15:00 | 690.90 | 693.35 | 693.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-15 10:15:00 | 699.50 | 690.88 | 691.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-15 10:15:00 | 699.50 | 690.88 | 691.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 10:15:00 | 699.50 | 690.88 | 691.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 11:00:00 | 699.50 | 690.88 | 691.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 11:15:00 | 698.25 | 692.35 | 692.52 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-02-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 12:15:00 | 699.45 | 693.77 | 693.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 13:15:00 | 703.45 | 695.71 | 694.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 10:15:00 | 695.35 | 697.86 | 695.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 10:15:00 | 695.35 | 697.86 | 695.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 10:15:00 | 695.35 | 697.86 | 695.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 10:45:00 | 696.25 | 697.86 | 695.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 11:15:00 | 696.00 | 697.49 | 695.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 11:30:00 | 696.25 | 697.49 | 695.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 12:15:00 | 697.95 | 697.58 | 696.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 13:00:00 | 697.95 | 697.58 | 696.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 13:15:00 | 692.95 | 696.65 | 695.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 13:45:00 | 693.00 | 696.65 | 695.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 14:15:00 | 695.40 | 696.40 | 695.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 09:15:00 | 701.20 | 696.85 | 696.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 12:15:00 | 701.10 | 699.38 | 697.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 10:15:00 | 694.60 | 697.59 | 697.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 10:15:00 | 694.60 | 697.59 | 697.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 15:15:00 | 693.00 | 695.39 | 696.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 10:15:00 | 696.80 | 693.59 | 694.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 10:15:00 | 696.80 | 693.59 | 694.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 10:15:00 | 696.80 | 693.59 | 694.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 11:15:00 | 699.90 | 693.59 | 694.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 11:15:00 | 698.55 | 694.58 | 694.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 11:30:00 | 699.55 | 694.58 | 694.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2024-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 12:15:00 | 699.75 | 695.62 | 695.30 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 12:15:00 | 693.60 | 694.94 | 695.11 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 13:15:00 | 728.70 | 701.69 | 698.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-26 14:15:00 | 754.60 | 712.27 | 703.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 10:15:00 | 757.20 | 759.19 | 741.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-28 11:00:00 | 757.20 | 759.19 | 741.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 15:15:00 | 744.20 | 751.48 | 743.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-29 09:15:00 | 746.45 | 751.48 | 743.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 09:15:00 | 748.85 | 750.96 | 744.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-29 10:15:00 | 753.80 | 750.96 | 744.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-29 10:45:00 | 753.95 | 751.71 | 745.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-04 12:15:00 | 760.00 | 768.01 | 768.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 12:15:00 | 760.00 | 768.01 | 768.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 13:15:00 | 752.95 | 762.31 | 765.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 767.75 | 741.60 | 748.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 09:15:00 | 767.75 | 741.60 | 748.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 767.75 | 741.60 | 748.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 09:30:00 | 772.20 | 741.60 | 748.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 10:15:00 | 759.00 | 745.08 | 749.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 11:15:00 | 742.85 | 745.08 | 749.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 705.71 | 717.87 | 727.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-12 14:15:00 | 704.55 | 704.44 | 716.15 | SL hit (close>ema200) qty=0.50 sl=704.44 alert=retest2 |

### Cycle 59 — BUY (started 2024-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 14:15:00 | 651.65 | 643.52 | 642.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-27 15:15:00 | 651.95 | 645.21 | 643.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 10:15:00 | 640.10 | 644.93 | 643.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 10:15:00 | 640.10 | 644.93 | 643.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 10:15:00 | 640.10 | 644.93 | 643.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 11:00:00 | 640.10 | 644.93 | 643.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 11:15:00 | 638.70 | 643.68 | 643.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 11:30:00 | 635.20 | 643.68 | 643.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2024-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 12:15:00 | 638.20 | 642.59 | 642.91 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 09:15:00 | 671.05 | 647.09 | 644.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 10:15:00 | 676.10 | 652.89 | 647.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 11:15:00 | 725.80 | 726.70 | 717.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-05 12:00:00 | 725.80 | 726.70 | 717.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 14:15:00 | 716.15 | 722.97 | 717.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 15:00:00 | 716.15 | 722.97 | 717.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 15:15:00 | 715.10 | 721.40 | 717.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 09:15:00 | 718.95 | 721.40 | 717.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-08 09:15:00 | 712.10 | 719.54 | 717.21 | SL hit (close<static) qty=1.00 sl=713.10 alert=retest2 |

### Cycle 62 — SELL (started 2024-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 13:15:00 | 710.55 | 715.17 | 715.61 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 10:15:00 | 718.55 | 716.24 | 715.94 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-04-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 09:15:00 | 708.00 | 715.43 | 716.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 12:15:00 | 705.55 | 712.96 | 714.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 698.40 | 694.51 | 701.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 698.40 | 694.51 | 701.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 698.40 | 694.51 | 701.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:00:00 | 698.40 | 694.51 | 701.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 700.10 | 695.63 | 701.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 12:15:00 | 695.55 | 696.04 | 700.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-18 11:15:00 | 715.00 | 701.94 | 701.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2024-04-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 11:15:00 | 715.00 | 701.94 | 701.40 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-04-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 13:15:00 | 683.25 | 699.75 | 700.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 14:15:00 | 672.50 | 694.30 | 698.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-22 09:15:00 | 682.00 | 681.02 | 686.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-22 09:15:00 | 682.00 | 681.02 | 686.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 682.00 | 681.02 | 686.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 09:45:00 | 683.45 | 681.02 | 686.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 14:15:00 | 685.25 | 683.11 | 685.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 15:00:00 | 685.25 | 683.11 | 685.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 15:15:00 | 687.00 | 683.88 | 685.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 09:15:00 | 685.95 | 683.88 | 685.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 09:15:00 | 684.05 | 683.92 | 685.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 10:15:00 | 688.20 | 683.92 | 685.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 10:15:00 | 692.85 | 685.70 | 686.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 10:45:00 | 693.05 | 685.70 | 686.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2024-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 11:15:00 | 695.30 | 687.62 | 687.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 14:15:00 | 700.50 | 691.38 | 689.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 10:15:00 | 715.10 | 715.19 | 706.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 10:45:00 | 715.15 | 715.19 | 706.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 14:15:00 | 708.05 | 717.12 | 713.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 15:00:00 | 708.05 | 717.12 | 713.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 15:15:00 | 711.00 | 715.90 | 713.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 09:15:00 | 717.80 | 715.90 | 713.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 09:15:00 | 715.75 | 722.91 | 722.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-02 09:15:00 | 714.40 | 721.21 | 721.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 09:15:00 | 714.40 | 721.21 | 721.59 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 15:15:00 | 723.00 | 721.59 | 721.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-03 09:15:00 | 732.10 | 723.70 | 722.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-06 09:15:00 | 740.35 | 741.08 | 734.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-06 10:00:00 | 740.35 | 741.08 | 734.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 11:15:00 | 805.15 | 754.62 | 741.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 11:30:00 | 768.40 | 754.62 | 741.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 925.00 | 924.57 | 916.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-14 12:00:00 | 928.60 | 924.75 | 918.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-17 10:15:00 | 921.25 | 934.17 | 934.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 10:15:00 | 921.25 | 934.17 | 934.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-17 14:15:00 | 918.20 | 926.25 | 930.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-21 09:15:00 | 930.65 | 917.53 | 923.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 930.65 | 917.53 | 923.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 930.65 | 917.53 | 923.66 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 09:15:00 | 939.45 | 924.94 | 924.76 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 10:15:00 | 919.50 | 926.18 | 926.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 13:15:00 | 905.35 | 919.84 | 923.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 862.30 | 853.26 | 867.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 862.30 | 853.26 | 867.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 862.30 | 853.26 | 867.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:45:00 | 870.05 | 853.26 | 867.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 870.40 | 856.69 | 867.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:45:00 | 872.05 | 856.69 | 867.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 876.80 | 860.71 | 868.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 12:00:00 | 876.80 | 860.71 | 868.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 883.85 | 871.40 | 871.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:15:00 | 883.95 | 871.40 | 871.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 09:15:00 | 880.20 | 873.16 | 872.56 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 12:15:00 | 864.50 | 874.35 | 875.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-03 10:15:00 | 856.00 | 868.32 | 871.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 798.50 | 795.14 | 815.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 13:00:00 | 798.50 | 795.14 | 815.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 793.75 | 790.91 | 806.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 794.05 | 790.91 | 806.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 795.40 | 783.25 | 794.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 10:00:00 | 795.40 | 783.25 | 794.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 815.00 | 789.60 | 796.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 11:00:00 | 815.00 | 789.60 | 796.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 11:15:00 | 817.70 | 795.22 | 797.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 11:45:00 | 816.50 | 795.22 | 797.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2024-06-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 13:15:00 | 822.35 | 802.37 | 800.86 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 10:15:00 | 800.10 | 804.09 | 804.63 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 12:15:00 | 807.45 | 802.07 | 801.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-12 15:15:00 | 810.60 | 805.52 | 803.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 14:15:00 | 803.45 | 806.55 | 805.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-13 14:15:00 | 803.45 | 806.55 | 805.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 803.45 | 806.55 | 805.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 14:45:00 | 801.20 | 806.55 | 805.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 810.50 | 807.34 | 805.71 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-06-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 13:15:00 | 798.05 | 804.22 | 804.89 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-06-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 15:15:00 | 809.90 | 805.56 | 805.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 09:15:00 | 830.15 | 810.48 | 807.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 15:15:00 | 827.00 | 827.14 | 821.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-20 09:15:00 | 832.45 | 827.14 | 821.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 830.25 | 827.77 | 822.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 836.35 | 827.09 | 824.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 13:00:00 | 846.30 | 833.86 | 828.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 11:15:00 | 820.00 | 835.47 | 833.34 | SL hit (close<static) qty=1.00 sl=822.35 alert=retest2 |

### Cycle 80 — SELL (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 10:15:00 | 830.10 | 831.99 | 832.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 13:15:00 | 825.20 | 830.39 | 831.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 13:15:00 | 778.60 | 769.82 | 783.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 13:45:00 | 778.30 | 769.82 | 783.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 779.35 | 771.72 | 782.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 14:45:00 | 782.20 | 771.72 | 782.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 785.65 | 775.41 | 782.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:00:00 | 785.65 | 775.41 | 782.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 786.75 | 777.68 | 782.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 11:15:00 | 782.40 | 777.68 | 782.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 11:15:00 | 796.20 | 780.56 | 779.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 11:15:00 | 796.20 | 780.56 | 779.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 12:15:00 | 800.00 | 784.45 | 781.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 09:15:00 | 835.45 | 837.80 | 825.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 09:45:00 | 834.65 | 837.80 | 825.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 827.90 | 835.82 | 825.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 827.90 | 835.82 | 825.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 828.85 | 834.66 | 829.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:00:00 | 828.85 | 834.66 | 829.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 826.00 | 832.93 | 829.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:30:00 | 826.00 | 832.93 | 829.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 828.45 | 828.49 | 828.20 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2024-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 12:15:00 | 825.60 | 827.83 | 827.94 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 09:15:00 | 833.35 | 828.16 | 827.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 11:15:00 | 843.95 | 831.92 | 829.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 14:15:00 | 830.40 | 834.76 | 831.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 14:15:00 | 830.40 | 834.76 | 831.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 830.40 | 834.76 | 831.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 15:00:00 | 830.40 | 834.76 | 831.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 830.00 | 833.81 | 831.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 09:15:00 | 838.00 | 833.81 | 831.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 11:00:00 | 836.50 | 833.79 | 832.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 13:30:00 | 833.65 | 832.29 | 831.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 15:00:00 | 833.15 | 832.46 | 831.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 834.10 | 832.79 | 832.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 831.10 | 832.79 | 832.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 832.90 | 832.81 | 832.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:45:00 | 831.20 | 832.81 | 832.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 831.40 | 832.53 | 832.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:45:00 | 831.20 | 832.53 | 832.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 835.80 | 833.18 | 832.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 12:45:00 | 843.15 | 835.70 | 834.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 14:15:00 | 830.15 | 833.89 | 833.54 | SL hit (close<static) qty=1.00 sl=830.60 alert=retest2 |

### Cycle 84 — SELL (started 2024-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 15:15:00 | 830.00 | 833.11 | 833.22 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 09:15:00 | 836.40 | 833.77 | 833.51 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 830.00 | 833.01 | 833.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 10:15:00 | 829.60 | 830.31 | 831.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 11:15:00 | 830.65 | 830.38 | 831.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 11:15:00 | 830.65 | 830.38 | 831.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 830.65 | 830.38 | 831.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:00:00 | 830.65 | 830.38 | 831.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2024-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 12:15:00 | 836.60 | 831.62 | 831.55 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 13:15:00 | 830.50 | 831.40 | 831.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 14:15:00 | 830.00 | 831.12 | 831.33 | Break + close below crossover candle low |

### Cycle 89 — BUY (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 11:15:00 | 838.35 | 832.33 | 831.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 12:15:00 | 845.05 | 834.87 | 832.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 10:15:00 | 867.85 | 872.52 | 860.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 11:00:00 | 867.85 | 872.52 | 860.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 863.65 | 870.75 | 861.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:00:00 | 863.65 | 870.75 | 861.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 863.85 | 869.37 | 861.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:30:00 | 862.40 | 869.37 | 861.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 862.30 | 867.02 | 861.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 15:00:00 | 862.30 | 867.02 | 861.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 863.95 | 866.41 | 861.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 865.90 | 866.41 | 861.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 15:15:00 | 860.10 | 875.51 | 876.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 15:15:00 | 860.10 | 875.51 | 876.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 842.40 | 862.19 | 866.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 11:15:00 | 867.45 | 861.20 | 865.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 11:15:00 | 867.45 | 861.20 | 865.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 11:15:00 | 867.45 | 861.20 | 865.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 11:45:00 | 865.05 | 861.20 | 865.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 12:15:00 | 882.70 | 865.50 | 867.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 13:00:00 | 882.70 | 865.50 | 867.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 13:15:00 | 883.35 | 869.07 | 868.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-02 14:15:00 | 894.00 | 874.06 | 870.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 865.00 | 877.10 | 873.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 10:15:00 | 865.00 | 877.10 | 873.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 865.00 | 877.10 | 873.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 10:45:00 | 866.00 | 877.10 | 873.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 11:15:00 | 871.35 | 875.95 | 873.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 13:15:00 | 873.90 | 874.28 | 872.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 15:15:00 | 865.95 | 871.96 | 872.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 15:15:00 | 865.95 | 871.96 | 872.07 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 09:15:00 | 901.05 | 877.78 | 874.70 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 10:15:00 | 854.45 | 893.09 | 897.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 11:15:00 | 845.90 | 883.65 | 892.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 856.80 | 835.92 | 850.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 856.80 | 835.92 | 850.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 856.80 | 835.92 | 850.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:45:00 | 858.50 | 835.92 | 850.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 857.45 | 840.23 | 851.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:00:00 | 857.45 | 840.23 | 851.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 860.55 | 844.29 | 851.97 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 877.95 | 859.31 | 857.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 10:15:00 | 890.60 | 871.64 | 864.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 15:15:00 | 879.50 | 881.72 | 873.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 09:15:00 | 881.55 | 881.72 | 873.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 904.05 | 896.49 | 887.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 10:45:00 | 919.15 | 900.76 | 889.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 09:15:00 | 882.75 | 904.10 | 906.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 09:15:00 | 882.75 | 904.10 | 906.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 14:15:00 | 875.65 | 888.18 | 897.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 10:15:00 | 889.90 | 885.96 | 893.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-27 11:00:00 | 889.90 | 885.96 | 893.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 886.30 | 883.75 | 888.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 10:45:00 | 878.90 | 883.32 | 888.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 11:15:00 | 878.50 | 883.32 | 888.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 11:45:00 | 880.00 | 881.54 | 887.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 12:15:00 | 836.00 | 849.03 | 860.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-30 14:15:00 | 848.60 | 846.86 | 857.71 | SL hit (close>ema200) qty=0.50 sl=846.86 alert=retest2 |

### Cycle 97 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 10:15:00 | 884.00 | 852.40 | 852.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 15:15:00 | 890.00 | 876.38 | 865.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 921.35 | 927.69 | 910.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 10:00:00 | 921.35 | 927.69 | 910.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 919.00 | 924.63 | 913.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 12:45:00 | 918.30 | 924.63 | 913.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 13:15:00 | 916.10 | 922.92 | 913.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:45:00 | 915.90 | 922.92 | 913.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 915.25 | 921.39 | 913.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:30:00 | 916.55 | 921.39 | 913.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 917.00 | 920.51 | 913.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 888.00 | 920.51 | 913.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 882.15 | 912.84 | 910.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 10:00:00 | 882.15 | 912.84 | 910.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 10:15:00 | 881.50 | 906.57 | 908.30 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 12:15:00 | 919.00 | 904.06 | 902.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 10:15:00 | 923.00 | 914.68 | 909.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 917.00 | 917.33 | 911.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 14:15:00 | 915.00 | 916.86 | 912.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 915.00 | 916.86 | 912.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:30:00 | 914.95 | 916.86 | 912.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 915.00 | 916.49 | 912.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:15:00 | 923.90 | 916.49 | 912.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 956.05 | 924.40 | 916.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 11:15:00 | 986.00 | 952.52 | 937.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 12:00:00 | 982.00 | 958.41 | 941.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 13:15:00 | 983.00 | 979.79 | 964.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 15:15:00 | 982.45 | 978.65 | 967.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 966.00 | 976.73 | 968.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:45:00 | 968.05 | 976.73 | 968.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 973.30 | 976.04 | 968.67 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-18 11:15:00 | 963.05 | 967.12 | 967.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 11:15:00 | 963.05 | 967.12 | 967.17 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 13:15:00 | 969.80 | 967.48 | 967.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 14:15:00 | 981.95 | 970.38 | 968.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 09:15:00 | 957.40 | 968.20 | 967.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 957.40 | 968.20 | 967.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 957.40 | 968.20 | 967.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 957.40 | 968.20 | 967.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 10:15:00 | 956.05 | 965.77 | 966.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 11:15:00 | 941.65 | 960.95 | 964.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 973.15 | 954.33 | 958.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 973.15 | 954.33 | 958.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 973.15 | 954.33 | 958.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:00:00 | 973.15 | 954.33 | 958.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 979.50 | 959.36 | 960.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:30:00 | 981.00 | 959.36 | 960.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 11:15:00 | 991.45 | 965.78 | 963.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 12:15:00 | 1006.00 | 973.83 | 967.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 09:15:00 | 974.85 | 982.10 | 974.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 09:15:00 | 974.85 | 982.10 | 974.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 974.85 | 982.10 | 974.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:30:00 | 969.20 | 982.10 | 974.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 978.10 | 981.30 | 974.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 10:15:00 | 990.65 | 975.01 | 974.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 12:00:00 | 985.75 | 978.61 | 975.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 12:30:00 | 984.20 | 980.32 | 976.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 09:15:00 | 1001.85 | 979.90 | 977.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 996.10 | 1010.58 | 1002.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 996.10 | 1010.58 | 1002.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 995.00 | 1007.46 | 1002.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 986.90 | 1007.46 | 1002.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-27 10:15:00 | 965.00 | 994.58 | 996.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 10:15:00 | 965.00 | 994.58 | 996.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 12:15:00 | 963.00 | 972.70 | 982.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 977.75 | 970.48 | 977.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 977.75 | 970.48 | 977.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 977.75 | 970.48 | 977.68 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2024-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 13:15:00 | 990.85 | 981.58 | 981.31 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 967.90 | 978.96 | 980.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 958.05 | 974.78 | 978.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 872.25 | 868.70 | 895.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 872.25 | 868.70 | 895.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 894.15 | 877.05 | 887.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:45:00 | 896.85 | 877.05 | 887.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 894.75 | 880.59 | 888.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 11:00:00 | 894.75 | 880.59 | 888.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 923.10 | 897.31 | 894.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 11:15:00 | 926.95 | 915.21 | 907.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 11:15:00 | 986.70 | 987.68 | 972.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 11:45:00 | 990.10 | 987.68 | 972.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 1010.50 | 1025.38 | 1012.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:45:00 | 1007.10 | 1025.38 | 1012.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 1012.95 | 1022.90 | 1012.63 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2024-10-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 09:15:00 | 999.85 | 1009.36 | 1009.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 989.95 | 1003.82 | 1006.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 1002.85 | 993.98 | 999.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 09:15:00 | 1002.85 | 993.98 | 999.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 1002.85 | 993.98 | 999.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:45:00 | 1003.15 | 993.98 | 999.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 1006.10 | 996.41 | 1000.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:15:00 | 988.00 | 1000.49 | 1001.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 10:15:00 | 995.35 | 986.79 | 988.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 1007.70 | 990.97 | 990.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2024-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 10:15:00 | 1007.70 | 990.97 | 990.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 11:15:00 | 1036.00 | 999.97 | 994.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 09:15:00 | 1047.00 | 1048.05 | 1033.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 11:15:00 | 1037.15 | 1043.86 | 1034.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 11:15:00 | 1037.15 | 1043.86 | 1034.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:45:00 | 1024.95 | 1043.86 | 1034.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 1076.60 | 1052.91 | 1042.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 13:15:00 | 1084.00 | 1065.86 | 1051.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 14:30:00 | 1084.15 | 1070.63 | 1056.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 1099.85 | 1077.82 | 1062.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 10:00:00 | 1083.50 | 1078.98 | 1065.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-06 10:15:00 | 1192.40 | 1132.69 | 1108.36 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 12:15:00 | 1200.35 | 1214.25 | 1214.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 13:15:00 | 1196.60 | 1210.72 | 1212.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1206.40 | 1192.66 | 1198.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 1206.40 | 1192.66 | 1198.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 1206.40 | 1192.66 | 1198.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 09:30:00 | 1197.30 | 1192.66 | 1198.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 1182.10 | 1190.55 | 1196.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 11:30:00 | 1177.05 | 1188.91 | 1195.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:30:00 | 1163.70 | 1185.97 | 1191.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 10:15:00 | 1177.20 | 1185.97 | 1191.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 11:00:00 | 1177.85 | 1184.35 | 1190.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 1183.80 | 1184.34 | 1189.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:30:00 | 1184.85 | 1184.34 | 1189.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 1232.00 | 1191.00 | 1190.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 1232.00 | 1191.00 | 1190.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 10:15:00 | 1260.05 | 1204.81 | 1196.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 09:15:00 | 1205.55 | 1251.71 | 1229.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 09:15:00 | 1205.55 | 1251.71 | 1229.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 1205.55 | 1251.71 | 1229.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:00:00 | 1205.55 | 1251.71 | 1229.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 1206.25 | 1242.62 | 1227.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 11:15:00 | 1214.25 | 1242.62 | 1227.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-26 09:15:00 | 1335.68 | 1293.03 | 1278.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 14:15:00 | 1503.45 | 1511.46 | 1512.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 15:15:00 | 1500.15 | 1509.20 | 1511.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 1519.50 | 1511.26 | 1512.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 1519.50 | 1511.26 | 1512.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1519.50 | 1511.26 | 1512.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 1536.30 | 1511.26 | 1512.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 1501.00 | 1509.21 | 1511.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 14:15:00 | 1498.00 | 1508.05 | 1510.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 15:15:00 | 1498.00 | 1506.83 | 1509.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 11:15:00 | 1523.00 | 1505.03 | 1507.06 | SL hit (close>static) qty=1.00 sl=1518.95 alert=retest2 |

### Cycle 113 — BUY (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 12:15:00 | 1525.15 | 1509.05 | 1508.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 09:15:00 | 1564.95 | 1527.61 | 1518.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 12:15:00 | 1480.00 | 1528.47 | 1521.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 12:15:00 | 1480.00 | 1528.47 | 1521.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 1480.00 | 1528.47 | 1521.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 13:00:00 | 1480.00 | 1528.47 | 1521.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 13:15:00 | 1483.70 | 1519.52 | 1518.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 13:30:00 | 1469.35 | 1519.52 | 1518.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2024-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 14:15:00 | 1493.60 | 1514.33 | 1516.14 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 11:15:00 | 1538.35 | 1518.37 | 1516.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-20 09:15:00 | 1573.95 | 1537.60 | 1527.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-23 09:15:00 | 1532.40 | 1568.07 | 1552.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 09:15:00 | 1532.40 | 1568.07 | 1552.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 1532.40 | 1568.07 | 1552.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 10:00:00 | 1532.40 | 1568.07 | 1552.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 1530.65 | 1560.59 | 1550.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 10:30:00 | 1521.50 | 1560.59 | 1550.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 1553.60 | 1553.87 | 1550.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-24 12:00:00 | 1593.50 | 1561.34 | 1554.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 14:00:00 | 1581.40 | 1609.63 | 1591.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 10:15:00 | 1553.60 | 1577.30 | 1580.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 10:15:00 | 1553.60 | 1577.30 | 1580.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 12:15:00 | 1537.80 | 1554.21 | 1565.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 10:15:00 | 1507.80 | 1500.12 | 1519.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-01 10:45:00 | 1512.70 | 1500.12 | 1519.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 14:15:00 | 1507.75 | 1496.95 | 1511.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 14:45:00 | 1524.15 | 1496.95 | 1511.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 1511.00 | 1499.76 | 1511.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:15:00 | 1567.00 | 1499.76 | 1511.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 1561.65 | 1512.14 | 1515.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:00:00 | 1561.65 | 1512.14 | 1515.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 1589.85 | 1527.68 | 1522.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 12:15:00 | 1592.65 | 1550.92 | 1534.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 1607.60 | 1630.02 | 1599.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 10:00:00 | 1607.60 | 1630.02 | 1599.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 1572.50 | 1618.51 | 1597.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 1572.50 | 1618.51 | 1597.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 1591.35 | 1613.08 | 1596.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 1578.65 | 1613.08 | 1596.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 1564.00 | 1598.44 | 1592.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:45:00 | 1562.00 | 1598.44 | 1592.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 1563.50 | 1587.27 | 1588.15 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 09:15:00 | 1605.45 | 1590.90 | 1589.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 10:15:00 | 1625.95 | 1597.91 | 1593.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-07 11:15:00 | 1567.90 | 1591.91 | 1590.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 11:15:00 | 1567.90 | 1591.91 | 1590.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1567.90 | 1591.91 | 1590.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:30:00 | 1579.95 | 1591.91 | 1590.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 12:15:00 | 1563.50 | 1586.23 | 1588.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 14:15:00 | 1554.90 | 1576.77 | 1583.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 09:15:00 | 1606.75 | 1578.97 | 1583.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 09:15:00 | 1606.75 | 1578.97 | 1583.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1606.75 | 1578.97 | 1583.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:30:00 | 1554.95 | 1579.41 | 1582.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 13:45:00 | 1568.75 | 1577.15 | 1581.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 11:15:00 | 1605.60 | 1582.38 | 1580.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 11:15:00 | 1605.60 | 1582.38 | 1580.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-09 15:15:00 | 1632.00 | 1600.29 | 1590.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 1543.50 | 1588.93 | 1586.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 09:15:00 | 1543.50 | 1588.93 | 1586.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 1543.50 | 1588.93 | 1586.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 1543.50 | 1588.93 | 1586.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 1572.50 | 1585.65 | 1584.92 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 12:15:00 | 1580.30 | 1584.04 | 1584.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 14:15:00 | 1552.25 | 1574.64 | 1579.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 09:15:00 | 1431.50 | 1429.79 | 1464.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 12:15:00 | 1447.35 | 1434.15 | 1458.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 12:15:00 | 1447.35 | 1434.15 | 1458.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 13:00:00 | 1447.35 | 1434.15 | 1458.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 1455.55 | 1440.39 | 1453.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 09:15:00 | 1433.20 | 1455.94 | 1456.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 10:15:00 | 1445.15 | 1454.53 | 1456.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-17 14:15:00 | 1476.75 | 1449.98 | 1451.78 | SL hit (close>static) qty=1.00 sl=1467.90 alert=retest2 |

### Cycle 123 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 1473.35 | 1454.41 | 1453.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 12:15:00 | 1488.05 | 1463.80 | 1458.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 1446.55 | 1469.52 | 1463.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 1446.55 | 1469.52 | 1463.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 1446.55 | 1469.52 | 1463.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 1446.55 | 1469.52 | 1463.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1422.55 | 1460.13 | 1459.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 1422.55 | 1460.13 | 1459.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 11:15:00 | 1426.90 | 1453.48 | 1456.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 15:15:00 | 1412.00 | 1433.79 | 1445.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1369.60 | 1353.20 | 1386.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 1369.60 | 1353.20 | 1386.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1376.85 | 1357.93 | 1385.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 1386.95 | 1357.93 | 1385.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 1384.15 | 1366.88 | 1383.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:45:00 | 1388.75 | 1366.88 | 1383.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 1391.75 | 1371.86 | 1383.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 15:00:00 | 1391.75 | 1371.86 | 1383.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 1401.00 | 1377.69 | 1385.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 1400.00 | 1377.69 | 1385.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 1377.00 | 1373.88 | 1381.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 12:00:00 | 1377.00 | 1373.88 | 1381.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 12:15:00 | 1371.40 | 1373.38 | 1380.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 12:45:00 | 1371.00 | 1373.38 | 1380.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 11:15:00 | 1365.80 | 1355.66 | 1366.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 11:30:00 | 1380.85 | 1355.66 | 1366.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 12:15:00 | 1374.80 | 1359.49 | 1367.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 12:30:00 | 1369.25 | 1359.49 | 1367.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 13:15:00 | 1369.65 | 1361.52 | 1367.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 15:15:00 | 1360.00 | 1362.61 | 1367.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 1377.00 | 1358.83 | 1362.95 | SL hit (close>static) qty=1.00 sl=1374.80 alert=retest2 |

### Cycle 125 — BUY (started 2025-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 15:15:00 | 1375.00 | 1366.28 | 1365.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 09:15:00 | 1449.95 | 1383.01 | 1373.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 09:15:00 | 1610.55 | 1636.72 | 1596.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 09:15:00 | 1610.55 | 1636.72 | 1596.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 1610.55 | 1636.72 | 1596.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:45:00 | 1603.35 | 1636.72 | 1596.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 1590.90 | 1627.56 | 1595.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:00:00 | 1590.90 | 1627.56 | 1595.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1591.25 | 1620.30 | 1595.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 1577.00 | 1620.30 | 1595.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1716.35 | 1639.51 | 1606.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 13:30:00 | 1731.90 | 1661.06 | 1619.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:15:00 | 1770.90 | 1697.97 | 1649.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 10:30:00 | 1729.00 | 1735.81 | 1721.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 12:15:00 | 1725.95 | 1732.35 | 1721.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 12:15:00 | 1728.75 | 1731.63 | 1721.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 13:15:00 | 1719.45 | 1731.63 | 1721.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 13:15:00 | 1720.00 | 1729.30 | 1721.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 09:45:00 | 1741.50 | 1735.35 | 1726.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 12:15:00 | 1701.80 | 1728.90 | 1731.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-02-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 12:15:00 | 1701.80 | 1728.90 | 1731.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 1618.65 | 1693.44 | 1713.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 1579.60 | 1564.51 | 1600.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 10:45:00 | 1577.15 | 1564.51 | 1600.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 1612.25 | 1574.05 | 1601.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 1612.25 | 1574.05 | 1601.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1571.25 | 1573.49 | 1598.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 13:30:00 | 1567.65 | 1573.89 | 1596.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 09:15:00 | 1544.30 | 1575.89 | 1593.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 12:00:00 | 1557.15 | 1568.83 | 1585.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 15:15:00 | 1558.65 | 1536.06 | 1552.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 15:15:00 | 1558.65 | 1540.58 | 1553.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 09:15:00 | 1572.85 | 1540.58 | 1553.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 1554.85 | 1543.43 | 1553.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 10:45:00 | 1508.75 | 1534.74 | 1548.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 11:15:00 | 1489.27 | 1526.62 | 1543.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 1537.00 | 1520.46 | 1535.69 | SL hit (close>ema200) qty=0.50 sl=1520.46 alert=retest2 |

### Cycle 127 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 1593.25 | 1530.12 | 1529.35 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 09:15:00 | 1481.35 | 1527.77 | 1532.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 14:15:00 | 1474.30 | 1498.65 | 1514.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 11:15:00 | 1489.80 | 1484.39 | 1501.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-24 12:00:00 | 1489.80 | 1484.39 | 1501.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 1498.30 | 1487.18 | 1500.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:00:00 | 1498.30 | 1487.18 | 1500.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 1499.15 | 1489.57 | 1500.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:30:00 | 1486.85 | 1489.57 | 1500.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 1530.00 | 1497.66 | 1503.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 15:00:00 | 1530.00 | 1497.66 | 1503.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 1528.30 | 1503.78 | 1505.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:15:00 | 1525.00 | 1503.78 | 1505.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 1530.30 | 1509.09 | 1507.95 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 11:15:00 | 1488.40 | 1503.62 | 1505.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 1466.65 | 1493.60 | 1499.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 12:15:00 | 1468.00 | 1447.69 | 1464.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 12:15:00 | 1468.00 | 1447.69 | 1464.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 12:15:00 | 1468.00 | 1447.69 | 1464.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 12:45:00 | 1447.75 | 1447.69 | 1464.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 13:15:00 | 1472.40 | 1452.63 | 1465.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 13:45:00 | 1481.95 | 1452.63 | 1465.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 1548.80 | 1471.87 | 1472.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 1548.80 | 1471.87 | 1472.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-02-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-28 15:15:00 | 1548.30 | 1487.15 | 1479.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 09:15:00 | 1555.00 | 1518.98 | 1503.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 12:15:00 | 1503.65 | 1524.79 | 1510.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 12:15:00 | 1503.65 | 1524.79 | 1510.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 12:15:00 | 1503.65 | 1524.79 | 1510.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 13:00:00 | 1503.65 | 1524.79 | 1510.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 1550.00 | 1529.83 | 1514.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 13:00:00 | 1562.15 | 1546.24 | 1530.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 09:15:00 | 1577.20 | 1550.45 | 1536.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 10:45:00 | 1560.45 | 1553.49 | 1540.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 13:00:00 | 1563.90 | 1555.02 | 1543.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 13:15:00 | 1548.30 | 1553.67 | 1543.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 14:00:00 | 1548.30 | 1553.67 | 1543.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 1548.50 | 1552.64 | 1544.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 15:00:00 | 1548.50 | 1552.64 | 1544.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 1554.00 | 1552.91 | 1544.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 09:15:00 | 1559.85 | 1552.91 | 1544.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:00:00 | 1558.55 | 1554.04 | 1546.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 11:30:00 | 1555.50 | 1553.73 | 1547.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 12:30:00 | 1557.15 | 1554.04 | 1548.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 1548.90 | 1553.01 | 1548.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:45:00 | 1546.45 | 1553.01 | 1548.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 1549.10 | 1552.23 | 1548.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:00:00 | 1549.10 | 1552.23 | 1548.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 1553.55 | 1552.49 | 1548.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:15:00 | 1558.20 | 1552.49 | 1548.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 13:30:00 | 1555.70 | 1556.36 | 1552.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 14:30:00 | 1564.80 | 1557.24 | 1553.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 1512.50 | 1549.52 | 1550.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 1512.50 | 1549.52 | 1550.61 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 09:15:00 | 1555.45 | 1550.26 | 1549.89 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 13:15:00 | 1537.75 | 1547.48 | 1548.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 14:15:00 | 1535.90 | 1545.17 | 1547.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 1548.20 | 1543.35 | 1546.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 1548.20 | 1543.35 | 1546.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 1548.20 | 1543.35 | 1546.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:15:00 | 1549.10 | 1543.35 | 1546.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 1545.55 | 1543.79 | 1546.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:45:00 | 1544.75 | 1543.79 | 1546.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 1554.45 | 1545.92 | 1546.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 12:15:00 | 1559.75 | 1545.92 | 1546.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 1553.00 | 1547.34 | 1547.44 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2025-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 13:15:00 | 1555.70 | 1549.01 | 1548.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 14:15:00 | 1560.00 | 1551.21 | 1549.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 14:15:00 | 1705.80 | 1707.92 | 1670.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 15:00:00 | 1705.80 | 1707.92 | 1670.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 1686.60 | 1702.71 | 1674.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 09:45:00 | 1680.95 | 1702.71 | 1674.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 1663.20 | 1694.81 | 1673.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 11:00:00 | 1663.20 | 1694.81 | 1673.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 11:15:00 | 1704.85 | 1696.81 | 1676.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 12:30:00 | 1709.00 | 1699.22 | 1679.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 13:30:00 | 1707.35 | 1701.20 | 1682.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 14:45:00 | 1707.05 | 1702.49 | 1684.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:15:00 | 1724.65 | 1702.99 | 1686.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 1757.10 | 1713.81 | 1692.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 11:15:00 | 1767.85 | 1723.32 | 1698.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 12:45:00 | 1786.00 | 1737.99 | 1710.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 13:30:00 | 1776.95 | 1744.65 | 1715.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-21 14:15:00 | 1879.90 | 1772.44 | 1730.89 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 1690.10 | 1750.29 | 1751.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 15:15:00 | 1680.00 | 1712.70 | 1731.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 1697.70 | 1680.58 | 1699.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 1697.70 | 1680.58 | 1699.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 1697.70 | 1680.58 | 1699.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:00:00 | 1697.70 | 1680.58 | 1699.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 1677.95 | 1680.05 | 1697.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 14:15:00 | 1671.85 | 1679.18 | 1694.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 15:15:00 | 1631.05 | 1679.14 | 1692.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:45:00 | 1666.00 | 1670.18 | 1681.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-01 11:15:00 | 1588.26 | 1634.32 | 1658.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-02 09:15:00 | 1649.50 | 1620.67 | 1640.68 | SL hit (close>ema200) qty=0.50 sl=1620.67 alert=retest2 |

### Cycle 137 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 1681.65 | 1654.51 | 1652.14 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1494.30 | 1641.57 | 1655.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 12:15:00 | 1446.20 | 1553.26 | 1607.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 12:15:00 | 1524.70 | 1494.85 | 1542.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 12:45:00 | 1529.80 | 1494.85 | 1542.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 1526.70 | 1507.46 | 1540.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 14:30:00 | 1540.20 | 1507.46 | 1540.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 1416.00 | 1491.65 | 1527.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 11:00:00 | 1409.80 | 1475.28 | 1516.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 14:15:00 | 1537.65 | 1485.68 | 1485.87 | SL hit (close>static) qty=1.00 sl=1530.65 alert=retest2 |

### Cycle 139 — BUY (started 2025-04-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 15:15:00 | 1532.00 | 1494.94 | 1490.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1567.40 | 1509.43 | 1497.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 13:15:00 | 1545.80 | 1546.77 | 1531.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 13:30:00 | 1540.10 | 1546.77 | 1531.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 14:15:00 | 1525.40 | 1542.49 | 1531.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 15:00:00 | 1525.40 | 1542.49 | 1531.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 15:15:00 | 1528.10 | 1539.61 | 1530.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:15:00 | 1527.30 | 1539.61 | 1530.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1506.10 | 1532.91 | 1528.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:00:00 | 1506.10 | 1532.91 | 1528.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 1524.70 | 1531.27 | 1528.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 1528.60 | 1531.27 | 1528.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-22 09:15:00 | 1681.46 | 1613.06 | 1580.86 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 1705.00 | 1713.64 | 1713.92 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 1764.00 | 1723.13 | 1718.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 10:15:00 | 1785.70 | 1735.64 | 1724.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 12:15:00 | 1735.00 | 1735.75 | 1726.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 12:30:00 | 1731.10 | 1735.75 | 1726.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1798.40 | 1760.59 | 1742.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 12:00:00 | 1833.00 | 1778.99 | 1753.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 14:15:00 | 1709.50 | 1757.84 | 1762.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 1709.50 | 1757.84 | 1762.46 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 10:15:00 | 1818.30 | 1769.51 | 1766.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 11:15:00 | 1822.10 | 1780.03 | 1771.46 | Break + close above crossover candle high |

### Cycle 144 — SELL (started 2025-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 12:15:00 | 1700.50 | 1764.12 | 1765.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 13:15:00 | 1693.00 | 1749.90 | 1758.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 14:15:00 | 1592.70 | 1568.53 | 1616.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 15:00:00 | 1592.70 | 1568.53 | 1616.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1634.40 | 1586.88 | 1616.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 11:30:00 | 1622.50 | 1599.07 | 1617.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-13 09:15:00 | 1661.80 | 1627.41 | 1625.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 09:15:00 | 1661.80 | 1627.41 | 1625.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 11:15:00 | 1688.20 | 1653.55 | 1646.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 10:15:00 | 1673.50 | 1677.32 | 1663.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 11:00:00 | 1673.50 | 1677.32 | 1663.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 13:15:00 | 1667.30 | 1675.22 | 1666.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 13:45:00 | 1671.50 | 1675.22 | 1666.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 1661.30 | 1672.43 | 1665.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 14:30:00 | 1662.00 | 1672.43 | 1665.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 1662.90 | 1670.53 | 1665.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:15:00 | 1627.00 | 1670.53 | 1665.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 1621.80 | 1660.78 | 1661.54 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 1677.30 | 1647.84 | 1646.09 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 11:15:00 | 1639.00 | 1652.15 | 1652.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 15:15:00 | 1632.10 | 1642.41 | 1647.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 1656.20 | 1645.17 | 1647.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 1656.20 | 1645.17 | 1647.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1656.20 | 1645.17 | 1647.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 10:45:00 | 1641.10 | 1644.53 | 1647.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:15:00 | 1641.90 | 1644.53 | 1647.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 12:15:00 | 1559.04 | 1594.21 | 1617.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 12:15:00 | 1559.81 | 1594.21 | 1617.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-29 09:15:00 | 1571.60 | 1548.38 | 1566.81 | SL hit (close>ema200) qty=0.50 sl=1548.38 alert=retest2 |

### Cycle 149 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 1570.90 | 1533.71 | 1529.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 12:15:00 | 1574.80 | 1547.98 | 1536.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 15:15:00 | 1547.80 | 1549.99 | 1540.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 09:15:00 | 1541.80 | 1549.99 | 1540.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1551.20 | 1550.23 | 1541.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:45:00 | 1539.10 | 1550.23 | 1541.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1627.40 | 1653.70 | 1642.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1627.40 | 1653.70 | 1642.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 1627.20 | 1648.40 | 1640.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:15:00 | 1624.00 | 1648.40 | 1640.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 1624.00 | 1643.52 | 1639.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 1633.50 | 1643.52 | 1639.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 1637.40 | 1642.21 | 1639.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 11:00:00 | 1637.40 | 1642.21 | 1639.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 1626.40 | 1639.05 | 1638.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 12:00:00 | 1626.40 | 1639.05 | 1638.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2025-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 12:15:00 | 1626.50 | 1636.54 | 1637.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 1611.80 | 1631.67 | 1634.77 | Break + close below crossover candle low |

### Cycle 151 — BUY (started 2025-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 10:15:00 | 1674.60 | 1640.26 | 1638.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 15:15:00 | 1685.00 | 1664.13 | 1652.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 13:15:00 | 1689.70 | 1692.38 | 1673.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 13:45:00 | 1688.70 | 1692.38 | 1673.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 1673.10 | 1691.31 | 1681.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 1671.90 | 1691.31 | 1681.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 1660.00 | 1685.05 | 1679.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:00:00 | 1660.00 | 1685.05 | 1679.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 1684.00 | 1684.91 | 1680.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 1675.30 | 1684.91 | 1680.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1686.40 | 1685.21 | 1681.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:30:00 | 1685.30 | 1685.21 | 1681.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1682.40 | 1684.65 | 1681.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 1684.70 | 1684.65 | 1681.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 1675.70 | 1682.86 | 1680.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 1675.70 | 1682.86 | 1680.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 1655.80 | 1677.45 | 1678.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 1650.10 | 1665.87 | 1672.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 1667.60 | 1664.04 | 1669.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 12:00:00 | 1667.60 | 1664.04 | 1669.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1638.50 | 1636.44 | 1646.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 11:15:00 | 1627.20 | 1636.17 | 1645.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 12:30:00 | 1630.60 | 1632.16 | 1641.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 09:30:00 | 1631.90 | 1622.18 | 1633.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 11:00:00 | 1628.90 | 1623.53 | 1632.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 1631.00 | 1625.02 | 1632.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:00:00 | 1631.00 | 1625.02 | 1632.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 1616.00 | 1623.22 | 1631.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 13:15:00 | 1611.10 | 1623.22 | 1631.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 09:30:00 | 1605.50 | 1616.60 | 1625.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 11:15:00 | 1614.70 | 1616.96 | 1624.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 09:15:00 | 1670.40 | 1614.03 | 1611.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 09:15:00 | 1670.40 | 1614.03 | 1611.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 10:15:00 | 1686.80 | 1628.58 | 1618.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 14:15:00 | 1693.90 | 1701.68 | 1678.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 15:00:00 | 1693.90 | 1701.68 | 1678.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1690.60 | 1698.24 | 1680.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 1690.60 | 1698.24 | 1680.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1682.50 | 1697.41 | 1689.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 1682.50 | 1697.41 | 1689.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1671.00 | 1692.13 | 1687.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:45:00 | 1664.50 | 1692.13 | 1687.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 1681.60 | 1684.60 | 1684.67 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 14:15:00 | 1688.70 | 1684.94 | 1684.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 1707.90 | 1689.86 | 1687.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 13:15:00 | 1698.90 | 1701.47 | 1694.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 14:00:00 | 1698.90 | 1701.47 | 1694.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 1698.60 | 1700.90 | 1694.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 1733.40 | 1696.74 | 1695.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-11 10:15:00 | 1906.74 | 1839.19 | 1804.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 1907.90 | 1913.92 | 1914.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 1893.70 | 1909.88 | 1912.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 11:15:00 | 1901.90 | 1901.34 | 1905.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 11:15:00 | 1901.90 | 1901.34 | 1905.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1901.90 | 1901.34 | 1905.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:30:00 | 1903.30 | 1901.34 | 1905.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1910.00 | 1903.07 | 1906.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:30:00 | 1910.00 | 1903.07 | 1906.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 1914.00 | 1905.26 | 1906.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:45:00 | 1912.70 | 1905.26 | 1906.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 14:15:00 | 1923.00 | 1908.80 | 1908.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 09:15:00 | 1957.60 | 1919.91 | 1913.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 12:15:00 | 1928.20 | 1929.25 | 1920.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 13:00:00 | 1928.20 | 1929.25 | 1920.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 1917.50 | 1927.71 | 1921.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 1917.50 | 1927.71 | 1921.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 1908.20 | 1923.81 | 1919.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 1898.00 | 1923.81 | 1919.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1919.70 | 1922.99 | 1919.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 11:45:00 | 1925.00 | 1924.94 | 1921.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 12:30:00 | 1926.40 | 1928.69 | 1923.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 1897.00 | 1933.99 | 1934.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 1897.00 | 1933.99 | 1934.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 11:15:00 | 1895.00 | 1926.19 | 1930.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 10:15:00 | 1917.80 | 1911.25 | 1919.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 10:15:00 | 1917.80 | 1911.25 | 1919.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 1917.80 | 1911.25 | 1919.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:30:00 | 1890.90 | 1911.25 | 1919.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — BUY (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 11:15:00 | 2109.10 | 1950.82 | 1937.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 09:15:00 | 2167.00 | 2093.79 | 2062.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 2149.00 | 2161.20 | 2120.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 09:45:00 | 2144.30 | 2161.20 | 2120.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 2182.40 | 2230.64 | 2214.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 2182.40 | 2230.64 | 2214.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 2170.00 | 2218.51 | 2210.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:30:00 | 2167.10 | 2218.51 | 2210.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2025-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 13:15:00 | 2160.80 | 2199.59 | 2202.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 2122.00 | 2184.07 | 2195.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 2110.80 | 2103.60 | 2139.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 2110.80 | 2103.60 | 2139.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 2170.30 | 2118.61 | 2139.98 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 13:15:00 | 2178.00 | 2151.95 | 2151.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 12:15:00 | 2193.30 | 2164.85 | 2158.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 2278.00 | 2285.56 | 2242.26 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 12:30:00 | 2344.20 | 2301.09 | 2260.02 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 10:30:00 | 2332.80 | 2322.94 | 2288.74 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 14:30:00 | 2340.00 | 2319.38 | 2297.94 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 2294.00 | 2314.30 | 2297.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-14 15:15:00 | 2294.00 | 2314.30 | 2297.59 | SL hit (close<ema400) qty=1.00 sl=2297.59 alert=retest1 |

### Cycle 162 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 2379.50 | 2427.01 | 2430.12 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 15:15:00 | 2385.90 | 2362.35 | 2360.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 2465.00 | 2382.88 | 2369.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 10:15:00 | 2459.70 | 2461.14 | 2427.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 10:30:00 | 2463.90 | 2461.14 | 2427.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 2442.90 | 2458.36 | 2434.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:45:00 | 2441.50 | 2458.36 | 2434.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 2470.80 | 2460.85 | 2437.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:15:00 | 2459.10 | 2460.85 | 2437.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 2459.10 | 2460.50 | 2439.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 2485.90 | 2460.50 | 2439.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-09 14:15:00 | 2734.49 | 2687.57 | 2652.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 10:15:00 | 2372.00 | 2589.09 | 2614.28 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 2426.00 | 2370.09 | 2366.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 12:15:00 | 2457.00 | 2415.37 | 2402.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 09:15:00 | 2459.40 | 2471.60 | 2437.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 10:00:00 | 2459.40 | 2471.60 | 2437.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 2417.50 | 2460.78 | 2435.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:00:00 | 2417.50 | 2460.78 | 2435.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 2457.00 | 2460.02 | 2437.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:30:00 | 2418.80 | 2460.02 | 2437.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 2429.70 | 2453.96 | 2437.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:45:00 | 2425.20 | 2453.96 | 2437.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 2441.70 | 2451.51 | 2437.47 | EMA400 retest candle locked (from upside) |

### Cycle 166 — SELL (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 15:15:00 | 2378.00 | 2426.88 | 2428.10 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 09:15:00 | 2487.50 | 2429.07 | 2426.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-26 09:15:00 | 2599.90 | 2498.37 | 2475.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 2511.20 | 2518.01 | 2489.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 12:00:00 | 2511.20 | 2518.01 | 2489.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 12:15:00 | 2465.00 | 2507.41 | 2487.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:00:00 | 2465.00 | 2507.41 | 2487.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 2466.30 | 2499.19 | 2485.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:30:00 | 2477.20 | 2499.19 | 2485.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 2495.10 | 2485.09 | 2481.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:45:00 | 2485.30 | 2485.09 | 2481.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 2499.00 | 2497.51 | 2489.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 2464.50 | 2497.51 | 2489.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 2442.30 | 2486.47 | 2485.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:00:00 | 2442.30 | 2486.47 | 2485.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 10:15:00 | 2464.60 | 2482.09 | 2483.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 11:15:00 | 2410.00 | 2467.68 | 2476.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 14:15:00 | 2458.00 | 2451.35 | 2465.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 15:00:00 | 2458.00 | 2451.35 | 2465.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 2443.40 | 2449.54 | 2462.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 2466.70 | 2449.54 | 2462.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 2452.40 | 2438.34 | 2450.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:45:00 | 2450.00 | 2438.34 | 2450.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 2430.00 | 2436.67 | 2448.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 2488.70 | 2436.67 | 2448.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 2488.80 | 2447.10 | 2452.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:00:00 | 2488.80 | 2447.10 | 2452.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 2471.90 | 2452.06 | 2454.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 11:15:00 | 2460.00 | 2452.06 | 2454.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 14:15:00 | 2500.90 | 2461.43 | 2457.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 2500.90 | 2461.43 | 2457.66 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 13:15:00 | 2452.60 | 2472.03 | 2472.80 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 12:15:00 | 2493.90 | 2476.88 | 2474.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 15:15:00 | 2499.00 | 2486.78 | 2480.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 09:15:00 | 2456.60 | 2480.74 | 2478.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 2456.60 | 2480.74 | 2478.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 2456.60 | 2480.74 | 2478.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 2456.60 | 2480.74 | 2478.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 2473.20 | 2479.24 | 2477.58 | EMA400 retest candle locked (from upside) |

### Cycle 172 — SELL (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 12:15:00 | 2465.20 | 2474.57 | 2475.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 09:15:00 | 2435.00 | 2464.50 | 2470.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 15:15:00 | 2455.00 | 2449.69 | 2458.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 09:15:00 | 2481.80 | 2449.69 | 2458.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 2489.50 | 2457.65 | 2461.37 | EMA400 retest candle locked (from downside) |

### Cycle 173 — BUY (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 10:15:00 | 2496.30 | 2465.38 | 2464.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 09:15:00 | 2546.10 | 2498.31 | 2483.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 11:15:00 | 2500.00 | 2503.85 | 2488.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 14:15:00 | 2463.90 | 2495.10 | 2488.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 2463.90 | 2495.10 | 2488.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 2463.90 | 2495.10 | 2488.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 2452.00 | 2486.48 | 2485.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 2461.80 | 2486.48 | 2485.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 09:15:00 | 2442.10 | 2477.60 | 2481.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 10:15:00 | 2434.50 | 2468.98 | 2476.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 12:15:00 | 2462.40 | 2461.86 | 2471.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 12:15:00 | 2462.40 | 2461.86 | 2471.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 2462.40 | 2461.86 | 2471.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:30:00 | 2481.90 | 2461.86 | 2471.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 2467.80 | 2463.05 | 2471.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:45:00 | 2473.10 | 2463.05 | 2471.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 2490.00 | 2468.44 | 2473.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 2490.00 | 2468.44 | 2473.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 2475.40 | 2469.83 | 2473.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 2475.00 | 2469.83 | 2473.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 2497.80 | 2475.43 | 2475.65 | EMA400 retest candle locked (from downside) |

### Cycle 175 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 2485.90 | 2477.52 | 2476.58 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 11:15:00 | 2460.40 | 2474.10 | 2475.11 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 12:15:00 | 2501.90 | 2479.66 | 2477.55 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 2462.50 | 2478.88 | 2478.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 14:15:00 | 2453.00 | 2470.03 | 2474.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 2503.00 | 2473.27 | 2475.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 2503.00 | 2473.27 | 2475.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 2503.00 | 2473.27 | 2475.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:45:00 | 2512.70 | 2473.27 | 2475.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 2478.00 | 2474.22 | 2475.28 | EMA400 retest candle locked (from downside) |

### Cycle 179 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 2497.50 | 2478.87 | 2477.30 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 14:15:00 | 2454.40 | 2477.14 | 2477.21 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 2514.90 | 2482.11 | 2479.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 2588.00 | 2509.19 | 2492.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 13:15:00 | 2526.30 | 2527.83 | 2508.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 13:45:00 | 2527.90 | 2527.83 | 2508.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 2503.50 | 2522.97 | 2507.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 2503.50 | 2522.97 | 2507.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 2520.00 | 2522.37 | 2509.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 2542.40 | 2522.37 | 2509.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 12:30:00 | 2537.80 | 2531.81 | 2518.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-28 10:15:00 | 2796.64 | 2686.15 | 2616.09 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 182 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 2921.50 | 2975.90 | 2982.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 2870.00 | 2938.40 | 2962.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 13:15:00 | 2886.10 | 2873.93 | 2901.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:45:00 | 2887.80 | 2873.93 | 2901.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 2896.20 | 2878.38 | 2900.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 2915.10 | 2878.38 | 2900.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 2864.10 | 2875.53 | 2897.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 2952.60 | 2875.53 | 2897.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 2964.50 | 2893.32 | 2903.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:45:00 | 2968.60 | 2893.32 | 2903.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 3005.20 | 2915.70 | 2912.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 11:15:00 | 3040.00 | 2940.56 | 2924.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 15:15:00 | 2970.00 | 2974.21 | 2948.57 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 09:15:00 | 3130.80 | 2974.21 | 2948.57 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 2907.50 | 2960.87 | 2944.83 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 2907.50 | 2960.87 | 2944.83 | SL hit (close<ema400) qty=1.00 sl=2944.83 alert=retest1 |

### Cycle 184 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 3091.70 | 3157.68 | 3163.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 3048.70 | 3092.46 | 3113.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 11:15:00 | 3083.80 | 3044.40 | 3068.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 11:15:00 | 3083.80 | 3044.40 | 3068.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 3083.80 | 3044.40 | 3068.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:00:00 | 3083.80 | 3044.40 | 3068.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 3123.40 | 3060.20 | 3073.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:30:00 | 3116.70 | 3060.20 | 3073.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 3115.50 | 3077.30 | 3078.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:30:00 | 3148.10 | 3077.30 | 3078.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — BUY (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 10:15:00 | 3128.90 | 3087.62 | 3083.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 11:15:00 | 3150.00 | 3100.09 | 3089.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 12:15:00 | 3063.00 | 3092.68 | 3086.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 12:15:00 | 3063.00 | 3092.68 | 3086.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 3063.00 | 3092.68 | 3086.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:00:00 | 3063.00 | 3092.68 | 3086.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 3074.00 | 3088.94 | 3085.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:45:00 | 3069.40 | 3088.94 | 3085.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 3083.90 | 3090.39 | 3086.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 3164.80 | 3090.39 | 3086.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 3157.70 | 3103.85 | 3093.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 14:45:00 | 3173.80 | 3144.73 | 3120.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 10:15:00 | 3041.50 | 3123.63 | 3116.94 | SL hit (close<static) qty=1.00 sl=3068.20 alert=retest2 |

### Cycle 186 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 3025.30 | 3103.97 | 3108.61 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 3154.80 | 3104.46 | 3101.76 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 15:15:00 | 3089.00 | 3101.79 | 3102.00 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 3128.00 | 3107.03 | 3104.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 10:15:00 | 3146.20 | 3114.86 | 3108.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 11:15:00 | 3112.80 | 3114.45 | 3108.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 11:15:00 | 3112.80 | 3114.45 | 3108.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 3112.80 | 3114.45 | 3108.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:45:00 | 3111.50 | 3114.45 | 3108.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 3075.00 | 3106.56 | 3105.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:00:00 | 3075.00 | 3106.56 | 3105.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 13:15:00 | 3083.10 | 3101.87 | 3103.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 14:15:00 | 3065.90 | 3094.67 | 3100.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 09:15:00 | 3057.20 | 3025.76 | 3051.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 3057.20 | 3025.76 | 3051.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 3057.20 | 3025.76 | 3051.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:45:00 | 3068.90 | 3025.76 | 3051.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 3015.20 | 3023.65 | 3047.82 | EMA400 retest candle locked (from downside) |

### Cycle 191 — BUY (started 2025-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 15:15:00 | 3098.00 | 3057.21 | 3055.98 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 15:15:00 | 2992.00 | 3048.46 | 3056.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 09:15:00 | 2963.20 | 3031.41 | 3047.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 15:15:00 | 2970.00 | 2962.84 | 2998.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 15:15:00 | 2970.00 | 2962.84 | 2998.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 2970.00 | 2962.84 | 2998.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 2933.40 | 2962.84 | 2998.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 2786.73 | 2875.32 | 2938.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-09 09:15:00 | 2640.06 | 2805.75 | 2887.86 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 193 — BUY (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 13:15:00 | 2787.20 | 2743.57 | 2742.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 09:15:00 | 2837.60 | 2775.72 | 2758.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 11:15:00 | 2750.10 | 2775.28 | 2761.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 11:15:00 | 2750.10 | 2775.28 | 2761.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 2750.10 | 2775.28 | 2761.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:00:00 | 2750.10 | 2775.28 | 2761.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 2766.50 | 2773.53 | 2761.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 2766.50 | 2773.53 | 2761.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 2746.90 | 2768.20 | 2760.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 2746.90 | 2768.20 | 2760.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 2748.50 | 2764.26 | 2759.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 2748.50 | 2764.26 | 2759.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 2748.60 | 2761.63 | 2759.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 2748.60 | 2761.63 | 2759.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 2756.90 | 2760.68 | 2759.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:15:00 | 2751.10 | 2760.68 | 2759.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 2742.50 | 2757.05 | 2757.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 2729.60 | 2751.56 | 2755.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 10:15:00 | 2712.00 | 2702.41 | 2718.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 10:15:00 | 2712.00 | 2702.41 | 2718.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 2712.00 | 2702.41 | 2718.42 | EMA400 retest candle locked (from downside) |

### Cycle 195 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 2753.00 | 2724.83 | 2723.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 2833.10 | 2746.48 | 2733.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 2840.00 | 2846.77 | 2801.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 10:00:00 | 2840.00 | 2846.77 | 2801.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 2791.00 | 2831.01 | 2801.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 2802.00 | 2831.01 | 2801.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 2794.80 | 2823.77 | 2801.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:30:00 | 2784.70 | 2823.77 | 2801.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 2776.90 | 2814.40 | 2799.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:00:00 | 2776.90 | 2814.40 | 2799.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 2789.00 | 2804.69 | 2797.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 2818.40 | 2804.69 | 2797.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 15:15:00 | 2782.40 | 2796.58 | 2797.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 2782.40 | 2796.58 | 2797.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 10:15:00 | 2749.10 | 2785.44 | 2791.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 2730.50 | 2727.11 | 2748.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 09:15:00 | 2730.50 | 2727.11 | 2748.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 2730.50 | 2727.11 | 2748.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:45:00 | 2738.40 | 2727.11 | 2748.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 2758.10 | 2733.30 | 2749.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:00:00 | 2758.10 | 2733.30 | 2749.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 2778.30 | 2742.30 | 2751.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:30:00 | 2778.00 | 2742.30 | 2751.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 2764.40 | 2749.39 | 2753.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 2764.40 | 2749.39 | 2753.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 2764.00 | 2752.32 | 2754.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 2764.00 | 2752.32 | 2754.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 2762.80 | 2754.41 | 2755.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 2737.30 | 2754.41 | 2755.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2735.80 | 2750.69 | 2753.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:15:00 | 2773.90 | 2750.69 | 2753.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 2772.50 | 2755.05 | 2755.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 2771.60 | 2755.05 | 2755.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 2777.80 | 2759.60 | 2757.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 2791.40 | 2765.96 | 2760.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 2833.20 | 2870.88 | 2845.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 2833.20 | 2870.88 | 2845.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 2833.20 | 2870.88 | 2845.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:45:00 | 2833.10 | 2870.88 | 2845.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 2844.70 | 2865.65 | 2845.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:45:00 | 2859.20 | 2865.65 | 2845.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 2890.00 | 2870.52 | 2849.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:15:00 | 2915.00 | 2870.52 | 2849.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 2905.70 | 2891.55 | 2868.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 14:15:00 | 2902.50 | 2881.02 | 2872.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:00:00 | 2907.90 | 2900.21 | 2884.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 2875.20 | 2895.21 | 2883.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 2875.20 | 2895.21 | 2883.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 2875.10 | 2891.19 | 2882.99 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-07 14:15:00 | 2815.10 | 2868.65 | 2874.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — SELL (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 14:15:00 | 2815.10 | 2868.65 | 2874.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 15:15:00 | 2808.00 | 2856.52 | 2868.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 2704.40 | 2684.27 | 2733.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 15:15:00 | 2706.90 | 2689.92 | 2714.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 2706.90 | 2689.92 | 2714.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 2746.40 | 2689.92 | 2714.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 2752.50 | 2702.44 | 2717.57 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 11:15:00 | 2781.70 | 2729.29 | 2727.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 13:15:00 | 2788.50 | 2749.53 | 2737.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 09:15:00 | 2726.00 | 2759.52 | 2746.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 2726.00 | 2759.52 | 2746.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 2726.00 | 2759.52 | 2746.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 2726.00 | 2759.52 | 2746.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 2710.40 | 2749.70 | 2743.34 | EMA400 retest candle locked (from upside) |

### Cycle 200 — SELL (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 12:15:00 | 2712.00 | 2738.05 | 2738.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 13:15:00 | 2694.10 | 2729.26 | 2734.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 11:15:00 | 2734.00 | 2721.91 | 2727.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 11:15:00 | 2734.00 | 2721.91 | 2727.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 2734.00 | 2721.91 | 2727.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:30:00 | 2732.80 | 2721.91 | 2727.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 2724.60 | 2722.45 | 2727.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 15:00:00 | 2713.20 | 2721.14 | 2726.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 10:15:00 | 2704.30 | 2718.91 | 2724.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 2577.54 | 2653.98 | 2686.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 2569.09 | 2653.98 | 2686.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-21 09:15:00 | 2441.88 | 2554.99 | 2616.00 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 201 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 2612.80 | 2526.46 | 2519.10 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 11:15:00 | 2388.00 | 2513.99 | 2515.65 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 2632.70 | 2493.59 | 2484.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 10:15:00 | 2651.70 | 2525.21 | 2499.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 2603.80 | 2606.57 | 2559.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 09:30:00 | 2590.00 | 2606.57 | 2559.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 2602.00 | 2601.72 | 2568.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 2611.10 | 2601.72 | 2568.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 2570.00 | 2591.92 | 2572.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 2495.00 | 2591.92 | 2572.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2478.40 | 2569.21 | 2563.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:30:00 | 2477.30 | 2569.21 | 2563.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 2462.40 | 2547.85 | 2554.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 13:15:00 | 2443.50 | 2507.71 | 2532.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 2549.90 | 2502.28 | 2522.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 2549.90 | 2502.28 | 2522.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2549.90 | 2502.28 | 2522.90 | EMA400 retest candle locked (from downside) |

### Cycle 205 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 2612.70 | 2549.49 | 2541.57 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2026-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 11:15:00 | 2453.00 | 2528.10 | 2537.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 12:15:00 | 2431.90 | 2508.86 | 2528.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 14:15:00 | 2282.10 | 2280.58 | 2369.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-05 14:45:00 | 2302.00 | 2280.58 | 2369.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 2372.20 | 2303.77 | 2364.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 2412.60 | 2303.77 | 2364.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 2341.40 | 2311.30 | 2362.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 12:30:00 | 2321.00 | 2311.67 | 2354.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 09:15:00 | 2204.95 | 2271.04 | 2320.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-11 09:15:00 | 2088.90 | 2145.17 | 2196.15 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 207 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 1820.50 | 1774.36 | 1768.11 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 10:15:00 | 1734.20 | 1763.27 | 1767.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 12:15:00 | 1730.90 | 1752.18 | 1761.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1652.80 | 1641.70 | 1663.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 1652.80 | 1641.70 | 1663.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1652.80 | 1641.70 | 1663.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 1659.20 | 1641.70 | 1663.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1650.60 | 1646.12 | 1662.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 1681.60 | 1646.12 | 1662.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1707.50 | 1658.40 | 1666.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 1707.50 | 1658.40 | 1666.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 1735.20 | 1673.76 | 1672.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 1747.10 | 1688.43 | 1679.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 10:15:00 | 1709.00 | 1709.19 | 1695.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 11:00:00 | 1709.00 | 1709.19 | 1695.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 1694.00 | 1706.15 | 1695.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 12:00:00 | 1694.00 | 1706.15 | 1695.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 1688.40 | 1702.60 | 1694.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:00:00 | 1688.40 | 1702.60 | 1694.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 1707.10 | 1703.50 | 1695.60 | EMA400 retest candle locked (from upside) |

### Cycle 210 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1672.30 | 1689.49 | 1691.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1649.00 | 1675.97 | 1684.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1724.90 | 1676.11 | 1681.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1724.90 | 1676.11 | 1681.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1724.90 | 1676.11 | 1681.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 1724.90 | 1676.11 | 1681.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 1757.20 | 1692.32 | 1688.12 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-03-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 12:15:00 | 1679.20 | 1691.23 | 1692.49 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2026-03-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-23 13:15:00 | 1703.40 | 1693.66 | 1693.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 11:15:00 | 1722.60 | 1702.61 | 1697.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1729.80 | 1745.22 | 1730.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1729.80 | 1745.22 | 1730.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1729.80 | 1745.22 | 1730.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:30:00 | 1729.60 | 1745.22 | 1730.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1744.10 | 1745.00 | 1732.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 1741.50 | 1745.00 | 1732.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 1733.00 | 1742.60 | 1732.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 1733.00 | 1742.60 | 1732.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 1726.50 | 1739.38 | 1731.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:45:00 | 1722.70 | 1739.38 | 1731.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 1728.00 | 1737.10 | 1731.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 1728.00 | 1737.10 | 1731.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 1732.90 | 1736.26 | 1731.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 1732.90 | 1736.26 | 1731.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 1738.00 | 1736.61 | 1732.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 1723.50 | 1736.61 | 1732.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1704.10 | 1730.11 | 1729.51 | EMA400 retest candle locked (from upside) |

### Cycle 214 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 1708.50 | 1725.79 | 1727.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 1687.00 | 1710.59 | 1719.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1721.40 | 1695.86 | 1709.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1721.40 | 1695.86 | 1709.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1721.40 | 1695.86 | 1709.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 1743.20 | 1695.86 | 1709.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1707.80 | 1698.25 | 1708.95 | EMA400 retest candle locked (from downside) |

### Cycle 215 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 1751.70 | 1718.37 | 1716.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 14:15:00 | 1754.00 | 1725.50 | 1719.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 1661.10 | 1717.34 | 1717.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 1661.10 | 1717.34 | 1717.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 1661.10 | 1717.34 | 1717.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:45:00 | 1677.10 | 1717.34 | 1717.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 216 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1678.70 | 1709.61 | 1713.74 | EMA200 below EMA400 |

### Cycle 217 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 1741.00 | 1713.53 | 1710.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 15:15:00 | 1743.20 | 1726.66 | 1717.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 15:15:00 | 1759.90 | 1766.66 | 1748.07 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1807.90 | 1766.66 | 1748.07 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 1829.80 | 1851.10 | 1825.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 1831.00 | 1851.10 | 1825.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 1824.50 | 1845.78 | 1825.74 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-09 14:15:00 | 1824.50 | 1845.78 | 1825.74 | SL hit (close<ema400) qty=1.00 sl=1825.74 alert=retest1 |

### Cycle 218 — SELL (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 10:15:00 | 1802.50 | 1824.90 | 1825.52 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-04-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 12:15:00 | 1831.30 | 1826.04 | 1825.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 13:15:00 | 1834.70 | 1827.77 | 1826.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 14:15:00 | 1822.20 | 1826.66 | 1826.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 14:15:00 | 1822.20 | 1826.66 | 1826.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 1822.20 | 1826.66 | 1826.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 15:00:00 | 1822.20 | 1826.66 | 1826.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 1826.00 | 1826.52 | 1826.28 | EMA400 retest candle locked (from upside) |

### Cycle 220 — SELL (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 09:15:00 | 1802.20 | 1821.66 | 1824.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-15 14:15:00 | 1782.80 | 1809.82 | 1817.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 09:15:00 | 1817.60 | 1807.71 | 1814.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 1817.60 | 1807.71 | 1814.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1817.60 | 1807.71 | 1814.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 10:15:00 | 1799.70 | 1807.71 | 1814.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:15:00 | 1799.90 | 1807.99 | 1814.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 11:15:00 | 1836.10 | 1812.73 | 1811.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 221 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 11:15:00 | 1836.10 | 1812.73 | 1811.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 12:15:00 | 1884.00 | 1826.98 | 1817.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 1794.80 | 1835.87 | 1826.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 1794.80 | 1835.87 | 1826.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 1794.80 | 1835.87 | 1826.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 10:00:00 | 1794.80 | 1835.87 | 1826.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 1783.20 | 1825.34 | 1822.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 11:00:00 | 1783.20 | 1825.34 | 1822.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — SELL (started 2026-04-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 12:15:00 | 1798.00 | 1817.05 | 1819.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 14:15:00 | 1773.90 | 1804.91 | 1813.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 12:15:00 | 1796.30 | 1777.07 | 1786.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 12:15:00 | 1796.30 | 1777.07 | 1786.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 1796.30 | 1777.07 | 1786.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:00:00 | 1796.30 | 1777.07 | 1786.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 1797.40 | 1781.13 | 1787.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:30:00 | 1805.60 | 1781.13 | 1787.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 1781.70 | 1781.25 | 1787.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 1764.10 | 1781.80 | 1786.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 1675.89 | 1713.53 | 1742.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 10:15:00 | 1721.40 | 1703.95 | 1720.35 | SL hit (close>ema200) qty=0.50 sl=1703.95 alert=retest2 |

### Cycle 223 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 1714.10 | 1657.13 | 1653.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 1756.90 | 1677.09 | 1662.74 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-29 09:15:00 | 432.50 | 2023-05-29 09:15:00 | 426.50 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2023-06-01 12:15:00 | 419.95 | 2023-06-02 13:15:00 | 421.00 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2023-06-02 10:00:00 | 421.95 | 2023-06-02 13:15:00 | 421.00 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2023-06-02 12:00:00 | 421.00 | 2023-06-02 13:15:00 | 421.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2023-06-13 09:30:00 | 529.50 | 2023-06-14 13:15:00 | 510.50 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2023-06-21 10:30:00 | 486.70 | 2023-06-27 09:15:00 | 485.25 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2023-06-21 11:00:00 | 486.25 | 2023-06-27 11:15:00 | 483.00 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2023-06-21 11:45:00 | 486.55 | 2023-06-27 11:15:00 | 483.00 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2023-06-21 12:30:00 | 487.00 | 2023-06-27 11:15:00 | 483.00 | STOP_HIT | 1.00 | 0.82% |
| SELL | retest2 | 2023-06-26 11:30:00 | 475.00 | 2023-06-27 11:15:00 | 483.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2023-06-30 09:15:00 | 483.95 | 2023-07-07 12:15:00 | 495.35 | STOP_HIT | 1.00 | 2.36% |
| BUY | retest2 | 2023-06-30 10:00:00 | 482.15 | 2023-07-07 12:15:00 | 495.35 | STOP_HIT | 1.00 | 2.74% |
| BUY | retest2 | 2023-06-30 11:15:00 | 485.55 | 2023-07-07 12:15:00 | 495.35 | STOP_HIT | 1.00 | 2.02% |
| SELL | retest2 | 2023-07-18 11:30:00 | 510.05 | 2023-07-20 11:15:00 | 530.00 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2023-07-19 11:15:00 | 513.40 | 2023-07-20 11:15:00 | 530.00 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2023-07-19 14:30:00 | 513.20 | 2023-07-20 11:15:00 | 530.00 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2023-08-09 12:15:00 | 530.05 | 2023-08-11 15:15:00 | 526.30 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2023-08-24 09:15:00 | 510.20 | 2023-09-01 10:15:00 | 561.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-11 11:00:00 | 637.75 | 2023-10-18 15:15:00 | 670.15 | STOP_HIT | 1.00 | 5.08% |
| BUY | retest2 | 2023-10-12 09:45:00 | 636.20 | 2023-10-18 15:15:00 | 670.15 | STOP_HIT | 1.00 | 5.34% |
| BUY | retest2 | 2023-11-01 09:15:00 | 676.00 | 2023-11-01 14:15:00 | 663.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2023-11-01 12:00:00 | 676.80 | 2023-11-01 14:15:00 | 663.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2023-11-23 12:30:00 | 789.55 | 2023-11-24 10:15:00 | 819.05 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2023-12-01 14:45:00 | 776.20 | 2023-12-06 09:15:00 | 737.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-04 13:45:00 | 778.60 | 2023-12-06 09:15:00 | 739.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-01 14:45:00 | 776.20 | 2023-12-07 09:15:00 | 740.20 | STOP_HIT | 0.50 | 4.64% |
| SELL | retest2 | 2023-12-04 13:45:00 | 778.60 | 2023-12-07 09:15:00 | 740.20 | STOP_HIT | 0.50 | 4.93% |
| SELL | retest2 | 2023-12-29 13:30:00 | 714.05 | 2024-01-01 10:15:00 | 723.55 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-01-02 10:15:00 | 714.15 | 2024-01-04 11:15:00 | 723.70 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-01-02 11:15:00 | 714.85 | 2024-01-04 11:15:00 | 723.70 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-01-02 12:00:00 | 714.35 | 2024-01-04 11:15:00 | 723.70 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-01-11 12:15:00 | 707.05 | 2024-01-17 09:15:00 | 701.15 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2024-02-02 12:30:00 | 719.65 | 2024-02-08 12:15:00 | 683.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-05 09:30:00 | 720.55 | 2024-02-08 12:15:00 | 684.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-02 12:30:00 | 719.65 | 2024-02-09 13:15:00 | 691.50 | STOP_HIT | 0.50 | 3.91% |
| SELL | retest2 | 2024-02-05 09:30:00 | 720.55 | 2024-02-09 13:15:00 | 691.50 | STOP_HIT | 0.50 | 4.03% |
| BUY | retest2 | 2024-02-20 09:15:00 | 701.20 | 2024-02-21 10:15:00 | 694.60 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-02-20 12:15:00 | 701.10 | 2024-02-21 10:15:00 | 694.60 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-02-29 10:15:00 | 753.80 | 2024-03-04 12:15:00 | 760.00 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2024-02-29 10:45:00 | 753.95 | 2024-03-04 12:15:00 | 760.00 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2024-03-07 11:15:00 | 742.85 | 2024-03-12 09:15:00 | 705.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 11:15:00 | 742.85 | 2024-03-12 14:15:00 | 704.55 | STOP_HIT | 0.50 | 5.16% |
| BUY | retest2 | 2024-04-08 09:15:00 | 718.95 | 2024-04-08 09:15:00 | 712.10 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-04-16 12:15:00 | 695.55 | 2024-04-18 11:15:00 | 715.00 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2024-04-29 09:15:00 | 717.80 | 2024-05-02 09:15:00 | 714.40 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-05-02 09:15:00 | 715.75 | 2024-05-02 09:15:00 | 714.40 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-05-14 12:00:00 | 928.60 | 2024-05-17 10:15:00 | 921.25 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-06-21 09:15:00 | 836.35 | 2024-06-24 11:15:00 | 820.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2024-06-21 13:00:00 | 846.30 | 2024-06-24 11:15:00 | 820.00 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2024-07-01 11:15:00 | 782.40 | 2024-07-03 11:15:00 | 796.20 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-07-12 09:15:00 | 838.00 | 2024-07-16 14:15:00 | 830.15 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-07-12 11:00:00 | 836.50 | 2024-07-16 15:15:00 | 830.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-07-12 13:30:00 | 833.65 | 2024-07-16 15:15:00 | 830.00 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2024-07-12 15:00:00 | 833.15 | 2024-07-16 15:15:00 | 830.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-07-16 12:45:00 | 843.15 | 2024-07-16 15:15:00 | 830.00 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-07-26 09:15:00 | 865.90 | 2024-07-30 15:15:00 | 860.10 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-08-05 13:15:00 | 873.90 | 2024-08-05 15:15:00 | 865.95 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-08-21 10:45:00 | 919.15 | 2024-08-26 09:15:00 | 882.75 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2024-08-28 10:45:00 | 878.90 | 2024-08-30 12:15:00 | 836.00 | PARTIAL | 0.50 | 4.88% |
| SELL | retest2 | 2024-08-28 10:45:00 | 878.90 | 2024-08-30 14:15:00 | 848.60 | STOP_HIT | 0.50 | 3.45% |
| SELL | retest2 | 2024-08-28 11:15:00 | 878.50 | 2024-09-02 10:15:00 | 834.95 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2024-08-28 11:45:00 | 880.00 | 2024-09-02 10:15:00 | 834.57 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2024-08-28 11:15:00 | 878.50 | 2024-09-02 13:15:00 | 844.95 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2024-08-28 11:45:00 | 880.00 | 2024-09-02 13:15:00 | 844.95 | STOP_HIT | 0.50 | 3.98% |
| BUY | retest2 | 2024-09-13 11:15:00 | 986.00 | 2024-09-18 11:15:00 | 963.05 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2024-09-13 12:00:00 | 982.00 | 2024-09-18 11:15:00 | 963.05 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-09-16 13:15:00 | 983.00 | 2024-09-18 11:15:00 | 963.05 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-09-16 15:15:00 | 982.45 | 2024-09-18 11:15:00 | 963.05 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-09-24 10:15:00 | 990.65 | 2024-09-27 10:15:00 | 965.00 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2024-09-24 12:00:00 | 985.75 | 2024-09-27 10:15:00 | 965.00 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-09-24 12:30:00 | 984.20 | 2024-09-27 10:15:00 | 965.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2024-09-25 09:15:00 | 1001.85 | 2024-09-27 10:15:00 | 965.00 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2024-10-24 09:15:00 | 988.00 | 2024-10-28 10:15:00 | 1007.70 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-10-28 10:15:00 | 995.35 | 2024-10-28 10:15:00 | 1007.70 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-10-31 13:15:00 | 1084.00 | 2024-11-06 10:15:00 | 1192.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-31 14:30:00 | 1084.15 | 2024-11-06 10:15:00 | 1192.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-01 18:00:00 | 1099.85 | 2024-11-06 10:15:00 | 1191.85 | TARGET_HIT | 1.00 | 8.36% |
| BUY | retest2 | 2024-11-04 10:00:00 | 1083.50 | 2024-11-07 10:15:00 | 1209.84 | TARGET_HIT | 1.00 | 11.66% |
| SELL | retest2 | 2024-11-14 11:30:00 | 1177.05 | 2024-11-19 09:15:00 | 1232.00 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2024-11-18 09:30:00 | 1163.70 | 2024-11-19 09:15:00 | 1232.00 | STOP_HIT | 1.00 | -5.87% |
| SELL | retest2 | 2024-11-18 10:15:00 | 1177.20 | 2024-11-19 09:15:00 | 1232.00 | STOP_HIT | 1.00 | -4.66% |
| SELL | retest2 | 2024-11-18 11:00:00 | 1177.85 | 2024-11-19 09:15:00 | 1232.00 | STOP_HIT | 1.00 | -4.60% |
| BUY | retest2 | 2024-11-21 11:15:00 | 1214.25 | 2024-11-26 09:15:00 | 1335.68 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-16 14:15:00 | 1498.00 | 2024-12-17 11:15:00 | 1523.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-12-16 15:15:00 | 1498.00 | 2024-12-17 11:15:00 | 1523.00 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-12-24 12:00:00 | 1593.50 | 2024-12-27 10:15:00 | 1553.60 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2024-12-26 14:00:00 | 1581.40 | 2024-12-27 10:15:00 | 1553.60 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-01-08 12:30:00 | 1554.95 | 2025-01-09 11:15:00 | 1605.60 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2025-01-08 13:45:00 | 1568.75 | 2025-01-09 11:15:00 | 1605.60 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-01-17 09:15:00 | 1433.20 | 2025-01-17 14:15:00 | 1476.75 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-01-17 10:15:00 | 1445.15 | 2025-01-17 14:15:00 | 1476.75 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-01-20 09:45:00 | 1442.00 | 2025-01-20 10:15:00 | 1473.35 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-01-27 15:15:00 | 1360.00 | 2025-01-28 12:15:00 | 1377.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-02-01 13:30:00 | 1731.90 | 2025-02-07 12:15:00 | 1701.80 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-02-03 10:15:00 | 1770.90 | 2025-02-07 12:15:00 | 1701.80 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2025-02-05 10:30:00 | 1729.00 | 2025-02-07 12:15:00 | 1701.80 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-02-05 12:15:00 | 1725.95 | 2025-02-07 12:15:00 | 1701.80 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-02-06 09:45:00 | 1741.50 | 2025-02-07 12:15:00 | 1701.80 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-02-12 13:30:00 | 1567.65 | 2025-02-17 11:15:00 | 1489.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 13:30:00 | 1567.65 | 2025-02-17 14:15:00 | 1537.00 | STOP_HIT | 0.50 | 1.96% |
| SELL | retest2 | 2025-02-13 09:15:00 | 1544.30 | 2025-02-19 09:15:00 | 1479.29 | PARTIAL | 0.50 | 4.21% |
| SELL | retest2 | 2025-02-13 09:15:00 | 1544.30 | 2025-02-19 09:15:00 | 1539.10 | STOP_HIT | 0.50 | 0.34% |
| SELL | retest2 | 2025-02-13 12:00:00 | 1557.15 | 2025-02-19 09:15:00 | 1480.72 | PARTIAL | 0.50 | 4.91% |
| SELL | retest2 | 2025-02-13 12:00:00 | 1557.15 | 2025-02-19 09:15:00 | 1539.10 | STOP_HIT | 0.50 | 1.16% |
| SELL | retest2 | 2025-02-14 15:15:00 | 1558.65 | 2025-02-19 10:15:00 | 1593.25 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-02-17 10:45:00 | 1508.75 | 2025-02-19 10:15:00 | 1593.25 | STOP_HIT | 1.00 | -5.60% |
| SELL | retest2 | 2025-02-18 09:30:00 | 1526.15 | 2025-02-19 10:15:00 | 1593.25 | STOP_HIT | 1.00 | -4.40% |
| SELL | retest2 | 2025-02-18 10:15:00 | 1522.15 | 2025-02-19 10:15:00 | 1593.25 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2025-02-18 11:00:00 | 1522.75 | 2025-02-19 10:15:00 | 1593.25 | STOP_HIT | 1.00 | -4.63% |
| BUY | retest2 | 2025-03-05 13:00:00 | 1562.15 | 2025-03-11 09:15:00 | 1512.50 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2025-03-06 09:15:00 | 1577.20 | 2025-03-11 09:15:00 | 1512.50 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2025-03-06 10:45:00 | 1560.45 | 2025-03-11 09:15:00 | 1512.50 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2025-03-06 13:00:00 | 1563.90 | 2025-03-11 09:15:00 | 1512.50 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2025-03-07 09:15:00 | 1559.85 | 2025-03-11 09:15:00 | 1512.50 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-03-07 10:00:00 | 1558.55 | 2025-03-11 09:15:00 | 1512.50 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2025-03-07 11:30:00 | 1555.50 | 2025-03-11 09:15:00 | 1512.50 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-03-07 12:30:00 | 1557.15 | 2025-03-11 09:15:00 | 1512.50 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-03-10 09:15:00 | 1558.20 | 2025-03-11 09:15:00 | 1512.50 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-03-10 13:30:00 | 1555.70 | 2025-03-11 09:15:00 | 1512.50 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-03-10 14:30:00 | 1564.80 | 2025-03-11 09:15:00 | 1512.50 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest2 | 2025-03-20 12:30:00 | 1709.00 | 2025-03-21 14:15:00 | 1879.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-20 13:30:00 | 1707.35 | 2025-03-21 14:15:00 | 1878.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-20 14:45:00 | 1707.05 | 2025-03-21 14:15:00 | 1877.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-21 09:15:00 | 1724.65 | 2025-03-25 11:15:00 | 1690.10 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-03-21 11:15:00 | 1767.85 | 2025-03-25 11:15:00 | 1690.10 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest2 | 2025-03-21 12:45:00 | 1786.00 | 2025-03-25 11:15:00 | 1690.10 | STOP_HIT | 1.00 | -5.37% |
| BUY | retest2 | 2025-03-21 13:30:00 | 1776.95 | 2025-03-25 11:15:00 | 1690.10 | STOP_HIT | 1.00 | -4.89% |
| BUY | retest2 | 2025-03-21 15:00:00 | 1883.60 | 2025-03-25 11:15:00 | 1690.10 | STOP_HIT | 1.00 | -10.27% |
| SELL | retest2 | 2025-03-27 14:15:00 | 1671.85 | 2025-04-01 11:15:00 | 1588.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-27 14:15:00 | 1671.85 | 2025-04-02 09:15:00 | 1649.50 | STOP_HIT | 0.50 | 1.34% |
| SELL | retest2 | 2025-03-27 15:15:00 | 1631.05 | 2025-04-03 09:15:00 | 1681.65 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-03-28 13:45:00 | 1666.00 | 2025-04-03 09:15:00 | 1681.65 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-04-02 12:45:00 | 1670.95 | 2025-04-03 09:15:00 | 1681.65 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-04-09 11:00:00 | 1409.80 | 2025-04-11 14:15:00 | 1537.65 | STOP_HIT | 1.00 | -9.07% |
| BUY | retest2 | 2025-04-17 11:15:00 | 1528.60 | 2025-04-22 09:15:00 | 1681.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-05 12:00:00 | 1833.00 | 2025-05-06 14:15:00 | 1709.50 | STOP_HIT | 1.00 | -6.74% |
| SELL | retest2 | 2025-05-12 11:30:00 | 1622.50 | 2025-05-13 09:15:00 | 1661.80 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-05-26 10:45:00 | 1641.10 | 2025-05-27 12:15:00 | 1559.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-26 11:15:00 | 1641.90 | 2025-05-27 12:15:00 | 1559.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-26 10:45:00 | 1641.10 | 2025-05-29 09:15:00 | 1571.60 | STOP_HIT | 0.50 | 4.23% |
| SELL | retest2 | 2025-05-26 11:15:00 | 1641.90 | 2025-05-29 09:15:00 | 1571.60 | STOP_HIT | 0.50 | 4.28% |
| SELL | retest2 | 2025-06-24 11:15:00 | 1627.20 | 2025-06-30 09:15:00 | 1670.40 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-06-24 12:30:00 | 1630.60 | 2025-06-30 09:15:00 | 1670.40 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-06-25 09:30:00 | 1631.90 | 2025-06-30 09:15:00 | 1670.40 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-06-25 11:00:00 | 1628.90 | 2025-06-30 09:15:00 | 1670.40 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-06-25 13:15:00 | 1611.10 | 2025-06-30 09:15:00 | 1670.40 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2025-06-26 09:30:00 | 1605.50 | 2025-06-30 09:15:00 | 1670.40 | STOP_HIT | 1.00 | -4.04% |
| SELL | retest2 | 2025-06-26 11:15:00 | 1614.70 | 2025-06-30 09:15:00 | 1670.40 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2025-07-08 09:15:00 | 1733.40 | 2025-07-11 10:15:00 | 1906.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-23 11:45:00 | 1925.00 | 2025-07-25 10:15:00 | 1897.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-07-23 12:30:00 | 1926.40 | 2025-07-25 10:15:00 | 1897.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest1 | 2025-08-13 12:30:00 | 2344.20 | 2025-08-14 15:15:00 | 2294.00 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest1 | 2025-08-14 10:30:00 | 2332.80 | 2025-08-14 15:15:00 | 2294.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest1 | 2025-08-14 14:30:00 | 2340.00 | 2025-08-14 15:15:00 | 2294.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-08-18 09:15:00 | 2357.00 | 2025-08-22 09:15:00 | 2379.50 | STOP_HIT | 1.00 | 0.95% |
| BUY | retest2 | 2025-09-03 09:15:00 | 2485.90 | 2025-09-09 14:15:00 | 2734.49 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-03 11:15:00 | 2460.00 | 2025-10-03 14:15:00 | 2500.90 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-10-24 09:15:00 | 2542.40 | 2025-10-28 10:15:00 | 2796.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-24 12:30:00 | 2537.80 | 2025-10-28 10:15:00 | 2791.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2025-11-11 09:15:00 | 3130.80 | 2025-11-11 09:15:00 | 2907.50 | STOP_HIT | 1.00 | -7.13% |
| BUY | retest2 | 2025-11-11 11:15:00 | 2932.70 | 2025-11-17 11:15:00 | 3225.97 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-26 14:45:00 | 3173.80 | 2025-11-27 10:15:00 | 3041.50 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2025-12-08 09:15:00 | 2933.40 | 2025-12-08 13:15:00 | 2786.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 09:15:00 | 2933.40 | 2025-12-09 09:15:00 | 2640.06 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-12-24 09:15:00 | 2818.40 | 2025-12-24 15:15:00 | 2782.40 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2026-01-05 12:15:00 | 2915.00 | 2026-01-07 14:15:00 | 2815.10 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest2 | 2026-01-06 09:15:00 | 2905.70 | 2026-01-07 14:15:00 | 2815.10 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2026-01-06 14:15:00 | 2902.50 | 2026-01-07 14:15:00 | 2815.10 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2026-01-07 10:00:00 | 2907.90 | 2026-01-07 14:15:00 | 2815.10 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2026-01-16 15:00:00 | 2713.20 | 2026-01-20 10:15:00 | 2577.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 10:15:00 | 2704.30 | 2026-01-20 10:15:00 | 2569.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 15:00:00 | 2713.20 | 2026-01-21 09:15:00 | 2441.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-19 10:15:00 | 2704.30 | 2026-01-21 09:15:00 | 2433.87 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-06 12:30:00 | 2321.00 | 2026-02-09 09:15:00 | 2204.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-06 12:30:00 | 2321.00 | 2026-02-11 09:15:00 | 2088.90 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2026-04-08 09:15:00 | 1807.90 | 2026-04-09 14:15:00 | 1824.50 | STOP_HIT | 1.00 | 0.92% |
| BUY | retest2 | 2026-04-10 12:45:00 | 1842.70 | 2026-04-13 10:15:00 | 1802.50 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2026-04-10 15:15:00 | 1834.80 | 2026-04-13 10:15:00 | 1802.50 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-04-16 10:15:00 | 1799.70 | 2026-04-17 11:15:00 | 1836.10 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2026-04-16 11:15:00 | 1799.90 | 2026-04-17 11:15:00 | 1836.10 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2026-04-23 09:15:00 | 1764.10 | 2026-04-24 09:15:00 | 1675.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 09:15:00 | 1764.10 | 2026-04-27 10:15:00 | 1721.40 | STOP_HIT | 0.50 | 2.42% |
