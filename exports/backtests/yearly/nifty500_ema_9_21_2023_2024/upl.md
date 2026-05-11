# UPL Ltd. (UPL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 644.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 208 |
| ALERT1 | 150 |
| ALERT2 | 149 |
| ALERT2_SKIP | 68 |
| ALERT3 | 401 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 165 |
| PARTIAL | 15 |
| TARGET_HIT | 6 |
| STOP_HIT | 166 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 185 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 55 / 130
- **Target hits / Stop hits / Partials:** 6 / 165 / 14
- **Avg / median % per leg:** 0.31% / -0.68%
- **Sum % (uncompounded):** 57.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 93 | 26 | 28.0% | 0 | 92 | 1 | -0.22% | -20.8% |
| BUY @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 0 | 6 | 1 | 1.87% | 13.1% |
| BUY @ 3rd Alert (retest2) | 86 | 20 | 23.3% | 0 | 86 | 0 | -0.39% | -33.9% |
| SELL (all) | 92 | 29 | 31.5% | 6 | 73 | 13 | 0.86% | 78.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 92 | 29 | 31.5% | 6 | 73 | 13 | 0.86% | 78.8% |
| retest1 (combined) | 7 | 6 | 85.7% | 0 | 6 | 1 | 1.87% | 13.1% |
| retest2 (combined) | 178 | 49 | 27.5% | 6 | 159 | 13 | 0.25% | 44.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 10:15:00 | 649.74 | 644.27 | 643.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 11:15:00 | 651.13 | 645.64 | 644.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 09:15:00 | 650.51 | 652.64 | 650.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 09:15:00 | 650.51 | 652.64 | 650.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 650.51 | 652.64 | 650.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 09:30:00 | 650.61 | 652.64 | 650.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 10:15:00 | 648.64 | 651.84 | 650.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 11:00:00 | 648.64 | 651.84 | 650.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 11:15:00 | 647.54 | 650.98 | 650.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 12:15:00 | 647.49 | 650.98 | 650.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 13:15:00 | 648.93 | 650.24 | 649.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 13:45:00 | 647.78 | 650.24 | 649.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 14:15:00 | 648.35 | 649.86 | 649.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 14:45:00 | 647.92 | 649.86 | 649.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 10:15:00 | 658.04 | 660.22 | 658.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 11:00:00 | 658.04 | 660.22 | 658.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 11:15:00 | 658.14 | 659.81 | 658.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 11:45:00 | 657.61 | 659.81 | 658.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 12:15:00 | 660.06 | 659.86 | 658.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 13:15:00 | 660.34 | 659.86 | 658.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 13:45:00 | 660.44 | 660.18 | 658.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 11:30:00 | 660.34 | 660.32 | 659.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-31 15:15:00 | 655.16 | 659.14 | 659.07 | SL hit (close<static) qty=1.00 sl=657.85 alert=retest2 |

### Cycle 2 — SELL (started 2023-06-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-01 09:15:00 | 651.90 | 657.69 | 658.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-01 13:15:00 | 649.26 | 654.04 | 656.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-02 09:15:00 | 653.87 | 653.03 | 655.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 09:15:00 | 653.87 | 653.03 | 655.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 653.87 | 653.03 | 655.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-02 09:45:00 | 655.07 | 653.03 | 655.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 11:15:00 | 654.78 | 653.69 | 655.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-02 12:00:00 | 654.78 | 653.69 | 655.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 12:15:00 | 654.40 | 653.83 | 655.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-02 12:30:00 | 655.26 | 653.83 | 655.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 14:15:00 | 655.31 | 654.19 | 655.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-02 15:00:00 | 655.31 | 654.19 | 655.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 15:15:00 | 655.93 | 654.54 | 655.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-05 09:15:00 | 658.42 | 654.54 | 655.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2023-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 09:15:00 | 659.10 | 655.45 | 655.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 13:15:00 | 662.12 | 658.20 | 656.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-07 14:15:00 | 665.43 | 665.67 | 663.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-07 15:00:00 | 665.43 | 665.67 | 663.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 09:15:00 | 667.54 | 666.21 | 664.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-08 10:30:00 | 669.55 | 666.58 | 664.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-08 11:15:00 | 662.88 | 665.84 | 664.43 | SL hit (close<static) qty=1.00 sl=663.41 alert=retest2 |

### Cycle 4 — SELL (started 2023-06-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 15:15:00 | 662.02 | 663.72 | 663.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 12:15:00 | 656.79 | 661.91 | 662.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 10:15:00 | 653.00 | 652.56 | 655.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-13 11:00:00 | 653.00 | 652.56 | 655.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 12:15:00 | 654.73 | 653.09 | 655.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 13:00:00 | 654.73 | 653.09 | 655.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 13:15:00 | 655.35 | 653.54 | 655.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 13:30:00 | 655.07 | 653.54 | 655.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 14:15:00 | 654.30 | 653.69 | 655.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 14:45:00 | 655.40 | 653.69 | 655.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 15:15:00 | 655.16 | 653.99 | 655.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-14 09:15:00 | 654.68 | 653.99 | 655.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 09:15:00 | 653.92 | 653.97 | 654.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-15 11:30:00 | 650.75 | 653.94 | 654.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-15 12:15:00 | 650.70 | 653.94 | 654.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-16 09:15:00 | 659.82 | 654.42 | 654.43 | SL hit (close>static) qty=1.00 sl=655.88 alert=retest2 |

### Cycle 5 — BUY (started 2023-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 10:15:00 | 660.82 | 655.70 | 655.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 11:15:00 | 663.51 | 657.26 | 655.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-19 09:15:00 | 660.25 | 661.09 | 658.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-19 09:30:00 | 659.96 | 661.09 | 658.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 10:15:00 | 656.12 | 660.09 | 658.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 11:00:00 | 656.12 | 660.09 | 658.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 11:15:00 | 656.12 | 659.30 | 658.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 12:00:00 | 656.12 | 659.30 | 658.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 15:15:00 | 657.70 | 657.85 | 657.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 09:15:00 | 653.48 | 657.85 | 657.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2023-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 09:15:00 | 652.76 | 656.83 | 657.26 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-06-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 13:15:00 | 658.52 | 656.81 | 656.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 15:15:00 | 658.95 | 657.45 | 657.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 09:15:00 | 655.59 | 657.08 | 656.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 09:15:00 | 655.59 | 657.08 | 656.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 655.59 | 657.08 | 656.87 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-06-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 10:15:00 | 654.35 | 656.53 | 656.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 11:15:00 | 651.47 | 655.52 | 656.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 09:15:00 | 648.64 | 643.72 | 647.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 09:15:00 | 648.64 | 643.72 | 647.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 648.64 | 643.72 | 647.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 09:45:00 | 647.11 | 643.72 | 647.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 652.29 | 645.43 | 647.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 11:00:00 | 652.29 | 645.43 | 647.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 11:15:00 | 650.80 | 646.51 | 648.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 14:15:00 | 649.94 | 648.18 | 648.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-26 15:15:00 | 651.33 | 649.30 | 649.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2023-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 15:15:00 | 651.33 | 649.30 | 649.06 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 10:15:00 | 644.80 | 648.11 | 648.55 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-06-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 13:15:00 | 653.05 | 649.06 | 648.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 09:15:00 | 656.07 | 651.63 | 649.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 09:15:00 | 654.35 | 656.21 | 653.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-03 09:45:00 | 654.68 | 656.21 | 653.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 12:15:00 | 651.66 | 655.00 | 653.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 13:00:00 | 651.66 | 655.00 | 653.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 13:15:00 | 651.90 | 654.38 | 653.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 13:45:00 | 651.33 | 654.38 | 653.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 656.36 | 654.21 | 653.61 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 14:15:00 | 649.41 | 653.06 | 653.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-05 09:15:00 | 647.11 | 651.44 | 652.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-06 14:15:00 | 645.04 | 644.64 | 646.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-06 15:00:00 | 645.04 | 644.64 | 646.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 642.07 | 644.31 | 646.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-07 10:15:00 | 641.16 | 644.31 | 646.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-13 13:15:00 | 609.10 | 613.30 | 619.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-07-14 12:15:00 | 610.32 | 609.69 | 614.35 | SL hit (close>ema200) qty=0.50 sl=609.69 alert=retest2 |

### Cycle 13 — BUY (started 2023-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 12:15:00 | 617.70 | 615.45 | 615.36 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 11:15:00 | 613.63 | 615.39 | 615.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-19 13:15:00 | 612.57 | 614.14 | 614.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-20 11:15:00 | 613.82 | 613.74 | 614.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-20 11:30:00 | 613.77 | 613.74 | 614.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 612.38 | 613.39 | 613.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-21 10:30:00 | 609.60 | 612.50 | 613.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-03 09:15:00 | 579.12 | 591.16 | 595.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-08-04 11:15:00 | 580.68 | 580.16 | 585.46 | SL hit (close>ema200) qty=0.50 sl=580.16 alert=retest2 |

### Cycle 15 — BUY (started 2023-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 11:15:00 | 585.33 | 582.11 | 582.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 12:15:00 | 588.78 | 583.45 | 582.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 11:15:00 | 585.91 | 587.79 | 585.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 11:15:00 | 585.91 | 587.79 | 585.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 11:15:00 | 585.91 | 587.79 | 585.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 12:00:00 | 585.91 | 587.79 | 585.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 12:15:00 | 586.72 | 587.57 | 585.76 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 10:15:00 | 579.38 | 584.02 | 584.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 12:15:00 | 578.81 | 582.20 | 583.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 12:15:00 | 567.44 | 566.92 | 571.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-16 13:00:00 | 567.44 | 566.92 | 571.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 560.68 | 557.92 | 561.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 09:30:00 | 560.97 | 557.92 | 561.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 564.80 | 559.29 | 561.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 10:45:00 | 564.18 | 559.29 | 561.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 11:15:00 | 562.50 | 559.94 | 561.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 14:15:00 | 561.83 | 560.98 | 561.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 15:00:00 | 561.78 | 561.14 | 561.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-22 09:45:00 | 559.77 | 560.95 | 561.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 11:15:00 | 562.26 | 560.93 | 561.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-23 11:15:00 | 563.41 | 561.43 | 561.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2023-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 11:15:00 | 563.41 | 561.43 | 561.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-24 09:15:00 | 566.91 | 562.98 | 562.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 13:15:00 | 563.99 | 564.12 | 563.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-24 14:00:00 | 563.99 | 564.12 | 563.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 14:15:00 | 563.32 | 563.96 | 563.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 15:00:00 | 563.32 | 563.96 | 563.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 562.50 | 563.67 | 563.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 09:15:00 | 563.08 | 563.67 | 563.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 560.97 | 563.13 | 562.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 10:15:00 | 558.09 | 563.13 | 562.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 10:15:00 | 558.38 | 562.18 | 562.45 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 09:15:00 | 569.07 | 562.87 | 562.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 11:15:00 | 574.83 | 566.50 | 564.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 14:15:00 | 575.74 | 576.10 | 572.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-30 14:30:00 | 575.45 | 576.10 | 572.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 573.96 | 575.52 | 572.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:30:00 | 573.92 | 575.52 | 572.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 570.70 | 574.56 | 572.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 11:00:00 | 570.70 | 574.56 | 572.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 11:15:00 | 569.45 | 573.54 | 572.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 12:00:00 | 569.45 | 573.54 | 572.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 13:15:00 | 570.46 | 572.51 | 571.81 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-08-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 14:15:00 | 566.62 | 571.34 | 571.34 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 11:15:00 | 575.31 | 571.98 | 571.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 13:15:00 | 578.04 | 573.69 | 572.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 12:15:00 | 581.01 | 581.40 | 578.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-05 12:45:00 | 580.68 | 581.40 | 578.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 12:15:00 | 580.01 | 581.83 | 580.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 12:45:00 | 579.77 | 581.83 | 580.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 13:15:00 | 582.16 | 581.90 | 580.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 14:30:00 | 583.70 | 582.38 | 580.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 09:15:00 | 585.43 | 584.54 | 584.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 13:15:00 | 581.88 | 585.82 | 586.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2023-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 13:15:00 | 581.88 | 585.82 | 586.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 14:15:00 | 578.81 | 584.42 | 585.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 11:15:00 | 584.42 | 583.18 | 584.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 11:15:00 | 584.42 | 583.18 | 584.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 11:15:00 | 584.42 | 583.18 | 584.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 12:00:00 | 584.42 | 583.18 | 584.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 12:15:00 | 584.32 | 583.40 | 584.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 13:00:00 | 584.32 | 583.40 | 584.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 13:15:00 | 583.70 | 583.46 | 584.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 14:15:00 | 584.18 | 583.46 | 584.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 14:15:00 | 583.03 | 583.38 | 584.25 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-09-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 09:15:00 | 595.79 | 585.91 | 585.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 10:15:00 | 603.75 | 589.48 | 586.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 13:15:00 | 607.73 | 607.86 | 603.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-18 14:00:00 | 607.73 | 607.86 | 603.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 15:15:00 | 602.40 | 606.13 | 603.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 09:15:00 | 601.35 | 606.13 | 603.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 597.61 | 604.42 | 602.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 10:15:00 | 595.69 | 604.42 | 602.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2023-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 11:15:00 | 597.13 | 601.68 | 601.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 09:15:00 | 594.30 | 598.59 | 599.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 10:15:00 | 594.97 | 594.21 | 596.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 10:15:00 | 594.97 | 594.21 | 596.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 10:15:00 | 594.97 | 594.21 | 596.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 10:30:00 | 596.60 | 594.21 | 596.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 11:15:00 | 596.31 | 594.63 | 596.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 11:45:00 | 596.27 | 594.63 | 596.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 12:15:00 | 598.18 | 595.34 | 596.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 13:00:00 | 598.18 | 595.34 | 596.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 13:15:00 | 594.20 | 595.11 | 596.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 13:30:00 | 598.14 | 595.11 | 596.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 14:15:00 | 591.04 | 594.30 | 595.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 14:30:00 | 595.16 | 594.30 | 595.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 594.73 | 593.85 | 595.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 10:00:00 | 594.73 | 593.85 | 595.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 595.74 | 594.23 | 595.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 10:45:00 | 596.55 | 594.23 | 595.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 11:15:00 | 597.27 | 594.83 | 595.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 12:00:00 | 597.27 | 594.83 | 595.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 12:15:00 | 593.10 | 594.49 | 595.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 09:15:00 | 590.32 | 593.58 | 594.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 12:15:00 | 590.75 | 591.71 | 593.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-28 09:30:00 | 589.84 | 590.73 | 592.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-29 10:30:00 | 591.18 | 588.43 | 589.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-29 13:15:00 | 593.96 | 590.92 | 590.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2023-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 13:15:00 | 593.96 | 590.92 | 590.60 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 09:15:00 | 585.47 | 590.09 | 590.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-03 10:15:00 | 584.95 | 589.06 | 589.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 580.20 | 579.83 | 582.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 580.20 | 579.83 | 582.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 580.20 | 579.83 | 582.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 10:15:00 | 578.09 | 579.83 | 582.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 11:30:00 | 578.81 | 579.63 | 582.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 12:00:00 | 578.95 | 579.63 | 582.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 14:00:00 | 578.66 | 579.45 | 581.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 580.82 | 579.89 | 581.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-10-06 11:15:00 | 584.85 | 581.43 | 581.79 | SL hit (close>static) qty=1.00 sl=584.56 alert=retest2 |

### Cycle 27 — BUY (started 2023-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 12:15:00 | 585.23 | 582.19 | 582.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 13:15:00 | 585.62 | 582.88 | 582.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 580.58 | 583.66 | 583.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 580.58 | 583.66 | 583.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 580.58 | 583.66 | 583.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-09 14:45:00 | 585.47 | 583.49 | 583.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-10 09:15:00 | 586.63 | 583.52 | 583.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 10:15:00 | 596.65 | 600.36 | 600.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2023-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 10:15:00 | 596.65 | 600.36 | 600.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 12:15:00 | 595.35 | 598.74 | 599.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 09:15:00 | 516.65 | 513.67 | 519.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 09:15:00 | 516.65 | 513.67 | 519.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 516.65 | 513.67 | 519.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 09:45:00 | 516.79 | 513.67 | 519.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 13:15:00 | 517.94 | 515.67 | 518.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 13:45:00 | 518.71 | 515.67 | 518.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 14:15:00 | 518.71 | 516.28 | 518.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 15:00:00 | 518.71 | 516.28 | 518.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 15:15:00 | 518.71 | 516.76 | 518.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 09:15:00 | 525.47 | 516.76 | 518.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 09:15:00 | 527.54 | 518.92 | 519.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 09:45:00 | 526.14 | 518.92 | 519.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2023-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 10:15:00 | 525.38 | 520.21 | 519.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 09:15:00 | 528.78 | 525.37 | 522.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 11:15:00 | 529.74 | 530.32 | 527.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-07 11:45:00 | 529.60 | 530.32 | 527.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 531.23 | 532.01 | 530.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 09:45:00 | 531.28 | 532.01 | 530.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 529.31 | 531.47 | 530.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 11:00:00 | 529.31 | 531.47 | 530.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 11:15:00 | 528.30 | 530.83 | 530.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 11:45:00 | 528.30 | 530.83 | 530.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2023-11-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 13:15:00 | 527.68 | 529.75 | 529.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 09:15:00 | 525.14 | 528.16 | 529.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 15:15:00 | 526.14 | 525.99 | 527.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-12 18:15:00 | 533.34 | 525.99 | 527.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 532.67 | 527.33 | 527.84 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 10:15:00 | 529.50 | 528.30 | 528.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-13 15:15:00 | 530.94 | 529.31 | 528.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 09:15:00 | 535.11 | 536.90 | 534.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-16 09:30:00 | 536.12 | 536.90 | 534.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 10:15:00 | 535.35 | 536.59 | 534.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-16 10:30:00 | 534.06 | 536.59 | 534.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 537.32 | 538.37 | 537.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 09:45:00 | 537.56 | 538.37 | 537.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 10:15:00 | 539.09 | 538.51 | 537.33 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 14:15:00 | 534.06 | 536.67 | 536.80 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 11:15:00 | 537.80 | 536.89 | 536.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 14:15:00 | 539.91 | 537.69 | 537.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-23 13:15:00 | 542.45 | 543.39 | 541.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-23 14:00:00 | 542.45 | 543.39 | 541.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 540.10 | 542.58 | 541.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 09:45:00 | 539.81 | 542.58 | 541.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 539.67 | 541.99 | 541.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 11:00:00 | 539.67 | 541.99 | 541.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2023-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 12:15:00 | 536.12 | 540.63 | 541.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 09:15:00 | 535.50 | 538.01 | 539.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 12:15:00 | 538.14 | 537.57 | 538.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-28 13:00:00 | 538.14 | 537.57 | 538.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 14:15:00 | 540.73 | 538.34 | 539.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 14:30:00 | 540.68 | 538.34 | 539.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 15:15:00 | 540.77 | 538.82 | 539.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 09:15:00 | 540.49 | 538.82 | 539.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2023-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 09:15:00 | 543.65 | 539.79 | 539.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 11:15:00 | 544.56 | 541.41 | 540.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 12:15:00 | 545.09 | 545.09 | 543.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-30 12:30:00 | 544.95 | 545.09 | 543.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 12:15:00 | 561.01 | 564.43 | 562.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:00:00 | 561.01 | 564.43 | 562.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 13:15:00 | 559.48 | 563.44 | 562.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:30:00 | 557.37 | 563.44 | 562.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 14:15:00 | 573.29 | 575.01 | 571.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 14:45:00 | 572.62 | 575.01 | 571.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 15:15:00 | 572.19 | 574.44 | 571.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 09:15:00 | 573.96 | 574.44 | 571.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-13 10:15:00 | 571.61 | 573.57 | 571.82 | SL hit (close<static) qty=1.00 sl=571.71 alert=retest2 |

### Cycle 36 — SELL (started 2023-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 11:15:00 | 577.08 | 580.10 | 580.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-19 14:15:00 | 574.97 | 578.04 | 579.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 560.25 | 556.56 | 561.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 10:15:00 | 559.96 | 557.24 | 561.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 559.96 | 557.24 | 561.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:45:00 | 560.97 | 557.24 | 561.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 14:15:00 | 557.99 | 557.87 | 560.62 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 15:15:00 | 563.08 | 561.79 | 561.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 09:15:00 | 564.75 | 562.38 | 561.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 12:15:00 | 559.09 | 561.95 | 561.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 12:15:00 | 559.09 | 561.95 | 561.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 559.09 | 561.95 | 561.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 13:00:00 | 559.09 | 561.95 | 561.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2023-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 13:15:00 | 559.43 | 561.45 | 561.65 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-12-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 14:15:00 | 566.19 | 561.83 | 561.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 09:15:00 | 572.19 | 565.30 | 563.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 14:15:00 | 567.63 | 569.42 | 566.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-01 15:00:00 | 567.63 | 569.42 | 566.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 569.21 | 569.49 | 567.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:30:00 | 567.30 | 569.49 | 567.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 567.68 | 569.50 | 568.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 09:30:00 | 565.62 | 569.50 | 568.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 10:15:00 | 567.58 | 569.11 | 568.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 10:45:00 | 565.52 | 569.11 | 568.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 11:15:00 | 567.78 | 568.85 | 568.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 12:00:00 | 567.78 | 568.85 | 568.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 12:15:00 | 565.28 | 568.13 | 568.02 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2024-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 13:15:00 | 565.09 | 567.52 | 567.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 14:15:00 | 562.93 | 566.61 | 567.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 15:15:00 | 564.03 | 563.64 | 564.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-05 09:15:00 | 563.65 | 563.64 | 564.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 09:15:00 | 565.23 | 563.96 | 565.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-05 10:15:00 | 565.81 | 563.96 | 565.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 10:15:00 | 563.03 | 563.78 | 564.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-05 13:15:00 | 561.11 | 563.01 | 564.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 09:15:00 | 533.05 | 542.70 | 550.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-10 14:15:00 | 535.45 | 535.24 | 539.90 | SL hit (close>ema200) qty=0.50 sl=535.24 alert=retest2 |

### Cycle 41 — BUY (started 2024-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 13:15:00 | 544.56 | 540.97 | 540.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-15 11:15:00 | 545.76 | 542.57 | 541.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-15 14:15:00 | 543.89 | 543.92 | 542.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-15 15:00:00 | 543.89 | 543.92 | 542.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 544.03 | 544.13 | 542.85 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2024-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 13:15:00 | 540.39 | 542.39 | 542.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 09:15:00 | 537.18 | 541.18 | 541.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 531.66 | 526.13 | 530.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 531.66 | 526.13 | 530.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 531.66 | 526.13 | 530.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-23 12:45:00 | 519.29 | 525.31 | 527.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 09:30:00 | 519.96 | 520.86 | 522.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-30 11:00:00 | 520.01 | 518.81 | 519.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-30 12:00:00 | 520.44 | 519.14 | 519.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 12:15:00 | 519.05 | 519.12 | 519.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-30 14:15:00 | 516.79 | 519.16 | 519.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-31 09:30:00 | 517.61 | 516.52 | 517.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 09:15:00 | 493.33 | 506.58 | 510.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 09:15:00 | 493.96 | 506.58 | 510.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 09:15:00 | 494.01 | 506.58 | 510.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 09:15:00 | 494.42 | 506.58 | 510.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 09:15:00 | 490.95 | 506.58 | 510.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 09:15:00 | 491.73 | 506.58 | 510.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-02-05 11:15:00 | 467.36 | 492.94 | 503.53 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 43 — BUY (started 2024-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 12:15:00 | 455.54 | 448.89 | 448.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-13 13:15:00 | 457.80 | 450.67 | 448.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 13:15:00 | 476.02 | 477.05 | 473.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-20 14:00:00 | 476.02 | 477.05 | 473.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 10:15:00 | 477.99 | 477.14 | 474.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 10:45:00 | 476.17 | 477.14 | 474.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 13:15:00 | 474.87 | 476.97 | 475.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 14:00:00 | 474.87 | 476.97 | 475.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 14:15:00 | 472.76 | 476.13 | 475.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 15:00:00 | 472.76 | 476.13 | 475.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 15:15:00 | 471.76 | 475.26 | 474.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 09:15:00 | 465.42 | 475.26 | 474.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-02-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 09:15:00 | 466.38 | 473.48 | 474.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 09:15:00 | 462.55 | 468.65 | 470.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-01 09:15:00 | 454.06 | 453.40 | 457.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 09:15:00 | 454.06 | 453.40 | 457.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 454.06 | 453.40 | 457.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 11:15:00 | 451.37 | 455.33 | 456.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-05 09:15:00 | 464.90 | 457.06 | 456.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-05 09:15:00 | 464.90 | 457.06 | 456.59 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 14:15:00 | 454.73 | 456.64 | 456.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 448.45 | 454.78 | 455.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 13:15:00 | 454.15 | 451.91 | 453.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 13:15:00 | 454.15 | 451.91 | 453.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 13:15:00 | 454.15 | 451.91 | 453.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 14:00:00 | 454.15 | 451.91 | 453.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 454.01 | 452.33 | 453.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 14:45:00 | 454.63 | 452.33 | 453.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 15:15:00 | 455.26 | 452.92 | 453.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 09:15:00 | 464.13 | 452.92 | 453.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 09:15:00 | 463.94 | 455.12 | 454.87 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 10:15:00 | 456.79 | 458.28 | 458.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 13:15:00 | 454.49 | 456.83 | 457.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 13:15:00 | 439.38 | 438.90 | 444.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 13:45:00 | 440.15 | 438.90 | 444.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 15:15:00 | 443.60 | 440.58 | 444.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 09:15:00 | 450.08 | 440.58 | 444.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 447.87 | 442.04 | 444.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 09:45:00 | 446.53 | 442.04 | 444.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 10:15:00 | 449.41 | 443.51 | 444.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 10:45:00 | 448.45 | 443.51 | 444.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 11:15:00 | 451.80 | 445.17 | 445.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 12:00:00 | 451.80 | 445.17 | 445.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2024-03-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 12:15:00 | 455.21 | 447.18 | 446.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 14:15:00 | 455.83 | 449.81 | 447.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-18 09:15:00 | 446.48 | 450.35 | 448.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 09:15:00 | 446.48 | 450.35 | 448.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 446.48 | 450.35 | 448.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-18 09:45:00 | 447.06 | 450.35 | 448.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 10:15:00 | 446.10 | 449.50 | 448.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-18 11:00:00 | 446.10 | 449.50 | 448.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 14:15:00 | 446.67 | 447.72 | 447.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-18 15:15:00 | 446.67 | 447.72 | 447.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2024-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 15:15:00 | 446.67 | 447.51 | 447.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 09:15:00 | 445.86 | 447.18 | 447.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 440.68 | 438.58 | 441.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 440.68 | 438.58 | 441.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 440.68 | 438.58 | 441.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 11:30:00 | 437.22 | 438.08 | 440.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-22 10:15:00 | 444.18 | 439.30 | 439.78 | SL hit (close>static) qty=1.00 sl=443.12 alert=retest2 |

### Cycle 51 — BUY (started 2024-03-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 11:15:00 | 449.31 | 441.30 | 440.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 12:15:00 | 453.39 | 443.72 | 441.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 11:15:00 | 447.01 | 447.83 | 445.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-26 11:45:00 | 447.49 | 447.83 | 445.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 13:15:00 | 446.91 | 447.54 | 445.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 13:45:00 | 446.29 | 447.54 | 445.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 09:15:00 | 441.78 | 446.01 | 445.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 09:45:00 | 441.78 | 446.01 | 445.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 10:15:00 | 444.61 | 445.73 | 445.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 12:45:00 | 445.47 | 445.27 | 445.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 13:15:00 | 445.09 | 445.27 | 445.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-27 14:15:00 | 435.78 | 443.29 | 444.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2024-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 14:15:00 | 435.78 | 443.29 | 444.23 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 13:15:00 | 442.35 | 441.27 | 441.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 14:15:00 | 447.25 | 442.47 | 441.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 09:15:00 | 455.30 | 456.47 | 453.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-04 09:45:00 | 456.31 | 456.47 | 453.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 13:15:00 | 466.10 | 471.28 | 468.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-09 09:15:00 | 472.57 | 470.40 | 468.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-09 12:45:00 | 472.52 | 471.91 | 470.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-10 10:00:00 | 472.52 | 471.01 | 470.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 12:15:00 | 472.71 | 477.66 | 477.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 12:15:00 | 472.71 | 477.66 | 477.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 13:15:00 | 470.99 | 476.32 | 477.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 11:15:00 | 469.50 | 467.91 | 470.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 11:15:00 | 469.50 | 467.91 | 470.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 11:15:00 | 469.50 | 467.91 | 470.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 11:45:00 | 469.50 | 467.91 | 470.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 12:15:00 | 470.17 | 468.36 | 470.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 12:30:00 | 472.09 | 468.36 | 470.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 13:15:00 | 466.24 | 467.94 | 470.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 14:30:00 | 461.64 | 465.96 | 469.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 10:15:00 | 471.56 | 467.14 | 466.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2024-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 10:15:00 | 471.56 | 467.14 | 466.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 476.22 | 471.99 | 469.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 15:15:00 | 474.20 | 474.62 | 472.34 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 09:15:00 | 478.57 | 474.62 | 472.34 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 481.44 | 478.92 | 476.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 13:45:00 | 484.99 | 481.70 | 478.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 10:30:00 | 484.75 | 484.56 | 481.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-30 14:15:00 | 486.29 | 488.14 | 486.74 | SL hit (close<ema400) qty=1.00 sl=486.74 alert=retest1 |

### Cycle 56 — SELL (started 2024-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 10:15:00 | 481.92 | 485.91 | 486.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 14:15:00 | 477.51 | 482.73 | 484.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 460.10 | 458.41 | 463.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 10:00:00 | 460.10 | 458.41 | 463.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 460.10 | 458.45 | 461.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-09 09:30:00 | 461.88 | 458.45 | 461.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 10:15:00 | 456.60 | 458.08 | 460.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 11:15:00 | 454.63 | 458.08 | 460.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 12:00:00 | 455.16 | 457.49 | 460.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 12:30:00 | 455.26 | 456.52 | 459.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-10 11:15:00 | 461.35 | 455.26 | 456.99 | SL hit (close>static) qty=1.00 sl=460.87 alert=retest2 |

### Cycle 57 — BUY (started 2024-05-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 13:15:00 | 469.26 | 458.71 | 458.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-10 14:15:00 | 480.92 | 463.15 | 460.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 09:15:00 | 484.66 | 494.67 | 482.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 09:15:00 | 484.66 | 494.67 | 482.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 484.66 | 494.67 | 482.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:00:00 | 484.66 | 494.67 | 482.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 11:15:00 | 484.27 | 491.18 | 482.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 11:45:00 | 485.47 | 491.18 | 482.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 489.45 | 491.56 | 489.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 493.05 | 489.55 | 488.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 11:00:00 | 490.89 | 490.16 | 489.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 14:30:00 | 490.84 | 490.70 | 489.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:15:00 | 491.37 | 490.60 | 489.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 491.47 | 490.78 | 490.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:45:00 | 491.47 | 490.78 | 490.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 11:15:00 | 491.13 | 490.85 | 490.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:30:00 | 491.56 | 490.85 | 490.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 490.65 | 490.81 | 490.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:15:00 | 492.19 | 490.81 | 490.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 490.94 | 490.83 | 490.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 495.16 | 491.24 | 490.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 10:00:00 | 492.91 | 491.57 | 490.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 12:15:00 | 488.40 | 491.93 | 492.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 12:15:00 | 488.40 | 491.93 | 492.08 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 09:15:00 | 498.76 | 492.53 | 492.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 09:15:00 | 505.95 | 497.51 | 495.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 09:15:00 | 502.12 | 502.89 | 499.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-28 10:00:00 | 502.12 | 502.89 | 499.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 499.96 | 502.13 | 499.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:45:00 | 499.96 | 502.13 | 499.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 498.71 | 501.45 | 499.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:30:00 | 496.94 | 501.45 | 499.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 496.26 | 500.41 | 499.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:00:00 | 496.26 | 500.41 | 499.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2024-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 15:15:00 | 496.17 | 498.88 | 498.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 492.14 | 496.57 | 497.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 490.80 | 489.65 | 492.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 14:00:00 | 490.80 | 489.65 | 492.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 504.08 | 492.31 | 492.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:00:00 | 504.08 | 492.31 | 492.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 504.23 | 494.69 | 493.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 11:15:00 | 508.40 | 497.44 | 495.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 493.63 | 501.13 | 498.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 493.63 | 501.13 | 498.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 493.63 | 501.13 | 498.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 487.20 | 501.13 | 498.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 474.73 | 495.85 | 496.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 473.43 | 491.37 | 494.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 490.03 | 485.42 | 489.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 490.03 | 485.42 | 489.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 490.03 | 485.42 | 489.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 490.03 | 485.42 | 489.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 500.92 | 488.52 | 490.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 500.92 | 488.52 | 490.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 502.84 | 491.38 | 491.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:00:00 | 502.84 | 491.38 | 491.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 502.55 | 493.62 | 492.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 13:15:00 | 503.75 | 495.64 | 493.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 512.86 | 515.58 | 511.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 09:15:00 | 512.86 | 515.58 | 511.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 512.86 | 515.58 | 511.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:00:00 | 512.86 | 515.58 | 511.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 513.39 | 515.14 | 511.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:30:00 | 511.13 | 515.14 | 511.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 533.82 | 529.46 | 526.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:15:00 | 537.18 | 529.46 | 526.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 09:15:00 | 538.28 | 532.54 | 531.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 11:45:00 | 535.21 | 533.88 | 532.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 14:30:00 | 534.78 | 534.32 | 532.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 539.72 | 535.38 | 533.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 10:30:00 | 544.56 | 537.23 | 534.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 10:00:00 | 543.79 | 544.03 | 539.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 11:00:00 | 543.60 | 543.94 | 540.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 15:00:00 | 543.41 | 543.67 | 541.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 535.59 | 541.81 | 540.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 530.51 | 541.81 | 540.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 547.25 | 547.26 | 544.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:30:00 | 546.34 | 547.26 | 544.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 544.32 | 546.67 | 545.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 10:30:00 | 549.36 | 547.07 | 545.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 13:30:00 | 548.50 | 547.65 | 546.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 15:00:00 | 548.59 | 547.84 | 546.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 09:45:00 | 548.97 | 547.85 | 546.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 546.96 | 547.67 | 546.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:45:00 | 545.81 | 547.67 | 546.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 545.33 | 547.20 | 546.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:00:00 | 545.33 | 547.20 | 546.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-27 12:15:00 | 541.54 | 546.07 | 546.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 12:15:00 | 541.54 | 546.07 | 546.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 539.00 | 544.66 | 545.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 15:15:00 | 544.85 | 544.62 | 545.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 15:15:00 | 544.85 | 544.62 | 545.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 544.85 | 544.62 | 545.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 548.26 | 544.62 | 545.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 546.29 | 544.95 | 545.49 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-06-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 13:15:00 | 547.54 | 545.70 | 545.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 548.69 | 546.68 | 546.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 13:15:00 | 547.49 | 548.00 | 547.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-01 14:00:00 | 547.49 | 548.00 | 547.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 545.33 | 548.97 | 548.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 545.33 | 548.97 | 548.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 540.77 | 547.33 | 547.41 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 11:15:00 | 551.56 | 547.67 | 547.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 11:15:00 | 555.40 | 549.79 | 548.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 13:15:00 | 549.60 | 550.03 | 548.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 13:15:00 | 549.60 | 550.03 | 548.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 549.60 | 550.03 | 548.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 14:00:00 | 549.60 | 550.03 | 548.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 547.06 | 549.44 | 548.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 547.06 | 549.44 | 548.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 547.25 | 549.00 | 548.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 546.58 | 549.00 | 548.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 548.16 | 548.75 | 548.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 11:45:00 | 549.41 | 548.83 | 548.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 12:30:00 | 549.89 | 548.83 | 548.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 15:15:00 | 549.45 | 548.81 | 548.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 09:15:00 | 546.91 | 548.53 | 548.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 09:15:00 | 546.91 | 548.53 | 548.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 10:15:00 | 544.03 | 547.63 | 548.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 15:15:00 | 545.62 | 544.83 | 546.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 09:15:00 | 545.09 | 544.83 | 546.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 544.32 | 544.73 | 546.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:45:00 | 547.25 | 544.73 | 546.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 546.91 | 545.16 | 546.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:00:00 | 546.91 | 545.16 | 546.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 545.09 | 545.15 | 546.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:30:00 | 546.82 | 545.15 | 546.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 532.19 | 540.89 | 543.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 10:15:00 | 529.50 | 540.89 | 543.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 12:15:00 | 530.41 | 537.73 | 541.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 12:15:00 | 541.88 | 539.31 | 539.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 12:15:00 | 541.88 | 539.31 | 539.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 543.27 | 541.39 | 540.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 10:15:00 | 540.63 | 541.24 | 540.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 10:15:00 | 540.63 | 541.24 | 540.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 540.63 | 541.24 | 540.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 11:00:00 | 540.63 | 541.24 | 540.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 539.67 | 540.93 | 540.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:00:00 | 539.67 | 540.93 | 540.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 538.14 | 540.37 | 540.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 13:00:00 | 538.14 | 540.37 | 540.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2024-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 13:15:00 | 535.55 | 539.41 | 539.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 528.73 | 534.91 | 536.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 526.05 | 525.78 | 530.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 526.05 | 525.78 | 530.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 531.85 | 527.00 | 530.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 530.32 | 527.00 | 530.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 529.45 | 527.49 | 530.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:30:00 | 531.95 | 527.49 | 530.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 527.20 | 527.43 | 530.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 15:00:00 | 522.98 | 526.33 | 529.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:15:00 | 522.12 | 525.67 | 528.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 12:15:00 | 521.30 | 516.40 | 516.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2024-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 12:15:00 | 521.30 | 516.40 | 516.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 522.40 | 517.60 | 516.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 10:15:00 | 544.27 | 546.19 | 541.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 11:00:00 | 544.27 | 546.19 | 541.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 540.53 | 545.05 | 541.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 540.53 | 545.05 | 541.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 542.36 | 544.52 | 541.58 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 530.89 | 539.21 | 539.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 14:15:00 | 514.78 | 532.97 | 536.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 523.89 | 515.38 | 522.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 523.89 | 515.38 | 522.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 523.89 | 515.38 | 522.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 523.89 | 515.38 | 522.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 524.75 | 517.26 | 522.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 11:00:00 | 524.75 | 517.26 | 522.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 11:15:00 | 523.60 | 518.52 | 522.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:15:00 | 522.79 | 518.52 | 522.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 15:15:00 | 526.14 | 520.82 | 520.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 526.14 | 520.82 | 520.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 09:15:00 | 527.15 | 522.09 | 521.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 524.27 | 524.80 | 523.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 524.27 | 524.80 | 523.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 524.99 | 530.03 | 527.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 12:30:00 | 536.36 | 531.54 | 529.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 09:15:00 | 523.89 | 531.01 | 531.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 523.89 | 531.01 | 531.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 10:15:00 | 521.16 | 529.04 | 530.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 11:15:00 | 523.70 | 522.76 | 525.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 12:00:00 | 523.70 | 522.76 | 525.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 522.26 | 522.66 | 525.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:30:00 | 525.09 | 522.66 | 525.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 528.11 | 523.75 | 525.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:00:00 | 528.11 | 523.75 | 525.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 530.85 | 525.17 | 526.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 15:00:00 | 530.85 | 525.17 | 526.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 533.91 | 527.70 | 527.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 14:15:00 | 538.76 | 533.12 | 530.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 550.13 | 553.19 | 548.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 550.13 | 553.19 | 548.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 550.13 | 553.19 | 548.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 550.89 | 553.19 | 548.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 549.98 | 552.16 | 549.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 549.98 | 552.16 | 549.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 549.55 | 551.64 | 549.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 551.52 | 551.64 | 549.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 12:00:00 | 551.09 | 551.06 | 550.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 12:30:00 | 551.76 | 551.18 | 550.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 10:15:00 | 551.28 | 554.30 | 554.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 551.28 | 554.30 | 554.56 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 559.38 | 554.57 | 554.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 10:15:00 | 568.54 | 557.37 | 555.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 11:15:00 | 577.94 | 578.04 | 574.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 11:30:00 | 577.46 | 578.04 | 574.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 588.74 | 589.95 | 584.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 588.74 | 589.95 | 584.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 589.55 | 589.87 | 585.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:30:00 | 588.11 | 589.87 | 585.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 584.80 | 590.21 | 587.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:45:00 | 583.51 | 590.21 | 587.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 585.91 | 589.35 | 586.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 581.73 | 589.35 | 586.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 12:15:00 | 580.77 | 584.90 | 585.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 13:15:00 | 576.89 | 583.29 | 584.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 588.74 | 583.37 | 584.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 588.74 | 583.37 | 584.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 588.74 | 583.37 | 584.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:45:00 | 587.78 | 583.37 | 584.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 592.04 | 585.11 | 584.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 14:15:00 | 594.44 | 589.15 | 587.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 10:15:00 | 589.41 | 590.17 | 588.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 10:45:00 | 589.98 | 590.17 | 588.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 591.71 | 590.48 | 588.46 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 09:15:00 | 586.34 | 587.68 | 587.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 11:15:00 | 585.23 | 586.93 | 587.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 12:15:00 | 587.10 | 586.97 | 587.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 12:15:00 | 587.10 | 586.97 | 587.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 587.10 | 586.97 | 587.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 12:30:00 | 587.92 | 586.97 | 587.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 589.45 | 587.46 | 587.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:45:00 | 588.98 | 587.46 | 587.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 14:15:00 | 589.93 | 587.96 | 587.73 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-09-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 15:15:00 | 586.19 | 587.83 | 587.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 11:15:00 | 582.50 | 586.45 | 587.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 12:15:00 | 587.34 | 586.63 | 587.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 12:15:00 | 587.34 | 586.63 | 587.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 587.34 | 586.63 | 587.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:45:00 | 586.58 | 586.63 | 587.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 13:15:00 | 588.40 | 586.98 | 587.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:00:00 | 588.40 | 586.98 | 587.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 589.31 | 587.45 | 587.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 15:00:00 | 589.31 | 587.45 | 587.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2024-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 15:15:00 | 588.50 | 587.66 | 587.64 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 585.33 | 587.19 | 587.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 10:15:00 | 582.21 | 585.56 | 586.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 571.71 | 570.77 | 576.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 571.71 | 570.77 | 576.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 573.10 | 571.16 | 575.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:15:00 | 568.11 | 571.81 | 574.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:15:00 | 566.72 | 571.38 | 574.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 10:15:00 | 581.25 | 572.02 | 571.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 10:15:00 | 581.25 | 572.02 | 571.61 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 10:15:00 | 570.85 | 574.41 | 574.65 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 15:15:00 | 577.85 | 574.64 | 574.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 09:15:00 | 582.21 | 576.16 | 575.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 11:15:00 | 588.35 | 593.53 | 590.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 11:15:00 | 588.35 | 593.53 | 590.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 588.35 | 593.53 | 590.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 12:00:00 | 588.35 | 593.53 | 590.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 584.23 | 591.67 | 590.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:00:00 | 584.23 | 591.67 | 590.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 580.34 | 589.40 | 589.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:00:00 | 580.34 | 589.40 | 589.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2024-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 14:15:00 | 580.68 | 587.66 | 588.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 573.92 | 581.58 | 584.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 10:15:00 | 558.04 | 556.70 | 562.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-09 10:30:00 | 556.98 | 556.70 | 562.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 567.44 | 558.12 | 560.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:45:00 | 566.24 | 558.12 | 560.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 563.46 | 559.19 | 560.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:30:00 | 561.64 | 560.21 | 561.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 12:45:00 | 561.64 | 560.61 | 561.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 10:15:00 | 561.92 | 561.18 | 561.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:15:00 | 533.56 | 542.22 | 546.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:15:00 | 533.56 | 542.22 | 546.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:15:00 | 533.82 | 542.22 | 546.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-18 11:15:00 | 533.15 | 532.97 | 538.78 | SL hit (close>ema200) qty=0.50 sl=532.97 alert=retest2 |

### Cycle 89 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 513.34 | 510.17 | 509.98 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 504.51 | 509.20 | 509.62 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 15:15:00 | 513.00 | 509.85 | 509.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 516.46 | 511.17 | 510.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 524.37 | 529.17 | 524.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 524.37 | 529.17 | 524.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 524.37 | 529.17 | 524.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 524.37 | 529.17 | 524.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 523.70 | 528.07 | 524.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 523.70 | 528.07 | 524.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 529.50 | 528.36 | 524.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 15:00:00 | 530.65 | 528.79 | 525.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:15:00 | 533.77 | 528.92 | 526.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 12:30:00 | 530.32 | 530.10 | 527.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 14:15:00 | 535.69 | 538.89 | 538.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 14:15:00 | 535.69 | 538.89 | 538.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 15:15:00 | 532.48 | 537.61 | 538.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 520.49 | 513.40 | 522.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 520.49 | 513.40 | 522.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 520.49 | 513.40 | 522.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 09:15:00 | 500.34 | 511.03 | 517.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 12:15:00 | 500.25 | 498.65 | 504.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 498.52 | 501.81 | 504.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 12:15:00 | 513.43 | 506.08 | 505.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 12:15:00 | 513.43 | 506.08 | 505.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 517.32 | 511.31 | 508.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 09:15:00 | 530.61 | 531.53 | 525.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-22 09:30:00 | 530.08 | 531.53 | 525.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 529.12 | 531.05 | 525.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:30:00 | 524.75 | 531.05 | 525.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 558.00 | 548.78 | 541.81 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 09:15:00 | 539.55 | 547.27 | 547.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 11:15:00 | 538.95 | 544.52 | 546.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 13:15:00 | 544.70 | 544.18 | 545.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 13:15:00 | 544.70 | 544.18 | 545.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 544.70 | 544.18 | 545.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:00:00 | 544.70 | 544.18 | 545.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 544.95 | 544.33 | 545.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 15:00:00 | 544.95 | 544.33 | 545.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 544.55 | 544.38 | 545.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 542.15 | 544.38 | 545.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 547.60 | 545.02 | 545.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:00:00 | 547.60 | 545.02 | 545.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 548.55 | 545.73 | 545.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:15:00 | 550.00 | 545.73 | 545.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 11:15:00 | 554.25 | 547.43 | 546.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 10:15:00 | 556.00 | 552.76 | 550.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 555.25 | 564.43 | 561.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 09:15:00 | 555.25 | 564.43 | 561.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 555.25 | 564.43 | 561.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:00:00 | 555.25 | 564.43 | 561.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 555.25 | 562.59 | 560.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:15:00 | 553.85 | 562.59 | 560.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 557.90 | 559.62 | 559.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 15:00:00 | 557.90 | 559.62 | 559.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 559.60 | 559.57 | 559.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 10:45:00 | 565.30 | 560.38 | 559.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 09:15:00 | 552.35 | 559.89 | 560.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 552.35 | 559.89 | 560.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 13:15:00 | 550.60 | 554.14 | 556.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 563.90 | 555.10 | 556.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 563.90 | 555.10 | 556.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 563.90 | 555.10 | 556.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:00:00 | 563.90 | 555.10 | 556.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 562.35 | 556.55 | 556.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:30:00 | 565.00 | 556.55 | 556.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 11:15:00 | 560.00 | 557.24 | 556.93 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 14:15:00 | 552.00 | 555.84 | 556.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 545.50 | 553.44 | 555.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 545.25 | 544.32 | 547.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 14:00:00 | 545.25 | 544.32 | 547.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 550.45 | 545.55 | 548.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 550.45 | 545.55 | 548.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 549.65 | 546.37 | 548.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 547.10 | 546.37 | 548.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 547.35 | 546.57 | 547.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:30:00 | 547.55 | 546.57 | 547.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 549.15 | 547.09 | 547.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 14:00:00 | 549.15 | 547.09 | 547.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 548.30 | 547.33 | 547.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 15:15:00 | 547.00 | 547.33 | 547.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 09:15:00 | 550.60 | 547.93 | 548.07 | SL hit (close>static) qty=1.00 sl=549.40 alert=retest2 |

### Cycle 99 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 505.20 | 502.00 | 501.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 11:15:00 | 507.65 | 503.13 | 502.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 12:15:00 | 524.85 | 525.00 | 518.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 12:45:00 | 525.55 | 525.00 | 518.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 519.25 | 523.85 | 518.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:00:00 | 519.25 | 523.85 | 518.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 520.70 | 523.22 | 518.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:30:00 | 516.65 | 523.22 | 518.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 539.65 | 526.02 | 520.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 10:15:00 | 542.05 | 526.02 | 520.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 09:30:00 | 540.30 | 539.97 | 532.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 15:00:00 | 540.30 | 537.74 | 533.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 10:30:00 | 540.65 | 543.38 | 540.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 545.70 | 546.88 | 544.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-13 10:15:00 | 546.95 | 546.88 | 544.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-13 11:15:00 | 537.05 | 544.01 | 543.15 | SL hit (close<static) qty=1.00 sl=538.30 alert=retest2 |

### Cycle 100 — SELL (started 2025-01-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 13:15:00 | 535.35 | 541.23 | 541.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 14:15:00 | 533.35 | 539.66 | 541.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 540.65 | 538.71 | 540.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 10:15:00 | 540.65 | 538.71 | 540.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 540.65 | 538.71 | 540.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 540.65 | 538.71 | 540.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 538.95 | 538.76 | 540.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:15:00 | 537.40 | 538.76 | 540.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 13:15:00 | 542.25 | 539.15 | 540.06 | SL hit (close>static) qty=1.00 sl=541.50 alert=retest2 |

### Cycle 101 — BUY (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 09:15:00 | 546.60 | 541.60 | 541.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 10:15:00 | 550.25 | 543.33 | 541.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 15:15:00 | 545.30 | 545.94 | 544.00 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 09:15:00 | 550.60 | 545.94 | 544.00 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 12:15:00 | 545.80 | 547.35 | 545.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 13:00:00 | 545.80 | 547.35 | 545.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 13:15:00 | 548.50 | 547.58 | 545.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 09:15:00 | 550.00 | 547.11 | 545.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 09:15:00 | 547.65 | 549.93 | 548.45 | SL hit (close<ema400) qty=1.00 sl=548.45 alert=retest1 |

### Cycle 102 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 544.30 | 548.73 | 549.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 538.55 | 546.17 | 547.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 543.40 | 540.37 | 543.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 543.40 | 540.37 | 543.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 543.40 | 540.37 | 543.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 543.40 | 540.37 | 543.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 553.00 | 543.09 | 544.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 553.00 | 543.09 | 544.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 554.50 | 545.37 | 545.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 555.05 | 545.37 | 545.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 11:15:00 | 555.00 | 547.30 | 546.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 12:15:00 | 556.75 | 549.19 | 547.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 12:15:00 | 554.10 | 554.44 | 551.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 13:00:00 | 554.10 | 554.44 | 551.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 551.50 | 553.64 | 551.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:30:00 | 552.00 | 553.64 | 551.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 550.40 | 552.99 | 551.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:15:00 | 544.75 | 552.99 | 551.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 538.95 | 550.19 | 550.38 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 549.15 | 545.46 | 545.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 553.40 | 547.87 | 546.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 15:15:00 | 564.00 | 565.67 | 558.73 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:15:00 | 570.95 | 565.67 | 558.73 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 14:15:00 | 599.50 | 580.74 | 569.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-02-01 15:15:00 | 599.60 | 601.31 | 589.08 | SL hit (close<ema200) qty=0.50 sl=601.31 alert=retest1 |

### Cycle 106 — SELL (started 2025-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 12:15:00 | 633.70 | 637.97 | 638.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 621.95 | 633.25 | 635.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 612.95 | 612.34 | 620.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:00:00 | 612.95 | 612.34 | 620.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 619.90 | 615.09 | 619.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 619.90 | 615.09 | 619.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 621.90 | 616.45 | 620.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:15:00 | 616.40 | 616.45 | 620.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 622.10 | 617.58 | 620.34 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 636.20 | 623.73 | 622.81 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 617.20 | 623.42 | 623.63 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 13:15:00 | 626.20 | 621.71 | 621.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-17 14:15:00 | 633.40 | 624.05 | 622.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-18 12:15:00 | 626.60 | 627.44 | 625.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-18 12:45:00 | 628.20 | 627.44 | 625.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 642.50 | 631.43 | 627.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-19 10:15:00 | 643.70 | 631.43 | 627.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 10:45:00 | 645.00 | 645.56 | 642.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 15:00:00 | 648.40 | 644.95 | 643.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:30:00 | 644.45 | 644.50 | 643.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 645.40 | 644.68 | 643.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:15:00 | 644.90 | 644.68 | 643.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 645.85 | 644.91 | 643.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:45:00 | 643.00 | 644.91 | 643.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 645.20 | 644.97 | 643.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 651.60 | 645.18 | 644.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 10:15:00 | 641.95 | 646.26 | 646.04 | SL hit (close<static) qty=1.00 sl=643.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 11:15:00 | 639.45 | 644.89 | 645.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 12:15:00 | 636.60 | 643.24 | 644.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 642.55 | 642.06 | 643.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 14:15:00 | 642.55 | 642.06 | 643.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 642.55 | 642.06 | 643.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 15:00:00 | 642.55 | 642.06 | 643.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 640.60 | 641.77 | 643.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 625.00 | 641.77 | 643.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 13:15:00 | 629.25 | 622.84 | 622.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 629.25 | 622.84 | 622.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 632.50 | 626.20 | 624.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 13:15:00 | 625.65 | 627.25 | 625.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 13:15:00 | 625.65 | 627.25 | 625.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 13:15:00 | 625.65 | 627.25 | 625.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 13:30:00 | 626.25 | 627.25 | 625.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 624.70 | 626.74 | 625.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 14:30:00 | 624.05 | 626.74 | 625.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 624.45 | 626.28 | 625.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:15:00 | 625.00 | 626.28 | 625.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 629.25 | 631.63 | 629.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:00:00 | 629.25 | 631.63 | 629.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 631.50 | 631.60 | 629.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 630.25 | 631.60 | 629.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 628.75 | 631.03 | 629.36 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 619.30 | 627.31 | 627.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 12:15:00 | 619.15 | 625.67 | 627.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 612.70 | 609.27 | 613.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 612.70 | 609.27 | 613.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 612.70 | 609.27 | 613.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:00:00 | 612.70 | 609.27 | 613.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 608.45 | 609.11 | 612.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:15:00 | 606.70 | 609.11 | 612.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:00:00 | 606.50 | 607.63 | 611.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 10:15:00 | 616.05 | 608.97 | 610.43 | SL hit (close>static) qty=1.00 sl=614.30 alert=retest2 |

### Cycle 113 — BUY (started 2025-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 12:15:00 | 620.90 | 613.25 | 612.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 13:15:00 | 625.85 | 615.77 | 613.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 14:15:00 | 656.85 | 657.31 | 649.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 15:00:00 | 656.85 | 657.31 | 649.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 656.50 | 659.16 | 655.57 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 11:15:00 | 652.55 | 654.04 | 654.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 650.50 | 653.17 | 653.75 | Break + close below crossover candle low |

### Cycle 115 — BUY (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 09:15:00 | 659.40 | 654.42 | 654.27 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 10:15:00 | 652.20 | 654.42 | 654.57 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 659.50 | 655.44 | 655.02 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 12:15:00 | 630.25 | 650.40 | 652.77 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 654.10 | 643.22 | 643.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 655.05 | 647.29 | 645.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 637.90 | 646.40 | 645.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 637.90 | 646.40 | 645.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 637.90 | 646.40 | 645.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:00:00 | 637.90 | 646.40 | 645.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 643.65 | 645.85 | 645.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 11:45:00 | 646.10 | 645.96 | 645.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 10:30:00 | 648.05 | 647.95 | 646.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 13:15:00 | 640.45 | 645.11 | 645.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 13:15:00 | 640.45 | 645.11 | 645.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 14:15:00 | 637.80 | 643.65 | 644.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 617.00 | 616.83 | 625.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 617.00 | 616.83 | 625.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 630.75 | 616.18 | 617.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:45:00 | 628.95 | 616.18 | 617.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 639.60 | 620.87 | 619.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 643.05 | 625.30 | 622.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 651.85 | 656.52 | 650.15 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 10:30:00 | 664.15 | 657.80 | 651.32 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 12:30:00 | 660.75 | 658.31 | 652.70 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:15:00 | 661.20 | 657.30 | 653.58 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 663.30 | 658.50 | 654.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 14:30:00 | 668.65 | 663.44 | 658.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 671.20 | 671.31 | 666.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 666.40 | 670.33 | 666.51 | SL hit (close<ema400) qty=1.00 sl=666.51 alert=retest1 |

### Cycle 122 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 662.80 | 673.55 | 674.04 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 677.35 | 672.16 | 672.07 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 12:15:00 | 668.45 | 672.51 | 672.76 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 10:15:00 | 675.75 | 672.92 | 672.70 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 671.00 | 672.64 | 672.68 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 686.00 | 674.94 | 673.70 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 11:15:00 | 675.70 | 682.50 | 682.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 12:15:00 | 670.00 | 680.00 | 681.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 11:15:00 | 664.05 | 662.73 | 670.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 12:00:00 | 664.05 | 662.73 | 670.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 669.95 | 664.75 | 670.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 14:00:00 | 669.95 | 664.75 | 670.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 673.65 | 666.53 | 670.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:00:00 | 673.65 | 666.53 | 670.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 681.95 | 669.61 | 671.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 692.55 | 669.61 | 671.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 689.70 | 673.63 | 673.16 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 09:15:00 | 646.70 | 672.08 | 674.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 12:15:00 | 642.80 | 658.44 | 666.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 09:15:00 | 645.10 | 641.18 | 649.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-15 09:45:00 | 645.45 | 641.18 | 649.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 642.65 | 642.92 | 646.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:30:00 | 644.55 | 642.92 | 646.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 643.95 | 642.52 | 644.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 15:00:00 | 643.95 | 642.52 | 644.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 642.80 | 642.58 | 644.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:15:00 | 646.50 | 642.58 | 644.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 653.15 | 644.69 | 645.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:00:00 | 653.15 | 644.69 | 645.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 650.40 | 645.83 | 645.74 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-05-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 15:15:00 | 644.90 | 645.76 | 645.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 09:15:00 | 640.20 | 644.65 | 645.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 12:15:00 | 638.05 | 634.37 | 637.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 12:15:00 | 638.05 | 634.37 | 637.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 638.05 | 634.37 | 637.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:30:00 | 640.00 | 634.37 | 637.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 638.00 | 635.10 | 637.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:45:00 | 635.80 | 636.09 | 637.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 10:15:00 | 637.60 | 632.95 | 632.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 637.60 | 632.95 | 632.40 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 15:15:00 | 630.80 | 632.36 | 632.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 09:15:00 | 627.10 | 631.31 | 631.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 12:15:00 | 630.55 | 630.52 | 631.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 13:00:00 | 630.55 | 630.52 | 631.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 630.05 | 630.43 | 631.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:45:00 | 629.25 | 630.43 | 631.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 631.30 | 630.60 | 631.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 631.30 | 630.60 | 631.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 630.10 | 630.50 | 631.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 627.35 | 630.50 | 631.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 625.05 | 629.41 | 630.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:15:00 | 623.05 | 629.41 | 630.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 11:15:00 | 633.55 | 629.53 | 629.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 11:15:00 | 633.55 | 629.53 | 629.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 12:15:00 | 635.20 | 630.67 | 629.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 632.95 | 633.65 | 632.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 13:00:00 | 632.95 | 633.65 | 632.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 644.25 | 644.73 | 642.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 14:00:00 | 644.80 | 644.75 | 642.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 11:15:00 | 639.00 | 642.21 | 642.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 11:15:00 | 639.00 | 642.21 | 642.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 09:15:00 | 637.20 | 639.75 | 640.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 640.75 | 636.99 | 638.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 640.75 | 636.99 | 638.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 640.75 | 636.99 | 638.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 641.30 | 636.99 | 638.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 636.80 | 636.95 | 638.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 12:00:00 | 635.45 | 636.65 | 637.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 13:00:00 | 635.85 | 636.49 | 637.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 11:15:00 | 639.75 | 635.38 | 635.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 639.75 | 635.38 | 635.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 13:15:00 | 643.10 | 637.66 | 636.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 14:15:00 | 645.80 | 646.11 | 642.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 15:00:00 | 645.80 | 646.11 | 642.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 643.30 | 645.71 | 643.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:45:00 | 643.95 | 645.71 | 643.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 641.90 | 644.95 | 642.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:00:00 | 641.90 | 644.95 | 642.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 639.80 | 643.92 | 642.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 637.85 | 643.92 | 642.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 638.10 | 642.75 | 642.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:45:00 | 637.90 | 642.75 | 642.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2025-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 14:15:00 | 637.90 | 641.78 | 641.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 633.75 | 639.48 | 640.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 635.50 | 633.59 | 636.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 11:00:00 | 635.50 | 633.59 | 636.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 632.00 | 632.29 | 634.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:30:00 | 635.00 | 633.00 | 634.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 637.30 | 633.86 | 634.87 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 639.25 | 635.57 | 635.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 13:15:00 | 639.75 | 636.40 | 635.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 633.00 | 645.43 | 643.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 633.00 | 645.43 | 643.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 633.00 | 645.43 | 643.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 633.00 | 645.43 | 643.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 632.10 | 642.77 | 642.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:45:00 | 630.00 | 642.77 | 642.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 634.40 | 641.09 | 641.92 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 645.75 | 641.51 | 641.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 15:15:00 | 646.20 | 643.53 | 642.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 681.80 | 683.07 | 676.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:45:00 | 682.10 | 683.07 | 676.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 679.45 | 680.74 | 678.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:00:00 | 679.45 | 680.74 | 678.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 677.20 | 680.03 | 678.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:00:00 | 677.20 | 680.03 | 678.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 680.90 | 680.20 | 678.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:45:00 | 677.90 | 680.20 | 678.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 677.75 | 679.68 | 678.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 677.75 | 679.68 | 678.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 672.00 | 678.14 | 678.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 11:15:00 | 670.55 | 676.63 | 677.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 674.35 | 674.31 | 676.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 15:00:00 | 674.35 | 674.31 | 676.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 656.00 | 655.52 | 659.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 10:45:00 | 652.20 | 654.63 | 659.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 15:15:00 | 661.20 | 656.51 | 658.15 | SL hit (close>static) qty=1.00 sl=660.70 alert=retest2 |

### Cycle 143 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 671.95 | 661.68 | 660.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 679.00 | 672.45 | 669.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 10:15:00 | 687.10 | 687.39 | 680.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 11:00:00 | 687.10 | 687.39 | 680.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 694.85 | 689.21 | 684.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:15:00 | 697.30 | 689.21 | 684.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 11:15:00 | 719.45 | 724.10 | 724.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 11:15:00 | 719.45 | 724.10 | 724.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 10:15:00 | 714.30 | 720.62 | 722.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 10:15:00 | 712.25 | 711.30 | 715.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-01 10:30:00 | 710.90 | 711.30 | 715.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 711.30 | 694.46 | 704.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 711.30 | 694.46 | 704.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 708.00 | 697.17 | 704.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 12:00:00 | 705.25 | 698.79 | 704.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 13:15:00 | 716.15 | 704.26 | 706.22 | SL hit (close>static) qty=1.00 sl=712.40 alert=retest2 |

### Cycle 145 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 711.35 | 707.22 | 707.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 12:15:00 | 714.15 | 709.31 | 708.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 711.40 | 713.97 | 711.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 711.40 | 713.97 | 711.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 711.40 | 713.97 | 711.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 711.40 | 713.97 | 711.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 712.45 | 713.67 | 711.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 712.45 | 713.67 | 711.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 709.05 | 712.75 | 711.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:00:00 | 709.05 | 712.75 | 711.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 706.65 | 711.53 | 710.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:30:00 | 705.70 | 711.53 | 710.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 707.15 | 710.65 | 710.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:00:00 | 707.15 | 710.65 | 710.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 14:15:00 | 704.70 | 709.46 | 709.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 700.30 | 707.00 | 708.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 706.05 | 700.91 | 704.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 706.05 | 700.91 | 704.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 706.05 | 700.91 | 704.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 706.05 | 700.91 | 704.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 704.55 | 701.63 | 704.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 702.10 | 701.63 | 704.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 697.35 | 689.51 | 688.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 697.35 | 689.51 | 688.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 702.80 | 693.67 | 691.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 712.95 | 713.17 | 708.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:00:00 | 712.95 | 713.17 | 708.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 709.80 | 712.02 | 709.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:30:00 | 708.80 | 712.02 | 709.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 713.60 | 712.34 | 710.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:45:00 | 710.30 | 712.34 | 710.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 734.00 | 718.68 | 714.24 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 715.65 | 721.52 | 722.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 704.80 | 717.29 | 720.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 12:15:00 | 716.60 | 715.41 | 718.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 13:00:00 | 716.60 | 715.41 | 718.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 719.50 | 716.23 | 718.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:00:00 | 715.60 | 716.10 | 718.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 730.00 | 718.39 | 718.82 | SL hit (close>static) qty=1.00 sl=722.20 alert=retest2 |

### Cycle 149 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 726.95 | 720.11 | 719.56 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 11:15:00 | 711.60 | 719.65 | 720.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 10:15:00 | 706.40 | 713.55 | 715.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 696.80 | 692.88 | 699.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 696.80 | 692.88 | 699.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 696.80 | 692.88 | 699.38 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 15:15:00 | 705.00 | 701.60 | 701.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 715.40 | 704.36 | 702.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 11:15:00 | 702.85 | 704.76 | 703.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 11:15:00 | 702.85 | 704.76 | 703.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 702.85 | 704.76 | 703.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 702.85 | 704.76 | 703.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 701.00 | 704.01 | 703.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:00:00 | 701.00 | 704.01 | 703.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 699.25 | 703.06 | 702.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:00:00 | 699.25 | 703.06 | 702.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 704.85 | 703.73 | 703.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 699.55 | 703.73 | 703.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 706.10 | 704.20 | 703.40 | EMA400 retest candle locked (from upside) |

### Cycle 152 — SELL (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 11:15:00 | 694.75 | 701.99 | 702.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 12:15:00 | 693.30 | 700.25 | 701.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 12:15:00 | 698.00 | 697.43 | 699.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 13:00:00 | 698.00 | 697.43 | 699.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 701.20 | 698.18 | 699.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 701.20 | 698.18 | 699.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 701.15 | 698.78 | 699.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:30:00 | 700.90 | 698.78 | 699.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 704.50 | 700.25 | 700.13 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 10:15:00 | 698.30 | 699.86 | 699.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 11:15:00 | 695.45 | 698.98 | 699.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 10:15:00 | 698.55 | 695.69 | 697.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 10:15:00 | 698.55 | 695.69 | 697.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 698.55 | 695.69 | 697.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 698.55 | 695.69 | 697.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 689.55 | 694.46 | 696.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 12:15:00 | 688.95 | 694.46 | 696.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 687.35 | 693.45 | 695.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 11:15:00 | 700.50 | 694.30 | 695.11 | SL hit (close>static) qty=1.00 sl=699.40 alert=retest2 |

### Cycle 155 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 701.30 | 695.70 | 695.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 14:15:00 | 702.05 | 697.87 | 696.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 11:15:00 | 700.80 | 703.78 | 701.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 11:15:00 | 700.80 | 703.78 | 701.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 700.80 | 703.78 | 701.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:45:00 | 699.25 | 703.78 | 701.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 693.85 | 701.80 | 701.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 693.85 | 701.80 | 701.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 698.35 | 701.11 | 700.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:30:00 | 699.35 | 700.68 | 700.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 15:15:00 | 698.20 | 700.19 | 700.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 15:15:00 | 698.20 | 700.19 | 700.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 693.50 | 698.85 | 699.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 13:15:00 | 678.25 | 676.86 | 681.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 14:00:00 | 678.25 | 676.86 | 681.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 655.80 | 652.81 | 657.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:30:00 | 653.85 | 652.74 | 656.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:00:00 | 654.90 | 654.22 | 655.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 661.25 | 656.79 | 656.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 661.25 | 656.79 | 656.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 670.15 | 660.60 | 658.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 14:15:00 | 679.85 | 680.32 | 676.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 14:45:00 | 679.45 | 680.32 | 676.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 680.45 | 679.99 | 677.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:15:00 | 681.05 | 679.99 | 677.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 13:15:00 | 676.45 | 679.32 | 677.38 | SL hit (close<static) qty=1.00 sl=677.00 alert=retest2 |

### Cycle 158 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 671.40 | 675.87 | 676.14 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 677.95 | 675.74 | 675.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 681.70 | 676.93 | 676.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 678.15 | 678.73 | 677.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 678.15 | 678.73 | 677.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 678.15 | 678.73 | 677.39 | EMA400 retest candle locked (from upside) |

### Cycle 160 — SELL (started 2025-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 14:15:00 | 673.90 | 676.65 | 676.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 15:15:00 | 672.65 | 675.85 | 676.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 679.65 | 676.61 | 676.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 679.65 | 676.61 | 676.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 679.65 | 676.61 | 676.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:45:00 | 681.20 | 676.61 | 676.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 670.50 | 675.39 | 676.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:15:00 | 669.50 | 675.39 | 676.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:45:00 | 669.65 | 674.12 | 675.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 15:00:00 | 669.80 | 671.42 | 673.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 15:15:00 | 678.00 | 674.18 | 673.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 678.00 | 674.18 | 673.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 682.15 | 675.77 | 674.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 679.35 | 679.80 | 677.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 679.35 | 679.80 | 677.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 679.35 | 679.80 | 677.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 679.75 | 679.80 | 677.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 678.15 | 679.47 | 677.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 676.80 | 679.47 | 677.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 679.75 | 679.53 | 678.03 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 674.25 | 677.13 | 677.20 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 680.00 | 677.28 | 677.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 683.25 | 679.92 | 678.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 678.90 | 679.96 | 679.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 14:15:00 | 678.90 | 679.96 | 679.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 678.90 | 679.96 | 679.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 678.90 | 679.96 | 679.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 679.40 | 681.54 | 680.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 679.40 | 681.54 | 680.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 674.55 | 680.14 | 679.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 674.55 | 680.14 | 679.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 674.15 | 678.94 | 679.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 673.60 | 677.87 | 678.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 679.75 | 674.67 | 676.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 679.75 | 674.67 | 676.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 679.75 | 674.67 | 676.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 679.75 | 674.67 | 676.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 678.60 | 675.46 | 676.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:45:00 | 679.05 | 675.46 | 676.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 679.85 | 676.65 | 676.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:30:00 | 679.10 | 676.65 | 676.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 679.95 | 677.31 | 677.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 681.80 | 678.76 | 677.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 11:15:00 | 716.20 | 718.72 | 712.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 11:45:00 | 716.00 | 718.72 | 712.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 721.55 | 728.31 | 725.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 721.55 | 728.31 | 725.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 725.50 | 727.75 | 725.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 14:00:00 | 726.25 | 725.72 | 725.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 09:45:00 | 727.60 | 727.90 | 726.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 15:15:00 | 760.00 | 761.95 | 762.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 760.00 | 761.95 | 762.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 758.75 | 761.31 | 761.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 757.85 | 756.40 | 758.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 757.85 | 756.40 | 758.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 757.85 | 756.40 | 758.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:45:00 | 752.30 | 756.54 | 757.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 750.25 | 756.13 | 757.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 10:45:00 | 750.25 | 751.07 | 753.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:15:00 | 750.50 | 747.41 | 749.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 748.25 | 747.58 | 749.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 13:30:00 | 747.00 | 747.61 | 749.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:30:00 | 746.95 | 748.17 | 749.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 758.30 | 750.51 | 750.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 758.30 | 750.51 | 750.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 759.15 | 752.24 | 750.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 759.20 | 759.41 | 756.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 13:00:00 | 759.20 | 759.41 | 756.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 759.15 | 759.36 | 756.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 757.80 | 759.36 | 756.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 762.90 | 759.85 | 757.56 | EMA400 retest candle locked (from upside) |

### Cycle 168 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 755.25 | 757.68 | 757.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 754.00 | 756.94 | 757.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 11:15:00 | 748.00 | 747.19 | 750.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 11:45:00 | 748.00 | 747.19 | 750.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 747.50 | 745.20 | 748.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:15:00 | 754.80 | 745.20 | 748.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 753.55 | 746.87 | 748.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:30:00 | 755.10 | 746.87 | 748.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 755.00 | 748.49 | 749.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 757.95 | 748.49 | 749.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 754.95 | 749.78 | 749.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 14:15:00 | 757.05 | 752.52 | 751.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 754.90 | 756.16 | 754.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 754.90 | 756.16 | 754.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 754.90 | 756.16 | 754.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 754.90 | 756.16 | 754.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 752.00 | 755.32 | 753.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 752.00 | 755.32 | 753.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 745.80 | 753.42 | 753.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 745.80 | 753.42 | 753.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 736.25 | 749.99 | 751.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 15:15:00 | 733.80 | 740.88 | 743.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 743.35 | 740.99 | 742.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 11:15:00 | 743.35 | 740.99 | 742.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 743.35 | 740.99 | 742.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 743.35 | 740.99 | 742.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 745.40 | 741.88 | 742.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 746.40 | 741.88 | 742.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 743.00 | 743.20 | 743.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 749.60 | 743.20 | 743.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 749.45 | 744.45 | 743.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 11:15:00 | 753.35 | 747.40 | 745.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 758.00 | 758.49 | 752.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 758.00 | 758.49 | 752.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 751.30 | 756.29 | 752.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:00:00 | 751.30 | 756.29 | 752.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 749.35 | 754.90 | 752.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 749.35 | 754.90 | 752.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 747.55 | 750.85 | 751.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 745.10 | 748.16 | 749.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 747.25 | 745.50 | 747.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 747.25 | 745.50 | 747.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 747.05 | 745.81 | 747.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:45:00 | 747.70 | 745.81 | 747.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 744.40 | 745.53 | 747.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:15:00 | 741.80 | 745.53 | 747.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 15:15:00 | 742.30 | 745.31 | 747.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:45:00 | 743.40 | 744.35 | 746.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:30:00 | 742.95 | 743.61 | 745.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 749.15 | 744.72 | 745.77 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 749.15 | 744.72 | 745.77 | SL hit (close>static) qty=1.00 sl=747.35 alert=retest2 |

### Cycle 173 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 749.20 | 746.81 | 746.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 772.25 | 751.90 | 748.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 11:15:00 | 780.00 | 780.12 | 773.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:45:00 | 779.95 | 780.12 | 773.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 775.70 | 778.84 | 774.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 775.70 | 778.84 | 774.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 773.05 | 777.69 | 774.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 778.95 | 777.69 | 774.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 15:15:00 | 771.90 | 774.76 | 774.34 | SL hit (close<static) qty=1.00 sl=773.05 alert=retest2 |

### Cycle 174 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 769.65 | 774.42 | 774.45 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 779.90 | 775.31 | 774.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 13:15:00 | 782.90 | 777.28 | 775.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 790.75 | 792.52 | 787.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 10:00:00 | 790.75 | 792.52 | 787.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 788.35 | 791.68 | 787.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:00:00 | 788.35 | 791.68 | 787.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 799.25 | 803.68 | 800.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 799.25 | 803.68 | 800.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 805.70 | 804.08 | 800.77 | EMA400 retest candle locked (from upside) |

### Cycle 176 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 792.40 | 799.92 | 800.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 786.85 | 793.44 | 796.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 774.90 | 774.49 | 780.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 774.90 | 774.49 | 780.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 774.90 | 774.49 | 780.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:45:00 | 775.60 | 774.49 | 780.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 775.95 | 771.83 | 775.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:30:00 | 773.00 | 771.83 | 775.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 779.40 | 773.34 | 775.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:30:00 | 779.30 | 773.34 | 775.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 782.75 | 775.22 | 776.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:00:00 | 782.75 | 775.22 | 776.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 15:15:00 | 780.90 | 777.18 | 776.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 791.85 | 780.11 | 778.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 781.05 | 783.62 | 780.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 13:15:00 | 781.05 | 783.62 | 780.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 781.05 | 783.62 | 780.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 781.05 | 783.62 | 780.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 787.55 | 784.40 | 781.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 784.35 | 784.40 | 781.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 785.00 | 788.05 | 785.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 773.05 | 788.05 | 785.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 763.70 | 783.18 | 783.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 747.35 | 771.62 | 778.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 14:15:00 | 700.35 | 699.36 | 715.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 14:45:00 | 700.45 | 699.36 | 715.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 716.00 | 705.47 | 709.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:00:00 | 716.00 | 705.47 | 709.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 718.25 | 708.03 | 710.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:30:00 | 721.90 | 708.03 | 710.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — BUY (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 14:15:00 | 716.70 | 712.49 | 712.33 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 13:15:00 | 711.40 | 712.32 | 712.39 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 715.80 | 713.02 | 712.70 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 705.40 | 711.82 | 712.23 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 717.50 | 712.29 | 712.15 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 10:15:00 | 709.55 | 711.75 | 712.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 15:15:00 | 702.10 | 706.68 | 709.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 686.20 | 680.81 | 689.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 14:00:00 | 686.20 | 680.81 | 689.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 697.20 | 684.09 | 690.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 697.20 | 684.09 | 690.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 697.15 | 686.70 | 690.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 737.50 | 686.70 | 690.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 738.50 | 697.06 | 695.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 757.70 | 734.46 | 718.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 739.55 | 749.30 | 736.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 739.55 | 749.30 | 736.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 740.65 | 747.57 | 736.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 740.65 | 747.57 | 736.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 741.00 | 745.22 | 741.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:00:00 | 741.00 | 745.22 | 741.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 738.55 | 743.89 | 740.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 738.55 | 743.89 | 740.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 740.50 | 743.21 | 740.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 11:00:00 | 742.45 | 741.07 | 740.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 13:45:00 | 742.50 | 742.33 | 741.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 14:45:00 | 744.15 | 742.66 | 741.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:30:00 | 742.30 | 747.51 | 746.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 742.35 | 746.48 | 746.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:15:00 | 741.45 | 746.48 | 746.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-12 11:15:00 | 740.65 | 745.31 | 745.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 740.65 | 745.31 | 745.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 732.70 | 742.53 | 744.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 733.85 | 731.07 | 735.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 13:00:00 | 733.85 | 731.07 | 735.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 735.55 | 731.96 | 735.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:45:00 | 735.00 | 731.96 | 735.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 733.50 | 732.27 | 735.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:45:00 | 735.60 | 732.27 | 735.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 735.75 | 733.47 | 735.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 735.20 | 733.47 | 735.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 737.20 | 734.21 | 735.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:30:00 | 736.65 | 734.21 | 735.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 741.35 | 736.59 | 736.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 749.65 | 740.02 | 737.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 09:15:00 | 752.50 | 756.58 | 750.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-20 10:00:00 | 752.50 | 756.58 | 750.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 752.30 | 755.73 | 751.08 | EMA400 retest candle locked (from upside) |

### Cycle 188 — SELL (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 09:15:00 | 657.25 | 734.23 | 742.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 11:15:00 | 644.00 | 703.12 | 726.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 634.30 | 629.37 | 645.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-26 10:00:00 | 634.30 | 629.37 | 645.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 639.75 | 636.46 | 640.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:00:00 | 639.75 | 636.46 | 640.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 638.50 | 636.87 | 640.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:45:00 | 637.10 | 636.87 | 640.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 641.25 | 637.75 | 640.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:00:00 | 641.25 | 637.75 | 640.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 636.80 | 637.56 | 640.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:45:00 | 641.75 | 637.56 | 640.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 621.90 | 616.10 | 622.14 | EMA400 retest candle locked (from downside) |

### Cycle 189 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 631.40 | 623.58 | 623.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 633.55 | 625.57 | 624.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 628.85 | 629.80 | 627.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 14:15:00 | 628.85 | 629.80 | 627.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 628.85 | 629.80 | 627.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 628.85 | 629.80 | 627.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 626.20 | 629.08 | 627.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 610.80 | 629.08 | 627.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 611.30 | 625.53 | 625.79 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 629.70 | 625.02 | 624.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 631.25 | 627.22 | 625.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 13:15:00 | 627.85 | 628.94 | 627.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 13:15:00 | 627.85 | 628.94 | 627.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 627.85 | 628.94 | 627.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:45:00 | 628.20 | 628.94 | 627.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 624.60 | 628.07 | 627.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 624.60 | 628.07 | 627.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 624.00 | 627.26 | 626.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 621.30 | 627.26 | 626.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 621.40 | 626.08 | 626.38 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 10:15:00 | 629.10 | 626.69 | 626.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 11:15:00 | 631.85 | 627.72 | 627.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 15:15:00 | 625.55 | 628.29 | 627.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 15:15:00 | 625.55 | 628.29 | 627.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 625.55 | 628.29 | 627.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:15:00 | 626.30 | 628.29 | 627.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 611.00 | 624.83 | 626.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 11:15:00 | 609.90 | 619.57 | 623.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 609.05 | 608.66 | 613.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 611.50 | 608.66 | 613.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 612.40 | 609.34 | 613.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 612.40 | 609.34 | 613.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 615.75 | 610.62 | 613.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 615.75 | 610.62 | 613.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 616.30 | 611.76 | 613.66 | EMA400 retest candle locked (from downside) |

### Cycle 195 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 619.65 | 615.18 | 614.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 627.45 | 619.73 | 617.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 619.50 | 624.14 | 620.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 619.50 | 624.14 | 620.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 619.50 | 624.14 | 620.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:30:00 | 618.40 | 624.14 | 620.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 621.25 | 623.57 | 620.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:00:00 | 624.10 | 623.67 | 620.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 12:15:00 | 618.60 | 622.66 | 620.73 | SL hit (close<static) qty=1.00 sl=619.00 alert=retest2 |

### Cycle 196 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 612.55 | 619.16 | 619.39 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 632.00 | 620.60 | 619.93 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 607.50 | 620.65 | 621.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 603.60 | 615.85 | 618.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 613.20 | 608.77 | 613.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 613.20 | 608.77 | 613.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 613.20 | 608.77 | 613.47 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 620.25 | 615.04 | 615.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 629.70 | 618.92 | 616.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 622.95 | 623.56 | 620.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 628.40 | 623.56 | 620.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 613.00 | 621.45 | 619.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 613.00 | 621.45 | 619.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 606.65 | 618.49 | 618.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 602.70 | 615.33 | 617.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 594.40 | 583.05 | 593.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 594.40 | 583.05 | 593.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 594.40 | 583.05 | 593.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 594.40 | 583.05 | 593.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 591.70 | 584.78 | 593.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 579.80 | 593.62 | 595.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 14:30:00 | 588.85 | 588.77 | 591.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 588.80 | 589.24 | 591.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:45:00 | 589.55 | 589.79 | 591.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 594.95 | 590.82 | 591.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:45:00 | 596.15 | 590.82 | 591.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 598.95 | 592.45 | 592.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 598.95 | 592.45 | 592.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 605.80 | 596.53 | 594.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 627.30 | 641.12 | 636.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 627.30 | 641.12 | 636.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 627.30 | 641.12 | 636.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 634.65 | 639.87 | 636.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 12:15:00 | 654.65 | 659.14 | 659.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — SELL (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 12:15:00 | 654.65 | 659.14 | 659.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 13:15:00 | 653.65 | 658.05 | 658.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 635.95 | 635.37 | 641.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 10:15:00 | 641.70 | 635.37 | 641.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 641.60 | 636.61 | 641.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 641.05 | 636.61 | 641.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 643.15 | 637.92 | 641.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:45:00 | 643.10 | 637.92 | 641.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 641.40 | 638.62 | 641.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:30:00 | 643.05 | 638.62 | 641.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 641.50 | 639.19 | 641.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:45:00 | 641.25 | 639.19 | 641.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 640.30 | 639.41 | 641.50 | EMA400 retest candle locked (from downside) |

### Cycle 203 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 650.90 | 642.72 | 642.54 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 633.90 | 643.08 | 643.69 | EMA200 below EMA400 |

### Cycle 205 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 654.85 | 643.36 | 642.78 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 12:15:00 | 642.30 | 642.95 | 642.99 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 647.85 | 643.80 | 643.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 654.65 | 647.30 | 645.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 11:15:00 | 652.85 | 653.21 | 649.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 12:00:00 | 652.85 | 653.21 | 649.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 653.75 | 653.20 | 650.15 | EMA400 retest candle locked (from upside) |

### Cycle 208 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 644.50 | 649.10 | 649.27 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-30 13:15:00 | 660.34 | 2023-05-31 15:15:00 | 655.16 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2023-05-30 13:45:00 | 660.44 | 2023-05-31 15:15:00 | 655.16 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2023-05-31 11:30:00 | 660.34 | 2023-05-31 15:15:00 | 655.16 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2023-06-08 10:30:00 | 669.55 | 2023-06-08 11:15:00 | 662.88 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2023-06-15 11:30:00 | 650.75 | 2023-06-16 09:15:00 | 659.82 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2023-06-15 12:15:00 | 650.70 | 2023-06-16 09:15:00 | 659.82 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2023-06-26 14:15:00 | 649.94 | 2023-06-26 15:15:00 | 651.33 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2023-07-07 10:15:00 | 641.16 | 2023-07-13 13:15:00 | 609.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-07 10:15:00 | 641.16 | 2023-07-14 12:15:00 | 610.32 | STOP_HIT | 0.50 | 4.81% |
| SELL | retest2 | 2023-07-21 10:30:00 | 609.60 | 2023-08-03 09:15:00 | 579.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-21 10:30:00 | 609.60 | 2023-08-04 11:15:00 | 580.68 | STOP_HIT | 0.50 | 4.74% |
| SELL | retest2 | 2023-08-21 14:15:00 | 561.83 | 2023-08-23 11:15:00 | 563.41 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2023-08-21 15:00:00 | 561.78 | 2023-08-23 11:15:00 | 563.41 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2023-08-22 09:45:00 | 559.77 | 2023-08-23 11:15:00 | 563.41 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-08-23 11:15:00 | 562.26 | 2023-08-23 11:15:00 | 563.41 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2023-09-06 14:30:00 | 583.70 | 2023-09-12 13:15:00 | 581.88 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2023-09-11 09:15:00 | 585.43 | 2023-09-12 13:15:00 | 581.88 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2023-09-27 09:15:00 | 590.32 | 2023-09-29 13:15:00 | 593.96 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2023-09-27 12:15:00 | 590.75 | 2023-09-29 13:15:00 | 593.96 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2023-09-28 09:30:00 | 589.84 | 2023-09-29 13:15:00 | 593.96 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2023-09-29 10:30:00 | 591.18 | 2023-09-29 13:15:00 | 593.96 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2023-10-05 10:15:00 | 578.09 | 2023-10-06 11:15:00 | 584.85 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2023-10-05 11:30:00 | 578.81 | 2023-10-06 11:15:00 | 584.85 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2023-10-05 12:00:00 | 578.95 | 2023-10-06 11:15:00 | 584.85 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2023-10-05 14:00:00 | 578.66 | 2023-10-06 11:15:00 | 584.85 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2023-10-09 14:45:00 | 585.47 | 2023-10-18 10:15:00 | 596.65 | STOP_HIT | 1.00 | 1.91% |
| BUY | retest2 | 2023-10-10 09:15:00 | 586.63 | 2023-10-18 10:15:00 | 596.65 | STOP_HIT | 1.00 | 1.71% |
| BUY | retest2 | 2023-12-13 09:15:00 | 573.96 | 2023-12-13 10:15:00 | 571.61 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2023-12-13 13:30:00 | 574.11 | 2023-12-19 11:15:00 | 577.08 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2023-12-13 14:30:00 | 574.01 | 2023-12-19 11:15:00 | 577.08 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2023-12-14 13:00:00 | 574.20 | 2023-12-19 11:15:00 | 577.08 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2023-12-15 09:15:00 | 586.29 | 2023-12-19 11:15:00 | 577.08 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-01-05 13:15:00 | 561.11 | 2024-01-09 09:15:00 | 533.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-05 13:15:00 | 561.11 | 2024-01-10 14:15:00 | 535.45 | STOP_HIT | 0.50 | 4.57% |
| SELL | retest2 | 2024-01-23 12:45:00 | 519.29 | 2024-02-05 09:15:00 | 493.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-25 09:30:00 | 519.96 | 2024-02-05 09:15:00 | 493.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-30 11:00:00 | 520.01 | 2024-02-05 09:15:00 | 494.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-30 12:00:00 | 520.44 | 2024-02-05 09:15:00 | 494.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-30 14:15:00 | 516.79 | 2024-02-05 09:15:00 | 490.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-31 09:30:00 | 517.61 | 2024-02-05 09:15:00 | 491.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-23 12:45:00 | 519.29 | 2024-02-05 11:15:00 | 467.36 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-01-25 09:30:00 | 519.96 | 2024-02-05 11:15:00 | 467.96 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-01-30 11:00:00 | 520.01 | 2024-02-05 11:15:00 | 468.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-01-30 12:00:00 | 520.44 | 2024-02-05 11:15:00 | 468.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-01-30 14:15:00 | 516.79 | 2024-02-05 11:15:00 | 465.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-01-31 09:30:00 | 517.61 | 2024-02-05 11:15:00 | 465.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-04 11:15:00 | 451.37 | 2024-03-05 09:15:00 | 464.90 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2024-03-21 11:30:00 | 437.22 | 2024-03-22 10:15:00 | 444.18 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-03-27 12:45:00 | 445.47 | 2024-03-27 14:15:00 | 435.78 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2024-03-27 13:15:00 | 445.09 | 2024-03-27 14:15:00 | 435.78 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2024-04-09 09:15:00 | 472.57 | 2024-04-15 12:15:00 | 472.71 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2024-04-09 12:45:00 | 472.52 | 2024-04-15 12:15:00 | 472.71 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2024-04-10 10:00:00 | 472.52 | 2024-04-15 12:15:00 | 472.71 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2024-04-18 14:30:00 | 461.64 | 2024-04-22 10:15:00 | 471.56 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest1 | 2024-04-24 09:15:00 | 478.57 | 2024-04-30 14:15:00 | 486.29 | STOP_HIT | 1.00 | 1.61% |
| BUY | retest2 | 2024-04-25 13:45:00 | 484.99 | 2024-05-02 10:15:00 | 481.92 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-04-26 10:30:00 | 484.75 | 2024-05-02 10:15:00 | 481.92 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-05-09 11:15:00 | 454.63 | 2024-05-10 11:15:00 | 461.35 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-05-09 12:00:00 | 455.16 | 2024-05-10 11:15:00 | 461.35 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-05-09 12:30:00 | 455.26 | 2024-05-10 11:15:00 | 461.35 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-05-17 09:15:00 | 493.05 | 2024-05-23 12:15:00 | 488.40 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-05-17 11:00:00 | 490.89 | 2024-05-23 12:15:00 | 488.40 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-05-17 14:30:00 | 490.84 | 2024-05-23 12:15:00 | 488.40 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-05-18 09:15:00 | 491.37 | 2024-05-23 12:15:00 | 488.40 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-05-22 09:15:00 | 495.16 | 2024-05-23 12:15:00 | 488.40 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-05-22 10:00:00 | 492.91 | 2024-05-23 12:15:00 | 488.40 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-06-13 12:15:00 | 537.18 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2024-06-19 09:15:00 | 538.28 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2024-06-19 11:45:00 | 535.21 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest2 | 2024-06-19 14:30:00 | 534.78 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | 1.26% |
| BUY | retest2 | 2024-06-20 10:30:00 | 544.56 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-06-21 10:00:00 | 543.79 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-06-21 11:00:00 | 543.60 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-06-21 15:00:00 | 543.41 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2024-06-26 10:30:00 | 549.36 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-06-26 13:30:00 | 548.50 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-06-26 15:00:00 | 548.59 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-06-27 09:45:00 | 548.97 | 2024-06-27 12:15:00 | 541.54 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-07-05 11:45:00 | 549.41 | 2024-07-08 09:15:00 | 546.91 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-07-05 12:30:00 | 549.89 | 2024-07-08 09:15:00 | 546.91 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-07-05 15:15:00 | 549.45 | 2024-07-08 09:15:00 | 546.91 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2024-07-10 10:15:00 | 529.50 | 2024-07-12 12:15:00 | 541.88 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2024-07-10 12:15:00 | 530.41 | 2024-07-12 12:15:00 | 541.88 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-07-22 15:00:00 | 522.98 | 2024-07-26 12:15:00 | 521.30 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2024-07-23 11:15:00 | 522.12 | 2024-07-26 12:15:00 | 521.30 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2024-08-06 12:15:00 | 522.79 | 2024-08-07 15:15:00 | 526.14 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-08-12 12:30:00 | 536.36 | 2024-08-14 09:15:00 | 523.89 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2024-08-26 09:15:00 | 551.52 | 2024-08-29 10:15:00 | 551.28 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2024-08-26 12:00:00 | 551.09 | 2024-08-29 10:15:00 | 551.28 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2024-08-26 12:30:00 | 551.76 | 2024-08-29 10:15:00 | 551.28 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2024-09-20 13:15:00 | 568.11 | 2024-09-24 10:15:00 | 581.25 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2024-09-20 14:15:00 | 566.72 | 2024-09-24 10:15:00 | 581.25 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2024-10-10 11:30:00 | 561.64 | 2024-10-17 11:15:00 | 533.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-10 12:45:00 | 561.64 | 2024-10-17 11:15:00 | 533.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-11 10:15:00 | 561.92 | 2024-10-17 11:15:00 | 533.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-10 11:30:00 | 561.64 | 2024-10-18 11:15:00 | 533.15 | STOP_HIT | 0.50 | 5.07% |
| SELL | retest2 | 2024-10-10 12:45:00 | 561.64 | 2024-10-18 11:15:00 | 533.15 | STOP_HIT | 0.50 | 5.07% |
| SELL | retest2 | 2024-10-11 10:15:00 | 561.92 | 2024-10-18 11:15:00 | 533.15 | STOP_HIT | 0.50 | 5.12% |
| BUY | retest2 | 2024-11-04 15:00:00 | 530.65 | 2024-11-08 14:15:00 | 535.69 | STOP_HIT | 1.00 | 0.95% |
| BUY | retest2 | 2024-11-05 09:15:00 | 533.77 | 2024-11-08 14:15:00 | 535.69 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2024-11-05 12:30:00 | 530.32 | 2024-11-08 14:15:00 | 535.69 | STOP_HIT | 1.00 | 1.01% |
| SELL | retest2 | 2024-11-13 09:15:00 | 500.34 | 2024-11-18 12:15:00 | 513.43 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2024-11-14 12:15:00 | 500.25 | 2024-11-18 12:15:00 | 513.43 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-11-18 09:15:00 | 498.52 | 2024-11-18 12:15:00 | 513.43 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2024-12-06 10:45:00 | 565.30 | 2024-12-09 09:15:00 | 552.35 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-12-16 15:15:00 | 547.00 | 2024-12-17 09:15:00 | 550.60 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-12-17 11:45:00 | 544.70 | 2024-12-19 15:15:00 | 517.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 11:45:00 | 544.70 | 2024-12-23 11:15:00 | 509.80 | STOP_HIT | 0.50 | 6.41% |
| BUY | retest2 | 2025-01-07 10:15:00 | 542.05 | 2025-01-13 11:15:00 | 537.05 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-01-08 09:30:00 | 540.30 | 2025-01-13 13:15:00 | 535.35 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-01-08 15:00:00 | 540.30 | 2025-01-13 13:15:00 | 535.35 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-01-10 10:30:00 | 540.65 | 2025-01-13 13:15:00 | 535.35 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-01-13 10:15:00 | 546.95 | 2025-01-13 13:15:00 | 535.35 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-01-14 12:15:00 | 537.40 | 2025-01-14 13:15:00 | 542.25 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest1 | 2025-01-16 09:15:00 | 550.60 | 2025-01-20 09:15:00 | 547.65 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-01-17 09:15:00 | 550.00 | 2025-01-21 14:15:00 | 544.30 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-01-20 10:15:00 | 548.80 | 2025-01-21 14:15:00 | 544.30 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-01-21 10:45:00 | 549.00 | 2025-01-21 14:15:00 | 544.30 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-01-21 11:45:00 | 551.25 | 2025-01-21 14:15:00 | 544.30 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest1 | 2025-01-31 09:15:00 | 570.95 | 2025-01-31 14:15:00 | 599.50 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-01-31 09:15:00 | 570.95 | 2025-02-01 15:15:00 | 599.60 | STOP_HIT | 0.50 | 5.02% |
| BUY | retest2 | 2025-02-06 15:00:00 | 644.65 | 2025-02-10 12:15:00 | 633.70 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-02-07 11:30:00 | 644.85 | 2025-02-10 12:15:00 | 633.70 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-02-19 10:15:00 | 643.70 | 2025-02-27 10:15:00 | 641.95 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-02-21 10:45:00 | 645.00 | 2025-02-27 11:15:00 | 639.45 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-02-21 15:00:00 | 648.40 | 2025-02-27 11:15:00 | 639.45 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-02-24 11:30:00 | 644.45 | 2025-02-27 11:15:00 | 639.45 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-02-25 09:15:00 | 651.60 | 2025-02-27 11:15:00 | 639.45 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-02-28 09:15:00 | 625.00 | 2025-03-05 13:15:00 | 629.25 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-03-13 11:15:00 | 606.70 | 2025-03-17 10:15:00 | 616.05 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-03-13 14:00:00 | 606.50 | 2025-03-17 10:15:00 | 616.05 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-04-03 11:45:00 | 646.10 | 2025-04-04 13:15:00 | 640.45 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-04-04 10:30:00 | 648.05 | 2025-04-04 13:15:00 | 640.45 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest1 | 2025-04-17 10:30:00 | 664.15 | 2025-04-23 09:15:00 | 666.40 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest1 | 2025-04-17 12:30:00 | 660.75 | 2025-04-23 09:15:00 | 666.40 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest1 | 2025-04-21 09:15:00 | 661.20 | 2025-04-23 09:15:00 | 666.40 | STOP_HIT | 1.00 | 0.79% |
| BUY | retest2 | 2025-04-21 14:30:00 | 668.65 | 2025-04-25 11:15:00 | 662.80 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-04-23 09:15:00 | 671.20 | 2025-04-25 11:15:00 | 662.80 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-04-23 10:15:00 | 668.80 | 2025-04-25 11:15:00 | 662.80 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-04-23 10:45:00 | 668.00 | 2025-04-25 11:15:00 | 662.80 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-05-22 09:45:00 | 635.80 | 2025-05-28 10:15:00 | 637.60 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-05-30 10:15:00 | 623.05 | 2025-06-02 11:15:00 | 633.55 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-06-06 14:00:00 | 644.80 | 2025-06-09 11:15:00 | 639.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-06-12 12:00:00 | 635.45 | 2025-06-16 11:15:00 | 639.75 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-06-12 13:00:00 | 635.85 | 2025-06-16 11:15:00 | 639.75 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-07-14 10:45:00 | 652.20 | 2025-07-14 15:15:00 | 661.20 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-07-21 10:15:00 | 697.30 | 2025-07-30 11:15:00 | 719.45 | STOP_HIT | 1.00 | 3.18% |
| SELL | retest2 | 2025-08-04 12:00:00 | 705.25 | 2025-08-04 13:15:00 | 716.15 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-08-05 10:00:00 | 703.60 | 2025-08-05 10:15:00 | 711.35 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-08-08 09:15:00 | 702.10 | 2025-08-18 10:15:00 | 697.35 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest2 | 2025-08-29 15:00:00 | 715.60 | 2025-09-01 09:15:00 | 730.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-09-15 12:15:00 | 688.95 | 2025-09-16 11:15:00 | 700.50 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-09-16 09:15:00 | 687.35 | 2025-09-16 11:15:00 | 700.50 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-09-18 14:30:00 | 699.35 | 2025-09-18 15:15:00 | 698.20 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-09-30 10:30:00 | 653.85 | 2025-10-01 13:15:00 | 661.25 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-10-01 10:00:00 | 654.90 | 2025-10-01 13:15:00 | 661.25 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-10-08 12:15:00 | 681.05 | 2025-10-08 13:15:00 | 676.45 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-10-08 13:30:00 | 682.55 | 2025-10-08 14:15:00 | 674.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-10-14 11:15:00 | 669.50 | 2025-10-15 15:15:00 | 678.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-10-14 11:45:00 | 669.65 | 2025-10-15 15:15:00 | 678.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-10-14 15:00:00 | 669.80 | 2025-10-15 15:15:00 | 678.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-11-06 14:00:00 | 726.25 | 2025-11-18 15:15:00 | 760.00 | STOP_HIT | 1.00 | 4.65% |
| BUY | retest2 | 2025-11-07 09:45:00 | 727.60 | 2025-11-18 15:15:00 | 760.00 | STOP_HIT | 1.00 | 4.45% |
| SELL | retest2 | 2025-11-20 14:45:00 | 752.30 | 2025-11-26 09:15:00 | 758.30 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-11-21 09:15:00 | 750.25 | 2025-11-26 09:15:00 | 758.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-11-24 10:45:00 | 750.25 | 2025-11-26 09:15:00 | 758.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-11-25 12:15:00 | 750.50 | 2025-11-26 09:15:00 | 758.30 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-11-25 13:30:00 | 747.00 | 2025-11-26 09:15:00 | 758.30 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-11-25 14:30:00 | 746.95 | 2025-11-26 09:15:00 | 758.30 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-12-18 14:15:00 | 741.80 | 2025-12-19 13:15:00 | 749.15 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-12-18 15:15:00 | 742.30 | 2025-12-19 13:15:00 | 749.15 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-12-19 10:45:00 | 743.40 | 2025-12-19 13:15:00 | 749.15 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-19 12:30:00 | 742.95 | 2025-12-19 13:15:00 | 749.15 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-12-26 09:15:00 | 778.95 | 2025-12-26 15:15:00 | 771.90 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-12-29 09:15:00 | 776.55 | 2025-12-29 12:15:00 | 769.65 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2026-02-09 11:00:00 | 742.45 | 2026-02-12 11:15:00 | 740.65 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2026-02-09 13:45:00 | 742.50 | 2026-02-12 11:15:00 | 740.65 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2026-02-09 14:45:00 | 744.15 | 2026-02-12 11:15:00 | 740.65 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2026-02-12 09:30:00 | 742.30 | 2026-02-12 11:15:00 | 740.65 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2026-03-19 12:00:00 | 624.10 | 2026-03-19 12:15:00 | 618.60 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-04-02 09:15:00 | 579.80 | 2026-04-06 11:15:00 | 598.95 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2026-04-02 14:30:00 | 588.85 | 2026-04-06 11:15:00 | 598.95 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2026-04-06 09:15:00 | 588.80 | 2026-04-06 11:15:00 | 598.95 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2026-04-06 09:45:00 | 589.55 | 2026-04-06 11:15:00 | 598.95 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-04-13 10:45:00 | 634.65 | 2026-04-21 12:15:00 | 654.65 | STOP_HIT | 1.00 | 3.15% |
