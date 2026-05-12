# Fortis Healthcare Ltd. (FORTIS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 951.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 17 |
| PARTIAL | 4 |
| TARGET_HIT | 6 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 11
- **Target hits / Stop hits / Partials:** 6 / 14 / 4
- **Avg / median % per leg:** 2.73% / 2.91%
- **Sum % (uncompounded):** 65.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 5 | 100.0% | 5 | 0 | 0 | 10.00% | 50.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 5 | 100.0% | 5 | 0 | 0 | 10.00% | 50.0% |
| SELL (all) | 19 | 8 | 42.1% | 1 | 14 | 4 | 0.82% | 15.5% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.88% | -8.6% |
| SELL @ 3rd Alert (retest2) | 16 | 8 | 50.0% | 1 | 11 | 4 | 1.51% | 24.1% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.88% | -8.6% |
| retest2 (combined) | 21 | 13 | 61.9% | 6 | 11 | 4 | 3.53% | 74.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 13:15:00 | 400.90 | 409.38 | 409.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 15:15:00 | 400.00 | 409.19 | 409.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 10:15:00 | 411.70 | 408.35 | 408.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 10:15:00 | 411.70 | 408.35 | 408.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 10:15:00 | 411.70 | 408.35 | 408.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-26 11:00:00 | 411.70 | 408.35 | 408.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 11:15:00 | 409.85 | 408.36 | 408.87 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 09:15:00 | 429.00 | 409.46 | 409.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-05 09:15:00 | 430.00 | 413.35 | 411.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-13 09:15:00 | 437.90 | 439.29 | 429.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-13 10:00:00 | 437.90 | 439.29 | 429.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 434.25 | 451.87 | 441.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 434.25 | 451.87 | 441.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 420.80 | 451.56 | 441.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 420.80 | 451.56 | 441.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 428.00 | 450.80 | 441.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-05 09:45:00 | 424.05 | 450.80 | 441.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 433.50 | 450.63 | 441.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 11:15:00 | 441.35 | 450.63 | 441.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 12:30:00 | 438.80 | 450.37 | 441.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 13:00:00 | 437.90 | 450.37 | 441.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-13 14:15:00 | 482.68 | 454.55 | 445.14 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 10:15:00 | 636.70 | 655.67 | 655.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 11:15:00 | 632.55 | 655.44 | 655.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 10:15:00 | 661.00 | 649.71 | 652.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 10:15:00 | 661.00 | 649.71 | 652.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 661.00 | 649.71 | 652.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:00:00 | 661.00 | 649.71 | 652.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 661.30 | 649.82 | 652.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:30:00 | 658.80 | 649.82 | 652.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 646.85 | 650.57 | 652.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 14:15:00 | 645.00 | 650.57 | 652.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 641.05 | 650.48 | 652.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 13:15:00 | 612.75 | 647.51 | 651.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 609.00 | 646.43 | 650.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-03 12:15:00 | 624.95 | 624.28 | 635.52 | SL hit (close>ema200) qty=0.50 sl=624.28 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 13:15:00 | 686.25 | 637.60 | 637.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 10:15:00 | 691.85 | 654.36 | 648.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 662.65 | 662.88 | 653.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 14:30:00 | 663.10 | 662.88 | 653.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 655.50 | 662.76 | 653.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 11:00:00 | 659.00 | 662.73 | 653.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 14:30:00 | 661.95 | 662.72 | 654.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-22 09:15:00 | 724.90 | 672.39 | 661.74 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 908.15 | 965.74 | 965.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 12:15:00 | 905.20 | 965.14 | 965.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 11:15:00 | 907.45 | 907.37 | 929.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-22 12:00:00 | 907.45 | 907.37 | 929.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 922.85 | 902.18 | 921.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:00:00 | 922.85 | 902.18 | 921.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 915.05 | 902.30 | 921.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 14:30:00 | 914.55 | 902.43 | 921.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 906.75 | 902.56 | 921.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 10:15:00 | 913.00 | 902.70 | 920.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 13:30:00 | 915.00 | 903.20 | 920.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 928.55 | 903.70 | 920.86 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 928.55 | 903.70 | 920.86 | SL hit (close>static) qty=1.00 sl=923.05 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 09:15:00 | 956.15 | 902.29 | 902.29 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 10:15:00 | 876.85 | 902.88 | 902.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 11:15:00 | 876.20 | 902.62 | 902.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 845.30 | 842.24 | 864.89 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 10:15:00 | 841.05 | 842.24 | 864.89 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 13:30:00 | 843.65 | 842.29 | 864.47 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 09:15:00 | 843.05 | 843.42 | 863.35 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 866.85 | 844.29 | 863.01 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 866.85 | 844.29 | 863.01 | SL hit (close>ema400) qty=1.00 sl=863.01 alert=retest1 |

### Cycle 8 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 957.00 | 876.17 | 875.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 962.50 | 903.40 | 891.15 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-05 11:15:00 | 441.35 | 2024-06-13 14:15:00 | 482.68 | TARGET_HIT | 1.00 | 9.36% |
| BUY | retest2 | 2024-06-05 12:30:00 | 438.80 | 2024-06-13 14:15:00 | 481.69 | TARGET_HIT | 1.00 | 9.77% |
| BUY | retest2 | 2024-06-05 13:00:00 | 437.90 | 2024-06-13 15:15:00 | 485.49 | TARGET_HIT | 1.00 | 10.87% |
| SELL | retest2 | 2025-02-06 14:15:00 | 645.00 | 2025-02-11 13:15:00 | 612.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 641.05 | 2025-02-12 09:15:00 | 609.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 14:15:00 | 645.00 | 2025-03-03 12:15:00 | 624.95 | STOP_HIT | 0.50 | 3.11% |
| SELL | retest2 | 2025-02-10 09:15:00 | 641.05 | 2025-03-03 12:15:00 | 624.95 | STOP_HIT | 0.50 | 2.51% |
| SELL | retest2 | 2025-03-04 12:45:00 | 643.40 | 2025-03-05 10:15:00 | 665.45 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2025-03-06 15:15:00 | 643.55 | 2025-03-10 14:15:00 | 611.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-06 15:15:00 | 643.55 | 2025-03-20 10:15:00 | 624.85 | STOP_HIT | 0.50 | 2.91% |
| BUY | retest2 | 2025-05-09 11:00:00 | 659.00 | 2025-05-22 09:15:00 | 724.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-09 14:30:00 | 661.95 | 2025-05-22 10:15:00 | 728.15 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-02 14:30:00 | 914.55 | 2026-01-06 09:15:00 | 928.55 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-01-05 09:15:00 | 906.75 | 2026-01-06 09:15:00 | 928.55 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2026-01-05 10:15:00 | 913.00 | 2026-01-06 09:15:00 | 928.55 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-01-05 13:30:00 | 915.00 | 2026-01-06 09:15:00 | 928.55 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-01-08 12:45:00 | 921.35 | 2026-01-20 09:15:00 | 875.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 12:45:00 | 921.35 | 2026-01-27 12:15:00 | 829.22 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-11 10:30:00 | 921.70 | 2026-02-12 09:15:00 | 934.95 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-02-11 11:00:00 | 921.50 | 2026-02-12 09:15:00 | 934.95 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-02-12 14:30:00 | 923.10 | 2026-02-25 11:15:00 | 931.80 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest1 | 2026-04-08 10:15:00 | 841.05 | 2026-04-15 09:15:00 | 866.85 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest1 | 2026-04-08 13:30:00 | 843.65 | 2026-04-15 09:15:00 | 866.85 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest1 | 2026-04-13 09:15:00 | 843.05 | 2026-04-15 09:15:00 | 866.85 | STOP_HIT | 1.00 | -2.82% |
