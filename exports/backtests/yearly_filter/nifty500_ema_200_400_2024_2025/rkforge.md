# Ramkrishna Forgings Ltd. (RKFORGE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 607.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 31 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 2 |
| TARGET_HIT | 4 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 20
- **Target hits / Stop hits / Partials:** 4 / 20 / 2
- **Avg / median % per leg:** -0.55% / -1.97%
- **Sum % (uncompounded):** -14.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 2 | 11.1% | 2 | 16 | 0 | -1.92% | -34.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 2 | 11.1% | 2 | 16 | 0 | -1.92% | -34.6% |
| SELL (all) | 8 | 4 | 50.0% | 2 | 4 | 2 | 2.53% | 20.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 4 | 50.0% | 2 | 4 | 2 | 2.53% | 20.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 26 | 6 | 23.1% | 4 | 20 | 2 | -0.55% | -14.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 12:15:00 | 687.90 | 720.49 | 720.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 09:15:00 | 677.00 | 715.33 | 717.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 11:15:00 | 719.00 | 708.01 | 713.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 11:15:00 | 719.00 | 708.01 | 713.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 11:15:00 | 719.00 | 708.01 | 713.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 12:00:00 | 719.00 | 708.01 | 713.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 12:15:00 | 711.65 | 708.05 | 713.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 682.30 | 708.13 | 713.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 11:15:00 | 648.18 | 706.83 | 712.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 12:15:00 | 614.07 | 706.12 | 712.35 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 15:15:00 | 815.35 | 715.66 | 715.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 09:15:00 | 822.00 | 716.72 | 715.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 857.70 | 869.32 | 818.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-19 10:00:00 | 857.70 | 869.32 | 818.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 943.20 | 981.33 | 946.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 10:00:00 | 943.20 | 981.33 | 946.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 934.10 | 980.86 | 945.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 10:45:00 | 933.70 | 980.86 | 945.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 12:15:00 | 920.10 | 979.81 | 945.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 13:00:00 | 920.10 | 979.81 | 945.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 945.75 | 969.40 | 943.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 13:30:00 | 950.60 | 968.75 | 943.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 14:30:00 | 956.20 | 968.74 | 943.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-15 09:15:00 | 1045.66 | 975.07 | 948.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 09:15:00 | 904.20 | 952.53 | 952.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 13:15:00 | 894.05 | 950.53 | 951.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 11:15:00 | 981.55 | 926.34 | 935.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 11:15:00 | 981.55 | 926.34 | 935.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 11:15:00 | 981.55 | 926.34 | 935.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 12:00:00 | 981.55 | 926.34 | 935.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 12:15:00 | 984.90 | 926.92 | 936.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 13:15:00 | 979.55 | 926.92 | 936.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 10:15:00 | 940.75 | 928.27 | 936.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-13 10:45:00 | 942.70 | 928.27 | 936.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 11:15:00 | 939.80 | 928.39 | 936.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-13 11:45:00 | 940.10 | 928.39 | 936.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 13:15:00 | 941.90 | 928.61 | 936.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-13 14:00:00 | 941.90 | 928.61 | 936.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 14:15:00 | 931.40 | 928.64 | 936.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-13 14:45:00 | 941.10 | 928.64 | 936.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 945.55 | 928.79 | 936.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:45:00 | 945.75 | 928.79 | 936.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 942.45 | 928.93 | 936.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 10:45:00 | 943.15 | 928.93 | 936.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 940.15 | 929.04 | 936.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 15:00:00 | 935.85 | 929.43 | 936.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-15 09:15:00 | 958.05 | 929.84 | 936.96 | SL hit (close>static) qty=1.00 sl=946.55 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 13:15:00 | 569.90 | 526.86 | 526.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 11:15:00 | 571.00 | 528.89 | 527.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 10:15:00 | 544.00 | 544.18 | 537.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-24 11:00:00 | 544.00 | 544.18 | 537.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 540.10 | 544.09 | 537.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:30:00 | 536.55 | 544.09 | 537.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 537.45 | 544.04 | 537.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 537.45 | 544.04 | 537.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 539.70 | 543.99 | 537.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:00:00 | 539.70 | 543.99 | 537.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 550.05 | 544.56 | 538.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 15:00:00 | 557.40 | 544.99 | 538.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:30:00 | 555.30 | 545.25 | 538.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 10:30:00 | 558.55 | 545.37 | 539.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 12:00:00 | 554.80 | 545.46 | 539.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 544.50 | 548.98 | 542.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 544.50 | 548.98 | 542.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 539.50 | 548.88 | 542.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:00:00 | 539.50 | 548.88 | 542.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 560.60 | 549.00 | 542.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 14:30:00 | 561.90 | 549.23 | 542.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 14:15:00 | 531.90 | 548.64 | 542.89 | SL hit (close<static) qty=1.00 sl=538.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 10:15:00 | 480.70 | 537.60 | 537.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 468.90 | 528.85 | 533.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 525.10 | 519.54 | 527.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 525.10 | 519.54 | 527.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 525.10 | 519.54 | 527.36 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 13:15:00 | 547.40 | 532.07 | 532.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 551.45 | 532.53 | 532.24 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-04 09:15:00 | 682.30 | 2024-06-04 11:15:00 | 648.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 682.30 | 2024-06-04 12:15:00 | 614.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-06 11:45:00 | 701.95 | 2024-06-11 12:15:00 | 721.00 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2024-06-10 11:00:00 | 706.05 | 2024-06-11 12:15:00 | 721.00 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2024-06-10 12:00:00 | 702.90 | 2024-06-11 12:15:00 | 721.00 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2024-10-10 13:30:00 | 950.60 | 2024-10-15 09:15:00 | 1045.66 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-10 14:30:00 | 956.20 | 2024-10-15 09:15:00 | 1051.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-22 15:15:00 | 955.00 | 2024-10-24 14:15:00 | 949.25 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-10-23 09:45:00 | 956.20 | 2024-10-25 09:15:00 | 913.80 | STOP_HIT | 1.00 | -4.43% |
| BUY | retest2 | 2024-10-24 13:30:00 | 961.00 | 2024-10-25 09:15:00 | 913.80 | STOP_HIT | 1.00 | -4.91% |
| BUY | retest2 | 2024-11-07 09:30:00 | 961.75 | 2024-11-12 14:15:00 | 946.85 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-11-07 11:30:00 | 963.45 | 2024-11-12 14:15:00 | 946.85 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-11-07 13:30:00 | 960.30 | 2024-11-12 14:15:00 | 946.85 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-11-19 09:15:00 | 968.80 | 2024-11-19 14:15:00 | 944.80 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2024-11-19 10:15:00 | 963.80 | 2024-11-19 14:15:00 | 944.80 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-11-21 10:00:00 | 965.25 | 2024-12-16 10:15:00 | 950.10 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-11-21 13:00:00 | 962.45 | 2024-12-16 11:15:00 | 942.95 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-12-16 09:30:00 | 959.75 | 2024-12-16 11:15:00 | 942.95 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-01-14 15:00:00 | 935.85 | 2025-01-15 09:15:00 | 958.05 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-01-21 14:00:00 | 938.85 | 2025-01-22 11:15:00 | 891.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 14:00:00 | 938.85 | 2025-01-24 09:15:00 | 844.97 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-02 15:00:00 | 557.40 | 2026-03-18 14:15:00 | 531.90 | STOP_HIT | 1.00 | -4.57% |
| BUY | retest2 | 2026-03-05 09:30:00 | 555.30 | 2026-03-19 10:15:00 | 521.90 | STOP_HIT | 1.00 | -6.01% |
| BUY | retest2 | 2026-03-05 10:30:00 | 558.55 | 2026-03-19 10:15:00 | 521.90 | STOP_HIT | 1.00 | -6.56% |
| BUY | retest2 | 2026-03-05 12:00:00 | 554.80 | 2026-03-19 10:15:00 | 521.90 | STOP_HIT | 1.00 | -5.93% |
| BUY | retest2 | 2026-03-13 14:30:00 | 561.90 | 2026-03-19 10:15:00 | 521.90 | STOP_HIT | 1.00 | -7.12% |
