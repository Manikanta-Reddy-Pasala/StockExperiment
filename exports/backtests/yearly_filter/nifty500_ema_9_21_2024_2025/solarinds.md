# Solar Industries India Ltd. (SOLARINDS)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 16101.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 138 |
| ALERT1 | 92 |
| ALERT2 | 92 |
| ALERT2_SKIP | 55 |
| ALERT3 | 248 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 90 |
| PARTIAL | 16 |
| TARGET_HIT | 5 |
| STOP_HIT | 86 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 107 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 47 / 60
- **Target hits / Stop hits / Partials:** 5 / 86 / 16
- **Avg / median % per leg:** 1.19% / -0.54%
- **Sum % (uncompounded):** 126.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 46 | 9 | 19.6% | 3 | 43 | 0 | -0.24% | -11.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 46 | 9 | 19.6% | 3 | 43 | 0 | -0.24% | -11.1% |
| SELL (all) | 61 | 38 | 62.3% | 2 | 43 | 16 | 2.26% | 138.1% |
| SELL @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 3 | 1 | 1.63% | 6.5% |
| SELL @ 3rd Alert (retest2) | 57 | 35 | 61.4% | 2 | 40 | 15 | 2.31% | 131.6% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 3 | 1 | 1.63% | 6.5% |
| retest2 (combined) | 103 | 44 | 42.7% | 5 | 83 | 15 | 1.17% | 120.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 10:15:00 | 8662.80 | 8740.11 | 8745.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-14 09:15:00 | 8625.20 | 8686.83 | 8714.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 14:15:00 | 8636.10 | 8629.84 | 8671.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-14 15:00:00 | 8636.10 | 8629.84 | 8671.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 8783.55 | 8429.93 | 8455.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:45:00 | 8761.20 | 8429.93 | 8455.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 10:15:00 | 8973.80 | 8538.70 | 8502.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 09:15:00 | 9420.00 | 8992.36 | 8804.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 14:15:00 | 9984.10 | 10161.89 | 9951.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-24 15:00:00 | 9984.10 | 10161.89 | 9951.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 9980.00 | 10125.51 | 9953.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:15:00 | 9973.90 | 10125.51 | 9953.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 9939.55 | 10088.32 | 9952.51 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 14:15:00 | 9692.70 | 9855.42 | 9876.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 15:15:00 | 9689.75 | 9822.28 | 9859.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 09:15:00 | 9374.90 | 9326.69 | 9464.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 09:15:00 | 9374.90 | 9326.69 | 9464.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 9374.90 | 9326.69 | 9464.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:30:00 | 9489.20 | 9326.69 | 9464.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 9297.85 | 9298.63 | 9404.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:30:00 | 9398.65 | 9298.63 | 9404.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 9252.10 | 9257.79 | 9356.42 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 15:15:00 | 9745.25 | 9375.28 | 9370.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 9855.25 | 9471.27 | 9414.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 9719.90 | 9804.04 | 9656.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 9719.90 | 9804.04 | 9656.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 9719.90 | 9804.04 | 9656.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 9593.40 | 9804.04 | 9656.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 9092.95 | 9661.82 | 9605.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 9092.95 | 9661.82 | 9605.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 8277.65 | 9384.99 | 9484.43 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 9324.35 | 9117.62 | 9102.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 13:15:00 | 9351.80 | 9296.37 | 9231.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 09:15:00 | 9389.00 | 9484.94 | 9402.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-12 09:15:00 | 9389.00 | 9484.94 | 9402.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 9389.00 | 9484.94 | 9402.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 10:00:00 | 9389.00 | 9484.94 | 9402.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 9388.80 | 9465.71 | 9401.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 11:00:00 | 9388.80 | 9465.71 | 9401.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 9390.00 | 9450.57 | 9400.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 11:30:00 | 9386.25 | 9450.57 | 9400.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 13:15:00 | 9349.25 | 9419.65 | 9394.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 13:45:00 | 9350.00 | 9419.65 | 9394.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 9420.00 | 9417.87 | 9397.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 9572.15 | 9417.87 | 9397.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 11:15:00 | 9791.85 | 9844.64 | 9851.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 11:15:00 | 9791.85 | 9844.64 | 9851.09 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 14:15:00 | 10047.30 | 9860.93 | 9846.36 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 12:15:00 | 9855.00 | 9917.37 | 9925.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 9825.00 | 9900.52 | 9913.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 11:15:00 | 9892.00 | 9871.74 | 9890.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 11:15:00 | 9892.00 | 9871.74 | 9890.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 9892.00 | 9871.74 | 9890.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:00:00 | 9892.00 | 9871.74 | 9890.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 9825.00 | 9862.39 | 9884.84 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 14:15:00 | 10035.00 | 9903.02 | 9899.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 14:15:00 | 10097.80 | 9988.26 | 9951.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 12:15:00 | 10849.05 | 10851.65 | 10594.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 13:00:00 | 10849.05 | 10851.65 | 10594.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 12042.35 | 12123.69 | 11970.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 13:30:00 | 12166.50 | 12137.82 | 12014.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 09:30:00 | 12185.90 | 12087.77 | 12020.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 15:15:00 | 12490.00 | 12066.50 | 12030.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 15:15:00 | 12239.85 | 12229.28 | 12162.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 12239.85 | 12231.39 | 12169.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-12 13:15:00 | 11944.00 | 12129.98 | 12144.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2024-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 13:15:00 | 11944.00 | 12129.98 | 12144.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 09:15:00 | 11822.50 | 12028.72 | 12090.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 15:15:00 | 12110.00 | 11926.89 | 11995.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 15:15:00 | 12110.00 | 11926.89 | 11995.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 12110.00 | 11926.89 | 11995.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 13:30:00 | 11797.15 | 11856.73 | 11933.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 09:30:00 | 11715.05 | 11819.23 | 11894.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 10:15:00 | 11759.25 | 11819.23 | 11894.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 14:15:00 | 11207.29 | 11557.20 | 11728.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 14:15:00 | 11171.29 | 11557.20 | 11728.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 11129.30 | 11430.58 | 11640.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-07-22 09:15:00 | 10617.43 | 11037.38 | 11287.50 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 12 — BUY (started 2024-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 14:15:00 | 11014.80 | 10655.82 | 10610.42 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 09:15:00 | 10769.00 | 10796.25 | 10797.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 13:15:00 | 10680.40 | 10747.25 | 10772.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 10673.60 | 10460.61 | 10542.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 10673.60 | 10460.61 | 10542.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 10673.60 | 10460.61 | 10542.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 10673.60 | 10460.61 | 10542.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 10443.00 | 10457.09 | 10533.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:30:00 | 10285.00 | 10436.09 | 10503.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 12:15:00 | 10435.00 | 10209.77 | 10179.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 12:15:00 | 10435.00 | 10209.77 | 10179.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 13:15:00 | 10476.00 | 10263.01 | 10206.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 12:15:00 | 10372.70 | 10374.15 | 10300.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 12:45:00 | 10370.00 | 10374.15 | 10300.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 15:15:00 | 10300.00 | 10368.07 | 10317.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 15:00:00 | 10485.40 | 10379.96 | 10341.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 11:15:00 | 10265.40 | 10357.57 | 10344.80 | SL hit (close<static) qty=1.00 sl=10300.00 alert=retest2 |

### Cycle 15 — SELL (started 2024-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 12:15:00 | 10239.35 | 10333.93 | 10335.21 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 15:15:00 | 10475.00 | 10331.71 | 10322.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-23 09:15:00 | 10688.00 | 10402.97 | 10356.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 12:15:00 | 10442.75 | 10443.34 | 10389.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 12:45:00 | 10418.45 | 10443.34 | 10389.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 10600.00 | 10471.24 | 10411.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 15:15:00 | 10696.00 | 10471.24 | 10411.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:15:00 | 10625.00 | 10519.81 | 10478.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 09:15:00 | 10387.40 | 10471.41 | 10480.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 09:15:00 | 10387.40 | 10471.41 | 10480.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 11:15:00 | 10190.00 | 10348.08 | 10401.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 13:15:00 | 10359.45 | 10343.32 | 10389.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 13:15:00 | 10359.45 | 10343.32 | 10389.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 13:15:00 | 10359.45 | 10343.32 | 10389.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 14:00:00 | 10359.45 | 10343.32 | 10389.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 10425.60 | 10359.93 | 10389.22 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 10:15:00 | 10833.35 | 10474.61 | 10437.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 12:15:00 | 10856.30 | 10602.85 | 10505.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 10839.50 | 10935.50 | 10826.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 14:15:00 | 10839.50 | 10935.50 | 10826.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 10839.50 | 10935.50 | 10826.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 10839.50 | 10935.50 | 10826.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 10804.90 | 10909.38 | 10824.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 10925.65 | 10909.38 | 10824.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 10898.00 | 10907.11 | 10831.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 14:45:00 | 10965.25 | 10919.32 | 10863.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 09:30:00 | 11024.35 | 10925.08 | 10911.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 10:45:00 | 10974.05 | 10936.23 | 10918.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 09:45:00 | 10989.95 | 10987.95 | 10961.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 10836.25 | 10957.61 | 10950.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:00:00 | 10836.25 | 10957.61 | 10950.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-09 11:15:00 | 10829.85 | 10932.06 | 10939.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 11:15:00 | 10829.85 | 10932.06 | 10939.10 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 12:15:00 | 10950.15 | 10932.08 | 10931.14 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 14:15:00 | 10761.00 | 10901.90 | 10917.84 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 10:15:00 | 11048.25 | 10933.97 | 10927.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 15:15:00 | 11113.70 | 10975.68 | 10951.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 15:15:00 | 10911.00 | 11061.63 | 11021.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 15:15:00 | 10911.00 | 11061.63 | 11021.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 10911.00 | 11061.63 | 11021.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 13:30:00 | 11235.95 | 11133.20 | 11074.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 10:15:00 | 10868.35 | 11053.65 | 11054.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 10:15:00 | 10868.35 | 11053.65 | 11054.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 11:15:00 | 10819.95 | 11006.91 | 11033.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 13:15:00 | 10722.75 | 10721.29 | 10801.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-18 13:45:00 | 10710.85 | 10721.29 | 10801.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 10770.00 | 10731.03 | 10798.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 10770.00 | 10731.03 | 10798.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 10667.15 | 10718.26 | 10786.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:15:00 | 10815.25 | 10718.26 | 10786.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 10654.05 | 10705.41 | 10774.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:30:00 | 10850.40 | 10705.41 | 10774.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 10855.00 | 10735.22 | 10762.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 10855.00 | 10735.22 | 10762.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 10929.00 | 10773.98 | 10777.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 10960.00 | 10773.98 | 10777.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 09:15:00 | 10983.15 | 10815.81 | 10796.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 14:15:00 | 11073.70 | 10939.32 | 10872.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 11:15:00 | 10955.80 | 10966.39 | 10909.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 11:45:00 | 10950.00 | 10966.39 | 10909.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 13:15:00 | 10902.30 | 10952.39 | 10912.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 14:00:00 | 10902.30 | 10952.39 | 10912.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 14:15:00 | 11060.00 | 10973.91 | 10926.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 09:15:00 | 11261.15 | 10987.53 | 10936.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 14:15:00 | 11551.25 | 11563.44 | 11564.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2024-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 14:15:00 | 11551.25 | 11563.44 | 11564.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 10:15:00 | 11381.05 | 11491.99 | 11528.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 14:15:00 | 11521.75 | 11471.03 | 11503.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 14:15:00 | 11521.75 | 11471.03 | 11503.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 11521.75 | 11471.03 | 11503.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 11521.75 | 11471.03 | 11503.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 11507.10 | 11478.24 | 11504.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 11429.95 | 11478.24 | 11504.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 10858.45 | 10987.50 | 11172.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-07 14:15:00 | 10718.95 | 10712.84 | 10944.98 | SL hit (close>ema200) qty=0.50 sl=10712.84 alert=retest2 |

### Cycle 26 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 11380.00 | 10993.87 | 10970.01 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2024-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 13:15:00 | 11174.15 | 11227.71 | 11230.56 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 14:15:00 | 11324.55 | 11247.08 | 11239.10 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 11:15:00 | 11206.20 | 11234.67 | 11235.75 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 13:15:00 | 11286.90 | 11238.78 | 11237.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 14:15:00 | 11434.80 | 11277.98 | 11255.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 11304.00 | 11403.69 | 11351.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 11304.00 | 11403.69 | 11351.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 11304.00 | 11403.69 | 11351.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 11304.00 | 11403.69 | 11351.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 11268.25 | 11376.60 | 11344.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:15:00 | 11226.20 | 11376.60 | 11344.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 11257.50 | 11313.99 | 11320.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 14:15:00 | 11238.65 | 11298.92 | 11312.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 12:15:00 | 11271.30 | 11193.15 | 11243.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 12:15:00 | 11271.30 | 11193.15 | 11243.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 11271.30 | 11193.15 | 11243.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:00:00 | 11271.30 | 11193.15 | 11243.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 11361.70 | 11226.86 | 11254.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:00:00 | 11361.70 | 11226.86 | 11254.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 11436.70 | 11268.83 | 11270.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:45:00 | 11375.00 | 11268.83 | 11270.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2024-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 15:15:00 | 11372.05 | 11289.47 | 11280.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 10:15:00 | 11482.55 | 11342.96 | 11307.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 14:15:00 | 11290.45 | 11408.82 | 11358.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 14:15:00 | 11290.45 | 11408.82 | 11358.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 11290.45 | 11408.82 | 11358.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 15:00:00 | 11290.45 | 11408.82 | 11358.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 11200.90 | 11367.23 | 11344.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:15:00 | 11152.85 | 11367.23 | 11344.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2024-10-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 09:15:00 | 11040.00 | 11301.79 | 11316.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 10:15:00 | 10991.00 | 11239.63 | 11287.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 14:15:00 | 10919.50 | 10886.80 | 11008.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 15:00:00 | 10919.50 | 10886.80 | 11008.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 10734.15 | 10845.74 | 10967.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:00:00 | 10668.70 | 10810.33 | 10940.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 10:15:00 | 10135.26 | 10254.41 | 10393.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-29 14:15:00 | 10379.95 | 10252.16 | 10344.75 | SL hit (close>ema200) qty=0.50 sl=10252.16 alert=retest2 |

### Cycle 34 — BUY (started 2024-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 11:15:00 | 10567.30 | 10388.10 | 10384.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 10637.55 | 10437.99 | 10407.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 15:15:00 | 10482.25 | 10482.65 | 10438.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 09:15:00 | 10334.90 | 10482.65 | 10438.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 10265.00 | 10439.12 | 10422.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:45:00 | 10248.45 | 10439.12 | 10422.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 10:15:00 | 10270.90 | 10405.48 | 10409.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 11:15:00 | 10198.10 | 10364.00 | 10389.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 10430.00 | 10303.41 | 10340.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 10430.00 | 10303.41 | 10340.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 10430.00 | 10303.41 | 10340.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 10430.00 | 10303.41 | 10340.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 10300.00 | 10302.72 | 10337.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:15:00 | 10152.25 | 10302.72 | 10337.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 13:15:00 | 10190.70 | 10078.14 | 10071.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 13:15:00 | 10190.70 | 10078.14 | 10071.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 15:15:00 | 10224.00 | 10125.97 | 10095.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 10:15:00 | 10187.00 | 10196.60 | 10159.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 10:15:00 | 10187.00 | 10196.60 | 10159.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 10187.00 | 10196.60 | 10159.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 11:00:00 | 10300.00 | 10195.71 | 10173.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 14:00:00 | 10207.15 | 10214.36 | 10188.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 15:15:00 | 10209.00 | 10211.50 | 10189.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 11:15:00 | 10221.60 | 10210.93 | 10195.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 10182.00 | 10205.15 | 10193.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:00:00 | 10182.00 | 10205.15 | 10193.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 10169.70 | 10198.06 | 10191.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:45:00 | 10166.50 | 10198.06 | 10191.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-12 13:15:00 | 10142.65 | 10186.98 | 10187.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 10142.65 | 10186.98 | 10187.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 10000.05 | 10149.59 | 10170.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 9909.70 | 9862.94 | 9975.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 9909.70 | 9862.94 | 9975.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 9970.15 | 9884.38 | 9975.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 9985.40 | 9884.38 | 9975.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 10069.90 | 9921.48 | 9983.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 11:30:00 | 10025.00 | 9921.48 | 9983.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 10067.15 | 9950.62 | 9991.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 10067.15 | 9950.62 | 9991.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 10017.40 | 9963.97 | 9993.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 15:00:00 | 9983.20 | 9967.82 | 9992.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 09:15:00 | 10086.00 | 9999.80 | 10003.53 | SL hit (close>static) qty=1.00 sl=10079.95 alert=retest2 |

### Cycle 38 — BUY (started 2024-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 10:15:00 | 10065.50 | 10012.94 | 10009.17 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 12:15:00 | 9839.20 | 9981.24 | 9995.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 14:15:00 | 9745.05 | 9913.70 | 9961.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 10:15:00 | 9864.90 | 9864.37 | 9922.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 11:00:00 | 9864.90 | 9864.37 | 9922.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 9981.25 | 9885.13 | 9913.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:00:00 | 9981.25 | 9885.13 | 9913.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 9832.35 | 9874.57 | 9906.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 9899.00 | 9874.57 | 9906.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 9906.30 | 9880.92 | 9906.14 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2024-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 14:15:00 | 9969.70 | 9926.17 | 9921.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 09:15:00 | 9980.00 | 9942.04 | 9929.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 15:15:00 | 10000.00 | 10017.27 | 9979.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 15:15:00 | 10000.00 | 10017.27 | 9979.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 10000.00 | 10017.27 | 9979.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 09:15:00 | 10377.20 | 10017.27 | 9979.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 13:45:00 | 10081.40 | 10044.48 | 10010.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 14:15:00 | 9957.80 | 10027.14 | 10005.89 | SL hit (close<static) qty=1.00 sl=9975.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 15:15:00 | 10480.00 | 10605.29 | 10618.29 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 09:15:00 | 10628.15 | 10609.60 | 10607.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 14:15:00 | 10830.95 | 10683.73 | 10645.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 09:15:00 | 10769.60 | 10890.71 | 10833.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 10769.60 | 10890.71 | 10833.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 10769.60 | 10890.71 | 10833.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:00:00 | 10769.60 | 10890.71 | 10833.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 10800.00 | 10872.57 | 10830.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 09:15:00 | 10880.90 | 10818.87 | 10815.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 11:15:00 | 10789.95 | 10812.58 | 10813.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 11:15:00 | 10789.95 | 10812.58 | 10813.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 12:15:00 | 10765.40 | 10803.15 | 10809.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 14:15:00 | 10853.45 | 10808.69 | 10810.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 14:15:00 | 10853.45 | 10808.69 | 10810.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 10853.45 | 10808.69 | 10810.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 15:00:00 | 10853.45 | 10808.69 | 10810.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2024-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 15:15:00 | 10850.00 | 10816.96 | 10814.09 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 10740.80 | 10801.72 | 10807.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 10:15:00 | 10734.00 | 10788.18 | 10800.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 12:15:00 | 10467.30 | 10449.76 | 10541.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 12:45:00 | 10452.35 | 10449.76 | 10541.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 10522.85 | 10463.62 | 10532.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 15:00:00 | 10522.85 | 10463.62 | 10532.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 10515.65 | 10474.02 | 10530.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:15:00 | 10570.30 | 10474.02 | 10530.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 10503.70 | 10479.96 | 10528.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 11:45:00 | 10417.05 | 10476.13 | 10518.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 10:00:00 | 10418.00 | 10475.41 | 10503.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 14:15:00 | 9896.20 | 10080.60 | 10210.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 14:15:00 | 9897.10 | 10080.60 | 10210.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-24 09:15:00 | 10176.90 | 9835.11 | 9952.01 | SL hit (close>ema200) qty=0.50 sl=9835.11 alert=retest2 |

### Cycle 46 — BUY (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 14:15:00 | 10182.00 | 10021.62 | 10010.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 10:15:00 | 10245.00 | 10169.72 | 10108.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 12:15:00 | 10164.70 | 10175.15 | 10122.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-27 12:45:00 | 10169.75 | 10175.15 | 10122.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 10158.05 | 10172.91 | 10130.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 15:00:00 | 10158.05 | 10172.91 | 10130.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 10170.00 | 10172.33 | 10134.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 10059.50 | 10172.33 | 10134.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 10027.30 | 10143.33 | 10124.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:00:00 | 10027.30 | 10143.33 | 10124.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 10000.40 | 10114.74 | 10113.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:00:00 | 10000.40 | 10114.74 | 10113.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 11:15:00 | 10010.85 | 10093.96 | 10103.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 12:15:00 | 9963.10 | 10067.79 | 10090.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 9763.35 | 9754.08 | 9873.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 13:45:00 | 9779.95 | 9754.08 | 9873.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 9853.95 | 9779.12 | 9855.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:15:00 | 9887.10 | 9779.12 | 9855.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 9870.00 | 9797.30 | 9856.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:45:00 | 9910.00 | 9797.30 | 9856.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 9886.20 | 9815.08 | 9859.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 11:30:00 | 9874.95 | 9815.08 | 9859.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 13:15:00 | 9817.05 | 9828.58 | 9858.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 10:45:00 | 9782.45 | 9808.99 | 9840.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 10:30:00 | 9770.80 | 9742.65 | 9779.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 09:15:00 | 9724.75 | 9680.55 | 9677.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 09:15:00 | 9724.75 | 9680.55 | 9677.79 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 9643.85 | 9673.21 | 9674.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 11:15:00 | 9597.95 | 9658.16 | 9667.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 14:15:00 | 9636.60 | 9630.15 | 9650.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 14:15:00 | 9636.60 | 9630.15 | 9650.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 9636.60 | 9630.15 | 9650.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 9636.60 | 9630.15 | 9650.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 9700.20 | 9643.33 | 9652.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 9700.20 | 9643.33 | 9652.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 9682.20 | 9651.10 | 9655.20 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2025-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 11:15:00 | 9697.85 | 9660.45 | 9659.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-09 12:15:00 | 9729.70 | 9674.30 | 9665.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 14:15:00 | 9621.00 | 9672.85 | 9666.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 14:15:00 | 9621.00 | 9672.85 | 9666.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 9621.00 | 9672.85 | 9666.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 15:00:00 | 9621.00 | 9672.85 | 9666.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2025-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 15:15:00 | 9601.00 | 9658.48 | 9660.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 9508.80 | 9628.54 | 9647.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 12:15:00 | 9090.05 | 9083.03 | 9237.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 13:00:00 | 9090.05 | 9083.03 | 9237.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 9221.00 | 9128.61 | 9221.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 9353.15 | 9128.61 | 9221.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 9379.95 | 9178.88 | 9235.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:00:00 | 9379.95 | 9178.88 | 9235.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 9470.95 | 9237.29 | 9257.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 11:00:00 | 9470.95 | 9237.29 | 9257.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 9514.25 | 9292.68 | 9280.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 9554.80 | 9459.05 | 9378.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 13:15:00 | 9672.20 | 9675.26 | 9578.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 13:45:00 | 9686.85 | 9675.26 | 9578.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 9582.70 | 9696.58 | 9661.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 9582.70 | 9696.58 | 9661.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 9674.90 | 9692.24 | 9662.44 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2025-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 15:15:00 | 9564.75 | 9631.20 | 9639.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 9349.20 | 9574.80 | 9613.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 9493.90 | 9362.09 | 9453.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 9493.90 | 9362.09 | 9453.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 9493.90 | 9362.09 | 9453.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 9493.90 | 9362.09 | 9453.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 9496.60 | 9388.99 | 9457.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:15:00 | 9528.25 | 9388.99 | 9457.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2025-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 15:15:00 | 9594.90 | 9506.84 | 9496.07 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 9199.05 | 9442.62 | 9474.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 9096.15 | 9373.33 | 9440.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 14:15:00 | 9317.35 | 9274.00 | 9363.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-27 15:00:00 | 9317.35 | 9274.00 | 9363.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 15:15:00 | 9350.00 | 9289.20 | 9361.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 09:15:00 | 9181.15 | 9289.20 | 9361.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 9139.85 | 9259.33 | 9341.66 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 9450.30 | 9333.06 | 9332.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 11:15:00 | 9509.45 | 9368.34 | 9348.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 9955.85 | 10192.62 | 10051.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 9955.85 | 10192.62 | 10051.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 9955.85 | 10192.62 | 10051.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 9955.85 | 10192.62 | 10051.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 10047.65 | 10163.62 | 10051.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:00:00 | 10128.65 | 10156.63 | 10058.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 9856.95 | 10080.91 | 10039.98 | SL hit (close<static) qty=1.00 sl=9900.00 alert=retest2 |

### Cycle 57 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 9685.05 | 9958.79 | 9988.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 12:15:00 | 9565.95 | 9880.22 | 9950.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 14:15:00 | 9750.00 | 9630.49 | 9734.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 14:15:00 | 9750.00 | 9630.49 | 9734.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 9750.00 | 9630.49 | 9734.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 15:00:00 | 9750.00 | 9630.49 | 9734.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 9750.05 | 9654.40 | 9735.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 9848.70 | 9654.40 | 9735.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 11:15:00 | 9992.05 | 9797.97 | 9786.99 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 09:15:00 | 9686.40 | 9773.03 | 9784.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 11:15:00 | 9025.00 | 9625.17 | 9715.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 09:15:00 | 9150.55 | 9128.97 | 9291.33 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 12:00:00 | 9085.00 | 9119.09 | 9258.60 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 09:15:00 | 9040.30 | 9156.58 | 9234.81 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 09:45:00 | 9003.00 | 9126.45 | 9214.00 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 8630.75 | 8858.06 | 9012.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-02-12 11:15:00 | 8925.40 | 8867.49 | 8989.61 | SL hit (close>ema200) qty=0.50 sl=8867.49 alert=retest1 |

### Cycle 60 — BUY (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 12:15:00 | 8786.45 | 8706.09 | 8696.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 8857.25 | 8772.92 | 8734.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 8794.50 | 8847.29 | 8800.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 8794.50 | 8847.29 | 8800.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 8794.50 | 8847.29 | 8800.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 8794.50 | 8847.29 | 8800.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 8811.20 | 8840.08 | 8801.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:15:00 | 8845.00 | 8840.08 | 8801.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 8854.45 | 8842.95 | 8806.30 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 8729.00 | 8832.48 | 8836.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 8534.35 | 8718.56 | 8772.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 14:15:00 | 8716.30 | 8641.58 | 8704.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 14:15:00 | 8716.30 | 8641.58 | 8704.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 8716.30 | 8641.58 | 8704.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 8716.30 | 8641.58 | 8704.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 8769.95 | 8667.25 | 8710.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:15:00 | 8807.85 | 8667.25 | 8710.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 10:15:00 | 8802.00 | 8699.86 | 8717.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 11:00:00 | 8802.00 | 8699.86 | 8717.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 11:15:00 | 8760.45 | 8711.98 | 8721.72 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2025-03-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 12:15:00 | 8850.00 | 8739.58 | 8733.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-03 13:15:00 | 8898.05 | 8771.27 | 8748.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-11 09:15:00 | 9741.30 | 9754.25 | 9585.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-11 09:30:00 | 9716.95 | 9754.25 | 9585.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 9662.00 | 9733.89 | 9618.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:00:00 | 9662.00 | 9733.89 | 9618.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 9679.35 | 9747.81 | 9679.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:30:00 | 9650.00 | 9747.81 | 9679.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 9728.00 | 9743.84 | 9683.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 14:15:00 | 9753.95 | 9739.48 | 9687.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-24 09:15:00 | 10729.35 | 10572.18 | 10433.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 11028.05 | 11170.88 | 11177.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 11005.00 | 11137.70 | 11161.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 10753.75 | 10732.15 | 10878.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 10772.40 | 10732.15 | 10878.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 10690.00 | 10723.72 | 10861.34 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2025-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 15:15:00 | 10915.00 | 10875.33 | 10875.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 11285.80 | 10957.42 | 10912.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 13:15:00 | 12163.00 | 12164.10 | 11951.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 14:00:00 | 12163.00 | 12164.10 | 11951.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 12873.00 | 13065.43 | 12936.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 12873.00 | 13065.43 | 12936.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 12707.00 | 12993.74 | 12915.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 12707.00 | 12993.74 | 12915.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2025-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 14:15:00 | 12762.00 | 12866.64 | 12872.91 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 13089.00 | 12886.53 | 12879.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 12:15:00 | 13146.00 | 12974.62 | 12924.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 13311.00 | 13362.02 | 13220.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 11:15:00 | 13220.00 | 13330.73 | 13230.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 13220.00 | 13330.73 | 13230.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:45:00 | 13236.00 | 13330.73 | 13230.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 13203.00 | 13305.18 | 13227.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 12:45:00 | 13231.00 | 13305.18 | 13227.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 13:15:00 | 13205.00 | 13285.15 | 13225.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:00:00 | 13205.00 | 13285.15 | 13225.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 13185.00 | 13265.12 | 13222.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 15:00:00 | 13185.00 | 13265.12 | 13222.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 13150.00 | 13242.09 | 13215.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 13255.00 | 13242.09 | 13215.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 11:15:00 | 13033.00 | 13197.75 | 13201.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 11:15:00 | 13033.00 | 13197.75 | 13201.81 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 09:15:00 | 13378.00 | 13194.30 | 13171.62 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 13105.00 | 13209.94 | 13224.05 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 10:15:00 | 13331.00 | 13244.24 | 13237.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-09 11:15:00 | 13461.00 | 13287.59 | 13258.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-12 09:15:00 | 13189.00 | 13331.66 | 13297.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 13189.00 | 13331.66 | 13297.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 13189.00 | 13331.66 | 13297.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:30:00 | 13770.00 | 13572.82 | 13456.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 13781.00 | 13684.38 | 13572.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 11:15:00 | 13780.00 | 13691.52 | 13595.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 14:15:00 | 13705.00 | 13847.28 | 13852.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 14:15:00 | 13705.00 | 13847.28 | 13852.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 09:15:00 | 13644.00 | 13784.66 | 13821.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 13743.00 | 13681.46 | 13740.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 13743.00 | 13681.46 | 13740.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 13743.00 | 13681.46 | 13740.77 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 12:15:00 | 14293.00 | 13852.56 | 13807.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 14652.00 | 14152.43 | 13975.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 11:15:00 | 15699.00 | 15761.81 | 15426.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 12:00:00 | 15699.00 | 15761.81 | 15426.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 16013.00 | 16143.60 | 16004.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 16013.00 | 16143.60 | 16004.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 16085.00 | 16131.88 | 16011.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 16001.00 | 16131.88 | 16011.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 16079.00 | 16121.30 | 16017.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 16079.00 | 16121.30 | 16017.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 16015.00 | 16100.04 | 16017.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:45:00 | 16014.00 | 16100.04 | 16017.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 15970.00 | 16074.03 | 16013.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 14:15:00 | 16019.00 | 16074.03 | 16013.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 14:45:00 | 16048.00 | 16085.83 | 16024.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 13:15:00 | 16515.00 | 16667.78 | 16681.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 13:15:00 | 16515.00 | 16667.78 | 16681.87 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 16926.00 | 16691.12 | 16682.49 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2025-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 11:15:00 | 16616.00 | 16780.37 | 16783.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 12:15:00 | 16599.00 | 16744.10 | 16766.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 16747.00 | 16677.19 | 16719.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 16747.00 | 16677.19 | 16719.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 16747.00 | 16677.19 | 16719.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 16790.00 | 16677.19 | 16719.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 16653.00 | 16672.35 | 16713.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 11:30:00 | 16612.00 | 16658.28 | 16703.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 12:45:00 | 16625.00 | 16654.63 | 16697.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 13:15:00 | 16612.00 | 16654.63 | 16697.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 14:00:00 | 16601.00 | 16643.90 | 16689.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 16776.00 | 16673.29 | 16691.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 16776.00 | 16673.29 | 16691.43 | SL hit (close>static) qty=1.00 sl=16752.00 alert=retest2 |

### Cycle 76 — BUY (started 2025-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 11:15:00 | 16875.00 | 16735.51 | 16717.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 09:15:00 | 16920.00 | 16814.64 | 16765.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 11:15:00 | 17063.00 | 17115.55 | 17036.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 11:15:00 | 17063.00 | 17115.55 | 17036.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 17063.00 | 17115.55 | 17036.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 17095.00 | 17115.55 | 17036.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 17061.00 | 17104.64 | 17038.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 17000.00 | 17104.64 | 17038.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 17101.00 | 17103.91 | 17044.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 09:15:00 | 17174.00 | 17096.31 | 17051.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 11:15:00 | 16945.00 | 17067.81 | 17050.46 | SL hit (close<static) qty=1.00 sl=17032.00 alert=retest2 |

### Cycle 77 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 16820.00 | 17018.25 | 17029.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 09:15:00 | 16798.00 | 16916.01 | 16972.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 16929.00 | 16918.61 | 16968.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 16929.00 | 16918.61 | 16968.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 16929.00 | 16918.61 | 16968.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 16929.00 | 16918.61 | 16968.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 16958.00 | 16926.49 | 16967.66 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 17098.00 | 16991.54 | 16980.15 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 10:15:00 | 16876.00 | 16997.15 | 17010.81 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 11:15:00 | 17094.00 | 17022.75 | 17013.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 12:15:00 | 17136.00 | 17045.40 | 17024.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 17399.00 | 17441.83 | 17277.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 15:00:00 | 17399.00 | 17441.83 | 17277.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 17416.00 | 17540.36 | 17455.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 17416.00 | 17540.36 | 17455.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 17347.00 | 17501.69 | 17445.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:00:00 | 17347.00 | 17501.69 | 17445.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 13:15:00 | 17268.00 | 17407.16 | 17409.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 16980.00 | 17261.17 | 17336.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 16934.00 | 16904.75 | 17014.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 09:45:00 | 16930.00 | 16904.75 | 17014.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 16631.00 | 16819.37 | 16915.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:30:00 | 16618.00 | 16770.68 | 16874.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 16594.00 | 16708.82 | 16805.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:15:00 | 16567.00 | 16571.90 | 16627.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:15:00 | 15787.10 | 16043.12 | 16282.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:15:00 | 15764.30 | 16043.12 | 16282.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:15:00 | 15738.65 | 16043.12 | 16282.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 15392.00 | 15321.73 | 15570.83 | SL hit (close>ema200) qty=0.50 sl=15321.73 alert=retest2 |

### Cycle 82 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 14495.00 | 14240.45 | 14208.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 10:15:00 | 14687.00 | 14329.76 | 14251.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 14832.00 | 14948.08 | 14756.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 09:30:00 | 14810.00 | 14948.08 | 14756.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 14726.00 | 14903.67 | 14754.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:00:00 | 14726.00 | 14903.67 | 14754.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 14714.00 | 14865.73 | 14750.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:30:00 | 14697.00 | 14865.73 | 14750.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 14677.00 | 14827.99 | 14743.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:00:00 | 14677.00 | 14827.99 | 14743.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 14699.00 | 14802.19 | 14739.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:45:00 | 14648.00 | 14802.19 | 14739.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 14713.00 | 14815.56 | 14764.09 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 14445.00 | 14713.52 | 14724.91 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 14883.00 | 14720.62 | 14699.45 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 11:15:00 | 14625.00 | 14713.30 | 14720.35 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 14955.00 | 14706.73 | 14706.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 11:15:00 | 15066.00 | 14815.59 | 14758.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 11:15:00 | 14973.00 | 15072.31 | 14995.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 11:15:00 | 14973.00 | 15072.31 | 14995.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 14973.00 | 15072.31 | 14995.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:00:00 | 14973.00 | 15072.31 | 14995.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 15028.00 | 15063.45 | 14998.48 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 09:15:00 | 14793.00 | 14972.31 | 14973.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 10:15:00 | 14774.00 | 14932.65 | 14954.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 14641.00 | 14600.25 | 14705.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 10:00:00 | 14641.00 | 14600.25 | 14705.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 14837.00 | 14647.60 | 14717.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 14837.00 | 14647.60 | 14717.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 14724.00 | 14662.88 | 14718.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:30:00 | 14843.00 | 14662.88 | 14718.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 14965.00 | 14695.00 | 14708.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 14965.00 | 14695.00 | 14708.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 10:15:00 | 14959.00 | 14747.80 | 14731.22 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 12:15:00 | 14712.00 | 14760.24 | 14761.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 14646.00 | 14728.55 | 14745.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 13948.00 | 13929.77 | 14142.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 13948.00 | 13929.77 | 14142.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 14103.00 | 13944.13 | 14027.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 14103.00 | 13944.13 | 14027.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 14025.00 | 13960.31 | 14027.55 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 12:15:00 | 14126.00 | 14048.50 | 14046.84 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 13:15:00 | 14024.00 | 14043.60 | 14044.77 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 14255.00 | 14078.09 | 14059.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 14:15:00 | 14297.00 | 14190.03 | 14127.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 14129.00 | 14192.38 | 14140.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 14129.00 | 14192.38 | 14140.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 14129.00 | 14192.38 | 14140.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:30:00 | 14130.00 | 14192.38 | 14140.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 13891.00 | 14132.10 | 14117.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 13891.00 | 14132.10 | 14117.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 13878.00 | 14081.28 | 14095.64 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 15:15:00 | 14135.00 | 13973.81 | 13957.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 10:15:00 | 14170.00 | 14035.00 | 13989.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 11:15:00 | 14008.00 | 14029.60 | 13990.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 11:15:00 | 14008.00 | 14029.60 | 13990.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 14008.00 | 14029.60 | 13990.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:45:00 | 14009.00 | 14029.60 | 13990.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 13958.00 | 14015.28 | 13987.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:45:00 | 13976.00 | 14015.28 | 13987.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 14015.00 | 14015.22 | 13990.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 14097.00 | 14005.98 | 13990.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 14537.00 | 14640.43 | 14641.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 14537.00 | 14640.43 | 14641.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 14465.00 | 14578.60 | 14609.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 14130.00 | 14119.76 | 14243.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 13:15:00 | 14236.00 | 14174.21 | 14232.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 14236.00 | 14174.21 | 14232.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:30:00 | 14233.00 | 14174.21 | 14232.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 14108.00 | 14160.97 | 14221.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:45:00 | 14225.00 | 14160.97 | 14221.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 13972.00 | 14115.02 | 14189.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:45:00 | 13940.00 | 14079.02 | 14166.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:15:00 | 13887.00 | 14028.97 | 14127.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 14:15:00 | 13243.00 | 13442.76 | 13629.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 13388.00 | 13380.74 | 13499.25 | SL hit (close>ema200) qty=0.50 sl=13380.74 alert=retest2 |

### Cycle 96 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 13859.00 | 13555.22 | 13554.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 14064.00 | 13801.25 | 13692.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 14090.00 | 14119.70 | 14018.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:00:00 | 14090.00 | 14119.70 | 14018.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 13970.00 | 14089.76 | 14013.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 13970.00 | 14089.76 | 14013.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 13975.00 | 14066.81 | 14010.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 13:00:00 | 14010.00 | 14055.45 | 14010.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 14:30:00 | 14089.00 | 14028.41 | 14005.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 14009.00 | 14021.72 | 14004.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:45:00 | 14094.00 | 14029.38 | 14009.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 14097.00 | 14146.73 | 14110.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:45:00 | 14113.00 | 14146.73 | 14110.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 14120.00 | 14141.39 | 14110.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 14001.00 | 14141.39 | 14110.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 14097.00 | 14132.51 | 14109.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 14059.00 | 14132.51 | 14109.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 13992.00 | 14104.41 | 14098.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 13992.00 | 14104.41 | 14098.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-13 11:15:00 | 13999.00 | 14083.33 | 14089.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 13999.00 | 14083.33 | 14089.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 13984.00 | 14037.82 | 14057.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 13997.00 | 13993.94 | 14025.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 13997.00 | 13993.94 | 14025.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 13997.00 | 13993.94 | 14025.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 13987.00 | 13993.94 | 14025.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 14034.00 | 14001.96 | 14025.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 14023.00 | 14001.96 | 14025.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 14063.00 | 14014.16 | 14029.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 14077.00 | 14014.16 | 14029.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 14056.00 | 14028.74 | 14033.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 14071.00 | 14028.74 | 14033.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 14020.00 | 14028.86 | 14032.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:15:00 | 13992.00 | 14028.86 | 14032.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:45:00 | 13979.00 | 14016.91 | 14026.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 13:15:00 | 14080.00 | 14034.02 | 14032.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 14080.00 | 14034.02 | 14032.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 14095.00 | 14047.43 | 14039.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 13972.00 | 14040.54 | 14038.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 13972.00 | 14040.54 | 14038.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 13972.00 | 14040.54 | 14038.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 13972.00 | 14040.54 | 14038.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 14051.00 | 14042.63 | 14039.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 14011.00 | 14042.63 | 14039.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 14086.00 | 14051.31 | 14044.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:30:00 | 14088.00 | 14051.31 | 14044.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 15:15:00 | 13982.00 | 14037.44 | 14038.46 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 14096.00 | 14049.16 | 14043.69 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 11:15:00 | 14056.00 | 14071.08 | 14072.88 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 10:15:00 | 14097.00 | 14057.23 | 14055.98 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 13:15:00 | 14021.00 | 14050.46 | 14053.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 09:15:00 | 13986.00 | 14027.26 | 14041.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 11:15:00 | 13921.00 | 13903.84 | 13951.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 12:00:00 | 13921.00 | 13903.84 | 13951.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 13924.00 | 13904.70 | 13943.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:45:00 | 13935.00 | 13904.70 | 13943.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 13936.00 | 13910.96 | 13942.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:30:00 | 13970.00 | 13910.96 | 13942.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 13970.00 | 13922.77 | 13945.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 13928.00 | 13922.77 | 13945.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 13854.00 | 13909.01 | 13936.89 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 10:15:00 | 13958.00 | 13923.26 | 13918.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 12:15:00 | 14101.00 | 13971.89 | 13942.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 10:15:00 | 13951.00 | 14037.18 | 13995.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 10:15:00 | 13951.00 | 14037.18 | 13995.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 13951.00 | 14037.18 | 13995.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:45:00 | 13970.00 | 14037.18 | 13995.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 13879.00 | 14005.54 | 13984.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:45:00 | 13846.00 | 14005.54 | 13984.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 12:15:00 | 13748.00 | 13954.03 | 13963.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 14:15:00 | 13623.00 | 13845.26 | 13909.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 13449.00 | 13436.27 | 13578.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:00:00 | 13449.00 | 13436.27 | 13578.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 13566.00 | 13468.26 | 13568.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 13582.00 | 13468.26 | 13568.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 13595.00 | 13493.60 | 13571.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 13560.00 | 13493.60 | 13571.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 13648.00 | 13524.48 | 13578.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:15:00 | 13674.00 | 13524.48 | 13578.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 13635.00 | 13546.59 | 13583.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 13665.00 | 13546.59 | 13583.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 13521.00 | 13554.98 | 13581.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:15:00 | 13498.00 | 13554.98 | 13581.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 14:45:00 | 13450.00 | 13524.26 | 13562.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 13707.00 | 13537.89 | 13560.79 | SL hit (close>static) qty=1.00 sl=13648.00 alert=retest2 |

### Cycle 106 — BUY (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 10:15:00 | 13972.00 | 13624.71 | 13598.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 14116.00 | 13828.33 | 13723.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 13871.00 | 13990.60 | 13884.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 13871.00 | 13990.60 | 13884.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 13871.00 | 13990.60 | 13884.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 13871.00 | 13990.60 | 13884.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 13934.00 | 13979.28 | 13888.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:30:00 | 13812.00 | 13979.28 | 13888.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 13905.00 | 13960.70 | 13895.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:45:00 | 13894.00 | 13960.70 | 13895.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 13841.00 | 13936.76 | 13890.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 13841.00 | 13936.76 | 13890.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 13741.00 | 13897.61 | 13877.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 13741.00 | 13897.61 | 13877.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 13946.00 | 13887.24 | 13875.24 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 13736.00 | 13868.31 | 13877.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 11:15:00 | 13675.00 | 13810.72 | 13848.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 12:15:00 | 13847.00 | 13817.97 | 13848.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 12:15:00 | 13847.00 | 13817.97 | 13848.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 13847.00 | 13817.97 | 13848.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:45:00 | 13866.00 | 13817.97 | 13848.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 13892.00 | 13832.78 | 13852.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:00:00 | 13892.00 | 13832.78 | 13852.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 13767.00 | 13819.62 | 13844.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 11:00:00 | 13741.00 | 13792.97 | 13825.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 11:30:00 | 13727.00 | 13787.17 | 13819.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 12:30:00 | 13745.00 | 13778.34 | 13812.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 14:30:00 | 13749.00 | 13779.78 | 13806.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 13792.00 | 13782.22 | 13805.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 13827.00 | 13782.22 | 13805.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 13884.00 | 13802.58 | 13812.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:15:00 | 13926.00 | 13802.58 | 13812.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 13875.00 | 13817.06 | 13818.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:45:00 | 13908.00 | 13817.06 | 13818.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 13937.00 | 13841.05 | 13829.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 13937.00 | 13841.05 | 13829.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 12:15:00 | 13958.00 | 13864.44 | 13840.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 13791.00 | 13892.15 | 13866.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 13791.00 | 13892.15 | 13866.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 13791.00 | 13892.15 | 13866.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 13791.00 | 13892.15 | 13866.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 13756.00 | 13864.92 | 13856.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 13756.00 | 13864.92 | 13856.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2025-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 12:15:00 | 13820.00 | 13846.19 | 13849.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 13756.00 | 13829.08 | 13840.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 13467.00 | 13367.03 | 13475.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 13467.00 | 13367.03 | 13475.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 13467.00 | 13367.03 | 13475.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 13467.00 | 13367.03 | 13475.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 13475.00 | 13388.62 | 13475.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 13481.00 | 13388.62 | 13475.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 13476.00 | 13406.10 | 13475.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:30:00 | 13483.00 | 13406.10 | 13475.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 13436.00 | 13412.08 | 13472.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 14:30:00 | 13422.00 | 13429.33 | 13470.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:45:00 | 13418.00 | 13435.11 | 13463.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 09:15:00 | 12750.90 | 12964.21 | 13096.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 09:15:00 | 12747.10 | 12964.21 | 13096.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 11:15:00 | 12971.00 | 12952.57 | 13067.25 | SL hit (close>ema200) qty=0.50 sl=12952.57 alert=retest2 |

### Cycle 110 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 12198.00 | 11912.24 | 11891.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 10:15:00 | 12345.00 | 11998.79 | 11932.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 12502.00 | 12513.54 | 12355.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 12584.00 | 12513.54 | 12355.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 12475.00 | 12561.05 | 12456.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:30:00 | 12475.00 | 12561.05 | 12456.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 12472.00 | 12543.24 | 12458.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 12679.00 | 12543.24 | 12458.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 14:15:00 | 12426.00 | 12520.67 | 12486.88 | SL hit (close<static) qty=1.00 sl=12453.00 alert=retest2 |

### Cycle 111 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 12433.00 | 12474.56 | 12475.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 14:15:00 | 12337.00 | 12440.08 | 12459.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 12290.00 | 12141.54 | 12255.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 12290.00 | 12141.54 | 12255.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 12290.00 | 12141.54 | 12255.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 12247.00 | 12141.54 | 12255.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 12255.00 | 12164.23 | 12255.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 12263.00 | 12164.23 | 12255.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 12305.00 | 12192.39 | 12260.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 12325.00 | 12192.39 | 12260.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 12228.00 | 12199.51 | 12257.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:15:00 | 12300.00 | 12199.51 | 12257.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 12202.00 | 12200.01 | 12252.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 12165.00 | 12209.27 | 12242.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:45:00 | 12192.00 | 12197.29 | 12230.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:45:00 | 12183.00 | 12186.15 | 12210.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 13:15:00 | 12290.00 | 12234.07 | 12228.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 13:15:00 | 12290.00 | 12234.07 | 12228.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 12351.00 | 12257.45 | 12239.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 12614.00 | 12653.23 | 12523.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 11:00:00 | 12614.00 | 12653.23 | 12523.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 13202.00 | 13392.90 | 13258.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 13:00:00 | 13202.00 | 13392.90 | 13258.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 13147.00 | 13343.72 | 13248.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:00:00 | 13147.00 | 13343.72 | 13248.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 13195.00 | 13289.54 | 13238.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 13027.00 | 13289.54 | 13238.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 13072.00 | 13226.59 | 13217.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 11:00:00 | 13072.00 | 13226.59 | 13217.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2026-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 11:15:00 | 13063.00 | 13193.87 | 13203.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 11:15:00 | 12769.00 | 13004.86 | 13089.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 12828.00 | 12807.64 | 12942.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 09:45:00 | 12805.00 | 12807.64 | 12942.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 12900.00 | 12826.49 | 12927.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:45:00 | 12920.00 | 12826.49 | 12927.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 12935.00 | 12848.19 | 12928.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:00:00 | 12935.00 | 12848.19 | 12928.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 12845.00 | 12847.55 | 12920.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:30:00 | 12812.00 | 12887.93 | 12921.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 11:15:00 | 13005.00 | 12917.12 | 12929.48 | SL hit (close>static) qty=1.00 sl=12943.00 alert=retest2 |

### Cycle 114 — BUY (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 09:15:00 | 12972.00 | 12908.80 | 12902.65 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 12801.00 | 12881.25 | 12891.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 12638.00 | 12832.60 | 12868.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 12783.00 | 12607.91 | 12682.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 12783.00 | 12607.91 | 12682.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 12783.00 | 12607.91 | 12682.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 12783.00 | 12607.91 | 12682.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 12663.00 | 12618.93 | 12680.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 12630.00 | 12636.54 | 12682.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 13:15:00 | 12814.00 | 12692.59 | 12701.53 | SL hit (close>static) qty=1.00 sl=12791.00 alert=retest2 |

### Cycle 116 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 12889.00 | 12731.87 | 12718.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 15:15:00 | 12931.00 | 12771.70 | 12737.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 12779.00 | 12789.02 | 12755.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 11:45:00 | 12810.00 | 12789.02 | 12755.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 12701.00 | 12769.97 | 12752.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 12701.00 | 12769.97 | 12752.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 12658.00 | 12747.58 | 12744.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 12658.00 | 12747.58 | 12744.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 15:15:00 | 12660.00 | 12730.06 | 12736.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 12596.00 | 12671.81 | 12703.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 12737.00 | 12684.85 | 12706.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 12737.00 | 12684.85 | 12706.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 12737.00 | 12684.85 | 12706.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 12737.00 | 12684.85 | 12706.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 12832.00 | 12714.28 | 12717.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 13133.00 | 12714.28 | 12717.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 13147.00 | 12800.82 | 12756.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 13299.00 | 12900.46 | 12806.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 13386.00 | 13496.89 | 13230.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 11:00:00 | 13386.00 | 13496.89 | 13230.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 13221.00 | 13387.66 | 13241.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:00:00 | 13221.00 | 13387.66 | 13241.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 13320.00 | 13374.13 | 13248.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 13371.00 | 13351.32 | 13268.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 11:30:00 | 13365.00 | 13358.06 | 13279.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 12:30:00 | 13370.00 | 13341.85 | 13278.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 13:30:00 | 13369.00 | 13346.28 | 13286.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 13348.00 | 13487.88 | 13390.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 13348.00 | 13487.88 | 13390.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 13321.00 | 13454.50 | 13383.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 13:45:00 | 13431.00 | 13440.60 | 13383.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 13202.00 | 13392.88 | 13367.43 | SL hit (close<static) qty=1.00 sl=13210.00 alert=retest2 |

### Cycle 119 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 13120.00 | 13338.30 | 13344.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 12935.00 | 13213.28 | 13284.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 13205.00 | 13137.57 | 13218.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 13205.00 | 13137.57 | 13218.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 13274.00 | 13164.86 | 13223.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 13418.00 | 13164.86 | 13223.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 13377.00 | 13207.29 | 13237.19 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 13512.00 | 13268.23 | 13262.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 14:15:00 | 14028.00 | 13490.55 | 13374.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 11:15:00 | 13585.00 | 13639.18 | 13497.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 11:15:00 | 13585.00 | 13639.18 | 13497.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 13585.00 | 13639.18 | 13497.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:30:00 | 13548.00 | 13639.18 | 13497.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 13548.00 | 13620.94 | 13502.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:30:00 | 13451.00 | 13620.94 | 13502.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 13352.00 | 13553.83 | 13507.81 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 13325.00 | 13462.01 | 13471.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 13110.00 | 13337.79 | 13402.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 13086.00 | 13085.60 | 13212.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 13086.00 | 13085.60 | 13212.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 13086.00 | 13085.60 | 13212.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:45:00 | 13193.00 | 13085.60 | 13212.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 13213.00 | 13111.08 | 13212.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 13213.00 | 13111.08 | 13212.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 13298.00 | 13148.47 | 13220.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 13298.00 | 13148.47 | 13220.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 13328.00 | 13184.37 | 13229.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:00:00 | 13328.00 | 13184.37 | 13229.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2026-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 15:15:00 | 13375.00 | 13276.22 | 13265.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 12:15:00 | 13443.00 | 13363.70 | 13315.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 13376.00 | 13385.91 | 13343.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 13376.00 | 13385.91 | 13343.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 13376.00 | 13385.91 | 13343.27 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 13230.00 | 13341.37 | 13351.64 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 13:15:00 | 13425.00 | 13360.27 | 13358.58 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 13110.00 | 13320.28 | 13341.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 13062.00 | 13185.23 | 13260.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 13126.00 | 13069.99 | 13146.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 13126.00 | 13069.99 | 13146.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 13249.00 | 13112.84 | 13153.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 13252.00 | 13112.84 | 13153.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 13262.00 | 13142.67 | 13163.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 13240.00 | 13142.67 | 13163.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 13272.00 | 13184.11 | 13179.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 13284.00 | 13228.78 | 13204.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 13309.00 | 13353.11 | 13315.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 15:15:00 | 13309.00 | 13353.11 | 13315.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 13309.00 | 13353.11 | 13315.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 13408.00 | 13353.11 | 13315.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 13396.00 | 13361.69 | 13322.82 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 10:15:00 | 13211.00 | 13329.69 | 13331.90 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 13385.00 | 13337.23 | 13333.51 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 13206.00 | 13315.43 | 13324.52 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 13474.00 | 13340.06 | 13323.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 13563.00 | 13431.48 | 13380.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 13528.00 | 13647.00 | 13544.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 13528.00 | 13647.00 | 13544.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 13528.00 | 13647.00 | 13544.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 13528.00 | 13647.00 | 13544.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 13522.00 | 13622.00 | 13542.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:45:00 | 13482.00 | 13622.00 | 13542.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 13522.00 | 13602.00 | 13540.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:30:00 | 13541.00 | 13602.00 | 13540.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 13576.00 | 13596.80 | 13543.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:45:00 | 13602.00 | 13575.55 | 13543.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 15:15:00 | 13491.00 | 13558.64 | 13538.44 | SL hit (close<static) qty=1.00 sl=13516.00 alert=retest2 |

### Cycle 131 — SELL (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 15:15:00 | 14720.00 | 14798.90 | 14804.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 09:15:00 | 14512.00 | 14741.52 | 14777.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 14640.00 | 14566.88 | 14644.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 14640.00 | 14566.88 | 14644.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 14640.00 | 14566.88 | 14644.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 14640.00 | 14566.88 | 14644.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 14679.00 | 14589.30 | 14647.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:45:00 | 14741.00 | 14589.30 | 14647.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 14566.00 | 14584.64 | 14639.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 14474.00 | 14558.85 | 14612.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:45:00 | 14442.00 | 14521.28 | 14590.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:15:00 | 13750.30 | 14028.14 | 14197.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 14:15:00 | 14042.00 | 14003.93 | 14128.65 | SL hit (close>ema200) qty=0.50 sl=14003.93 alert=retest2 |

### Cycle 132 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 13025.00 | 12819.61 | 12816.90 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 12550.00 | 12815.73 | 12825.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 12315.00 | 12715.59 | 12778.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 12718.00 | 12321.07 | 12439.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 12718.00 | 12321.07 | 12439.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 12718.00 | 12321.07 | 12439.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 12718.00 | 12321.07 | 12439.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 12529.00 | 12362.65 | 12447.81 | EMA400 retest candle locked (from downside) |

### Cycle 134 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 12970.00 | 12578.10 | 12536.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 13014.00 | 12758.93 | 12665.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 14966.00 | 15012.30 | 14883.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 15:15:00 | 14850.00 | 14955.24 | 14909.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 14850.00 | 14955.24 | 14909.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:30:00 | 15084.00 | 14991.59 | 14930.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 14969.00 | 15002.79 | 14970.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 14:15:00 | 15045.00 | 15303.85 | 15308.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2026-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 14:15:00 | 15045.00 | 15303.85 | 15308.95 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 15394.00 | 15270.29 | 15269.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 15541.00 | 15366.28 | 15323.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 15342.00 | 15396.54 | 15351.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 15342.00 | 15396.54 | 15351.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 15342.00 | 15396.54 | 15351.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 15342.00 | 15396.54 | 15351.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 15400.00 | 15397.24 | 15355.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 15400.00 | 15397.24 | 15355.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 15410.00 | 15399.79 | 15360.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 15284.00 | 15399.79 | 15360.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 15225.00 | 15364.83 | 15348.37 | EMA400 retest candle locked (from upside) |

### Cycle 137 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 15210.00 | 15333.86 | 15335.79 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 15376.00 | 15342.29 | 15339.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 12:15:00 | 15440.00 | 15361.83 | 15348.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 13:15:00 | 15412.00 | 15442.07 | 15407.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 13:15:00 | 15412.00 | 15442.07 | 15407.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 15412.00 | 15442.07 | 15407.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 13:30:00 | 15389.00 | 15442.07 | 15407.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 15467.00 | 15447.06 | 15413.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 14:30:00 | 15464.00 | 15447.06 | 15413.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 15560.00 | 15477.32 | 15433.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 11:45:00 | 15613.00 | 15524.48 | 15463.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 10:00:00 | 15603.00 | 15700.95 | 15647.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-13 09:15:00 | 9572.15 | 2024-06-20 11:15:00 | 9791.85 | STOP_HIT | 1.00 | 2.30% |
| BUY | retest2 | 2024-07-09 13:30:00 | 12166.50 | 2024-07-12 13:15:00 | 11944.00 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2024-07-10 09:30:00 | 12185.90 | 2024-07-12 13:15:00 | 11944.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-07-10 15:15:00 | 12490.00 | 2024-07-12 13:15:00 | 11944.00 | STOP_HIT | 1.00 | -4.37% |
| BUY | retest2 | 2024-07-11 15:15:00 | 12239.85 | 2024-07-12 13:15:00 | 11944.00 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2024-07-16 13:30:00 | 11797.15 | 2024-07-18 14:15:00 | 11207.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-18 09:30:00 | 11715.05 | 2024-07-18 14:15:00 | 11171.29 | PARTIAL | 0.50 | 4.64% |
| SELL | retest2 | 2024-07-18 10:15:00 | 11759.25 | 2024-07-19 09:15:00 | 11129.30 | PARTIAL | 0.50 | 5.36% |
| SELL | retest2 | 2024-07-16 13:30:00 | 11797.15 | 2024-07-22 09:15:00 | 10617.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-18 09:30:00 | 11715.05 | 2024-07-22 09:15:00 | 11132.95 | STOP_HIT | 0.50 | 4.97% |
| SELL | retest2 | 2024-07-18 10:15:00 | 11759.25 | 2024-07-22 09:15:00 | 11132.95 | STOP_HIT | 0.50 | 5.33% |
| SELL | retest2 | 2024-08-06 14:30:00 | 10285.00 | 2024-08-16 12:15:00 | 10435.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-08-20 15:00:00 | 10485.40 | 2024-08-21 11:15:00 | 10265.40 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-08-23 15:15:00 | 10696.00 | 2024-08-28 09:15:00 | 10387.40 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2024-08-27 09:15:00 | 10625.00 | 2024-08-28 09:15:00 | 10387.40 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-09-04 14:45:00 | 10965.25 | 2024-09-09 11:15:00 | 10829.85 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-09-06 09:30:00 | 11024.35 | 2024-09-09 11:15:00 | 10829.85 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-09-06 10:45:00 | 10974.05 | 2024-09-09 11:15:00 | 10829.85 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-09-09 09:45:00 | 10989.95 | 2024-09-09 11:15:00 | 10829.85 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-09-13 13:30:00 | 11235.95 | 2024-09-16 10:15:00 | 10868.35 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2024-09-24 09:15:00 | 11261.15 | 2024-09-30 14:15:00 | 11551.25 | STOP_HIT | 1.00 | 2.58% |
| SELL | retest2 | 2024-10-03 09:15:00 | 11429.95 | 2024-10-07 09:15:00 | 10858.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 11429.95 | 2024-10-07 14:15:00 | 10718.95 | STOP_HIT | 0.50 | 6.22% |
| SELL | retest2 | 2024-10-24 11:00:00 | 10668.70 | 2024-10-29 10:15:00 | 10135.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 11:00:00 | 10668.70 | 2024-10-29 14:15:00 | 10379.95 | STOP_HIT | 0.50 | 2.71% |
| SELL | retest2 | 2024-11-04 09:15:00 | 10152.25 | 2024-11-06 13:15:00 | 10190.70 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-11-11 11:00:00 | 10300.00 | 2024-11-12 13:15:00 | 10142.65 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-11-11 14:00:00 | 10207.15 | 2024-11-12 13:15:00 | 10142.65 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-11-11 15:15:00 | 10209.00 | 2024-11-12 13:15:00 | 10142.65 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-11-12 11:15:00 | 10221.60 | 2024-11-12 13:15:00 | 10142.65 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-11-14 15:00:00 | 9983.20 | 2024-11-18 09:15:00 | 10086.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-11-25 09:15:00 | 10377.20 | 2024-11-25 14:15:00 | 9957.80 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest2 | 2024-11-25 13:45:00 | 10081.40 | 2024-11-25 14:15:00 | 9957.80 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-11-26 09:15:00 | 10117.00 | 2024-11-29 09:15:00 | 11128.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-11 09:15:00 | 10880.90 | 2024-12-11 11:15:00 | 10789.95 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-12-17 11:45:00 | 10417.05 | 2024-12-20 14:15:00 | 9896.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 10:00:00 | 10418.00 | 2024-12-20 14:15:00 | 9897.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 11:45:00 | 10417.05 | 2024-12-24 09:15:00 | 10176.90 | STOP_HIT | 0.50 | 2.31% |
| SELL | retest2 | 2024-12-18 10:00:00 | 10418.00 | 2024-12-24 09:15:00 | 10176.90 | STOP_HIT | 0.50 | 2.31% |
| SELL | retest2 | 2025-01-02 10:45:00 | 9782.45 | 2025-01-08 09:15:00 | 9724.75 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2025-01-03 10:30:00 | 9770.80 | 2025-01-08 09:15:00 | 9724.75 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2025-02-01 15:00:00 | 10128.65 | 2025-02-03 09:15:00 | 9856.95 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest1 | 2025-02-10 12:00:00 | 9085.00 | 2025-02-12 09:15:00 | 8630.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-10 12:00:00 | 9085.00 | 2025-02-12 11:15:00 | 8925.40 | STOP_HIT | 0.50 | 1.76% |
| SELL | retest1 | 2025-02-11 09:15:00 | 9040.30 | 2025-02-12 12:15:00 | 9033.05 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest1 | 2025-02-11 09:45:00 | 9003.00 | 2025-02-12 12:15:00 | 9033.05 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-02-13 13:30:00 | 8864.40 | 2025-02-19 12:15:00 | 8786.45 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-02-14 15:00:00 | 8859.00 | 2025-02-19 12:15:00 | 8786.45 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2025-03-12 14:15:00 | 9753.95 | 2025-03-24 09:15:00 | 10729.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-02 09:15:00 | 13255.00 | 2025-05-02 11:15:00 | 13033.00 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-05-14 09:30:00 | 13770.00 | 2025-05-19 14:15:00 | 13705.00 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-05-15 09:15:00 | 13781.00 | 2025-05-19 14:15:00 | 13705.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-05-15 11:15:00 | 13780.00 | 2025-05-19 14:15:00 | 13705.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-05-30 14:15:00 | 16019.00 | 2025-06-06 13:15:00 | 16515.00 | STOP_HIT | 1.00 | 3.10% |
| BUY | retest2 | 2025-05-30 14:45:00 | 16048.00 | 2025-06-06 13:15:00 | 16515.00 | STOP_HIT | 1.00 | 2.91% |
| SELL | retest2 | 2025-06-12 11:30:00 | 16612.00 | 2025-06-13 09:15:00 | 16776.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-06-12 12:45:00 | 16625.00 | 2025-06-13 09:15:00 | 16776.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-06-12 13:15:00 | 16612.00 | 2025-06-13 09:15:00 | 16776.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-06-12 14:00:00 | 16601.00 | 2025-06-13 09:15:00 | 16776.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-06-19 09:15:00 | 17174.00 | 2025-06-19 11:15:00 | 16945.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-07-07 11:30:00 | 16618.00 | 2025-07-11 09:15:00 | 15787.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-08 09:30:00 | 16594.00 | 2025-07-11 09:15:00 | 15764.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-09 14:15:00 | 16567.00 | 2025-07-11 09:15:00 | 15738.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-07 11:30:00 | 16618.00 | 2025-07-15 09:15:00 | 15392.00 | STOP_HIT | 0.50 | 7.38% |
| SELL | retest2 | 2025-07-08 09:30:00 | 16594.00 | 2025-07-15 09:15:00 | 15392.00 | STOP_HIT | 0.50 | 7.24% |
| SELL | retest2 | 2025-07-09 14:15:00 | 16567.00 | 2025-07-15 09:15:00 | 15392.00 | STOP_HIT | 0.50 | 7.09% |
| BUY | retest2 | 2025-09-12 09:15:00 | 14097.00 | 2025-09-22 10:15:00 | 14537.00 | STOP_HIT | 1.00 | 3.12% |
| SELL | retest2 | 2025-09-26 10:45:00 | 13940.00 | 2025-09-30 14:15:00 | 13243.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-26 10:45:00 | 13940.00 | 2025-10-01 14:15:00 | 13388.00 | STOP_HIT | 0.50 | 3.96% |
| SELL | retest2 | 2025-09-26 13:15:00 | 13887.00 | 2025-10-03 10:15:00 | 13859.00 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-10-08 13:00:00 | 14010.00 | 2025-10-13 11:15:00 | 13999.00 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-10-08 14:30:00 | 14089.00 | 2025-10-13 11:15:00 | 13999.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-10-09 09:15:00 | 14009.00 | 2025-10-13 11:15:00 | 13999.00 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-10-09 09:45:00 | 14094.00 | 2025-10-13 11:15:00 | 13999.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-10-16 10:15:00 | 13992.00 | 2025-10-16 13:15:00 | 14080.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-10-16 11:45:00 | 13979.00 | 2025-10-16 13:15:00 | 14080.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-11-10 13:15:00 | 13498.00 | 2025-11-11 09:15:00 | 13707.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-11-10 14:45:00 | 13450.00 | 2025-11-11 09:15:00 | 13707.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-11-19 11:00:00 | 13741.00 | 2025-11-20 11:15:00 | 13937.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-11-19 11:30:00 | 13727.00 | 2025-11-20 11:15:00 | 13937.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-11-19 12:30:00 | 13745.00 | 2025-11-20 11:15:00 | 13937.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-11-19 14:30:00 | 13749.00 | 2025-11-20 11:15:00 | 13937.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-11-26 14:30:00 | 13422.00 | 2025-12-04 09:15:00 | 12750.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 10:45:00 | 13418.00 | 2025-12-04 09:15:00 | 12747.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 14:30:00 | 13422.00 | 2025-12-04 11:15:00 | 12971.00 | STOP_HIT | 0.50 | 3.36% |
| SELL | retest2 | 2025-11-27 10:45:00 | 13418.00 | 2025-12-04 11:15:00 | 12971.00 | STOP_HIT | 0.50 | 3.33% |
| BUY | retest2 | 2025-12-26 09:15:00 | 12679.00 | 2025-12-26 14:15:00 | 12426.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2026-01-01 09:30:00 | 12165.00 | 2026-01-02 13:15:00 | 12290.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-01-01 11:45:00 | 12192.00 | 2026-01-02 13:15:00 | 12290.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-01-02 09:45:00 | 12183.00 | 2026-01-02 13:15:00 | 12290.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-01-16 09:30:00 | 12812.00 | 2026-01-16 11:15:00 | 13005.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2026-01-16 13:15:00 | 12764.00 | 2026-01-19 09:15:00 | 12956.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-01-16 15:00:00 | 12824.00 | 2026-01-19 09:15:00 | 12956.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-01-22 11:30:00 | 12630.00 | 2026-01-22 13:15:00 | 12814.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-01-30 10:30:00 | 13371.00 | 2026-02-01 14:15:00 | 13202.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-01-30 11:30:00 | 13365.00 | 2026-02-01 14:15:00 | 13202.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-01-30 12:30:00 | 13370.00 | 2026-02-01 14:15:00 | 13202.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-01-30 13:30:00 | 13369.00 | 2026-02-01 14:15:00 | 13202.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2026-02-01 13:45:00 | 13431.00 | 2026-02-01 15:15:00 | 13120.00 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2026-02-27 14:45:00 | 13602.00 | 2026-02-27 15:15:00 | 13491.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2026-03-02 09:15:00 | 13752.00 | 2026-03-06 09:15:00 | 15127.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-13 09:15:00 | 14474.00 | 2026-03-17 10:15:00 | 13750.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 09:15:00 | 14474.00 | 2026-03-17 14:15:00 | 14042.00 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2026-03-13 09:45:00 | 14442.00 | 2026-03-18 12:15:00 | 13719.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 09:45:00 | 14442.00 | 2026-03-20 14:15:00 | 12997.80 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-21 09:30:00 | 15084.00 | 2026-04-24 14:15:00 | 15045.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2026-04-22 09:15:00 | 14969.00 | 2026-04-24 14:15:00 | 15045.00 | STOP_HIT | 1.00 | 0.51% |
