# Abbott India Ltd. (ABBOTINDIA)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 26850.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 239 |
| ALERT1 | 169 |
| ALERT2 | 165 |
| ALERT2_SKIP | 94 |
| ALERT3 | 463 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 188 |
| PARTIAL | 11 |
| TARGET_HIT | 1 |
| STOP_HIT | 191 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 203 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 61 / 142
- **Target hits / Stop hits / Partials:** 1 / 191 / 11
- **Avg / median % per leg:** 0.15% / -0.63%
- **Sum % (uncompounded):** 31.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 75 | 17 | 22.7% | 1 | 74 | 0 | -0.28% | -20.7% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.77% | -1.5% |
| BUY @ 3rd Alert (retest2) | 73 | 17 | 23.3% | 1 | 72 | 0 | -0.26% | -19.1% |
| SELL (all) | 128 | 44 | 34.4% | 0 | 117 | 11 | 0.40% | 51.7% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 2 | 0 | 0.31% | 0.6% |
| SELL @ 3rd Alert (retest2) | 126 | 42 | 33.3% | 0 | 115 | 11 | 0.41% | 51.1% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 4 | 0 | -0.23% | -0.9% |
| retest2 (combined) | 199 | 59 | 29.6% | 1 | 187 | 11 | 0.16% | 32.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 10:15:00 | 21459.60 | 21113.15 | 21099.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 10:15:00 | 21527.70 | 21351.78 | 21242.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 13:15:00 | 21470.00 | 21496.30 | 21408.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 09:15:00 | 21500.00 | 21496.33 | 21430.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 21500.00 | 21496.33 | 21430.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-25 12:00:00 | 21585.20 | 21523.03 | 21454.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-25 13:15:00 | 21595.00 | 21530.00 | 21463.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-31 14:15:00 | 21739.90 | 21850.90 | 21860.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 14:15:00 | 21739.90 | 21850.90 | 21860.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 15:15:00 | 21700.00 | 21820.72 | 21846.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-01 12:15:00 | 21839.90 | 21811.53 | 21831.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 12:15:00 | 21839.90 | 21811.53 | 21831.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 12:15:00 | 21839.90 | 21811.53 | 21831.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 13:00:00 | 21839.90 | 21811.53 | 21831.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 13:15:00 | 21749.90 | 21799.20 | 21824.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 13:30:00 | 21820.20 | 21799.20 | 21824.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 10:15:00 | 21712.80 | 21709.76 | 21766.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-02 10:45:00 | 21726.10 | 21709.76 | 21766.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 12:15:00 | 21715.40 | 21694.74 | 21748.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-02 13:00:00 | 21715.40 | 21694.74 | 21748.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 13:15:00 | 21899.90 | 21735.77 | 21762.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-02 14:00:00 | 21899.90 | 21735.77 | 21762.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 14:15:00 | 21926.20 | 21773.86 | 21777.16 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 15:15:00 | 21931.80 | 21805.44 | 21791.22 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 11:15:00 | 21675.40 | 21797.06 | 21808.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 12:15:00 | 21609.00 | 21759.45 | 21790.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-06 14:15:00 | 21829.90 | 21764.36 | 21786.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 14:15:00 | 21829.90 | 21764.36 | 21786.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 14:15:00 | 21829.90 | 21764.36 | 21786.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-06 15:00:00 | 21829.90 | 21764.36 | 21786.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 15:15:00 | 21844.00 | 21780.29 | 21791.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 09:15:00 | 21885.30 | 21780.29 | 21791.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2023-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 09:15:00 | 21879.70 | 21800.17 | 21799.83 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-06-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 12:15:00 | 21730.80 | 21808.02 | 21817.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-08 14:15:00 | 21682.10 | 21773.18 | 21799.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-09 13:15:00 | 21700.00 | 21655.51 | 21716.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-09 13:15:00 | 21700.00 | 21655.51 | 21716.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 13:15:00 | 21700.00 | 21655.51 | 21716.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-09 14:00:00 | 21700.00 | 21655.51 | 21716.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 14:15:00 | 21803.40 | 21685.09 | 21724.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-09 15:00:00 | 21803.40 | 21685.09 | 21724.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 15:15:00 | 21746.80 | 21697.43 | 21726.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 09:45:00 | 21848.90 | 21729.94 | 21738.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2023-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 10:15:00 | 21870.00 | 21757.96 | 21750.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-12 14:15:00 | 21970.00 | 21855.15 | 21803.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 12:15:00 | 22111.40 | 22133.41 | 22042.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-14 13:15:00 | 22100.10 | 22133.41 | 22042.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 15:15:00 | 22109.30 | 22123.41 | 22060.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-15 09:15:00 | 22236.00 | 22123.41 | 22060.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-21 14:15:00 | 22707.90 | 22817.10 | 22828.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2023-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 14:15:00 | 22707.90 | 22817.10 | 22828.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 09:15:00 | 22578.20 | 22743.13 | 22790.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 13:15:00 | 22670.00 | 22571.81 | 22630.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 13:15:00 | 22670.00 | 22571.81 | 22630.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 13:15:00 | 22670.00 | 22571.81 | 22630.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 14:00:00 | 22670.00 | 22571.81 | 22630.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 14:15:00 | 22670.00 | 22591.45 | 22633.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 14:45:00 | 22689.30 | 22591.45 | 22633.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 15:15:00 | 22550.10 | 22583.18 | 22626.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 09:15:00 | 22718.00 | 22583.18 | 22626.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 22622.00 | 22590.94 | 22625.80 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 12:15:00 | 22750.10 | 22648.18 | 22645.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-26 14:15:00 | 22851.90 | 22707.65 | 22673.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-27 10:15:00 | 22710.00 | 22751.04 | 22706.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 10:15:00 | 22710.00 | 22751.04 | 22706.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 22710.00 | 22751.04 | 22706.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-27 12:30:00 | 22825.00 | 22732.91 | 22705.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-28 09:15:00 | 22840.30 | 22716.54 | 22704.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-11 12:15:00 | 23370.00 | 23482.68 | 23496.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2023-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-11 12:15:00 | 23370.00 | 23482.68 | 23496.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-11 13:15:00 | 23317.10 | 23449.56 | 23480.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 14:15:00 | 23695.70 | 23498.79 | 23499.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 14:15:00 | 23695.70 | 23498.79 | 23499.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 23695.70 | 23498.79 | 23499.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 14:45:00 | 23775.00 | 23498.79 | 23499.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2023-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 15:15:00 | 23661.00 | 23531.23 | 23514.60 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-07-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 13:15:00 | 23314.60 | 23508.55 | 23534.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 10:15:00 | 23189.40 | 23393.16 | 23467.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-17 10:15:00 | 23364.10 | 23335.26 | 23395.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 10:15:00 | 23364.10 | 23335.26 | 23395.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 10:15:00 | 23364.10 | 23335.26 | 23395.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 11:00:00 | 23364.10 | 23335.26 | 23395.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 11:15:00 | 23296.80 | 23327.57 | 23386.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 11:30:00 | 23324.40 | 23327.57 | 23386.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 23228.10 | 23243.40 | 23317.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-18 10:30:00 | 23134.10 | 23224.00 | 23301.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-18 15:15:00 | 23341.20 | 23264.52 | 23291.07 | SL hit (close>static) qty=1.00 sl=23339.40 alert=retest2 |

### Cycle 13 — BUY (started 2023-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 12:15:00 | 23480.00 | 23333.39 | 23316.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 15:15:00 | 23515.00 | 23399.76 | 23354.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 09:15:00 | 23422.60 | 23542.35 | 23476.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 09:15:00 | 23422.60 | 23542.35 | 23476.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 23422.60 | 23542.35 | 23476.00 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 12:15:00 | 23286.10 | 23423.25 | 23432.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 14:15:00 | 23215.10 | 23357.47 | 23399.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 15:15:00 | 23225.10 | 23192.80 | 23267.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-25 09:15:00 | 23118.00 | 23192.80 | 23267.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 23235.10 | 23007.77 | 23050.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 09:45:00 | 23195.90 | 23007.77 | 23050.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 10:15:00 | 23317.90 | 23069.79 | 23074.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 11:00:00 | 23317.90 | 23069.79 | 23074.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2023-07-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 11:15:00 | 23410.20 | 23137.87 | 23104.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 12:15:00 | 23705.80 | 23251.46 | 23159.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-31 15:15:00 | 23999.30 | 24011.95 | 23817.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-01 09:15:00 | 23988.50 | 24011.95 | 23817.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 12:15:00 | 23875.80 | 23975.05 | 23861.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 13:00:00 | 23875.80 | 23975.05 | 23861.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 13:15:00 | 23861.90 | 23952.42 | 23861.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 14:00:00 | 23861.90 | 23952.42 | 23861.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 14:15:00 | 23944.00 | 23950.73 | 23869.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-01 15:15:00 | 23980.00 | 23950.73 | 23869.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-02 10:15:00 | 23850.10 | 23927.02 | 23878.80 | SL hit (close<static) qty=1.00 sl=23851.30 alert=retest2 |

### Cycle 16 — SELL (started 2023-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-04 14:15:00 | 23899.90 | 24098.49 | 24109.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-07 11:15:00 | 23677.50 | 23966.47 | 24040.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-07 14:15:00 | 23984.40 | 23920.68 | 23996.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-07 15:00:00 | 23984.40 | 23920.68 | 23996.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 15:15:00 | 23801.00 | 23896.75 | 23978.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-08 09:15:00 | 24048.30 | 23896.75 | 23978.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 09:15:00 | 24137.60 | 23944.92 | 23993.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 14:45:00 | 23790.90 | 23906.49 | 23957.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-09 10:15:00 | 23735.40 | 23864.30 | 23927.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-09 14:45:00 | 23799.90 | 23806.53 | 23869.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-10 10:15:00 | 23989.60 | 23917.89 | 23911.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2023-08-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 10:15:00 | 23989.60 | 23917.89 | 23911.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-11 11:15:00 | 24100.00 | 23979.45 | 23947.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 12:15:00 | 23928.20 | 23969.20 | 23946.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 12:15:00 | 23928.20 | 23969.20 | 23946.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 12:15:00 | 23928.20 | 23969.20 | 23946.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 13:00:00 | 23928.20 | 23969.20 | 23946.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 13:15:00 | 23850.10 | 23945.38 | 23937.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 13:30:00 | 23815.90 | 23945.38 | 23937.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2023-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 14:15:00 | 23756.50 | 23907.61 | 23920.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 09:15:00 | 23350.00 | 23764.63 | 23851.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 14:15:00 | 23628.40 | 23596.44 | 23716.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-14 14:15:00 | 23628.40 | 23596.44 | 23716.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 14:15:00 | 23628.40 | 23596.44 | 23716.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 15:00:00 | 23628.40 | 23596.44 | 23716.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 12:15:00 | 23750.00 | 23621.08 | 23679.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 13:00:00 | 23750.00 | 23621.08 | 23679.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 13:15:00 | 23716.10 | 23640.08 | 23682.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 13:45:00 | 23800.00 | 23640.08 | 23682.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2023-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 15:15:00 | 23894.90 | 23731.85 | 23719.67 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-08-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 12:15:00 | 23684.00 | 23710.86 | 23712.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-17 15:15:00 | 23650.00 | 23689.94 | 23701.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 10:15:00 | 23549.90 | 23521.50 | 23586.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-21 11:00:00 | 23549.90 | 23521.50 | 23586.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 09:15:00 | 23349.90 | 23438.56 | 23511.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 10:00:00 | 23317.10 | 23444.93 | 23489.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 13:45:00 | 23325.10 | 23393.58 | 23448.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 14:15:00 | 23315.10 | 23393.58 | 23448.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 13:15:00 | 23321.00 | 23355.42 | 23398.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 14:15:00 | 23177.10 | 23183.61 | 23259.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-25 15:00:00 | 23177.10 | 23183.61 | 23259.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 23225.80 | 23187.95 | 23248.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-28 10:30:00 | 23115.00 | 23205.14 | 23250.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-28 11:15:00 | 23310.80 | 23226.27 | 23255.98 | SL hit (close>static) qty=1.00 sl=23299.90 alert=retest2 |

### Cycle 21 — BUY (started 2023-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 13:15:00 | 23350.00 | 23282.21 | 23278.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 14:15:00 | 23460.00 | 23374.60 | 23334.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 10:15:00 | 23381.00 | 23387.56 | 23351.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-30 10:45:00 | 23381.90 | 23387.56 | 23351.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 12:15:00 | 23427.10 | 23398.29 | 23362.77 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-08-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 10:15:00 | 23300.00 | 23346.34 | 23349.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 14:15:00 | 23108.40 | 23286.28 | 23320.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-05 11:15:00 | 22552.20 | 22480.35 | 22667.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-05 12:00:00 | 22552.20 | 22480.35 | 22667.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 12:15:00 | 22767.90 | 22537.86 | 22676.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-05 12:45:00 | 22759.10 | 22537.86 | 22676.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 13:15:00 | 22812.10 | 22592.71 | 22689.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-05 13:45:00 | 22824.10 | 22592.71 | 22689.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 15:15:00 | 22810.00 | 22671.72 | 22710.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-06 09:15:00 | 22872.90 | 22671.72 | 22710.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2023-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 10:15:00 | 23028.80 | 22772.23 | 22751.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 11:15:00 | 23127.00 | 22843.19 | 22785.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 10:15:00 | 23004.30 | 23014.14 | 22911.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-07 11:00:00 | 23004.30 | 23014.14 | 22911.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 15:15:00 | 23000.00 | 23046.92 | 22970.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-08 09:15:00 | 23095.60 | 23046.92 | 22970.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-08 11:15:00 | 22950.00 | 23032.19 | 22983.35 | SL hit (close<static) qty=1.00 sl=22956.00 alert=retest2 |

### Cycle 24 — SELL (started 2023-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-11 12:15:00 | 22926.70 | 22970.70 | 22971.76 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 15:15:00 | 23029.10 | 22970.96 | 22970.29 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 22933.20 | 22963.41 | 22966.92 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-12 10:15:00 | 23037.20 | 22978.17 | 22973.31 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 22907.80 | 22964.10 | 22967.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 22847.80 | 22940.84 | 22956.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-12 14:15:00 | 22933.60 | 22916.86 | 22941.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 14:15:00 | 22933.60 | 22916.86 | 22941.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 14:15:00 | 22933.60 | 22916.86 | 22941.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-12 15:00:00 | 22933.60 | 22916.86 | 22941.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 15:15:00 | 22842.60 | 22902.00 | 22932.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 09:15:00 | 22925.10 | 22902.00 | 22932.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 09:15:00 | 22953.20 | 22912.24 | 22934.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-13 11:30:00 | 22786.80 | 22876.20 | 22913.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-14 09:15:00 | 23009.90 | 22920.12 | 22917.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2023-09-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 09:15:00 | 23009.90 | 22920.12 | 22917.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 10:15:00 | 23200.00 | 23020.55 | 22969.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 10:15:00 | 23224.00 | 23244.66 | 23139.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-18 11:00:00 | 23224.00 | 23244.66 | 23139.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 12:15:00 | 23246.40 | 23234.70 | 23152.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 12:30:00 | 23156.10 | 23234.70 | 23152.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 13:15:00 | 23349.80 | 23257.72 | 23170.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 13:30:00 | 23250.10 | 23257.72 | 23170.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 23275.10 | 23275.51 | 23201.44 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-09-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 09:15:00 | 23039.80 | 23191.60 | 23194.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 12:15:00 | 22911.30 | 23102.76 | 23150.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 11:15:00 | 22554.10 | 22553.92 | 22726.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-25 12:00:00 | 22554.10 | 22553.92 | 22726.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 14:15:00 | 22699.10 | 22608.45 | 22710.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 14:30:00 | 22716.20 | 22608.45 | 22710.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 22576.40 | 22601.52 | 22689.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 10:15:00 | 22540.00 | 22601.52 | 22689.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-26 11:15:00 | 22871.90 | 22671.35 | 22706.93 | SL hit (close>static) qty=1.00 sl=22750.00 alert=retest2 |

### Cycle 31 — BUY (started 2023-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 11:15:00 | 23000.00 | 22624.34 | 22601.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 12:15:00 | 23102.10 | 22719.89 | 22647.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 09:15:00 | 22918.00 | 22928.38 | 22787.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 22918.00 | 22928.38 | 22787.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 22918.00 | 22928.38 | 22787.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 14:45:00 | 23269.70 | 23067.23 | 22917.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-04 14:15:00 | 22807.20 | 22886.17 | 22889.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2023-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 14:15:00 | 22807.20 | 22886.17 | 22889.85 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-10-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 11:15:00 | 22900.00 | 22890.86 | 22890.36 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-10-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 12:15:00 | 22857.60 | 22884.21 | 22887.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 13:15:00 | 22812.10 | 22869.79 | 22880.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 14:15:00 | 22869.90 | 22869.81 | 22879.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-05 15:00:00 | 22869.90 | 22869.81 | 22879.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 15:15:00 | 22909.00 | 22877.65 | 22882.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 09:15:00 | 22929.40 | 22877.65 | 22882.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2023-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 09:15:00 | 22974.50 | 22897.02 | 22890.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 11:15:00 | 23092.90 | 22939.90 | 22911.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-06 13:15:00 | 22926.10 | 22941.11 | 22917.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 13:15:00 | 22926.10 | 22941.11 | 22917.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 13:15:00 | 22926.10 | 22941.11 | 22917.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-06 13:45:00 | 22900.10 | 22941.11 | 22917.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 14:15:00 | 23040.20 | 22960.93 | 22928.38 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-10 10:15:00 | 22855.80 | 22922.41 | 22930.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-10 11:15:00 | 22700.00 | 22877.93 | 22909.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-11 09:15:00 | 22898.20 | 22837.73 | 22872.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 09:15:00 | 22898.20 | 22837.73 | 22872.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 22898.20 | 22837.73 | 22872.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 13:00:00 | 22750.00 | 22822.77 | 22857.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-19 13:15:00 | 22571.90 | 22444.26 | 22439.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2023-10-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-19 13:15:00 | 22571.90 | 22444.26 | 22439.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-19 14:15:00 | 22652.60 | 22485.93 | 22458.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-20 10:15:00 | 22512.90 | 22525.19 | 22487.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-20 11:00:00 | 22512.90 | 22525.19 | 22487.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 11:15:00 | 22422.10 | 22504.57 | 22481.10 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 13:15:00 | 22340.00 | 22463.21 | 22465.85 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 14:15:00 | 22599.80 | 22490.53 | 22478.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-23 11:15:00 | 22627.10 | 22529.48 | 22501.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-25 09:15:00 | 22568.80 | 22608.10 | 22556.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 09:15:00 | 22568.80 | 22608.10 | 22556.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 22568.80 | 22608.10 | 22556.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-25 09:30:00 | 22692.90 | 22608.10 | 22556.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 10:15:00 | 22487.40 | 22583.96 | 22550.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-25 11:00:00 | 22487.40 | 22583.96 | 22550.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 11:15:00 | 22477.10 | 22562.59 | 22543.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-25 12:00:00 | 22477.10 | 22562.59 | 22543.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 12:15:00 | 22425.20 | 22535.11 | 22532.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-25 13:45:00 | 22588.60 | 22547.35 | 22538.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-25 14:30:00 | 22562.20 | 22542.20 | 22536.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-26 09:15:00 | 22285.00 | 22483.85 | 22510.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2023-10-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 09:15:00 | 22285.00 | 22483.85 | 22510.93 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 12:15:00 | 22666.00 | 22495.46 | 22473.38 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 09:15:00 | 22475.90 | 22520.43 | 22522.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 09:15:00 | 22310.00 | 22441.37 | 22478.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 10:15:00 | 22377.00 | 22291.98 | 22357.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 10:15:00 | 22377.00 | 22291.98 | 22357.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 10:15:00 | 22377.00 | 22291.98 | 22357.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:00:00 | 22377.00 | 22291.98 | 22357.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 22361.90 | 22305.96 | 22357.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:45:00 | 22423.30 | 22305.96 | 22357.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 12:15:00 | 22437.00 | 22332.17 | 22364.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 13:00:00 | 22437.00 | 22332.17 | 22364.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 13:15:00 | 22374.10 | 22340.56 | 22365.62 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 11:15:00 | 22482.10 | 22394.04 | 22383.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 12:15:00 | 22609.00 | 22437.03 | 22403.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 12:15:00 | 24031.40 | 24125.80 | 23825.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-09 13:00:00 | 24031.40 | 24125.80 | 23825.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 23890.20 | 24046.76 | 23839.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 15:00:00 | 23890.20 | 24046.76 | 23839.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 15:15:00 | 23899.90 | 24017.39 | 23845.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 09:15:00 | 23975.00 | 24017.39 | 23845.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 23922.20 | 23998.35 | 23852.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 10:15:00 | 23832.00 | 23998.35 | 23852.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 10:15:00 | 23937.30 | 23986.14 | 23859.82 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 14:15:00 | 23679.00 | 23776.65 | 23786.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 09:15:00 | 23425.00 | 23705.69 | 23751.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 10:15:00 | 23152.50 | 23068.74 | 23226.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-16 11:00:00 | 23152.50 | 23068.74 | 23226.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 12:15:00 | 23265.00 | 23123.37 | 23225.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 13:00:00 | 23265.00 | 23123.37 | 23225.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 13:15:00 | 23235.30 | 23145.76 | 23226.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 14:00:00 | 23235.30 | 23145.76 | 23226.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 14:15:00 | 23304.20 | 23177.45 | 23233.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 14:30:00 | 23334.40 | 23177.45 | 23233.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2023-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 10:15:00 | 23372.90 | 23275.07 | 23268.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-17 13:15:00 | 23534.90 | 23374.71 | 23319.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-21 13:15:00 | 23915.10 | 23915.73 | 23752.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-21 14:00:00 | 23915.10 | 23915.73 | 23752.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 13:15:00 | 23811.30 | 23911.50 | 23834.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 14:00:00 | 23811.30 | 23911.50 | 23834.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 14:15:00 | 23890.00 | 23907.20 | 23839.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 12:45:00 | 23922.00 | 23884.55 | 23852.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-24 09:15:00 | 23995.80 | 23880.41 | 23857.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-24 09:45:00 | 23997.10 | 23893.01 | 23865.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-28 12:15:00 | 23835.80 | 23887.00 | 23891.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2023-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 12:15:00 | 23835.80 | 23887.00 | 23891.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 13:15:00 | 23708.20 | 23851.24 | 23874.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 09:15:00 | 23687.40 | 23647.38 | 23725.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-30 10:00:00 | 23687.40 | 23647.38 | 23725.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 12:15:00 | 23776.80 | 23664.78 | 23713.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-30 13:00:00 | 23776.80 | 23664.78 | 23713.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 13:15:00 | 23750.00 | 23681.83 | 23716.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-30 13:45:00 | 23810.00 | 23681.83 | 23716.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2023-11-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 15:15:00 | 23840.00 | 23738.39 | 23737.85 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2023-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-01 12:15:00 | 23657.60 | 23740.69 | 23740.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-04 09:15:00 | 23570.50 | 23681.09 | 23710.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-04 11:15:00 | 23806.90 | 23696.91 | 23712.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-04 11:15:00 | 23806.90 | 23696.91 | 23712.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 11:15:00 | 23806.90 | 23696.91 | 23712.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-04 12:00:00 | 23806.90 | 23696.91 | 23712.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 12:15:00 | 23515.10 | 23660.55 | 23694.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-04 13:45:00 | 23478.00 | 23604.52 | 23665.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-05 09:45:00 | 23505.00 | 23526.93 | 23610.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-18 13:15:00 | 22782.20 | 22769.68 | 22768.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2023-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 13:15:00 | 22782.20 | 22769.68 | 22768.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-19 10:15:00 | 22937.10 | 22808.27 | 22786.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 09:15:00 | 22777.00 | 22851.57 | 22823.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 09:15:00 | 22777.00 | 22851.57 | 22823.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 09:15:00 | 22777.00 | 22851.57 | 22823.75 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2023-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 11:15:00 | 22596.10 | 22773.39 | 22791.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 12:15:00 | 22549.40 | 22728.60 | 22769.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 22360.10 | 22358.64 | 22494.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 15:00:00 | 22360.10 | 22358.64 | 22494.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 22488.80 | 22388.09 | 22484.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:00:00 | 22488.80 | 22388.09 | 22484.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 22461.10 | 22402.69 | 22482.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:30:00 | 22465.60 | 22402.69 | 22482.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 11:15:00 | 22519.90 | 22426.13 | 22485.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 12:00:00 | 22519.90 | 22426.13 | 22485.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 12:15:00 | 22489.60 | 22438.83 | 22485.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 13:15:00 | 22574.90 | 22438.83 | 22485.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 13:15:00 | 22509.10 | 22452.88 | 22488.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 13:30:00 | 22536.10 | 22452.88 | 22488.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 14:15:00 | 22635.70 | 22489.45 | 22501.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 15:00:00 | 22635.70 | 22489.45 | 22501.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2023-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 15:15:00 | 22600.00 | 22511.56 | 22510.44 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2023-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-26 11:15:00 | 22356.20 | 22499.13 | 22506.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-26 13:15:00 | 22268.00 | 22434.65 | 22475.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-26 14:15:00 | 22487.80 | 22445.28 | 22476.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-26 14:15:00 | 22487.80 | 22445.28 | 22476.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 14:15:00 | 22487.80 | 22445.28 | 22476.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 15:00:00 | 22487.80 | 22445.28 | 22476.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 15:15:00 | 22500.00 | 22456.22 | 22478.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 09:15:00 | 22493.30 | 22456.22 | 22478.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 09:15:00 | 22556.70 | 22476.32 | 22485.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 10:00:00 | 22556.70 | 22476.32 | 22485.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2023-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 10:15:00 | 22598.80 | 22500.81 | 22495.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 15:15:00 | 22640.00 | 22591.08 | 22562.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 09:15:00 | 22578.00 | 22588.46 | 22563.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 09:15:00 | 22578.00 | 22588.46 | 22563.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 22578.00 | 22588.46 | 22563.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 14:15:00 | 22747.90 | 22595.87 | 22573.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-11 11:15:00 | 25022.69 | 24615.80 | 24311.41 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-01-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 10:15:00 | 25745.20 | 25754.94 | 25755.06 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-01-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 11:15:00 | 25771.70 | 25758.29 | 25756.58 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-01-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 12:15:00 | 25715.00 | 25749.63 | 25752.80 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-01-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 14:15:00 | 25801.80 | 25754.84 | 25754.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 14:15:00 | 25895.20 | 25786.91 | 25771.27 | Break + close above crossover candle high |

### Cycle 58 — SELL (started 2024-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 09:15:00 | 25540.10 | 25751.93 | 25758.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-20 10:15:00 | 24997.50 | 25601.04 | 25689.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-23 09:15:00 | 25704.60 | 25385.37 | 25504.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 25704.60 | 25385.37 | 25504.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 25704.60 | 25385.37 | 25504.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:45:00 | 25657.70 | 25385.37 | 25504.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 25660.20 | 25440.34 | 25518.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-23 12:15:00 | 25530.90 | 25470.89 | 25525.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-23 13:45:00 | 25541.80 | 25513.17 | 25536.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-23 15:15:00 | 25530.00 | 25520.76 | 25537.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 12:00:00 | 25414.30 | 25493.56 | 25519.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 12:15:00 | 25505.90 | 25496.03 | 25517.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 12:45:00 | 25490.00 | 25496.03 | 25517.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 25570.80 | 25510.98 | 25522.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:00:00 | 25570.80 | 25510.98 | 25522.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 25589.90 | 25526.77 | 25528.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 15:00:00 | 25589.90 | 25526.77 | 25528.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-01-24 15:15:00 | 25653.00 | 25552.01 | 25540.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2024-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 15:15:00 | 25653.00 | 25552.01 | 25540.08 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 09:15:00 | 25392.80 | 25520.17 | 25526.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 11:15:00 | 25173.00 | 25421.56 | 25478.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-29 09:15:00 | 25373.40 | 25265.09 | 25363.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-29 09:15:00 | 25373.40 | 25265.09 | 25363.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 25373.40 | 25265.09 | 25363.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 10:00:00 | 25373.40 | 25265.09 | 25363.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 10:15:00 | 25157.20 | 25243.51 | 25344.39 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 09:15:00 | 25683.10 | 25412.25 | 25387.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 09:15:00 | 25900.00 | 25576.55 | 25492.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 14:15:00 | 25883.70 | 25885.11 | 25704.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-31 15:00:00 | 25883.70 | 25885.11 | 25704.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 12:15:00 | 25835.90 | 25880.37 | 25772.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 13:15:00 | 25712.10 | 25880.37 | 25772.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 13:15:00 | 25807.60 | 25865.81 | 25775.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 13:30:00 | 25701.20 | 25865.81 | 25775.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 14:15:00 | 25864.10 | 25865.47 | 25783.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 09:15:00 | 27479.70 | 25852.60 | 25785.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-09 14:15:00 | 28082.20 | 28185.78 | 28193.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2024-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 14:15:00 | 28082.20 | 28185.78 | 28193.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 15:15:00 | 28060.00 | 28160.62 | 28181.76 | Break + close below crossover candle low |

### Cycle 63 — BUY (started 2024-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-12 09:15:00 | 28343.90 | 28197.28 | 28196.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-12 10:15:00 | 28371.30 | 28232.08 | 28212.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 12:15:00 | 29009.80 | 29109.17 | 28924.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-15 13:00:00 | 29009.80 | 29109.17 | 28924.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 29350.00 | 29456.57 | 29355.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 10:00:00 | 29350.00 | 29456.57 | 29355.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 29361.55 | 29437.57 | 29356.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 10:30:00 | 29356.75 | 29437.57 | 29356.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 11:15:00 | 29235.55 | 29397.17 | 29345.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 11:45:00 | 29215.80 | 29397.17 | 29345.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 12:15:00 | 29199.95 | 29357.72 | 29331.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 12:45:00 | 29167.70 | 29357.72 | 29331.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2024-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 13:15:00 | 29113.25 | 29308.83 | 29312.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 15:15:00 | 29042.50 | 29167.92 | 29227.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 15:15:00 | 28929.00 | 28913.52 | 29041.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-23 09:15:00 | 28877.00 | 28913.52 | 29041.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 14:15:00 | 28408.15 | 28394.59 | 28606.98 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-28 10:15:00 | 28807.35 | 28656.56 | 28643.26 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-02-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 09:15:00 | 28308.55 | 28606.40 | 28637.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-29 12:15:00 | 28211.95 | 28466.63 | 28560.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 15:15:00 | 28650.00 | 28435.66 | 28516.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 15:15:00 | 28650.00 | 28435.66 | 28516.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 28650.00 | 28435.66 | 28516.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-02 09:15:00 | 28050.55 | 28359.97 | 28443.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-02 12:15:00 | 28072.00 | 28293.99 | 28395.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 09:45:00 | 28120.15 | 28202.68 | 28334.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-13 15:15:00 | 27400.00 | 27236.62 | 27229.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2024-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-13 15:15:00 | 27400.00 | 27236.62 | 27229.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-14 09:15:00 | 28046.05 | 27398.50 | 27303.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-18 09:15:00 | 28048.85 | 28223.20 | 28013.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 09:15:00 | 28048.85 | 28223.20 | 28013.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 28048.85 | 28223.20 | 28013.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-18 10:00:00 | 28048.85 | 28223.20 | 28013.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 10:15:00 | 28111.10 | 28200.78 | 28022.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-18 10:45:00 | 27949.90 | 28200.78 | 28022.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 28292.60 | 28430.88 | 28258.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 11:00:00 | 28292.60 | 28430.88 | 28258.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 11:15:00 | 28306.80 | 28406.07 | 28262.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 11:30:00 | 28205.00 | 28406.07 | 28262.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 14:15:00 | 28285.55 | 28357.98 | 28274.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 14:45:00 | 28277.85 | 28357.98 | 28274.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 15:15:00 | 28203.55 | 28327.09 | 28268.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 09:15:00 | 28165.00 | 28327.09 | 28268.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 27941.65 | 28250.00 | 28238.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 10:00:00 | 27941.65 | 28250.00 | 28238.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2024-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 10:15:00 | 27640.05 | 28128.01 | 28183.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-21 12:15:00 | 27606.15 | 27711.10 | 27879.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 15:15:00 | 27755.00 | 27685.39 | 27822.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-22 09:15:00 | 27870.00 | 27685.39 | 27822.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 09:15:00 | 27582.40 | 27664.80 | 27800.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 11:00:00 | 27561.90 | 27644.22 | 27778.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 11:45:00 | 27550.00 | 27584.87 | 27739.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-01 09:15:00 | 27287.25 | 27030.98 | 27003.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2024-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 09:15:00 | 27287.25 | 27030.98 | 27003.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 11:15:00 | 27511.80 | 27179.50 | 27078.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-02 12:15:00 | 27390.60 | 27468.62 | 27315.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-02 13:00:00 | 27390.60 | 27468.62 | 27315.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 13:15:00 | 27323.85 | 27439.67 | 27316.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 13:45:00 | 27343.65 | 27439.67 | 27316.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 14:15:00 | 27288.15 | 27409.37 | 27314.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 15:00:00 | 27288.15 | 27409.37 | 27314.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 15:15:00 | 27290.00 | 27385.49 | 27311.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 09:15:00 | 27124.15 | 27385.49 | 27311.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 10:15:00 | 27150.00 | 27306.49 | 27286.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 11:00:00 | 27150.00 | 27306.49 | 27286.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2024-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 11:15:00 | 26952.65 | 27235.72 | 27256.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-03 14:15:00 | 26710.40 | 27056.41 | 27162.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 09:15:00 | 26701.00 | 26690.89 | 26852.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 09:15:00 | 26701.00 | 26690.89 | 26852.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 26701.00 | 26690.89 | 26852.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 09:30:00 | 26765.75 | 26690.89 | 26852.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 26800.00 | 26693.11 | 26772.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-08 10:00:00 | 26800.00 | 26693.11 | 26772.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 10:15:00 | 26759.35 | 26706.36 | 26771.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-08 11:15:00 | 26660.00 | 26706.36 | 26771.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-10 09:15:00 | 26862.50 | 26511.19 | 26564.25 | SL hit (close>static) qty=1.00 sl=26825.10 alert=retest2 |

### Cycle 71 — BUY (started 2024-04-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 10:15:00 | 27075.90 | 26624.13 | 26610.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 11:15:00 | 27188.05 | 26736.91 | 26663.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 12:15:00 | 26678.25 | 26919.65 | 26836.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 12:15:00 | 26678.25 | 26919.65 | 26836.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 12:15:00 | 26678.25 | 26919.65 | 26836.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 13:00:00 | 26678.25 | 26919.65 | 26836.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 13:15:00 | 26574.35 | 26850.59 | 26812.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 14:00:00 | 26574.35 | 26850.59 | 26812.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2024-04-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 14:15:00 | 26487.15 | 26777.90 | 26783.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 10:15:00 | 26178.00 | 26569.08 | 26678.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 11:15:00 | 26328.55 | 26253.08 | 26346.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 11:15:00 | 26328.55 | 26253.08 | 26346.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 11:15:00 | 26328.55 | 26253.08 | 26346.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 12:00:00 | 26328.55 | 26253.08 | 26346.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 12:15:00 | 26404.15 | 26283.30 | 26352.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 12:45:00 | 26324.80 | 26283.30 | 26352.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 13:15:00 | 26243.45 | 26275.33 | 26342.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 14:30:00 | 26151.85 | 26235.26 | 26317.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 14:45:00 | 26145.25 | 26154.52 | 26216.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 09:45:00 | 26143.10 | 26156.07 | 26207.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 12:15:00 | 26379.70 | 26264.28 | 26249.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2024-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 12:15:00 | 26379.70 | 26264.28 | 26249.56 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 09:15:00 | 25846.85 | 26204.18 | 26230.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-23 10:15:00 | 25794.50 | 26122.24 | 26190.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 13:15:00 | 25592.95 | 25451.28 | 25620.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 13:15:00 | 25592.95 | 25451.28 | 25620.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 13:15:00 | 25592.95 | 25451.28 | 25620.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 14:00:00 | 25592.95 | 25451.28 | 25620.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 14:15:00 | 25675.05 | 25496.04 | 25625.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 15:00:00 | 25675.05 | 25496.04 | 25625.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 15:15:00 | 25707.00 | 25538.23 | 25632.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-26 09:15:00 | 25820.00 | 25538.23 | 25632.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 25758.05 | 25582.19 | 25644.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-26 10:00:00 | 25758.05 | 25582.19 | 25644.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2024-04-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 11:15:00 | 25955.85 | 25711.20 | 25695.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-29 11:15:00 | 25982.95 | 25870.36 | 25798.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-02 09:15:00 | 26275.00 | 26350.33 | 26177.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-02 10:00:00 | 26275.00 | 26350.33 | 26177.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 11:15:00 | 26181.30 | 26487.21 | 26381.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 12:00:00 | 26181.30 | 26487.21 | 26381.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 12:15:00 | 26124.95 | 26414.76 | 26357.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 12:45:00 | 26080.35 | 26414.76 | 26357.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2024-05-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 15:15:00 | 26140.00 | 26288.59 | 26307.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 09:15:00 | 25870.10 | 26204.90 | 26267.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 25737.75 | 25705.62 | 25845.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 11:15:00 | 25839.80 | 25733.27 | 25833.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 25839.80 | 25733.27 | 25833.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 11:30:00 | 25851.80 | 25733.27 | 25833.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 25837.90 | 25754.20 | 25834.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:45:00 | 25835.50 | 25754.20 | 25834.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 25895.45 | 25782.45 | 25839.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 14:00:00 | 25895.45 | 25782.45 | 25839.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 14:15:00 | 25927.70 | 25811.50 | 25847.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 15:00:00 | 25927.70 | 25811.50 | 25847.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 12:15:00 | 25620.00 | 25724.72 | 25789.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-09 12:30:00 | 25710.00 | 25724.72 | 25789.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2024-05-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 09:15:00 | 26460.00 | 25828.00 | 25810.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 12:15:00 | 26814.60 | 26410.19 | 26201.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 13:15:00 | 26835.20 | 26888.94 | 26630.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-14 13:45:00 | 26825.20 | 26888.94 | 26630.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 26671.95 | 26812.57 | 26656.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:15:00 | 26592.90 | 26812.57 | 26656.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 26656.25 | 26781.30 | 26656.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:30:00 | 26640.00 | 26781.30 | 26656.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 11:15:00 | 26462.05 | 26717.45 | 26638.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 12:00:00 | 26462.05 | 26717.45 | 26638.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 12:15:00 | 26504.05 | 26674.77 | 26626.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 13:00:00 | 26504.05 | 26674.77 | 26626.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2024-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 14:15:00 | 26438.95 | 26583.80 | 26590.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 15:15:00 | 26360.00 | 26539.04 | 26569.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 26504.00 | 26426.78 | 26489.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 14:15:00 | 26504.00 | 26426.78 | 26489.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 26504.00 | 26426.78 | 26489.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 26504.00 | 26426.78 | 26489.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 26598.75 | 26461.17 | 26499.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 10:00:00 | 26481.00 | 26465.14 | 26498.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 10:30:00 | 26450.00 | 26466.11 | 26495.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 12:45:00 | 26362.90 | 26448.72 | 26482.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-18 11:15:00 | 26689.25 | 26516.19 | 26500.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2024-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 11:15:00 | 26689.25 | 26516.19 | 26500.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 12:15:00 | 26748.00 | 26562.56 | 26523.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 26388.30 | 26527.70 | 26510.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 26388.30 | 26527.70 | 26510.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 26388.30 | 26527.70 | 26510.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:30:00 | 26392.65 | 26527.70 | 26510.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 26410.00 | 26504.16 | 26501.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:30:00 | 26375.10 | 26504.16 | 26501.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 11:15:00 | 26409.75 | 26485.28 | 26493.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 12:15:00 | 26290.00 | 26393.78 | 26436.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 14:15:00 | 26145.00 | 26095.95 | 26213.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-23 15:00:00 | 26145.00 | 26095.95 | 26213.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 26158.40 | 26113.09 | 26201.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:45:00 | 26168.90 | 26113.09 | 26201.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 26319.30 | 26154.33 | 26212.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 11:00:00 | 26319.30 | 26154.33 | 26212.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 26170.05 | 26157.47 | 26208.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 15:15:00 | 26169.55 | 26199.91 | 26217.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 10:00:00 | 26100.50 | 26175.17 | 26203.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 12:45:00 | 26131.00 | 26216.14 | 26217.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 13:15:00 | 26230.65 | 26219.04 | 26218.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2024-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 13:15:00 | 26230.65 | 26219.04 | 26218.74 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 26162.75 | 26211.62 | 26216.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 11:15:00 | 26021.00 | 26157.24 | 26189.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 26046.90 | 26011.66 | 26091.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 10:00:00 | 26046.90 | 26011.66 | 26091.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 26173.85 | 26044.10 | 26098.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:45:00 | 26169.95 | 26044.10 | 26098.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 26220.00 | 26079.28 | 26109.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 11:45:00 | 26265.70 | 26079.28 | 26109.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 13:15:00 | 26340.00 | 26166.42 | 26146.25 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 25868.75 | 26106.36 | 26124.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 09:15:00 | 25820.80 | 25980.32 | 26045.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 12:15:00 | 25990.00 | 25945.81 | 26010.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 12:15:00 | 25990.00 | 25945.81 | 26010.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 25990.00 | 25945.81 | 26010.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:00:00 | 25990.00 | 25945.81 | 26010.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 25981.70 | 25952.99 | 26007.48 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 26178.20 | 26040.94 | 26024.52 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-03 15:15:00 | 25875.60 | 25995.03 | 26007.56 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 09:15:00 | 26124.00 | 26020.82 | 26018.15 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 25692.50 | 25955.16 | 25988.54 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 13:15:00 | 26217.70 | 26013.31 | 26007.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 14:15:00 | 26280.25 | 26066.70 | 26032.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 09:15:00 | 27132.80 | 27222.41 | 26823.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 10:00:00 | 27132.80 | 27222.41 | 26823.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 26999.95 | 27100.30 | 26907.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 14:30:00 | 26936.95 | 27100.30 | 26907.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 27725.95 | 27961.27 | 27785.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 10:00:00 | 27725.95 | 27961.27 | 27785.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 27544.70 | 27877.96 | 27763.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 10:45:00 | 27588.00 | 27877.96 | 27763.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2024-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 15:15:00 | 27525.00 | 27672.35 | 27691.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 12:15:00 | 27449.95 | 27566.12 | 27615.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 15:15:00 | 26948.00 | 26913.78 | 27045.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 09:15:00 | 27194.40 | 26913.78 | 27045.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 27331.20 | 26997.27 | 27071.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:30:00 | 27215.95 | 26997.27 | 27071.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 27285.70 | 27054.95 | 27090.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 11:30:00 | 27234.20 | 27090.77 | 27103.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 12:30:00 | 27221.05 | 27082.41 | 27098.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 10:15:00 | 27667.30 | 27032.38 | 26949.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 10:15:00 | 27667.30 | 27032.38 | 26949.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 13:15:00 | 27701.05 | 27329.76 | 27120.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 11:15:00 | 27500.00 | 27501.03 | 27300.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 14:15:00 | 27398.95 | 27489.22 | 27346.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 27398.95 | 27489.22 | 27346.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 27398.95 | 27489.22 | 27346.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 27280.00 | 27447.38 | 27340.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 09:15:00 | 27915.45 | 27447.38 | 27340.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 10:15:00 | 27942.00 | 28054.85 | 28056.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 10:15:00 | 27942.00 | 28054.85 | 28056.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 13:15:00 | 27833.20 | 27986.36 | 28022.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 09:15:00 | 28029.45 | 27666.28 | 27767.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 09:15:00 | 28029.45 | 27666.28 | 27767.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 28029.45 | 27666.28 | 27767.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:00:00 | 28029.45 | 27666.28 | 27767.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 28087.40 | 27750.51 | 27796.22 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2024-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 12:15:00 | 28369.05 | 27913.82 | 27864.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 13:15:00 | 28434.40 | 28017.94 | 27916.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 28210.75 | 28600.21 | 28489.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 09:15:00 | 28210.75 | 28600.21 | 28489.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 28210.75 | 28600.21 | 28489.08 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2024-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 11:15:00 | 27920.00 | 28401.90 | 28414.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 12:15:00 | 27720.00 | 28265.52 | 28351.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 28000.00 | 27835.28 | 28064.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 11:00:00 | 28000.00 | 27835.28 | 28064.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 27686.75 | 27827.19 | 27967.60 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2024-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 14:15:00 | 28018.65 | 27788.24 | 27787.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 15:15:00 | 28198.00 | 27870.19 | 27824.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 09:15:00 | 28246.15 | 28433.48 | 28231.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 09:15:00 | 28246.15 | 28433.48 | 28231.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 28246.15 | 28433.48 | 28231.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 10:00:00 | 28246.15 | 28433.48 | 28231.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 28152.05 | 28377.19 | 28224.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 11:00:00 | 28152.05 | 28377.19 | 28224.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 11:15:00 | 28080.00 | 28317.76 | 28211.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 12:00:00 | 28080.00 | 28317.76 | 28211.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 12:15:00 | 28014.00 | 28257.00 | 28193.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 12:45:00 | 27900.00 | 28257.00 | 28193.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 09:15:00 | 28024.75 | 28132.63 | 28146.37 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 13:15:00 | 28239.95 | 28163.15 | 28155.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 09:15:00 | 28428.70 | 28217.55 | 28182.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 14:15:00 | 28419.60 | 28440.55 | 28328.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 15:00:00 | 28419.60 | 28440.55 | 28328.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 28400.00 | 28432.44 | 28335.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:30:00 | 28531.00 | 28449.96 | 28352.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 11:30:00 | 28560.00 | 28453.52 | 28370.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 15:00:00 | 28450.00 | 28455.96 | 28393.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 09:15:00 | 28550.35 | 28432.74 | 28388.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 28492.15 | 28444.62 | 28397.74 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-02 14:15:00 | 28204.25 | 28408.45 | 28398.90 | SL hit (close<static) qty=1.00 sl=28332.00 alert=retest2 |

### Cycle 98 — SELL (started 2024-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 15:15:00 | 28244.30 | 28375.62 | 28384.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 27921.00 | 28278.12 | 28338.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 28520.00 | 28188.94 | 28246.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 28520.00 | 28188.94 | 28246.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 28520.00 | 28188.94 | 28246.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 28520.00 | 28188.94 | 28246.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 28370.25 | 28225.20 | 28258.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:00:00 | 28265.90 | 28233.34 | 28258.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 13:15:00 | 28410.00 | 28230.46 | 28210.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 28410.00 | 28230.46 | 28210.39 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 15:15:00 | 28000.00 | 28179.50 | 28190.40 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 28970.65 | 28337.73 | 28261.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 10:15:00 | 29199.95 | 28510.17 | 28346.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 11:15:00 | 28207.15 | 28449.57 | 28333.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 11:15:00 | 28207.15 | 28449.57 | 28333.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 11:15:00 | 28207.15 | 28449.57 | 28333.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 12:00:00 | 28207.15 | 28449.57 | 28333.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 12:15:00 | 27470.00 | 28253.65 | 28255.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 13:15:00 | 27254.00 | 28053.72 | 28164.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 13:15:00 | 27576.35 | 27573.45 | 27805.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-09 14:00:00 | 27576.35 | 27573.45 | 27805.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 27300.00 | 27146.54 | 27334.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:45:00 | 27335.95 | 27146.54 | 27334.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 27400.00 | 27197.23 | 27340.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:00:00 | 27400.00 | 27197.23 | 27340.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 27300.05 | 27217.79 | 27336.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 09:30:00 | 27276.10 | 27247.38 | 27316.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 10:00:00 | 27183.95 | 27247.38 | 27316.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 11:00:00 | 27265.00 | 27250.90 | 27312.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 10:45:00 | 27265.50 | 27149.64 | 27211.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 11:15:00 | 27700.10 | 27259.73 | 27256.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 27700.10 | 27259.73 | 27256.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 12:15:00 | 27921.05 | 27391.99 | 27316.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 10:15:00 | 27809.50 | 27937.14 | 27767.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 11:00:00 | 27809.50 | 27937.14 | 27767.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 27829.70 | 27915.65 | 27773.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 27745.00 | 27915.65 | 27773.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 27731.40 | 27878.80 | 27769.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:45:00 | 27788.80 | 27878.80 | 27769.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 27780.00 | 27859.04 | 27770.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 28031.10 | 27842.58 | 27778.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 14:15:00 | 29666.00 | 29908.58 | 29938.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 14:15:00 | 29666.00 | 29908.58 | 29938.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 15:15:00 | 29625.20 | 29754.41 | 29805.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 09:15:00 | 29872.35 | 29778.00 | 29811.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 29872.35 | 29778.00 | 29811.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 29872.35 | 29778.00 | 29811.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 29872.35 | 29778.00 | 29811.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 29879.90 | 29798.38 | 29817.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:30:00 | 29840.00 | 29798.38 | 29817.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 29908.45 | 29820.39 | 29826.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:45:00 | 29894.15 | 29820.39 | 29826.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2024-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 12:15:00 | 29905.30 | 29837.38 | 29833.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 14:15:00 | 29941.35 | 29872.24 | 29850.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 15:15:00 | 29692.85 | 29836.36 | 29836.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 15:15:00 | 29692.85 | 29836.36 | 29836.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 29692.85 | 29836.36 | 29836.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 29938.55 | 29836.36 | 29836.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 29899.40 | 29848.97 | 29842.08 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 12:15:00 | 29765.70 | 29826.39 | 29833.54 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 14:15:00 | 29909.95 | 29846.88 | 29841.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 15:15:00 | 29927.60 | 29863.03 | 29849.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 09:15:00 | 29789.80 | 29848.38 | 29844.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 29789.80 | 29848.38 | 29844.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 29789.80 | 29848.38 | 29844.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:00:00 | 29789.80 | 29848.38 | 29844.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 29821.90 | 29843.08 | 29842.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:45:00 | 29750.00 | 29843.08 | 29842.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 12:15:00 | 29894.40 | 29870.36 | 29855.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 12:45:00 | 29880.50 | 29870.36 | 29855.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 29881.75 | 29872.64 | 29858.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 13:30:00 | 29904.60 | 29872.64 | 29858.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 29800.00 | 29858.11 | 29852.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 15:00:00 | 29800.00 | 29858.11 | 29852.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2024-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 15:15:00 | 29780.00 | 29842.49 | 29846.23 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 12:15:00 | 29871.00 | 29849.03 | 29848.04 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 13:15:00 | 29750.90 | 29829.40 | 29839.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 14:15:00 | 29529.00 | 29769.32 | 29811.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 11:15:00 | 29766.50 | 29662.24 | 29736.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 11:15:00 | 29766.50 | 29662.24 | 29736.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 29766.50 | 29662.24 | 29736.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:45:00 | 29718.10 | 29662.24 | 29736.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 29656.10 | 29661.01 | 29729.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 13:15:00 | 29550.10 | 29661.01 | 29729.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 09:15:00 | 29586.90 | 29668.05 | 29714.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 11:15:00 | 29576.10 | 29650.53 | 29698.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 12:30:00 | 29585.20 | 29636.97 | 29683.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 29450.00 | 29576.43 | 29639.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 11:15:00 | 29380.10 | 29545.97 | 29619.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:15:00 | 28072.59 | 28724.09 | 29055.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:15:00 | 28107.56 | 28724.09 | 29055.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:15:00 | 28097.29 | 28724.09 | 29055.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:15:00 | 28105.94 | 28724.09 | 29055.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 11:15:00 | 27911.09 | 28556.55 | 28948.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-19 11:15:00 | 28012.40 | 27979.20 | 28393.77 | SL hit (close>ema200) qty=0.50 sl=27979.20 alert=retest2 |

### Cycle 111 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 11:15:00 | 28378.95 | 28228.46 | 28224.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 13:15:00 | 28428.00 | 28293.51 | 28255.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 10:15:00 | 28279.75 | 28309.70 | 28277.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 10:15:00 | 28279.75 | 28309.70 | 28277.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 28279.75 | 28309.70 | 28277.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:00:00 | 28279.75 | 28309.70 | 28277.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 28282.70 | 28304.30 | 28277.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:45:00 | 28273.75 | 28304.30 | 28277.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 28287.95 | 28301.03 | 28278.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 13:00:00 | 28287.95 | 28301.03 | 28278.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 28319.95 | 28304.82 | 28282.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:00:00 | 28319.95 | 28304.82 | 28282.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 28254.05 | 28294.66 | 28279.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:45:00 | 28251.55 | 28294.66 | 28279.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 28274.00 | 28290.53 | 28279.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:15:00 | 28359.95 | 28290.53 | 28279.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 28323.75 | 28297.17 | 28283.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 14:15:00 | 28538.70 | 28330.22 | 28304.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 09:15:00 | 28865.55 | 29042.97 | 29062.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 28865.55 | 29042.97 | 29062.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 28470.05 | 28888.69 | 28986.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 15:15:00 | 28086.00 | 28049.06 | 28253.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 09:15:00 | 28312.00 | 28101.65 | 28259.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 28312.00 | 28101.65 | 28259.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:45:00 | 28314.00 | 28101.65 | 28259.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 28550.00 | 28191.32 | 28285.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 28550.00 | 28191.32 | 28285.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 28538.65 | 28354.54 | 28342.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 15:15:00 | 28600.10 | 28403.65 | 28365.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 11:15:00 | 28556.45 | 28633.36 | 28553.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 11:15:00 | 28556.45 | 28633.36 | 28553.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 28556.45 | 28633.36 | 28553.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:00:00 | 28556.45 | 28633.36 | 28553.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 28447.00 | 28596.09 | 28543.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:00:00 | 28447.00 | 28596.09 | 28543.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 28345.90 | 28546.05 | 28525.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:00:00 | 28345.90 | 28546.05 | 28525.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 28364.20 | 28509.68 | 28510.83 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 10:15:00 | 28728.35 | 28551.76 | 28528.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 12:15:00 | 28816.05 | 28627.34 | 28567.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 14:15:00 | 28831.30 | 28885.54 | 28763.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 15:00:00 | 28831.30 | 28885.54 | 28763.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 15:15:00 | 28761.10 | 28860.65 | 28763.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:15:00 | 29150.00 | 28860.65 | 28763.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 12:30:00 | 28914.00 | 29017.71 | 28940.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 28699.85 | 28940.99 | 28928.54 | SL hit (close<static) qty=1.00 sl=28761.10 alert=retest2 |

### Cycle 116 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 28522.35 | 28857.26 | 28891.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 28272.55 | 28649.98 | 28780.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 09:15:00 | 28912.60 | 28640.40 | 28738.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 09:15:00 | 28912.60 | 28640.40 | 28738.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 28912.60 | 28640.40 | 28738.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:00:00 | 28912.60 | 28640.40 | 28738.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 29181.10 | 28748.54 | 28778.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 11:00:00 | 29181.10 | 28748.54 | 28778.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 11:15:00 | 28998.40 | 28798.51 | 28798.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 14:15:00 | 29199.55 | 28940.03 | 28867.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 14:15:00 | 29139.25 | 29179.51 | 29050.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-21 14:45:00 | 29142.45 | 29179.51 | 29050.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 29308.25 | 29210.52 | 29097.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:45:00 | 29215.35 | 29210.52 | 29097.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 13:15:00 | 29128.10 | 29182.28 | 29111.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 13:30:00 | 29070.00 | 29182.28 | 29111.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 14:15:00 | 28973.20 | 29140.46 | 29099.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 15:00:00 | 28973.20 | 29140.46 | 29099.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 15:15:00 | 29006.20 | 29113.61 | 29090.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:15:00 | 29093.55 | 29113.61 | 29090.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 29045.05 | 29091.46 | 29084.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:45:00 | 29002.75 | 29091.46 | 29084.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 29074.05 | 29087.98 | 29083.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:00:00 | 29074.05 | 29087.98 | 29083.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 29100.65 | 29090.52 | 29084.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:15:00 | 29019.05 | 29090.52 | 29084.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2024-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 13:15:00 | 28962.35 | 29064.88 | 29073.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 14:15:00 | 28441.00 | 28940.11 | 29016.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 14:15:00 | 28662.25 | 28645.10 | 28796.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 15:00:00 | 28662.25 | 28645.10 | 28796.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 28501.05 | 28629.06 | 28763.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 11:00:00 | 28416.55 | 28586.56 | 28732.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 12:00:00 | 28500.00 | 28569.25 | 28711.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 14:00:00 | 28463.95 | 28546.50 | 28676.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 14:45:00 | 28489.70 | 28536.50 | 28659.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 12:15:00 | 28485.65 | 28395.07 | 28530.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 12:30:00 | 28579.95 | 28395.07 | 28530.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 28498.90 | 28415.84 | 28527.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 15:00:00 | 28438.15 | 28420.30 | 28519.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 15:15:00 | 28342.00 | 28199.76 | 28313.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 12:15:00 | 28618.70 | 28407.11 | 28378.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 12:15:00 | 28618.70 | 28407.11 | 28378.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 11:15:00 | 28693.35 | 28467.79 | 28420.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 09:15:00 | 29300.60 | 29329.04 | 29072.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-05 10:00:00 | 29300.60 | 29329.04 | 29072.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 29135.05 | 29290.24 | 29078.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:45:00 | 29158.05 | 29290.24 | 29078.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 29288.70 | 29289.93 | 29097.37 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 09:15:00 | 28915.15 | 29111.29 | 29117.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 13:15:00 | 28240.00 | 28868.43 | 28997.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 09:15:00 | 28824.90 | 28680.79 | 28864.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 09:15:00 | 28824.90 | 28680.79 | 28864.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 28824.90 | 28680.79 | 28864.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:45:00 | 28807.95 | 28680.79 | 28864.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 28837.85 | 28696.92 | 28824.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 13:00:00 | 28837.85 | 28696.92 | 28824.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 28839.00 | 28725.33 | 28825.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 14:15:00 | 28595.85 | 28725.33 | 28825.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 12:00:00 | 28650.60 | 28664.80 | 28750.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 12:30:00 | 28659.90 | 28658.84 | 28739.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 13:00:00 | 28635.00 | 28658.84 | 28739.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 14:15:00 | 28730.75 | 28678.81 | 28735.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 14:30:00 | 28700.00 | 28678.81 | 28735.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 15:15:00 | 28731.00 | 28689.25 | 28734.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:15:00 | 28884.00 | 28689.25 | 28734.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 28883.65 | 28728.13 | 28748.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:45:00 | 28934.05 | 28728.13 | 28748.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-12 10:15:00 | 29002.50 | 28783.00 | 28771.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 10:15:00 | 29002.50 | 28783.00 | 28771.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-12 11:15:00 | 29102.65 | 28846.93 | 28801.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 13:15:00 | 28842.00 | 28859.88 | 28816.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 13:15:00 | 28842.00 | 28859.88 | 28816.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 28842.00 | 28859.88 | 28816.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 14:00:00 | 28842.00 | 28859.88 | 28816.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 28590.00 | 28805.91 | 28795.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 15:00:00 | 28590.00 | 28805.91 | 28795.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2024-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 15:15:00 | 28511.55 | 28747.04 | 28769.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 28208.50 | 28639.33 | 28718.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 27404.95 | 27363.72 | 27598.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 09:30:00 | 27465.55 | 27363.72 | 27598.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 27399.30 | 27172.22 | 27283.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:00:00 | 27399.30 | 27172.22 | 27283.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 27341.85 | 27206.15 | 27288.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:45:00 | 27311.30 | 27206.15 | 27288.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 27524.55 | 27329.64 | 27334.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:30:00 | 27595.95 | 27329.64 | 27334.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 27647.95 | 27393.30 | 27362.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 14:15:00 | 27703.95 | 27455.43 | 27393.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 27463.70 | 27681.02 | 27595.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 09:15:00 | 27463.70 | 27681.02 | 27595.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 27463.70 | 27681.02 | 27595.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 10:00:00 | 27463.70 | 27681.02 | 27595.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 27639.65 | 27672.75 | 27599.11 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2024-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 10:15:00 | 27348.65 | 27531.95 | 27554.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 11:15:00 | 27200.00 | 27465.56 | 27522.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 13:15:00 | 27442.80 | 27437.58 | 27498.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 13:15:00 | 27442.80 | 27437.58 | 27498.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 27442.80 | 27437.58 | 27498.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:45:00 | 27462.35 | 27437.58 | 27498.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 27441.75 | 27438.42 | 27493.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 14:30:00 | 27522.00 | 27438.42 | 27493.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 27561.35 | 27463.00 | 27499.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 10:45:00 | 27337.10 | 27457.52 | 27490.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 13:00:00 | 27337.30 | 27432.52 | 27473.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 15:00:00 | 27367.15 | 27423.03 | 27462.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 10:15:00 | 27784.10 | 27495.14 | 27484.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 10:15:00 | 27784.10 | 27495.14 | 27484.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 27984.45 | 27741.86 | 27628.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 10:15:00 | 28636.45 | 28733.32 | 28486.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 10:45:00 | 28687.95 | 28733.32 | 28486.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 28899.95 | 28766.65 | 28524.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:30:00 | 28535.90 | 28766.65 | 28524.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 28907.20 | 29055.69 | 28912.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 11:30:00 | 29172.00 | 29080.06 | 28948.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 13:45:00 | 29158.45 | 29112.61 | 28987.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 14:45:00 | 29165.60 | 29120.08 | 29002.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 29182.35 | 29125.66 | 29015.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 29122.55 | 29125.04 | 29025.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 10:45:00 | 29265.25 | 29136.40 | 29083.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 12:15:00 | 28880.10 | 29068.45 | 29060.37 | SL hit (close<static) qty=1.00 sl=28888.05 alert=retest2 |

### Cycle 126 — SELL (started 2024-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 13:15:00 | 28826.05 | 29019.97 | 29039.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 28788.50 | 28940.78 | 28995.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 28683.35 | 28628.46 | 28745.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 15:00:00 | 28683.35 | 28628.46 | 28745.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 28635.05 | 28629.78 | 28735.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 28597.20 | 28629.78 | 28735.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 28617.40 | 28627.30 | 28724.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:30:00 | 28480.00 | 28591.14 | 28699.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 11:15:00 | 28745.25 | 28332.70 | 28317.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2024-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 11:15:00 | 28745.25 | 28332.70 | 28317.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 13:15:00 | 29000.00 | 28551.33 | 28424.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 12:15:00 | 28633.40 | 28823.59 | 28653.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 12:15:00 | 28633.40 | 28823.59 | 28653.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 28633.40 | 28823.59 | 28653.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:00:00 | 28633.40 | 28823.59 | 28653.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 28756.55 | 28810.18 | 28662.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:30:00 | 28550.00 | 28810.18 | 28662.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 28666.65 | 28781.47 | 28662.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 28666.65 | 28781.47 | 28662.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 28611.05 | 28747.39 | 28658.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:15:00 | 28475.55 | 28747.39 | 28658.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 28677.15 | 28733.34 | 28659.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:30:00 | 28463.70 | 28733.34 | 28659.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 28820.75 | 28750.82 | 28674.52 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2024-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 09:15:00 | 28480.75 | 28615.82 | 28633.32 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 15:15:00 | 28750.00 | 28600.26 | 28587.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 29489.05 | 28778.02 | 28669.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 15:15:00 | 29625.00 | 29685.49 | 29393.60 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 09:30:00 | 29893.95 | 29701.87 | 29427.58 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 10:30:00 | 29924.80 | 29751.50 | 29475.08 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 29795.50 | 29784.55 | 29619.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:30:00 | 29773.85 | 29784.55 | 29619.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 13:15:00 | 29895.45 | 29916.91 | 29811.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 13:30:00 | 29840.30 | 29916.91 | 29811.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 29680.15 | 29870.43 | 29816.73 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-03 09:15:00 | 29680.15 | 29870.43 | 29816.73 | SL hit (close<ema400) qty=1.00 sl=29816.73 alert=retest1 |

### Cycle 130 — SELL (started 2025-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 11:15:00 | 29462.40 | 29733.48 | 29760.37 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-06 13:15:00 | 29893.00 | 29717.89 | 29708.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-06 15:15:00 | 29970.60 | 29805.72 | 29752.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-07 14:15:00 | 29971.65 | 30011.30 | 29906.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 14:15:00 | 29971.65 | 30011.30 | 29906.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 29971.65 | 30011.30 | 29906.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:00:00 | 29971.65 | 30011.30 | 29906.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 30000.00 | 30007.23 | 29922.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:45:00 | 29939.10 | 30007.23 | 29922.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 29912.80 | 29988.35 | 29921.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 11:15:00 | 29759.50 | 29988.35 | 29921.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 11:15:00 | 29784.90 | 29947.66 | 29909.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 11:45:00 | 29782.90 | 29947.66 | 29909.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 12:15:00 | 29620.95 | 29882.32 | 29883.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 15:15:00 | 29500.00 | 29717.29 | 29799.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 10:15:00 | 29750.00 | 29714.47 | 29783.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 10:15:00 | 29750.00 | 29714.47 | 29783.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 29750.00 | 29714.47 | 29783.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 11:00:00 | 29750.00 | 29714.47 | 29783.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 11:15:00 | 29690.00 | 29709.58 | 29775.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 11:45:00 | 29725.15 | 29709.58 | 29775.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 29825.00 | 29732.66 | 29779.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 12:45:00 | 29816.25 | 29732.66 | 29779.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 29778.10 | 29741.75 | 29779.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 15:15:00 | 29600.00 | 29737.15 | 29773.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:45:00 | 29475.15 | 29654.48 | 29728.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 09:15:00 | 28120.00 | 28296.25 | 28581.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 09:15:00 | 28001.39 | 28296.25 | 28581.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-17 09:15:00 | 27620.90 | 27481.28 | 27785.08 | SL hit (close>ema200) qty=0.50 sl=27481.28 alert=retest2 |

### Cycle 133 — BUY (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 12:15:00 | 27871.75 | 27767.94 | 27760.21 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 27659.50 | 27747.18 | 27752.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 27465.05 | 27672.00 | 27715.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 11:15:00 | 27679.90 | 27670.98 | 27707.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 11:45:00 | 27644.10 | 27670.98 | 27707.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 12:15:00 | 27607.95 | 27658.38 | 27698.65 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 27738.60 | 27720.90 | 27719.17 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 10:15:00 | 27645.10 | 27729.15 | 27737.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 27395.00 | 27612.18 | 27677.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 12:15:00 | 25833.35 | 25807.72 | 26220.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 13:00:00 | 25833.35 | 25807.72 | 26220.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 26085.10 | 25758.43 | 26054.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:45:00 | 26152.25 | 25758.43 | 26054.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 26006.15 | 25807.97 | 26049.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 11:30:00 | 25948.90 | 25835.71 | 26040.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 12:00:00 | 25946.65 | 25835.71 | 26040.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 11:15:00 | 26200.00 | 26045.03 | 26060.21 | SL hit (close>static) qty=1.00 sl=26174.90 alert=retest2 |

### Cycle 137 — BUY (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 14:15:00 | 26210.90 | 26090.75 | 26078.49 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 25950.40 | 26065.59 | 26079.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 15:15:00 | 25887.95 | 26030.06 | 26062.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 13:15:00 | 25960.75 | 25948.55 | 26003.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 13:15:00 | 25960.75 | 25948.55 | 26003.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 13:15:00 | 25960.75 | 25948.55 | 26003.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:30:00 | 25891.20 | 25948.55 | 26003.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 14:15:00 | 25999.15 | 25958.67 | 26002.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 15:00:00 | 25999.15 | 25958.67 | 26002.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 25958.00 | 25958.54 | 25998.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:15:00 | 26083.80 | 25958.54 | 25998.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 26001.20 | 25967.07 | 25998.97 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 11:15:00 | 26155.85 | 26039.75 | 26028.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 26518.90 | 26169.13 | 26096.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 09:15:00 | 29383.90 | 29507.98 | 29137.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-11 10:00:00 | 29383.90 | 29507.98 | 29137.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 12:15:00 | 29171.10 | 29388.81 | 29171.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 13:00:00 | 29171.10 | 29388.81 | 29171.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 13:15:00 | 29104.10 | 29331.86 | 29164.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 14:00:00 | 29104.10 | 29331.86 | 29164.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 14:15:00 | 29114.95 | 29288.48 | 29160.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 14:30:00 | 29074.90 | 29288.48 | 29160.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 15:15:00 | 29117.20 | 29254.23 | 29156.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-12 09:15:00 | 28472.05 | 29254.23 | 29156.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 09:15:00 | 28290.65 | 29061.51 | 29077.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 27920.80 | 28454.14 | 28684.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 09:15:00 | 28019.00 | 27969.42 | 28262.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 10:15:00 | 28055.00 | 27969.42 | 28262.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 28223.30 | 28020.20 | 28258.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:45:00 | 28229.55 | 28020.20 | 28258.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 28245.35 | 28065.23 | 28257.71 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2025-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 09:15:00 | 28607.95 | 28376.61 | 28352.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 11:15:00 | 29015.55 | 28586.24 | 28457.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 11:15:00 | 29505.75 | 29525.84 | 29228.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 12:00:00 | 29505.75 | 29525.84 | 29228.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 29006.75 | 29383.83 | 29269.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 29006.75 | 29383.83 | 29269.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 28880.00 | 29283.06 | 29234.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 28880.00 | 29283.06 | 29234.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 11:15:00 | 28796.70 | 29185.79 | 29194.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 28671.35 | 28978.91 | 29079.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 12:15:00 | 29318.45 | 29005.79 | 29062.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 12:15:00 | 29318.45 | 29005.79 | 29062.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 29318.45 | 29005.79 | 29062.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:45:00 | 29389.90 | 29005.79 | 29062.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 29448.00 | 29094.23 | 29097.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:45:00 | 29510.00 | 29094.23 | 29097.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 14:15:00 | 29822.30 | 29239.84 | 29163.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-25 09:15:00 | 30421.85 | 29565.87 | 29330.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-25 15:15:00 | 29942.05 | 30065.44 | 29743.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-27 09:30:00 | 29872.00 | 30088.35 | 29783.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 13:15:00 | 29580.00 | 29982.87 | 29832.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 14:00:00 | 29580.00 | 29982.87 | 29832.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 30000.00 | 29986.29 | 29848.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-28 15:00:00 | 30369.95 | 29951.48 | 29872.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-03 10:15:00 | 30199.65 | 30033.64 | 29929.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-07 15:15:00 | 30651.95 | 31128.07 | 31187.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 15:15:00 | 30651.95 | 31128.07 | 31187.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 30483.90 | 30863.66 | 31025.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 09:15:00 | 30902.55 | 30794.20 | 30946.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 10:00:00 | 30902.55 | 30794.20 | 30946.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 30781.95 | 30791.75 | 30931.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 11:30:00 | 30610.00 | 30770.61 | 30909.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 12:15:00 | 30674.00 | 30770.61 | 30909.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 12:45:00 | 30673.45 | 30736.49 | 30881.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 10:15:00 | 30450.00 | 29974.13 | 29963.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2025-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 10:15:00 | 30450.00 | 29974.13 | 29963.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 09:15:00 | 30510.00 | 30283.97 | 30149.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 13:15:00 | 30201.00 | 30343.98 | 30230.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 13:15:00 | 30201.00 | 30343.98 | 30230.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 13:15:00 | 30201.00 | 30343.98 | 30230.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 13:30:00 | 30232.85 | 30343.98 | 30230.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 14:15:00 | 30217.85 | 30318.76 | 30229.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 15:15:00 | 30189.20 | 30318.76 | 30229.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 15:15:00 | 30189.20 | 30292.85 | 30225.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:30:00 | 30250.00 | 30244.28 | 30209.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 11:15:00 | 30325.25 | 30228.56 | 30205.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 14:30:00 | 30362.75 | 30285.11 | 30244.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 30621.40 | 30248.09 | 30230.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 30495.95 | 30297.66 | 30255.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:30:00 | 30319.95 | 30297.66 | 30255.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 31172.95 | 31093.32 | 30880.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:30:00 | 30817.85 | 31093.32 | 30880.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 30970.00 | 31042.56 | 30892.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:45:00 | 30872.00 | 31042.56 | 30892.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 30678.50 | 30969.75 | 30873.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:30:00 | 30642.85 | 30969.75 | 30873.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 30923.00 | 30960.40 | 30877.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:45:00 | 30664.70 | 30960.40 | 30877.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 30636.80 | 30895.68 | 30855.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 15:00:00 | 30636.80 | 30895.68 | 30855.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 30600.00 | 30836.54 | 30832.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 30371.35 | 30836.54 | 30832.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-27 09:15:00 | 30519.40 | 30773.12 | 30804.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 09:15:00 | 30519.40 | 30773.12 | 30804.00 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 11:15:00 | 30980.00 | 30736.03 | 30725.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 12:15:00 | 30990.70 | 30786.97 | 30749.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 13:15:00 | 30556.05 | 30740.78 | 30732.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 13:15:00 | 30556.05 | 30740.78 | 30732.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 30556.05 | 30740.78 | 30732.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 13:45:00 | 30540.00 | 30740.78 | 30732.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 14:15:00 | 30349.85 | 30662.60 | 30697.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 15:15:00 | 30245.30 | 30579.14 | 30656.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 09:15:00 | 30021.00 | 29927.94 | 30206.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 30021.00 | 29927.94 | 30206.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 30021.00 | 29927.94 | 30206.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:15:00 | 29704.30 | 29940.82 | 30099.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 28219.08 | 29195.60 | 29584.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-07 15:15:00 | 29107.55 | 28932.60 | 29253.18 | SL hit (close>ema200) qty=0.50 sl=28932.60 alert=retest2 |

### Cycle 149 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 29591.80 | 29326.50 | 29293.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 29604.85 | 29382.17 | 29321.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 14:15:00 | 30755.00 | 30852.65 | 30408.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 15:00:00 | 30755.00 | 30852.65 | 30408.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 30635.00 | 30752.18 | 30589.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:45:00 | 30930.00 | 30769.75 | 30612.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 15:15:00 | 30915.00 | 30966.21 | 30790.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 11:30:00 | 30915.00 | 30973.42 | 30848.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 14:15:00 | 30490.00 | 30831.83 | 30855.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 14:15:00 | 30490.00 | 30831.83 | 30855.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 10:15:00 | 30150.00 | 30576.85 | 30722.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 11:15:00 | 29815.00 | 29812.80 | 30029.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 12:00:00 | 29815.00 | 29812.80 | 30029.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 29735.00 | 29783.05 | 29931.38 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 11:15:00 | 30180.00 | 29949.97 | 29935.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 12:15:00 | 30275.00 | 30014.97 | 29966.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 30465.00 | 30564.97 | 30398.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 10:15:00 | 30405.00 | 30532.97 | 30398.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 30405.00 | 30532.97 | 30398.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:45:00 | 30400.00 | 30532.97 | 30398.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 30355.00 | 30497.38 | 30394.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:45:00 | 30330.00 | 30497.38 | 30394.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 30300.00 | 30457.90 | 30386.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 12:30:00 | 30290.00 | 30457.90 | 30386.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 30250.00 | 30416.32 | 30373.92 | EMA400 retest candle locked (from upside) |

### Cycle 152 — SELL (started 2025-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 15:15:00 | 30025.00 | 30299.45 | 30325.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 29660.00 | 30171.56 | 30265.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 12:15:00 | 30200.00 | 30064.16 | 30182.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 12:15:00 | 30200.00 | 30064.16 | 30182.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 30200.00 | 30064.16 | 30182.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:45:00 | 30220.00 | 30064.16 | 30182.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 30315.00 | 30114.33 | 30194.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 30315.00 | 30114.33 | 30194.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 30315.00 | 30154.46 | 30205.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:45:00 | 30420.00 | 30154.46 | 30205.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 30170.00 | 30157.57 | 30202.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 30230.00 | 30157.57 | 30202.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 30115.00 | 30149.05 | 30194.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 14:45:00 | 29885.00 | 30168.90 | 30193.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 12:00:00 | 29970.00 | 29996.72 | 30092.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 10:15:00 | 30405.00 | 30153.87 | 30126.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 30405.00 | 30153.87 | 30126.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 31000.00 | 30482.18 | 30389.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 13:15:00 | 30500.00 | 30544.48 | 30454.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 13:45:00 | 30520.00 | 30544.48 | 30454.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 30835.00 | 31125.06 | 31028.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:00:00 | 30835.00 | 31125.06 | 31028.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 31050.00 | 31110.05 | 31030.46 | EMA400 retest candle locked (from upside) |

### Cycle 154 — SELL (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 12:15:00 | 30815.00 | 30987.09 | 30995.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 13:15:00 | 30790.00 | 30947.67 | 30977.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 10:15:00 | 30930.00 | 30873.27 | 30926.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 10:15:00 | 30930.00 | 30873.27 | 30926.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 30930.00 | 30873.27 | 30926.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:00:00 | 30930.00 | 30873.27 | 30926.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 30790.00 | 30856.62 | 30913.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 14:45:00 | 30495.00 | 30696.19 | 30824.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 15:15:00 | 30620.00 | 30380.77 | 30372.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 15:15:00 | 30620.00 | 30380.77 | 30372.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 09:15:00 | 30635.00 | 30431.61 | 30396.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 15:15:00 | 31300.00 | 31674.10 | 31525.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 15:15:00 | 31300.00 | 31674.10 | 31525.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 31300.00 | 31674.10 | 31525.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 11:30:00 | 31920.00 | 31734.66 | 31590.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 13:00:00 | 31945.00 | 31776.73 | 31622.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:15:00 | 31920.00 | 31772.48 | 31660.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 15:15:00 | 31560.00 | 31645.41 | 31655.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 31560.00 | 31645.41 | 31655.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 14:15:00 | 31525.00 | 31613.04 | 31636.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 31600.00 | 31460.84 | 31514.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 31600.00 | 31460.84 | 31514.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 31600.00 | 31460.84 | 31514.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 31615.00 | 31460.84 | 31514.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 31770.00 | 31522.67 | 31537.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 31770.00 | 31522.67 | 31537.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 31675.00 | 31553.14 | 31550.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 14:15:00 | 31845.00 | 31644.25 | 31608.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 09:15:00 | 31655.00 | 31675.32 | 31630.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 31655.00 | 31675.32 | 31630.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 31655.00 | 31675.32 | 31630.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:15:00 | 31640.00 | 31675.32 | 31630.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 31655.00 | 31671.25 | 31632.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 12:30:00 | 31800.00 | 31661.20 | 31634.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 15:15:00 | 31555.00 | 31624.42 | 31623.19 | SL hit (close<static) qty=1.00 sl=31565.00 alert=retest2 |

### Cycle 158 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 31465.00 | 31632.57 | 31633.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 13:15:00 | 31435.00 | 31593.05 | 31615.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 31520.00 | 31369.92 | 31469.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 11:15:00 | 31520.00 | 31369.92 | 31469.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 31520.00 | 31369.92 | 31469.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 31520.00 | 31369.92 | 31469.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 31425.00 | 31380.94 | 31465.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:15:00 | 31535.00 | 31380.94 | 31465.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 31540.00 | 31412.75 | 31471.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:00:00 | 31540.00 | 31412.75 | 31471.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 31585.00 | 31447.20 | 31482.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:45:00 | 31625.00 | 31447.20 | 31482.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 31410.00 | 31180.88 | 31282.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 31410.00 | 31180.88 | 31282.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 31445.00 | 31233.70 | 31297.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:45:00 | 31455.00 | 31233.70 | 31297.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 31625.00 | 31351.37 | 31342.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 32525.00 | 31807.09 | 31619.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 34225.00 | 34405.42 | 33701.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:00:00 | 34225.00 | 34405.42 | 33701.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 34670.00 | 34812.09 | 34458.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 34860.00 | 34812.09 | 34458.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:00:00 | 34855.00 | 34772.34 | 34498.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 14:15:00 | 34750.00 | 35036.96 | 35044.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 14:15:00 | 34750.00 | 35036.96 | 35044.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 10:15:00 | 34690.00 | 34891.16 | 34968.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 10:15:00 | 34485.00 | 34302.59 | 34475.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 10:15:00 | 34485.00 | 34302.59 | 34475.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 34485.00 | 34302.59 | 34475.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:45:00 | 34475.00 | 34302.59 | 34475.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 34275.00 | 34297.07 | 34457.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:00:00 | 34090.00 | 34255.66 | 34423.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 34690.00 | 34443.50 | 34416.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 09:15:00 | 34690.00 | 34443.50 | 34416.18 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 34370.00 | 34409.11 | 34414.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 14:15:00 | 34270.00 | 34366.23 | 34392.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 15:15:00 | 34290.00 | 34276.41 | 34321.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 15:15:00 | 34290.00 | 34276.41 | 34321.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 34290.00 | 34276.41 | 34321.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 34435.00 | 34276.41 | 34321.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 34380.00 | 34297.13 | 34326.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 10:30:00 | 34180.00 | 34264.70 | 34309.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 09:30:00 | 34210.00 | 34186.96 | 34241.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 11:45:00 | 34160.00 | 34216.66 | 34246.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 12:15:00 | 34175.00 | 34141.35 | 34139.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 12:15:00 | 34175.00 | 34141.35 | 34139.49 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 12:15:00 | 34100.00 | 34143.66 | 34144.98 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 13:15:00 | 34260.00 | 34166.92 | 34155.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 14:15:00 | 34750.00 | 34283.54 | 34209.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 34700.00 | 34741.95 | 34538.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 14:45:00 | 34560.00 | 34741.95 | 34538.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 34640.00 | 34721.56 | 34548.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 34230.00 | 34721.56 | 34548.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 34135.00 | 34604.25 | 34510.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:45:00 | 34205.00 | 34604.25 | 34510.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 34030.00 | 34418.32 | 34437.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 33945.00 | 34266.32 | 34361.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 33325.00 | 33295.52 | 33622.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 12:15:00 | 33700.00 | 33295.52 | 33622.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 34240.00 | 33484.41 | 33678.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 34240.00 | 33484.41 | 33678.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 34380.00 | 33663.53 | 33742.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:30:00 | 34385.00 | 33663.53 | 33742.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 34720.00 | 33874.82 | 33831.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 12:15:00 | 34995.00 | 34388.74 | 34128.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 34675.00 | 34746.42 | 34406.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 09:45:00 | 34665.00 | 34746.42 | 34406.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 34370.00 | 34668.70 | 34503.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 34370.00 | 34668.70 | 34503.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 34485.00 | 34631.96 | 34501.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:45:00 | 34560.00 | 34580.57 | 34489.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 33980.00 | 34460.45 | 34443.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:45:00 | 33955.00 | 34460.45 | 34443.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 33925.00 | 34353.36 | 34396.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 33745.00 | 34158.35 | 34295.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 34095.00 | 33912.09 | 34097.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 11:15:00 | 34095.00 | 33912.09 | 34097.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 34095.00 | 33912.09 | 34097.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 34095.00 | 33912.09 | 34097.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 34105.00 | 33950.67 | 34097.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 13:15:00 | 34055.00 | 33950.67 | 34097.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:15:00 | 32352.25 | 32757.99 | 33161.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 32725.00 | 32677.06 | 32962.15 | SL hit (close>ema200) qty=0.50 sl=32677.06 alert=retest2 |

### Cycle 169 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 33020.00 | 32908.48 | 32898.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 33320.00 | 32990.79 | 32937.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 14:15:00 | 33005.00 | 33291.97 | 33136.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 14:15:00 | 33005.00 | 33291.97 | 33136.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 33005.00 | 33291.97 | 33136.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 33005.00 | 33291.97 | 33136.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 32250.00 | 33083.58 | 33055.79 | EMA400 retest candle locked (from upside) |

### Cycle 170 — SELL (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 09:15:00 | 32680.00 | 33002.86 | 33021.63 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 11:15:00 | 33650.00 | 33111.03 | 33066.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 12:15:00 | 33720.00 | 33232.83 | 33125.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 32900.00 | 33201.81 | 33132.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 14:15:00 | 32900.00 | 33201.81 | 33132.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 32900.00 | 33201.81 | 33132.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 32900.00 | 33201.81 | 33132.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 32850.00 | 33131.45 | 33106.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 33195.00 | 33131.45 | 33106.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 13:15:00 | 33010.00 | 33100.59 | 33101.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 33010.00 | 33100.59 | 33101.49 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 14:15:00 | 33300.00 | 33140.47 | 33119.53 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 32680.00 | 33040.70 | 33082.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 12:15:00 | 32655.00 | 32963.56 | 33043.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 14:15:00 | 33095.00 | 32780.91 | 32855.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 14:15:00 | 33095.00 | 32780.91 | 32855.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 33095.00 | 32780.91 | 32855.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 33095.00 | 32780.91 | 32855.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 32900.00 | 32804.73 | 32859.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 32920.00 | 32804.73 | 32859.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 32730.00 | 32709.88 | 32776.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 10:15:00 | 32570.00 | 32664.37 | 32717.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 15:15:00 | 32275.00 | 31852.85 | 31803.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 32275.00 | 31852.85 | 31803.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 32395.00 | 31961.28 | 31857.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 13:15:00 | 32410.00 | 32421.20 | 32238.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:30:00 | 32420.00 | 32421.20 | 32238.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 32490.00 | 32447.57 | 32283.12 | EMA400 retest candle locked (from upside) |

### Cycle 176 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 32155.00 | 32271.55 | 32275.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 31500.00 | 32117.24 | 32204.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 31510.00 | 31462.02 | 31766.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:30:00 | 31455.00 | 31462.02 | 31766.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 31145.00 | 30915.43 | 31140.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 31145.00 | 30915.43 | 31140.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 31295.00 | 30991.35 | 31154.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:45:00 | 31300.00 | 30991.35 | 31154.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 31320.00 | 31057.08 | 31169.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:30:00 | 31340.00 | 31057.08 | 31169.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 31240.00 | 31081.33 | 31160.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:30:00 | 31265.00 | 31081.33 | 31160.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 30890.00 | 31043.06 | 31135.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 31280.00 | 31043.06 | 31135.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 31330.00 | 31100.45 | 31153.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 31425.00 | 31100.45 | 31153.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 31145.00 | 31114.09 | 31150.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 14:15:00 | 31000.00 | 31107.02 | 31140.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 12:30:00 | 31025.00 | 31057.61 | 31095.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 10:15:00 | 31290.00 | 31090.49 | 31092.15 | SL hit (close>static) qty=1.00 sl=31245.00 alert=retest2 |

### Cycle 177 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 31340.00 | 31140.39 | 31114.68 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 14:15:00 | 30930.00 | 31094.84 | 31102.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 15:15:00 | 30885.00 | 31052.87 | 31082.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 10:15:00 | 31065.00 | 31046.84 | 31073.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 10:15:00 | 31065.00 | 31046.84 | 31073.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 31065.00 | 31046.84 | 31073.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:30:00 | 31060.00 | 31046.84 | 31073.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 31120.00 | 31061.47 | 31078.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:00:00 | 31120.00 | 31061.47 | 31078.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 31140.00 | 31077.18 | 31083.72 | EMA400 retest candle locked (from downside) |

### Cycle 179 — BUY (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 13:15:00 | 31145.00 | 31090.74 | 31089.29 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 30960.00 | 31064.59 | 31077.54 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 31145.00 | 31079.67 | 31079.63 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 30910.00 | 31054.47 | 31069.86 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 31220.00 | 31078.98 | 31073.49 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 30945.00 | 31052.19 | 31061.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 10:15:00 | 30855.00 | 30966.69 | 31011.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 11:15:00 | 30640.00 | 30627.83 | 30780.60 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-22 12:15:00 | 30520.00 | 30627.83 | 30780.60 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-22 13:30:00 | 30480.00 | 30579.61 | 30731.69 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 30340.00 | 30324.94 | 30426.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:00:00 | 30175.00 | 30294.95 | 30403.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 10:15:00 | 30405.00 | 30246.62 | 30330.87 | SL hit (close>ema400) qty=1.00 sl=30330.87 alert=retest1 |

### Cycle 185 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 30015.00 | 29687.45 | 29687.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 30095.00 | 29852.03 | 29772.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 14:15:00 | 29985.00 | 30059.29 | 29922.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 14:15:00 | 29985.00 | 30059.29 | 29922.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 29985.00 | 30059.29 | 29922.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 11:30:00 | 30410.00 | 30157.57 | 30010.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 12:00:00 | 30400.00 | 30157.57 | 30010.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 14:15:00 | 29825.00 | 30023.55 | 30036.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 29825.00 | 30023.55 | 30036.10 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 11:15:00 | 30100.00 | 30048.84 | 30042.64 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 29875.00 | 30030.73 | 30037.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 15:15:00 | 29865.00 | 29997.58 | 30021.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 11:15:00 | 29830.00 | 29752.97 | 29845.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 12:00:00 | 29830.00 | 29752.97 | 29845.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 29815.00 | 29765.38 | 29842.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 13:00:00 | 29815.00 | 29765.38 | 29842.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 29905.00 | 29793.30 | 29848.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:00:00 | 29905.00 | 29793.30 | 29848.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 30095.00 | 29853.64 | 29870.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 30095.00 | 29853.64 | 29870.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — BUY (started 2025-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 15:15:00 | 30255.00 | 29933.91 | 29905.80 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 12:15:00 | 29915.00 | 29973.06 | 29977.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 09:15:00 | 29755.00 | 29879.29 | 29928.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 10:15:00 | 29900.00 | 29883.44 | 29925.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 10:15:00 | 29900.00 | 29883.44 | 29925.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 29900.00 | 29883.44 | 29925.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:30:00 | 29925.00 | 29883.44 | 29925.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 29810.00 | 29854.80 | 29892.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 11:30:00 | 29650.00 | 29768.85 | 29825.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 13:15:00 | 30125.00 | 29845.06 | 29850.31 | SL hit (close>static) qty=1.00 sl=29995.00 alert=retest2 |

### Cycle 191 — BUY (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 14:15:00 | 29920.00 | 29860.05 | 29856.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 14:15:00 | 30300.00 | 30008.03 | 29929.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 12:15:00 | 29975.00 | 30035.22 | 29971.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 12:15:00 | 29975.00 | 30035.22 | 29971.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 29975.00 | 30035.22 | 29971.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 15:15:00 | 30100.00 | 30032.54 | 29981.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 12:15:00 | 29870.00 | 29955.86 | 29961.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 29870.00 | 29955.86 | 29961.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 09:15:00 | 29775.00 | 29906.59 | 29935.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 29945.00 | 29839.70 | 29876.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 29945.00 | 29839.70 | 29876.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 29945.00 | 29839.70 | 29876.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 29945.00 | 29839.70 | 29876.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 29935.00 | 29858.76 | 29882.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:30:00 | 29965.00 | 29858.76 | 29882.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 29920.00 | 29871.01 | 29885.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 29920.00 | 29871.01 | 29885.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 29865.00 | 29869.81 | 29883.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:30:00 | 29915.00 | 29869.81 | 29883.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 29900.00 | 29875.84 | 29885.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:45:00 | 29950.00 | 29875.84 | 29885.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 29790.00 | 29858.68 | 29876.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:30:00 | 29770.00 | 29858.68 | 29876.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 29790.00 | 29844.94 | 29868.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 29905.00 | 29844.94 | 29868.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 29730.00 | 29821.95 | 29856.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:30:00 | 29865.00 | 29821.95 | 29856.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 29705.00 | 29750.09 | 29806.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:30:00 | 29630.00 | 29708.16 | 29771.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 11:00:00 | 29675.00 | 29701.53 | 29762.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 13:15:00 | 29345.00 | 29302.81 | 29300.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — BUY (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 13:15:00 | 29345.00 | 29302.81 | 29300.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 10:15:00 | 29735.00 | 29407.25 | 29350.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 13:15:00 | 29500.00 | 29501.31 | 29415.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 14:00:00 | 29500.00 | 29501.31 | 29415.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 29560.00 | 29513.05 | 29428.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 29560.00 | 29513.05 | 29428.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 29500.00 | 29510.44 | 29434.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:45:00 | 29340.00 | 29473.35 | 29424.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 29345.00 | 29447.68 | 29417.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:00:00 | 29345.00 | 29447.68 | 29417.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — SELL (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 11:15:00 | 29075.00 | 29373.15 | 29386.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 12:15:00 | 29050.00 | 29308.52 | 29355.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 11:15:00 | 29255.00 | 29087.28 | 29191.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 11:15:00 | 29255.00 | 29087.28 | 29191.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 29255.00 | 29087.28 | 29191.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:00:00 | 29255.00 | 29087.28 | 29191.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 29400.00 | 29149.82 | 29210.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 13:00:00 | 29400.00 | 29149.82 | 29210.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 29520.00 | 29223.86 | 29238.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:00:00 | 29520.00 | 29223.86 | 29238.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 29530.00 | 29285.09 | 29264.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 10:15:00 | 29600.00 | 29416.36 | 29335.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 15:15:00 | 29450.00 | 29518.01 | 29426.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 15:15:00 | 29450.00 | 29518.01 | 29426.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 29450.00 | 29518.01 | 29426.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 29500.00 | 29518.01 | 29426.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 29410.00 | 29496.40 | 29425.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:00:00 | 29410.00 | 29496.40 | 29425.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 29545.00 | 29506.12 | 29436.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:15:00 | 29290.00 | 29506.12 | 29436.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 29295.00 | 29463.90 | 29423.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 15:00:00 | 29605.00 | 29459.80 | 29428.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:30:00 | 29550.00 | 29502.27 | 29454.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:00:00 | 29560.00 | 29502.27 | 29454.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 14:15:00 | 29580.00 | 29501.77 | 29471.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 29530.00 | 29507.42 | 29476.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 29530.00 | 29507.42 | 29476.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 29465.00 | 29513.75 | 29486.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:45:00 | 29510.00 | 29513.75 | 29486.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 29360.00 | 29483.00 | 29474.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:00:00 | 29360.00 | 29483.00 | 29474.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 29365.00 | 29459.40 | 29464.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 29365.00 | 29459.40 | 29464.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 29280.00 | 29423.52 | 29447.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 29410.00 | 29401.05 | 29432.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 14:15:00 | 29410.00 | 29401.05 | 29432.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 29410.00 | 29401.05 | 29432.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 29410.00 | 29401.05 | 29432.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 29210.00 | 29350.67 | 29403.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 29340.00 | 29350.67 | 29403.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 29530.00 | 29378.43 | 29406.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:00:00 | 29530.00 | 29378.43 | 29406.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — BUY (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 12:15:00 | 29635.00 | 29429.74 | 29426.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 10:15:00 | 29670.00 | 29550.94 | 29493.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 29600.00 | 29631.23 | 29567.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 29600.00 | 29631.23 | 29567.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 29600.00 | 29631.23 | 29567.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:30:00 | 29575.00 | 29631.23 | 29567.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 29705.00 | 29645.98 | 29579.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:30:00 | 29615.00 | 29645.98 | 29579.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 29630.00 | 29642.79 | 29584.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:30:00 | 29475.00 | 29642.79 | 29584.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 29550.00 | 29624.23 | 29581.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:30:00 | 29590.00 | 29624.23 | 29581.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 29540.00 | 29607.38 | 29577.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 15:15:00 | 29620.00 | 29595.91 | 29575.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 29485.00 | 29577.58 | 29570.60 | SL hit (close<static) qty=1.00 sl=29500.00 alert=retest2 |

### Cycle 198 — SELL (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 11:15:00 | 29535.00 | 29561.45 | 29564.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 12:15:00 | 29445.00 | 29538.16 | 29553.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 14:15:00 | 29605.00 | 29524.62 | 29542.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 14:15:00 | 29605.00 | 29524.62 | 29542.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 29605.00 | 29524.62 | 29542.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 29605.00 | 29524.62 | 29542.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 29625.00 | 29544.70 | 29550.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 29690.00 | 29544.70 | 29550.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 29305.00 | 29496.81 | 29527.51 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 12:15:00 | 29615.00 | 29528.57 | 29524.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 15:15:00 | 29780.00 | 29603.43 | 29562.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 09:15:00 | 29565.00 | 29595.74 | 29562.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 29565.00 | 29595.74 | 29562.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 29565.00 | 29595.74 | 29562.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:30:00 | 29415.00 | 29595.74 | 29562.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 29555.00 | 29587.59 | 29561.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:00:00 | 29555.00 | 29587.59 | 29561.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 29550.00 | 29580.08 | 29560.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:45:00 | 29575.00 | 29580.08 | 29560.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 29650.00 | 29594.06 | 29568.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 13:30:00 | 29810.00 | 29644.25 | 29593.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 12:15:00 | 29550.00 | 29762.45 | 29772.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 29550.00 | 29762.45 | 29772.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 13:15:00 | 29455.00 | 29700.96 | 29743.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 11:15:00 | 29535.00 | 29507.67 | 29617.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 11:45:00 | 29535.00 | 29507.67 | 29617.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 29670.00 | 29550.91 | 29618.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 29670.00 | 29550.91 | 29618.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 30045.00 | 29649.73 | 29657.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 30045.00 | 29649.73 | 29657.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — BUY (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 15:15:00 | 30010.00 | 29721.78 | 29689.51 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 29360.00 | 29674.21 | 29685.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 29270.00 | 29549.50 | 29623.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 29115.00 | 29057.57 | 29233.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:00:00 | 29115.00 | 29057.57 | 29233.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 29000.00 | 28971.59 | 29082.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 13:15:00 | 28885.00 | 28982.42 | 29068.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 14:00:00 | 28890.00 | 28963.93 | 29052.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:15:00 | 27440.75 | 27729.91 | 27811.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:15:00 | 27445.50 | 27729.91 | 27811.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 10:15:00 | 27850.00 | 27753.93 | 27814.79 | SL hit (close>ema200) qty=0.50 sl=27753.93 alert=retest2 |

### Cycle 203 — BUY (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 14:15:00 | 27965.00 | 27862.13 | 27853.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 10:15:00 | 28050.00 | 27923.89 | 27885.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 09:15:00 | 27960.00 | 28010.73 | 27959.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 27960.00 | 28010.73 | 27959.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 27960.00 | 28010.73 | 27959.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:00:00 | 27960.00 | 28010.73 | 27959.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 28020.00 | 28012.58 | 27965.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 11:30:00 | 28090.00 | 28028.06 | 27976.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 14:00:00 | 28055.00 | 28032.16 | 27987.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 14:45:00 | 28090.00 | 28043.73 | 27996.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 15:15:00 | 27950.00 | 28024.98 | 27992.46 | SL hit (close<static) qty=1.00 sl=27960.00 alert=retest2 |

### Cycle 204 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 28500.00 | 28679.77 | 28701.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 09:15:00 | 28290.00 | 28546.20 | 28630.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 28155.00 | 28142.44 | 28275.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 15:00:00 | 28155.00 | 28142.44 | 28275.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 28300.00 | 28175.96 | 28267.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:00:00 | 28300.00 | 28175.96 | 28267.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 28360.00 | 28212.77 | 28275.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:45:00 | 28405.00 | 28212.77 | 28275.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — BUY (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 14:15:00 | 28420.00 | 28325.55 | 28315.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 09:15:00 | 28600.00 | 28393.15 | 28348.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 10:15:00 | 28365.00 | 28387.52 | 28350.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 11:00:00 | 28365.00 | 28387.52 | 28350.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 28360.00 | 28382.02 | 28351.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 28360.00 | 28382.02 | 28351.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 28230.00 | 28351.61 | 28340.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:45:00 | 28155.00 | 28351.61 | 28340.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 28350.00 | 28351.29 | 28341.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:45:00 | 28470.00 | 28377.03 | 28353.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:30:00 | 28485.00 | 28429.30 | 28383.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 14:15:00 | 28395.00 | 28410.42 | 28389.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 15:15:00 | 28400.00 | 28393.33 | 28383.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 28400.00 | 28394.67 | 28384.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 28155.00 | 28394.67 | 28384.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 28055.00 | 28326.73 | 28354.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 28055.00 | 28326.73 | 28354.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 10:15:00 | 27900.00 | 28241.39 | 28313.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 15:15:00 | 28140.00 | 28125.75 | 28218.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 15:15:00 | 28140.00 | 28125.75 | 28218.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 28140.00 | 28125.75 | 28218.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 09:15:00 | 27785.00 | 28125.75 | 28218.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 11:30:00 | 28035.00 | 28094.63 | 28178.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 14:15:00 | 28310.00 | 28151.25 | 28184.91 | SL hit (close>static) qty=1.00 sl=28250.00 alert=retest2 |

### Cycle 207 — BUY (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 09:15:00 | 28360.00 | 28227.20 | 28215.71 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 14:15:00 | 28005.00 | 28201.41 | 28214.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 13:15:00 | 27900.00 | 28119.38 | 28167.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 28030.00 | 28029.36 | 28107.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-16 10:00:00 | 28030.00 | 28029.36 | 28107.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 28040.00 | 28034.79 | 28097.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:30:00 | 27860.00 | 27983.83 | 28068.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 11:15:00 | 27730.00 | 27598.20 | 27590.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 27730.00 | 27598.20 | 27590.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 10:15:00 | 27815.00 | 27705.02 | 27654.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 27685.00 | 27701.02 | 27656.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 12:00:00 | 27685.00 | 27701.02 | 27656.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 27780.00 | 27749.32 | 27693.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:45:00 | 27705.00 | 27749.32 | 27693.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 27465.00 | 27713.36 | 27687.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:30:00 | 27540.00 | 27713.36 | 27687.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 27465.00 | 27663.69 | 27667.67 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 27715.00 | 27650.49 | 27649.26 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 12:15:00 | 27575.00 | 27636.91 | 27643.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 09:15:00 | 26880.00 | 27476.59 | 27566.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 11:15:00 | 26925.00 | 26874.54 | 27101.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 11:30:00 | 26910.00 | 26874.54 | 27101.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 27365.00 | 26991.11 | 27116.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:00:00 | 27365.00 | 26991.11 | 27116.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 27725.00 | 27137.89 | 27171.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 15:00:00 | 27725.00 | 27137.89 | 27171.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — BUY (started 2026-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 15:15:00 | 27595.00 | 27229.31 | 27209.92 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 27180.00 | 27246.35 | 27247.22 | EMA200 below EMA400 |

### Cycle 215 — BUY (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 10:15:00 | 27305.00 | 27258.08 | 27252.47 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 27130.00 | 27232.47 | 27241.34 | EMA200 below EMA400 |

### Cycle 217 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 27340.00 | 27253.42 | 27248.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 27500.00 | 27327.67 | 27285.39 | Break + close above crossover candle high |

### Cycle 218 — SELL (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 12:15:00 | 26870.00 | 27244.51 | 27255.45 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 27185.00 | 27115.89 | 27114.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 27375.00 | 27167.71 | 27138.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 10:15:00 | 27375.00 | 27414.99 | 27325.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 11:00:00 | 27375.00 | 27414.99 | 27325.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 27310.00 | 27382.47 | 27332.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:30:00 | 27290.00 | 27382.47 | 27332.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 27290.00 | 27363.98 | 27328.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 27290.00 | 27363.98 | 27328.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 27315.00 | 27354.18 | 27327.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 27235.00 | 27354.18 | 27327.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 220 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 27085.00 | 27300.35 | 27305.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 12:15:00 | 26890.00 | 27139.62 | 27223.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 26555.00 | 26536.51 | 26692.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 10:00:00 | 26555.00 | 26536.51 | 26692.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 26455.00 | 26427.35 | 26544.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 26430.00 | 26427.35 | 26544.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 26635.00 | 26479.13 | 26532.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:30:00 | 26565.00 | 26479.13 | 26532.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 26680.00 | 26519.30 | 26545.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 26715.00 | 26519.30 | 26545.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 221 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 26585.00 | 26564.16 | 26562.40 | EMA200 above EMA400 |

### Cycle 222 — SELL (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 14:15:00 | 26450.00 | 26544.45 | 26554.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 26415.00 | 26505.05 | 26534.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 26470.00 | 26442.57 | 26477.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 10:15:00 | 26470.00 | 26442.57 | 26477.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 26470.00 | 26442.57 | 26477.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:45:00 | 26535.00 | 26442.57 | 26477.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 26530.00 | 26460.05 | 26481.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 26530.00 | 26460.05 | 26481.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 26390.00 | 26446.04 | 26473.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 26440.00 | 26446.04 | 26473.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 26500.00 | 26418.06 | 26448.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 26490.00 | 26418.06 | 26448.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 26500.00 | 26434.45 | 26452.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 14:15:00 | 26410.00 | 26455.44 | 26459.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 26500.00 | 26464.35 | 26462.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 223 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 26500.00 | 26464.35 | 26462.99 | EMA200 above EMA400 |

### Cycle 224 — SELL (started 2026-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 15:15:00 | 26425.00 | 26456.48 | 26459.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 26300.00 | 26425.18 | 26445.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 26365.00 | 26354.99 | 26397.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 14:15:00 | 26365.00 | 26354.99 | 26397.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 26365.00 | 26354.99 | 26397.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 14:45:00 | 26395.00 | 26354.99 | 26397.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 26340.00 | 26353.60 | 26389.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:45:00 | 26290.00 | 26355.90 | 26384.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 13:00:00 | 26290.00 | 26342.72 | 26375.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:30:00 | 26305.00 | 26347.80 | 26361.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 26205.00 | 26365.03 | 26366.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 26245.00 | 26341.03 | 26355.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-27 13:15:00 | 26500.00 | 26321.19 | 26333.55 | SL hit (close>static) qty=1.00 sl=26485.00 alert=retest2 |

### Cycle 225 — BUY (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 14:15:00 | 26525.00 | 26361.95 | 26350.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-02 12:15:00 | 26720.00 | 26510.07 | 26434.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-05 13:15:00 | 27560.00 | 27756.77 | 27354.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-05 13:30:00 | 27545.00 | 27756.77 | 27354.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 27395.00 | 27635.43 | 27394.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 27275.00 | 27635.43 | 27394.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 27350.00 | 27578.34 | 27390.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 27355.00 | 27578.34 | 27390.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 27375.00 | 27537.67 | 27389.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 12:15:00 | 27420.00 | 27537.67 | 27389.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 14:15:00 | 27420.00 | 27494.71 | 27394.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 26925.00 | 27324.21 | 27337.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 226 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 26925.00 | 27324.21 | 27337.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 13:15:00 | 26625.00 | 26855.11 | 26948.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 11:15:00 | 26750.00 | 26694.33 | 26822.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-13 11:30:00 | 26695.00 | 26694.33 | 26822.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 26965.00 | 26675.86 | 26775.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 14:45:00 | 27055.00 | 26675.86 | 26775.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 26990.00 | 26738.69 | 26794.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:15:00 | 26775.00 | 26738.69 | 26794.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 26870.00 | 26764.95 | 26801.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:30:00 | 26560.00 | 26689.21 | 26747.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 09:30:00 | 26625.00 | 26559.75 | 26657.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 13:00:00 | 26550.00 | 26559.79 | 26632.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 26615.00 | 26427.18 | 26490.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 26485.00 | 26438.74 | 26490.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 26590.00 | 26438.74 | 26490.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 26480.00 | 26446.99 | 26489.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:30:00 | 26415.00 | 26446.99 | 26489.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 26620.00 | 26481.59 | 26501.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:00:00 | 26620.00 | 26481.59 | 26501.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-20 12:15:00 | 26730.00 | 26531.28 | 26521.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 227 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 26730.00 | 26531.28 | 26521.87 | EMA200 above EMA400 |

### Cycle 228 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 25945.00 | 26478.16 | 26507.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 25640.00 | 26310.53 | 26428.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 26180.00 | 25784.82 | 26000.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 26180.00 | 25784.82 | 26000.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 26180.00 | 25784.82 | 26000.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 26180.00 | 25784.82 | 26000.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 26530.00 | 25933.85 | 26048.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 26530.00 | 25933.85 | 26048.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 229 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 26785.00 | 26189.87 | 26150.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 27035.00 | 26466.11 | 26289.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 26395.00 | 26637.97 | 26466.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 14:15:00 | 26395.00 | 26637.97 | 26466.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 26395.00 | 26637.97 | 26466.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 15:00:00 | 26395.00 | 26637.97 | 26466.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 26505.00 | 26611.38 | 26470.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 26195.00 | 26528.10 | 26445.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 26290.00 | 26480.48 | 26430.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 26225.00 | 26480.48 | 26430.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 230 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 26265.00 | 26375.81 | 26390.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 26090.00 | 26318.64 | 26362.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 26490.00 | 26087.52 | 26163.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 26490.00 | 26087.52 | 26163.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 26490.00 | 26087.52 | 26163.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 26375.00 | 26087.52 | 26163.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 26310.00 | 26132.02 | 26176.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:15:00 | 26205.00 | 26132.02 | 26176.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 26455.00 | 26232.29 | 26216.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 231 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 26455.00 | 26232.29 | 26216.62 | EMA200 above EMA400 |

### Cycle 232 — SELL (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 09:15:00 | 26025.00 | 26248.72 | 26260.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 12:15:00 | 25900.00 | 26121.74 | 26195.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 26140.00 | 25852.87 | 25952.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 26140.00 | 25852.87 | 25952.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 26140.00 | 25852.87 | 25952.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 26200.00 | 25852.87 | 25952.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 25875.00 | 25857.30 | 25945.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 12:45:00 | 25825.00 | 25865.67 | 25934.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:30:00 | 25825.00 | 25863.23 | 25921.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 10:45:00 | 25780.00 | 25871.45 | 25910.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 12:30:00 | 25800.00 | 25852.93 | 25895.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 25790.00 | 25831.88 | 25877.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 25790.00 | 25831.88 | 25877.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 25815.00 | 25817.80 | 25862.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 25575.00 | 25804.70 | 25836.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 25945.00 | 25816.26 | 25814.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 233 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 25945.00 | 25816.26 | 25814.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 26090.00 | 25871.01 | 25839.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-15 13:15:00 | 25780.00 | 25876.65 | 25849.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 13:15:00 | 25780.00 | 25876.65 | 25849.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 13:15:00 | 25780.00 | 25876.65 | 25849.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:00:00 | 25780.00 | 25876.65 | 25849.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 25990.00 | 25899.32 | 25862.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:30:00 | 25850.00 | 25899.32 | 25862.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 15:15:00 | 25900.00 | 25899.45 | 25865.55 | EMA400 retest candle locked (from upside) |

### Cycle 234 — SELL (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 09:15:00 | 25795.00 | 25863.03 | 25868.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 10:15:00 | 25530.00 | 25729.70 | 25791.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 25535.00 | 25486.88 | 25575.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 25535.00 | 25486.88 | 25575.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 25535.00 | 25486.88 | 25575.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:45:00 | 25620.00 | 25486.88 | 25575.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 25500.00 | 25489.50 | 25568.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:45:00 | 25495.00 | 25489.50 | 25568.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 25505.00 | 25492.60 | 25562.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 14:45:00 | 25450.00 | 25480.61 | 25539.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:00:00 | 25465.00 | 25472.59 | 25525.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:30:00 | 25400.00 | 25459.07 | 25514.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 12:45:00 | 25455.00 | 25409.31 | 25449.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 25250.00 | 25377.44 | 25431.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 14:15:00 | 25220.00 | 25377.44 | 25431.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:45:00 | 25235.00 | 25344.49 | 25371.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 15:15:00 | 25550.00 | 25356.33 | 25362.53 | SL hit (close>static) qty=1.00 sl=25485.00 alert=retest2 |

### Cycle 235 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 25515.00 | 25388.06 | 25376.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 25740.00 | 25458.45 | 25409.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 25485.00 | 25500.40 | 25448.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 14:15:00 | 25485.00 | 25500.40 | 25448.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 25485.00 | 25500.40 | 25448.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:30:00 | 25460.00 | 25500.40 | 25448.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 25510.00 | 25502.32 | 25454.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 25345.00 | 25502.32 | 25454.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 25350.00 | 25471.85 | 25445.01 | EMA400 retest candle locked (from upside) |

### Cycle 236 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 25365.00 | 25427.79 | 25428.23 | EMA200 below EMA400 |

### Cycle 237 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 25675.00 | 25473.12 | 25446.50 | EMA200 above EMA400 |

### Cycle 238 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 25390.00 | 25454.76 | 25461.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 11:15:00 | 25345.00 | 25432.81 | 25450.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 15:15:00 | 25385.00 | 25379.14 | 25415.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 09:15:00 | 25500.00 | 25379.14 | 25415.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 25550.00 | 25413.31 | 25427.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:45:00 | 25630.00 | 25413.31 | 25427.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 239 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 25570.00 | 25444.65 | 25440.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 25835.00 | 25562.18 | 25498.38 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-17 12:00:00 | 21102.10 | 2023-05-22 09:15:00 | 21418.90 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2023-05-17 14:15:00 | 21135.40 | 2023-05-22 09:15:00 | 21418.90 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2023-05-18 12:45:00 | 21064.10 | 2023-05-22 09:15:00 | 21418.90 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2023-05-25 12:00:00 | 21585.20 | 2023-05-31 14:15:00 | 21739.90 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2023-05-25 13:15:00 | 21595.00 | 2023-05-31 14:15:00 | 21739.90 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2023-06-15 09:15:00 | 22236.00 | 2023-06-21 14:15:00 | 22707.90 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest2 | 2023-06-27 12:30:00 | 22825.00 | 2023-07-11 12:15:00 | 23370.00 | STOP_HIT | 1.00 | 2.39% |
| BUY | retest2 | 2023-06-28 09:15:00 | 22840.30 | 2023-07-11 12:15:00 | 23370.00 | STOP_HIT | 1.00 | 2.32% |
| SELL | retest2 | 2023-07-18 10:30:00 | 23134.10 | 2023-07-18 15:15:00 | 23341.20 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2023-08-01 15:15:00 | 23980.00 | 2023-08-02 10:15:00 | 23850.10 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2023-08-02 11:15:00 | 24071.30 | 2023-08-02 12:15:00 | 23830.70 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2023-08-02 12:15:00 | 24003.60 | 2023-08-02 12:15:00 | 23830.70 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2023-08-02 14:15:00 | 23966.90 | 2023-08-04 14:15:00 | 23899.90 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2023-08-08 14:45:00 | 23790.90 | 2023-08-10 10:15:00 | 23989.60 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2023-08-09 10:15:00 | 23735.40 | 2023-08-10 10:15:00 | 23989.60 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2023-08-09 14:45:00 | 23799.90 | 2023-08-10 10:15:00 | 23989.60 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2023-08-23 10:00:00 | 23317.10 | 2023-08-28 11:15:00 | 23310.80 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2023-08-23 13:45:00 | 23325.10 | 2023-08-28 13:15:00 | 23350.00 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2023-08-23 14:15:00 | 23315.10 | 2023-08-28 13:15:00 | 23350.00 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2023-08-24 13:15:00 | 23321.00 | 2023-08-28 13:15:00 | 23350.00 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2023-08-28 10:30:00 | 23115.00 | 2023-08-28 13:15:00 | 23350.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2023-09-08 09:15:00 | 23095.60 | 2023-09-08 11:15:00 | 22950.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2023-09-11 09:15:00 | 23080.50 | 2023-09-11 11:15:00 | 22911.50 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2023-09-13 11:30:00 | 22786.80 | 2023-09-14 09:15:00 | 23009.90 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2023-09-26 10:15:00 | 22540.00 | 2023-09-26 11:15:00 | 22871.90 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2023-09-27 09:45:00 | 22525.40 | 2023-09-29 10:15:00 | 22840.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2023-09-28 09:15:00 | 22550.20 | 2023-09-29 10:15:00 | 22840.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2023-09-28 10:00:00 | 22554.10 | 2023-09-29 10:15:00 | 22840.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2023-10-03 14:45:00 | 23269.70 | 2023-10-04 14:15:00 | 22807.20 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2023-10-11 13:00:00 | 22750.00 | 2023-10-19 13:15:00 | 22571.90 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2023-10-25 13:45:00 | 22588.60 | 2023-10-26 09:15:00 | 22285.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2023-10-25 14:30:00 | 22562.20 | 2023-10-26 09:15:00 | 22285.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2023-11-23 12:45:00 | 23922.00 | 2023-11-28 12:15:00 | 23835.80 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2023-11-24 09:15:00 | 23995.80 | 2023-11-28 12:15:00 | 23835.80 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2023-11-24 09:45:00 | 23997.10 | 2023-11-28 12:15:00 | 23835.80 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2023-12-04 13:45:00 | 23478.00 | 2023-12-18 13:15:00 | 22782.20 | STOP_HIT | 1.00 | 2.96% |
| SELL | retest2 | 2023-12-05 09:45:00 | 23505.00 | 2023-12-18 13:15:00 | 22782.20 | STOP_HIT | 1.00 | 3.08% |
| BUY | retest2 | 2023-12-29 14:15:00 | 22747.90 | 2024-01-11 11:15:00 | 25022.69 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-01-23 12:15:00 | 25530.90 | 2024-01-24 15:15:00 | 25653.00 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2024-01-23 13:45:00 | 25541.80 | 2024-01-24 15:15:00 | 25653.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-01-23 15:15:00 | 25530.00 | 2024-01-24 15:15:00 | 25653.00 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2024-01-24 12:00:00 | 25414.30 | 2024-01-24 15:15:00 | 25653.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-02-02 09:15:00 | 27479.70 | 2024-02-09 14:15:00 | 28082.20 | STOP_HIT | 1.00 | 2.19% |
| SELL | retest2 | 2024-03-02 09:15:00 | 28050.55 | 2024-03-13 15:15:00 | 27400.00 | STOP_HIT | 1.00 | 2.32% |
| SELL | retest2 | 2024-03-02 12:15:00 | 28072.00 | 2024-03-13 15:15:00 | 27400.00 | STOP_HIT | 1.00 | 2.39% |
| SELL | retest2 | 2024-03-04 09:45:00 | 28120.15 | 2024-03-13 15:15:00 | 27400.00 | STOP_HIT | 1.00 | 2.56% |
| SELL | retest2 | 2024-03-22 11:00:00 | 27561.90 | 2024-04-01 09:15:00 | 27287.25 | STOP_HIT | 1.00 | 1.00% |
| SELL | retest2 | 2024-03-22 11:45:00 | 27550.00 | 2024-04-01 09:15:00 | 27287.25 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2024-04-08 11:15:00 | 26660.00 | 2024-04-10 09:15:00 | 26862.50 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-04-18 14:30:00 | 26151.85 | 2024-04-22 12:15:00 | 26379.70 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-04-19 14:45:00 | 26145.25 | 2024-04-22 12:15:00 | 26379.70 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-04-22 09:45:00 | 26143.10 | 2024-04-22 12:15:00 | 26379.70 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-05-17 10:00:00 | 26481.00 | 2024-05-18 11:15:00 | 26689.25 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-05-17 10:30:00 | 26450.00 | 2024-05-18 11:15:00 | 26689.25 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-05-17 12:45:00 | 26362.90 | 2024-05-18 11:15:00 | 26689.25 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-05-24 15:15:00 | 26169.55 | 2024-05-27 13:15:00 | 26230.65 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2024-05-27 10:00:00 | 26100.50 | 2024-05-27 13:15:00 | 26230.65 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-05-27 12:45:00 | 26131.00 | 2024-05-27 13:15:00 | 26230.65 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2024-06-21 11:30:00 | 27234.20 | 2024-06-26 10:15:00 | 27667.30 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-06-21 12:30:00 | 27221.05 | 2024-06-26 10:15:00 | 27667.30 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-06-28 09:15:00 | 27915.45 | 2024-07-11 10:15:00 | 27942.00 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2024-08-01 09:30:00 | 28531.00 | 2024-08-02 14:15:00 | 28204.25 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-08-01 11:30:00 | 28560.00 | 2024-08-02 14:15:00 | 28204.25 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-08-01 15:00:00 | 28450.00 | 2024-08-02 14:15:00 | 28204.25 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-08-02 09:15:00 | 28550.35 | 2024-08-02 14:15:00 | 28204.25 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-08-06 12:00:00 | 28265.90 | 2024-08-07 13:15:00 | 28410.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-08-14 09:30:00 | 27276.10 | 2024-08-16 11:15:00 | 27700.10 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-08-14 10:00:00 | 27183.95 | 2024-08-16 11:15:00 | 27700.10 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-08-14 11:00:00 | 27265.00 | 2024-08-16 11:15:00 | 27700.10 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-08-16 10:45:00 | 27265.50 | 2024-08-16 11:15:00 | 27700.10 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-08-21 09:15:00 | 28031.10 | 2024-09-03 14:15:00 | 29666.00 | STOP_HIT | 1.00 | 5.83% |
| SELL | retest2 | 2024-09-12 13:15:00 | 29550.10 | 2024-09-18 10:15:00 | 28072.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-13 09:15:00 | 29586.90 | 2024-09-18 10:15:00 | 28107.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-13 11:15:00 | 29576.10 | 2024-09-18 10:15:00 | 28097.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-13 12:30:00 | 29585.20 | 2024-09-18 10:15:00 | 28105.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-16 11:15:00 | 29380.10 | 2024-09-18 11:15:00 | 27911.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-12 13:15:00 | 29550.10 | 2024-09-19 11:15:00 | 28012.40 | STOP_HIT | 0.50 | 5.20% |
| SELL | retest2 | 2024-09-13 09:15:00 | 29586.90 | 2024-09-19 11:15:00 | 28012.40 | STOP_HIT | 0.50 | 5.32% |
| SELL | retest2 | 2024-09-13 11:15:00 | 29576.10 | 2024-09-19 11:15:00 | 28012.40 | STOP_HIT | 0.50 | 5.29% |
| SELL | retest2 | 2024-09-13 12:30:00 | 29585.20 | 2024-09-19 11:15:00 | 28012.40 | STOP_HIT | 0.50 | 5.32% |
| SELL | retest2 | 2024-09-16 11:15:00 | 29380.10 | 2024-09-19 11:15:00 | 28012.40 | STOP_HIT | 0.50 | 4.66% |
| BUY | retest2 | 2024-09-25 14:15:00 | 28538.70 | 2024-10-03 09:15:00 | 28865.55 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2024-10-15 09:15:00 | 29150.00 | 2024-10-17 09:15:00 | 28699.85 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-10-16 12:30:00 | 28914.00 | 2024-10-17 09:15:00 | 28699.85 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-10-25 11:00:00 | 28416.55 | 2024-10-30 12:15:00 | 28618.70 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-10-25 12:00:00 | 28500.00 | 2024-10-30 12:15:00 | 28618.70 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2024-10-25 14:00:00 | 28463.95 | 2024-10-30 12:15:00 | 28618.70 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-10-25 14:45:00 | 28489.70 | 2024-10-30 12:15:00 | 28618.70 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2024-10-28 15:00:00 | 28438.15 | 2024-10-30 12:15:00 | 28618.70 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-10-29 15:15:00 | 28342.00 | 2024-10-30 12:15:00 | 28618.70 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-11-08 14:15:00 | 28595.85 | 2024-11-12 10:15:00 | 29002.50 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-11-11 12:00:00 | 28650.60 | 2024-11-12 10:15:00 | 29002.50 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-11-11 12:30:00 | 28659.90 | 2024-11-12 10:15:00 | 29002.50 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-11-11 13:00:00 | 28635.00 | 2024-11-12 10:15:00 | 29002.50 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-11-28 10:45:00 | 27337.10 | 2024-11-29 10:15:00 | 27784.10 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-11-28 13:00:00 | 27337.30 | 2024-11-29 10:15:00 | 27784.10 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-11-28 15:00:00 | 27367.15 | 2024-11-29 10:15:00 | 27784.10 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-12-09 11:30:00 | 29172.00 | 2024-12-11 12:15:00 | 28880.10 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-12-09 13:45:00 | 29158.45 | 2024-12-11 12:15:00 | 28880.10 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-12-09 14:45:00 | 29165.60 | 2024-12-11 12:15:00 | 28880.10 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-12-10 09:15:00 | 29182.35 | 2024-12-11 12:15:00 | 28880.10 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-12-11 10:45:00 | 29265.25 | 2024-12-11 12:15:00 | 28880.10 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-12-16 10:30:00 | 28480.00 | 2024-12-19 11:15:00 | 28745.25 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest1 | 2024-12-31 09:30:00 | 29893.95 | 2025-01-03 09:15:00 | 29680.15 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest1 | 2024-12-31 10:30:00 | 29924.80 | 2025-01-03 09:15:00 | 29680.15 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-01-09 15:15:00 | 29600.00 | 2025-01-15 09:15:00 | 28120.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 09:45:00 | 29475.15 | 2025-01-15 09:15:00 | 28001.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 15:15:00 | 29600.00 | 2025-01-17 09:15:00 | 27620.90 | STOP_HIT | 0.50 | 6.69% |
| SELL | retest2 | 2025-01-10 09:45:00 | 29475.15 | 2025-01-17 09:15:00 | 27620.90 | STOP_HIT | 0.50 | 6.29% |
| SELL | retest2 | 2025-01-30 11:30:00 | 25948.90 | 2025-01-31 11:15:00 | 26200.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-01-30 12:00:00 | 25946.65 | 2025-01-31 11:15:00 | 26200.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-02-28 15:00:00 | 30369.95 | 2025-03-07 15:15:00 | 30651.95 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2025-03-03 10:15:00 | 30199.65 | 2025-03-07 15:15:00 | 30651.95 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest2 | 2025-03-11 11:30:00 | 30610.00 | 2025-03-19 10:15:00 | 30450.00 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2025-03-11 12:15:00 | 30674.00 | 2025-03-19 10:15:00 | 30450.00 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2025-03-11 12:45:00 | 30673.45 | 2025-03-19 10:15:00 | 30450.00 | STOP_HIT | 1.00 | 0.73% |
| BUY | retest2 | 2025-03-21 09:30:00 | 30250.00 | 2025-03-27 09:15:00 | 30519.40 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest2 | 2025-03-21 11:15:00 | 30325.25 | 2025-03-27 09:15:00 | 30519.40 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2025-03-21 14:30:00 | 30362.75 | 2025-03-27 09:15:00 | 30519.40 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2025-03-24 09:15:00 | 30621.40 | 2025-03-27 09:15:00 | 30519.40 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-04-04 09:15:00 | 29704.30 | 2025-04-07 09:15:00 | 28219.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 09:15:00 | 29704.30 | 2025-04-07 15:15:00 | 29107.55 | STOP_HIT | 0.50 | 2.01% |
| BUY | retest2 | 2025-04-21 09:45:00 | 30930.00 | 2025-04-23 14:15:00 | 30490.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-04-21 15:15:00 | 30915.00 | 2025-04-23 14:15:00 | 30490.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-04-22 11:30:00 | 30915.00 | 2025-04-23 14:15:00 | 30490.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-05-08 14:45:00 | 29885.00 | 2025-05-12 10:15:00 | 30405.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-05-09 12:00:00 | 29970.00 | 2025-05-12 10:15:00 | 30405.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-05-26 14:45:00 | 30495.00 | 2025-05-30 15:15:00 | 30620.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-06-09 11:30:00 | 31920.00 | 2025-06-11 15:15:00 | 31560.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-06-09 13:00:00 | 31945.00 | 2025-06-11 15:15:00 | 31560.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-06-10 09:15:00 | 31920.00 | 2025-06-11 15:15:00 | 31560.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-06-18 12:30:00 | 31800.00 | 2025-06-18 15:15:00 | 31555.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-06-19 09:15:00 | 31775.00 | 2025-06-19 12:15:00 | 31465.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-06-19 10:00:00 | 31770.00 | 2025-06-19 12:15:00 | 31465.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-07-02 09:15:00 | 34860.00 | 2025-07-04 14:15:00 | 34750.00 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-07-02 11:00:00 | 34855.00 | 2025-07-04 14:15:00 | 34750.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-07-09 13:00:00 | 34090.00 | 2025-07-11 09:15:00 | 34690.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-07-16 10:30:00 | 34180.00 | 2025-07-22 12:15:00 | 34175.00 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-07-17 09:30:00 | 34210.00 | 2025-07-22 12:15:00 | 34175.00 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-07-17 11:45:00 | 34160.00 | 2025-07-22 12:15:00 | 34175.00 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-08-04 13:15:00 | 34055.00 | 2025-08-07 09:15:00 | 32352.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-04 13:15:00 | 34055.00 | 2025-08-07 14:15:00 | 32725.00 | STOP_HIT | 0.50 | 3.91% |
| BUY | retest2 | 2025-08-14 09:15:00 | 33195.00 | 2025-08-14 13:15:00 | 33010.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-08-22 10:15:00 | 32570.00 | 2025-09-01 15:15:00 | 32275.00 | STOP_HIT | 1.00 | 0.91% |
| SELL | retest2 | 2025-09-11 14:15:00 | 31000.00 | 2025-09-15 10:15:00 | 31290.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-09-12 12:30:00 | 31025.00 | 2025-09-15 10:15:00 | 31290.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest1 | 2025-09-22 12:15:00 | 30520.00 | 2025-09-25 10:15:00 | 30405.00 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest1 | 2025-09-22 13:30:00 | 30480.00 | 2025-09-25 10:15:00 | 30405.00 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-09-24 13:00:00 | 30175.00 | 2025-10-01 12:15:00 | 30015.00 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest2 | 2025-09-25 13:00:00 | 30195.00 | 2025-10-01 12:15:00 | 30015.00 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2025-10-06 11:30:00 | 30410.00 | 2025-10-07 14:15:00 | 29825.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-10-06 12:00:00 | 30400.00 | 2025-10-07 14:15:00 | 29825.00 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-10-20 11:30:00 | 29650.00 | 2025-10-20 13:15:00 | 30125.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-10-23 15:15:00 | 30100.00 | 2025-10-24 12:15:00 | 29870.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-10-30 09:30:00 | 29630.00 | 2025-11-04 13:15:00 | 29345.00 | STOP_HIT | 1.00 | 0.96% |
| SELL | retest2 | 2025-10-30 11:00:00 | 29675.00 | 2025-11-04 13:15:00 | 29345.00 | STOP_HIT | 1.00 | 1.11% |
| BUY | retest2 | 2025-11-12 15:00:00 | 29605.00 | 2025-11-14 11:15:00 | 29365.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-11-13 09:30:00 | 29550.00 | 2025-11-14 11:15:00 | 29365.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-11-13 10:00:00 | 29560.00 | 2025-11-14 11:15:00 | 29365.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-11-13 14:15:00 | 29580.00 | 2025-11-14 11:15:00 | 29365.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-11-19 15:15:00 | 29620.00 | 2025-11-20 09:15:00 | 29485.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-11-20 10:30:00 | 29580.00 | 2025-11-20 11:15:00 | 29535.00 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2025-11-20 11:15:00 | 29565.00 | 2025-11-20 11:15:00 | 29535.00 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-11-25 13:30:00 | 29810.00 | 2025-11-27 12:15:00 | 29550.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-12-05 13:15:00 | 28885.00 | 2025-12-18 09:15:00 | 27440.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 14:00:00 | 28890.00 | 2025-12-18 09:15:00 | 27445.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 13:15:00 | 28885.00 | 2025-12-18 10:15:00 | 27850.00 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2025-12-05 14:00:00 | 28890.00 | 2025-12-18 10:15:00 | 27850.00 | STOP_HIT | 0.50 | 3.60% |
| BUY | retest2 | 2025-12-22 11:30:00 | 28090.00 | 2025-12-22 15:15:00 | 27950.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-12-22 14:00:00 | 28055.00 | 2025-12-22 15:15:00 | 27950.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-12-22 14:45:00 | 28090.00 | 2025-12-22 15:15:00 | 27950.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-12-23 11:30:00 | 28100.00 | 2026-01-01 13:15:00 | 28500.00 | STOP_HIT | 1.00 | 1.42% |
| BUY | retest2 | 2025-12-24 14:45:00 | 28255.00 | 2026-01-01 13:15:00 | 28500.00 | STOP_HIT | 1.00 | 0.87% |
| BUY | retest2 | 2026-01-07 14:45:00 | 28470.00 | 2026-01-09 09:15:00 | 28055.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-01-08 09:30:00 | 28485.00 | 2026-01-09 09:15:00 | 28055.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2026-01-08 14:15:00 | 28395.00 | 2026-01-09 09:15:00 | 28055.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-01-08 15:15:00 | 28400.00 | 2026-01-09 09:15:00 | 28055.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-01-12 09:15:00 | 27785.00 | 2026-01-12 14:15:00 | 28310.00 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-01-12 11:30:00 | 28035.00 | 2026-01-12 14:15:00 | 28310.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-01-16 12:30:00 | 27860.00 | 2026-01-22 11:15:00 | 27730.00 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2026-02-23 14:15:00 | 26410.00 | 2026-02-23 14:15:00 | 26500.00 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2026-02-25 11:45:00 | 26290.00 | 2026-02-27 13:15:00 | 26500.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-02-25 13:00:00 | 26290.00 | 2026-02-27 13:15:00 | 26500.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-02-26 12:30:00 | 26305.00 | 2026-02-27 13:15:00 | 26500.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-02-27 09:15:00 | 26205.00 | 2026-02-27 13:15:00 | 26500.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2026-03-06 12:15:00 | 27420.00 | 2026-03-09 09:15:00 | 26925.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2026-03-06 14:15:00 | 27420.00 | 2026-03-09 09:15:00 | 26925.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-03-17 11:30:00 | 26560.00 | 2026-03-20 12:15:00 | 26730.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-03-18 09:30:00 | 26625.00 | 2026-03-20 12:15:00 | 26730.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2026-03-18 13:00:00 | 26550.00 | 2026-03-20 12:15:00 | 26730.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2026-03-20 09:15:00 | 26615.00 | 2026-03-20 12:15:00 | 26730.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2026-04-01 11:15:00 | 26205.00 | 2026-04-01 12:15:00 | 26455.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-04-08 12:45:00 | 25825.00 | 2026-04-15 10:15:00 | 25945.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2026-04-08 14:30:00 | 25825.00 | 2026-04-15 10:15:00 | 25945.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2026-04-09 10:45:00 | 25780.00 | 2026-04-15 10:15:00 | 25945.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-04-09 12:30:00 | 25800.00 | 2026-04-15 10:15:00 | 25945.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-04-13 09:15:00 | 25575.00 | 2026-04-15 10:15:00 | 25945.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-04-22 14:45:00 | 25450.00 | 2026-04-28 15:15:00 | 25550.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2026-04-23 10:00:00 | 25465.00 | 2026-04-28 15:15:00 | 25550.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2026-04-23 10:30:00 | 25400.00 | 2026-04-29 09:15:00 | 25515.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-04-24 12:45:00 | 25455.00 | 2026-04-29 09:15:00 | 25515.00 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2026-04-24 14:15:00 | 25220.00 | 2026-04-29 09:15:00 | 25515.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-04-28 10:45:00 | 25235.00 | 2026-04-29 09:15:00 | 25515.00 | STOP_HIT | 1.00 | -1.11% |
