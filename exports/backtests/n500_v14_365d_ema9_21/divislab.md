# Divi's Laboratories Ltd. (DIVISLAB)

## Backtest Summary

- **Window:** 2025-07-14 09:15:00 → 2026-05-08 15:15:00 (1409 bars)
- **Last close:** 6705.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 54 |
| ALERT1 | 39 |
| ALERT2 | 39 |
| ALERT2_SKIP | 22 |
| ALERT3 | 101 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 53 |
| PARTIAL | 3 |
| TARGET_HIT | 4 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 56 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 44
- **Target hits / Stop hits / Partials:** 4 / 49 / 3
- **Avg / median % per leg:** 0.41% / -0.66%
- **Sum % (uncompounded):** 23.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 4 | 19.0% | 4 | 17 | 0 | 0.84% | 17.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 4 | 19.0% | 4 | 17 | 0 | 0.84% | 17.6% |
| SELL (all) | 35 | 8 | 22.9% | 0 | 32 | 3 | 0.16% | 5.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 35 | 8 | 22.9% | 0 | 32 | 3 | 0.16% | 5.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 56 | 12 | 21.4% | 4 | 49 | 3 | 0.41% | 23.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 6729.00 | 6779.88 | 6780.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 10:15:00 | 6698.50 | 6763.60 | 6773.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 6672.00 | 6641.75 | 6671.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 6672.00 | 6641.75 | 6671.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 6672.00 | 6641.75 | 6671.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:45:00 | 6674.50 | 6641.75 | 6671.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 6645.00 | 6642.40 | 6669.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:45:00 | 6653.50 | 6642.40 | 6669.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 6651.50 | 6648.68 | 6664.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 6651.50 | 6648.68 | 6664.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 6624.00 | 6642.83 | 6658.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:15:00 | 6619.00 | 6642.83 | 6658.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 11:00:00 | 6607.50 | 6635.77 | 6654.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:15:00 | 6619.00 | 6636.32 | 6650.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:45:00 | 6619.50 | 6634.06 | 6647.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 6624.00 | 6631.00 | 6644.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 13:30:00 | 6580.00 | 6616.30 | 6626.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 09:15:00 | 6509.00 | 6605.21 | 6619.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 6661.50 | 6600.16 | 6609.75 | SL hit (close>static) qty=1.00 sl=6652.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 6661.50 | 6600.16 | 6609.75 | SL hit (close>static) qty=1.00 sl=6652.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 6678.00 | 6625.22 | 6620.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 6678.00 | 6625.22 | 6620.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 6678.00 | 6625.22 | 6620.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 6678.00 | 6625.22 | 6620.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 6678.00 | 6625.22 | 6620.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 6717.00 | 6651.46 | 6633.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 12:15:00 | 6646.00 | 6659.95 | 6642.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 12:15:00 | 6646.00 | 6659.95 | 6642.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 6646.00 | 6659.95 | 6642.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 12:45:00 | 6624.50 | 6659.95 | 6642.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 6643.00 | 6656.56 | 6642.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:30:00 | 6651.00 | 6656.56 | 6642.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 6658.00 | 6656.85 | 6644.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 14:45:00 | 6625.00 | 6656.85 | 6644.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 6644.00 | 6654.28 | 6644.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 6646.00 | 6654.28 | 6644.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 6599.00 | 6643.22 | 6640.11 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 11:15:00 | 6591.50 | 6628.84 | 6633.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 6468.50 | 6584.14 | 6609.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 6456.50 | 6435.85 | 6495.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 11:45:00 | 6469.50 | 6435.85 | 6495.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 6493.50 | 6447.38 | 6494.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 6493.50 | 6447.38 | 6494.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 6509.50 | 6459.81 | 6496.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 6509.50 | 6459.81 | 6496.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 6485.00 | 6464.84 | 6495.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 09:45:00 | 6467.50 | 6468.90 | 6491.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 11:15:00 | 6462.00 | 6475.12 | 6492.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 13:15:00 | 6144.12 | 6280.83 | 6368.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 13:15:00 | 6138.90 | 6280.83 | 6368.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 6124.00 | 6113.26 | 6213.31 | SL hit (close>ema200) qty=0.50 sl=6113.26 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 6124.00 | 6113.26 | 6213.31 | SL hit (close>ema200) qty=0.50 sl=6113.26 alert=retest2 |

### Cycle 4 — BUY (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 12:15:00 | 6097.00 | 6031.22 | 6025.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 09:15:00 | 6117.50 | 6075.74 | 6050.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 15:15:00 | 6154.50 | 6157.35 | 6125.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 09:15:00 | 6135.00 | 6157.35 | 6125.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 6136.00 | 6153.08 | 6126.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 6131.00 | 6153.08 | 6126.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 6167.50 | 6155.96 | 6130.47 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 14:15:00 | 6077.00 | 6117.52 | 6118.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 09:15:00 | 6007.50 | 6089.75 | 6105.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 6027.00 | 6023.51 | 6053.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 10:45:00 | 6030.50 | 6023.51 | 6053.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 6052.00 | 6027.75 | 6048.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:00:00 | 6052.00 | 6027.75 | 6048.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 6030.00 | 6028.20 | 6046.53 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 09:15:00 | 6166.00 | 6056.05 | 6056.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 09:15:00 | 6219.50 | 6141.36 | 6105.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 6165.00 | 6207.89 | 6166.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 6165.00 | 6207.89 | 6166.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 6165.00 | 6207.89 | 6166.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 6141.00 | 6207.89 | 6166.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 6200.00 | 6206.31 | 6169.58 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 6129.50 | 6165.94 | 6167.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 6112.50 | 6143.32 | 6155.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 6175.00 | 6149.66 | 6157.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 6175.00 | 6149.66 | 6157.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 6175.00 | 6149.66 | 6157.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 6175.00 | 6149.66 | 6157.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 6154.00 | 6150.52 | 6157.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 6136.00 | 6150.52 | 6157.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:45:00 | 6138.50 | 6148.80 | 6155.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:00:00 | 6113.50 | 6141.74 | 6151.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 10:15:00 | 6137.50 | 6117.76 | 6118.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 10:15:00 | 6167.50 | 6127.71 | 6122.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 10:15:00 | 6167.50 | 6127.71 | 6122.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 10:15:00 | 6167.50 | 6127.71 | 6122.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 10:15:00 | 6167.50 | 6127.71 | 6122.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 6167.50 | 6127.71 | 6122.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 11:15:00 | 6168.00 | 6135.77 | 6126.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 6150.00 | 6151.64 | 6139.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 6150.00 | 6151.64 | 6139.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 6150.00 | 6151.64 | 6139.06 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 6129.50 | 6132.22 | 6132.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 6105.00 | 6126.77 | 6130.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 13:15:00 | 6044.00 | 6026.48 | 6059.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 13:15:00 | 6044.00 | 6026.48 | 6059.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 6044.00 | 6026.48 | 6059.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:45:00 | 6054.00 | 6026.48 | 6059.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 6023.00 | 6025.78 | 6055.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:30:00 | 6051.50 | 6025.78 | 6055.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 5924.50 | 5995.85 | 6023.58 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 6069.00 | 6028.21 | 6023.65 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 6004.00 | 6025.70 | 6026.08 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 6055.00 | 6031.84 | 6028.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 6096.00 | 6052.73 | 6040.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 6063.00 | 6086.04 | 6071.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 13:15:00 | 6063.00 | 6086.04 | 6071.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 6063.00 | 6086.04 | 6071.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 6063.00 | 6086.04 | 6071.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 6049.50 | 6078.73 | 6069.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:30:00 | 6051.50 | 6078.73 | 6069.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 6085.00 | 6074.22 | 6069.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:15:00 | 6112.50 | 6074.27 | 6070.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 6090.50 | 6133.46 | 6134.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 6090.50 | 6133.46 | 6134.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 6045.00 | 6109.14 | 6122.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 15:15:00 | 5715.00 | 5707.30 | 5779.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 09:15:00 | 5713.50 | 5707.30 | 5779.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 5709.00 | 5702.36 | 5738.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 5725.50 | 5702.36 | 5738.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 5731.00 | 5705.90 | 5728.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 5731.00 | 5705.90 | 5728.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 5711.50 | 5707.02 | 5726.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:30:00 | 5718.50 | 5707.02 | 5726.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 5793.00 | 5724.77 | 5731.51 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 5825.00 | 5744.82 | 5740.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 5850.50 | 5776.38 | 5755.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 5826.50 | 5829.32 | 5799.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 12:30:00 | 5814.50 | 5829.32 | 5799.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 5816.00 | 5826.24 | 5805.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 5872.00 | 5826.24 | 5805.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-10 12:15:00 | 6459.20 | 6249.08 | 6162.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 6556.00 | 6588.83 | 6590.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 6547.50 | 6572.29 | 6582.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 12:15:00 | 6587.50 | 6573.37 | 6580.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 12:15:00 | 6587.50 | 6573.37 | 6580.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 6587.50 | 6573.37 | 6580.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:45:00 | 6575.00 | 6573.37 | 6580.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 6560.00 | 6570.69 | 6578.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:30:00 | 6541.00 | 6563.00 | 6574.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 09:15:00 | 6610.00 | 6509.77 | 6499.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 09:15:00 | 6610.00 | 6509.77 | 6499.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 10:15:00 | 6685.00 | 6544.82 | 6516.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 11:15:00 | 6765.00 | 6787.53 | 6737.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 11:45:00 | 6775.00 | 6787.53 | 6737.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 6801.50 | 6803.82 | 6766.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 14:30:00 | 6876.00 | 6824.93 | 6790.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 10:00:00 | 6850.00 | 6837.96 | 6803.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 10:30:00 | 6851.00 | 6837.17 | 6806.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:15:00 | 6854.50 | 6837.17 | 6806.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 6715.00 | 6812.73 | 6797.81 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-07 11:15:00 | 6715.00 | 6812.73 | 6797.81 | SL hit (close<static) qty=1.00 sl=6764.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 11:15:00 | 6715.00 | 6812.73 | 6797.81 | SL hit (close<static) qty=1.00 sl=6764.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 11:15:00 | 6715.00 | 6812.73 | 6797.81 | SL hit (close<static) qty=1.00 sl=6764.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 11:15:00 | 6715.00 | 6812.73 | 6797.81 | SL hit (close<static) qty=1.00 sl=6764.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-11-07 12:00:00 | 6715.00 | 6812.73 | 6797.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 6731.00 | 6796.39 | 6791.74 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 6636.00 | 6764.31 | 6777.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 6513.50 | 6639.58 | 6688.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 11:15:00 | 6563.50 | 6526.54 | 6560.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 11:15:00 | 6563.50 | 6526.54 | 6560.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 6563.50 | 6526.54 | 6560.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:45:00 | 6567.50 | 6526.54 | 6560.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 6592.50 | 6539.73 | 6563.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:30:00 | 6586.00 | 6539.73 | 6563.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 6599.00 | 6551.58 | 6566.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 13:45:00 | 6623.00 | 6551.58 | 6566.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 10:15:00 | 6595.50 | 6576.77 | 6575.42 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 6541.50 | 6571.03 | 6573.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 6520.00 | 6560.83 | 6568.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 15:15:00 | 6546.00 | 6527.32 | 6540.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 15:15:00 | 6546.00 | 6527.32 | 6540.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 6546.00 | 6527.32 | 6540.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:15:00 | 6541.00 | 6527.32 | 6540.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 6475.00 | 6516.85 | 6534.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 10:30:00 | 6462.50 | 6509.78 | 6529.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 6458.50 | 6498.08 | 6516.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 10:00:00 | 6466.00 | 6491.66 | 6512.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 11:00:00 | 6439.50 | 6481.23 | 6505.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 6458.00 | 6462.11 | 6483.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:30:00 | 6433.50 | 6454.70 | 6470.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:15:00 | 6426.00 | 6389.41 | 6396.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 6474.00 | 6412.26 | 6405.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 6474.00 | 6412.26 | 6405.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 6474.00 | 6412.26 | 6405.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 6474.00 | 6412.26 | 6405.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 6474.00 | 6412.26 | 6405.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 6474.00 | 6412.26 | 6405.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 6474.00 | 6412.26 | 6405.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 6495.50 | 6439.82 | 6420.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 6482.50 | 6485.53 | 6455.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 10:30:00 | 6489.00 | 6485.53 | 6455.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 6456.50 | 6479.72 | 6455.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:45:00 | 6456.50 | 6479.72 | 6455.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 6456.50 | 6475.08 | 6455.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 6456.50 | 6475.08 | 6455.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 6491.50 | 6478.36 | 6459.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 14:15:00 | 6495.50 | 6478.36 | 6459.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 6499.00 | 6481.49 | 6462.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:30:00 | 6500.00 | 6480.99 | 6465.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 11:15:00 | 6449.00 | 6475.08 | 6465.61 | SL hit (close<static) qty=1.00 sl=6452.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-28 11:15:00 | 6449.00 | 6475.08 | 6465.61 | SL hit (close<static) qty=1.00 sl=6452.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-28 11:15:00 | 6449.00 | 6475.08 | 6465.61 | SL hit (close<static) qty=1.00 sl=6452.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 6498.00 | 6467.03 | 6463.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 6464.00 | 6466.42 | 6463.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:45:00 | 6452.00 | 6466.42 | 6463.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 6455.00 | 6464.14 | 6463.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 6455.00 | 6464.14 | 6463.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-01 11:15:00 | 6435.50 | 6458.41 | 6460.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 6435.50 | 6458.41 | 6460.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 14:15:00 | 6424.00 | 6447.51 | 6454.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 11:15:00 | 6425.00 | 6405.39 | 6420.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 11:15:00 | 6425.00 | 6405.39 | 6420.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 6425.00 | 6405.39 | 6420.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:45:00 | 6428.50 | 6405.39 | 6420.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 6415.50 | 6407.41 | 6420.29 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2025-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 15:15:00 | 6460.00 | 6429.51 | 6428.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 10:15:00 | 6483.00 | 6443.96 | 6435.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 15:15:00 | 6420.00 | 6452.73 | 6444.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 15:15:00 | 6420.00 | 6452.73 | 6444.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 6420.00 | 6452.73 | 6444.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 6402.50 | 6452.73 | 6444.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 6404.00 | 6442.98 | 6440.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 15:00:00 | 6471.50 | 6452.25 | 6446.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 10:15:00 | 6384.00 | 6434.72 | 6439.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 6384.00 | 6434.72 | 6439.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 6342.00 | 6407.18 | 6425.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 6362.50 | 6319.44 | 6351.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 6362.50 | 6319.44 | 6351.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 6362.50 | 6319.44 | 6351.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:00:00 | 6362.50 | 6319.44 | 6351.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 6370.00 | 6329.55 | 6352.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:30:00 | 6378.50 | 6329.55 | 6352.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 6381.50 | 6339.94 | 6355.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:45:00 | 6387.00 | 6339.94 | 6355.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 6411.00 | 6349.45 | 6353.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 6411.00 | 6349.45 | 6353.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 6483.00 | 6376.16 | 6365.72 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 12:15:00 | 6375.50 | 6400.40 | 6400.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 13:15:00 | 6357.00 | 6391.72 | 6396.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 14:15:00 | 6341.50 | 6335.10 | 6358.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 15:00:00 | 6341.50 | 6335.10 | 6358.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 6361.50 | 6340.12 | 6356.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:30:00 | 6359.50 | 6340.12 | 6356.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 6316.00 | 6335.30 | 6352.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:15:00 | 6310.50 | 6330.74 | 6349.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 10:15:00 | 6392.00 | 6331.84 | 6338.40 | SL hit (close>static) qty=1.00 sl=6371.50 alert=retest2 |

### Cycle 26 — BUY (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 11:15:00 | 6389.50 | 6343.37 | 6343.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 6565.50 | 6406.39 | 6374.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 6495.50 | 6499.53 | 6463.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 6508.00 | 6499.53 | 6463.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 6483.50 | 6496.52 | 6474.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 13:30:00 | 6492.00 | 6493.12 | 6474.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 6523.00 | 6488.07 | 6475.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 6460.00 | 6498.12 | 6496.84 | SL hit (close<static) qty=1.00 sl=6469.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 6460.00 | 6498.12 | 6496.84 | SL hit (close<static) qty=1.00 sl=6469.50 alert=retest2 |

### Cycle 27 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 6448.00 | 6488.10 | 6492.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 6425.50 | 6475.58 | 6486.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 6480.50 | 6383.67 | 6404.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 6480.50 | 6383.67 | 6404.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 6480.50 | 6383.67 | 6404.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:00:00 | 6346.00 | 6367.93 | 6382.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 10:15:00 | 6348.00 | 6354.01 | 6370.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 10:45:00 | 6350.00 | 6354.91 | 6369.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 15:15:00 | 6392.00 | 6376.02 | 6375.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 15:15:00 | 6392.00 | 6376.02 | 6375.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 15:15:00 | 6392.00 | 6376.02 | 6375.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 15:15:00 | 6392.00 | 6376.02 | 6375.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 6417.00 | 6384.22 | 6378.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 12:15:00 | 6392.50 | 6394.93 | 6386.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 13:00:00 | 6392.50 | 6394.93 | 6386.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 6360.50 | 6388.05 | 6383.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 6360.50 | 6388.05 | 6383.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 6366.00 | 6383.64 | 6382.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:15:00 | 6348.50 | 6383.64 | 6382.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 15:15:00 | 6348.50 | 6376.61 | 6379.08 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 6506.00 | 6402.49 | 6390.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 10:15:00 | 6543.50 | 6430.69 | 6404.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 6623.50 | 6633.65 | 6579.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 11:00:00 | 6623.50 | 6633.65 | 6579.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 6601.50 | 6627.22 | 6581.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:45:00 | 6594.00 | 6627.22 | 6581.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 6597.50 | 6621.79 | 6590.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 6597.50 | 6621.79 | 6590.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 6608.00 | 6619.03 | 6591.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 6573.00 | 6619.03 | 6591.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 6604.00 | 6616.02 | 6592.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 6550.50 | 6616.02 | 6592.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 6626.00 | 6618.02 | 6595.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:30:00 | 6601.50 | 6618.02 | 6595.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 6584.50 | 6610.51 | 6596.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:30:00 | 6582.50 | 6610.51 | 6596.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 6573.00 | 6603.01 | 6594.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 13:45:00 | 6587.00 | 6603.01 | 6594.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 6624.50 | 6607.31 | 6596.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:30:00 | 6591.00 | 6607.31 | 6596.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 6399.00 | 6565.68 | 6579.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 10:15:00 | 6331.00 | 6368.07 | 6411.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 6125.00 | 6040.90 | 6087.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 6125.00 | 6040.90 | 6087.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 6125.00 | 6040.90 | 6087.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 6125.00 | 6040.90 | 6087.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 6069.00 | 6046.52 | 6085.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:45:00 | 6065.50 | 6053.22 | 6085.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:15:00 | 6068.00 | 6056.87 | 6083.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:30:00 | 6061.00 | 6061.50 | 6076.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 10:15:00 | 6068.00 | 6041.58 | 6057.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 6043.50 | 6041.96 | 6055.81 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 6098.00 | 6060.17 | 6059.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 6098.00 | 6060.17 | 6059.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 6098.00 | 6060.17 | 6059.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 6098.00 | 6060.17 | 6059.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 6098.00 | 6060.17 | 6059.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 6127.00 | 6080.55 | 6069.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 6112.50 | 6134.07 | 6103.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 6112.50 | 6134.07 | 6103.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 6112.50 | 6134.07 | 6103.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 6112.50 | 6134.07 | 6103.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 6101.50 | 6127.56 | 6103.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 6090.50 | 6127.56 | 6103.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 6036.00 | 6109.25 | 6096.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 6036.00 | 6109.25 | 6096.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 12:15:00 | 6004.50 | 6088.30 | 6088.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 5984.50 | 6027.35 | 6040.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 5968.50 | 5967.92 | 5997.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 09:15:00 | 6188.00 | 5967.92 | 5997.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 34 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 6241.50 | 6022.64 | 6020.13 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 14:15:00 | 6040.00 | 6088.44 | 6091.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 6000.50 | 6066.30 | 6080.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 12:15:00 | 6116.00 | 6059.91 | 6072.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 12:15:00 | 6116.00 | 6059.91 | 6072.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 6116.00 | 6059.91 | 6072.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 13:00:00 | 6116.00 | 6059.91 | 6072.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 6115.50 | 6071.03 | 6076.39 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2026-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 14:15:00 | 6137.50 | 6084.32 | 6081.95 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 5981.50 | 6071.07 | 6076.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 12:15:00 | 5950.50 | 6018.97 | 6049.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 6014.00 | 6013.66 | 6041.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 15:00:00 | 6014.00 | 6013.66 | 6041.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 6043.00 | 6021.34 | 6040.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 6057.50 | 6021.34 | 6040.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 6090.00 | 6047.98 | 6049.67 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 6100.50 | 6058.48 | 6054.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 6111.00 | 6068.99 | 6059.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 6153.00 | 6157.85 | 6124.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 6203.50 | 6157.85 | 6124.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 6197.50 | 6165.78 | 6130.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:45:00 | 6234.50 | 6179.72 | 6140.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 12:45:00 | 6240.00 | 6209.16 | 6161.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 6099.00 | 6198.04 | 6201.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 6099.00 | 6198.04 | 6201.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 6099.00 | 6198.04 | 6201.89 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 6213.00 | 6188.67 | 6186.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 6264.00 | 6211.33 | 6198.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 6280.00 | 6293.21 | 6261.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 15:00:00 | 6280.00 | 6293.21 | 6261.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 6283.50 | 6288.36 | 6264.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 11:45:00 | 6315.00 | 6288.89 | 6268.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 13:30:00 | 6309.00 | 6294.49 | 6274.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 6339.00 | 6293.43 | 6277.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 09:15:00 | 6252.00 | 6276.55 | 6277.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 09:15:00 | 6252.00 | 6276.55 | 6277.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 09:15:00 | 6252.00 | 6276.55 | 6277.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 6252.00 | 6276.55 | 6277.96 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 6338.50 | 6286.48 | 6279.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 6376.00 | 6304.38 | 6288.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 10:15:00 | 6427.00 | 6432.15 | 6392.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 10:45:00 | 6423.00 | 6432.15 | 6392.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 6403.00 | 6428.20 | 6403.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 6403.00 | 6428.20 | 6403.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 6395.50 | 6421.66 | 6402.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 6321.50 | 6421.66 | 6402.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 6323.50 | 6402.03 | 6395.37 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 6342.00 | 6390.02 | 6390.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 6259.00 | 6361.79 | 6376.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 6327.50 | 6313.88 | 6338.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 6327.50 | 6313.88 | 6338.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 6327.50 | 6313.88 | 6338.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:30:00 | 6355.00 | 6313.88 | 6338.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 6352.00 | 6321.51 | 6339.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:30:00 | 6347.50 | 6321.51 | 6339.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 6315.00 | 6320.21 | 6337.70 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 6418.50 | 6348.33 | 6344.41 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 6244.00 | 6334.21 | 6343.93 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 09:15:00 | 6432.50 | 6341.63 | 6338.47 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 6233.00 | 6352.40 | 6364.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 6169.00 | 6279.31 | 6317.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 6074.00 | 6045.54 | 6129.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 6079.50 | 6045.54 | 6129.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 6116.00 | 6062.99 | 6123.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 6118.00 | 6062.99 | 6123.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 6111.00 | 6072.59 | 6121.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 6074.00 | 6072.59 | 6121.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 15:00:00 | 6090.50 | 6083.05 | 6111.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 6166.00 | 6105.08 | 6115.01 | SL hit (close>static) qty=1.00 sl=6147.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 6166.00 | 6105.08 | 6115.01 | SL hit (close>static) qty=1.00 sl=6147.00 alert=retest2 |

### Cycle 48 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 6166.00 | 6125.25 | 6122.95 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 6085.00 | 6123.71 | 6124.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 6039.50 | 6097.08 | 6111.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 6057.00 | 6041.38 | 6073.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 6057.00 | 6041.38 | 6073.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 6057.00 | 6041.38 | 6073.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:45:00 | 6072.00 | 6041.38 | 6073.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 6112.50 | 6055.60 | 6077.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 6109.50 | 6055.60 | 6077.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 6116.50 | 6067.78 | 6080.68 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 13:15:00 | 6136.50 | 6097.88 | 6093.14 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 6009.00 | 6081.00 | 6086.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-24 10:15:00 | 5973.00 | 6013.52 | 6042.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 6038.50 | 6013.55 | 6036.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 6038.50 | 6013.55 | 6036.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 6038.50 | 6013.55 | 6036.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 6051.50 | 6013.55 | 6036.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 6041.50 | 6019.14 | 6037.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 6057.50 | 6019.14 | 6037.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 6034.00 | 6022.11 | 6037.01 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 6073.50 | 6046.42 | 6045.21 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 15:15:00 | 6015.00 | 6044.73 | 6046.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 09:15:00 | 5994.00 | 6034.58 | 6041.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 10:15:00 | 6035.50 | 6034.77 | 6041.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 10:15:00 | 6035.50 | 6034.77 | 6041.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 6035.50 | 6034.77 | 6041.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 6035.50 | 6034.77 | 6041.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 6024.50 | 6027.65 | 6035.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:45:00 | 6043.00 | 6027.65 | 6035.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 6045.00 | 6024.53 | 6032.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 5909.50 | 6024.53 | 6032.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 5949.00 | 5976.03 | 5991.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:00:00 | 5933.50 | 5974.48 | 5988.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 5651.55 | 5887.37 | 5940.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 5847.50 | 5845.61 | 5900.10 | SL hit (close>ema200) qty=0.50 sl=5845.61 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 5909.50 | 5858.99 | 5856.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 5909.50 | 5858.99 | 5856.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 5909.50 | 5858.99 | 5856.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 5996.00 | 5944.69 | 5915.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 6031.50 | 6058.34 | 6000.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 6031.50 | 6058.34 | 6000.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 6031.50 | 6058.34 | 6000.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 6063.00 | 6058.34 | 6000.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 6063.00 | 6058.67 | 6006.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:45:00 | 6052.50 | 6059.74 | 6011.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-05 14:15:00 | 6657.75 | 6618.02 | 6581.35 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-05-05 15:15:00 | 6669.30 | 6628.41 | 6589.41 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-05-05 15:15:00 | 6669.30 | 6628.41 | 6589.41 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-07-24 10:15:00 | 6619.00 | 2025-07-29 12:15:00 | 6661.50 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-07-24 11:00:00 | 6607.50 | 2025-07-29 12:15:00 | 6661.50 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-07-24 14:15:00 | 6619.00 | 2025-07-29 14:15:00 | 6678.00 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-07-24 14:45:00 | 6619.50 | 2025-07-29 14:15:00 | 6678.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-07-28 13:30:00 | 6580.00 | 2025-07-29 14:15:00 | 6678.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-07-29 09:15:00 | 6509.00 | 2025-07-29 14:15:00 | 6678.00 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-08-05 09:45:00 | 6467.50 | 2025-08-06 13:15:00 | 6144.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 11:15:00 | 6462.00 | 2025-08-06 13:15:00 | 6138.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 09:45:00 | 6467.50 | 2025-08-07 14:15:00 | 6124.00 | STOP_HIT | 0.50 | 5.31% |
| SELL | retest2 | 2025-08-05 11:15:00 | 6462.00 | 2025-08-07 14:15:00 | 6124.00 | STOP_HIT | 0.50 | 5.23% |
| SELL | retest2 | 2025-08-29 12:15:00 | 6136.00 | 2025-09-03 10:15:00 | 6167.50 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-08-29 13:45:00 | 6138.50 | 2025-09-03 10:15:00 | 6167.50 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-08-29 15:00:00 | 6113.50 | 2025-09-03 10:15:00 | 6167.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-09-03 10:15:00 | 6137.50 | 2025-09-03 10:15:00 | 6167.50 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-09-18 14:15:00 | 6112.50 | 2025-09-22 14:15:00 | 6090.50 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-10-07 09:15:00 | 5872.00 | 2025-10-10 12:15:00 | 6459.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-27 09:30:00 | 6541.00 | 2025-10-30 09:15:00 | 6610.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-11-06 14:30:00 | 6876.00 | 2025-11-07 11:15:00 | 6715.00 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-11-07 10:00:00 | 6850.00 | 2025-11-07 11:15:00 | 6715.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-11-07 10:30:00 | 6851.00 | 2025-11-07 11:15:00 | 6715.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-11-07 11:15:00 | 6854.50 | 2025-11-07 11:15:00 | 6715.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-11-18 10:30:00 | 6462.50 | 2025-11-26 09:15:00 | 6474.00 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-11-19 09:15:00 | 6458.50 | 2025-11-26 09:15:00 | 6474.00 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-11-19 10:00:00 | 6466.00 | 2025-11-26 09:15:00 | 6474.00 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-11-19 11:00:00 | 6439.50 | 2025-11-26 09:15:00 | 6474.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-11-21 09:30:00 | 6433.50 | 2025-11-26 09:15:00 | 6474.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-11-25 14:15:00 | 6426.00 | 2025-11-26 09:15:00 | 6474.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-11-27 14:15:00 | 6495.50 | 2025-11-28 11:15:00 | 6449.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-11-27 15:15:00 | 6499.00 | 2025-11-28 11:15:00 | 6449.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-11-28 09:30:00 | 6500.00 | 2025-11-28 11:15:00 | 6449.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-12-01 09:15:00 | 6498.00 | 2025-12-01 11:15:00 | 6435.50 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-05 15:00:00 | 6471.50 | 2025-12-08 10:15:00 | 6384.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-12-17 12:15:00 | 6310.50 | 2025-12-18 10:15:00 | 6392.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-12-23 13:30:00 | 6492.00 | 2025-12-26 12:15:00 | 6460.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-12-24 09:15:00 | 6523.00 | 2025-12-26 12:15:00 | 6460.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2026-01-01 13:00:00 | 6346.00 | 2026-01-02 15:15:00 | 6392.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-01-02 10:15:00 | 6348.00 | 2026-01-02 15:15:00 | 6392.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-01-02 10:45:00 | 6350.00 | 2026-01-02 15:15:00 | 6392.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-01-22 11:45:00 | 6065.50 | 2026-01-28 09:15:00 | 6098.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-01-22 13:15:00 | 6068.00 | 2026-01-28 09:15:00 | 6098.00 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2026-01-23 10:30:00 | 6061.00 | 2026-01-28 09:15:00 | 6098.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-01-27 10:15:00 | 6068.00 | 2026-01-28 09:15:00 | 6098.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2026-02-11 10:45:00 | 6234.50 | 2026-02-13 09:15:00 | 6099.00 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2026-02-11 12:45:00 | 6240.00 | 2026-02-13 09:15:00 | 6099.00 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2026-02-20 11:45:00 | 6315.00 | 2026-02-24 09:15:00 | 6252.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-02-20 13:30:00 | 6309.00 | 2026-02-24 09:15:00 | 6252.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2026-02-23 09:15:00 | 6339.00 | 2026-02-24 09:15:00 | 6252.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-03-17 11:15:00 | 6074.00 | 2026-03-18 10:15:00 | 6166.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2026-03-17 15:00:00 | 6090.50 | 2026-03-18 10:15:00 | 6166.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-03-30 09:15:00 | 5909.50 | 2026-04-02 09:15:00 | 5651.55 | PARTIAL | 0.50 | 4.37% |
| SELL | retest2 | 2026-03-30 09:15:00 | 5909.50 | 2026-04-02 13:15:00 | 5847.50 | STOP_HIT | 0.50 | 1.05% |
| SELL | retest2 | 2026-04-01 10:45:00 | 5949.00 | 2026-04-08 09:15:00 | 5909.50 | STOP_HIT | 1.00 | 0.66% |
| SELL | retest2 | 2026-04-01 13:00:00 | 5933.50 | 2026-04-08 09:15:00 | 5909.50 | STOP_HIT | 1.00 | 0.40% |
| BUY | retest2 | 2026-04-13 10:15:00 | 6063.00 | 2026-05-05 14:15:00 | 6657.75 | TARGET_HIT | 1.00 | 9.81% |
| BUY | retest2 | 2026-04-13 10:45:00 | 6063.00 | 2026-05-05 15:15:00 | 6669.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 11:45:00 | 6052.50 | 2026-05-05 15:15:00 | 6669.30 | TARGET_HIT | 1.00 | 10.19% |
