# Blue Dart Express Ltd. (BLUEDART)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 5695.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 74 |
| ALERT1 | 44 |
| ALERT2 | 44 |
| ALERT2_SKIP | 21 |
| ALERT3 | 109 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 52 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 54 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 22 / 33
- **Target hits / Stop hits / Partials:** 1 / 51 / 3
- **Avg / median % per leg:** 0.76% / -0.54%
- **Sum % (uncompounded):** 41.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 11 | 40.7% | 1 | 26 | 0 | 1.32% | 35.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 27 | 11 | 40.7% | 1 | 26 | 0 | 1.32% | 35.6% |
| SELL (all) | 28 | 11 | 39.3% | 0 | 25 | 3 | 0.22% | 6.2% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.70% | -1.4% |
| SELL @ 3rd Alert (retest2) | 26 | 11 | 42.3% | 0 | 23 | 3 | 0.29% | 7.6% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.70% | -1.4% |
| retest2 (combined) | 53 | 22 | 41.5% | 1 | 49 | 3 | 0.81% | 43.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 6904.50 | 6938.65 | 6940.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 13:15:00 | 6870.00 | 6919.45 | 6931.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 7020.50 | 6923.15 | 6928.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 7020.50 | 6923.15 | 6928.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 7020.50 | 6923.15 | 6928.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:00:00 | 7020.50 | 6923.15 | 6928.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 7000.00 | 6938.52 | 6934.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 7129.50 | 7025.46 | 6992.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 6868.50 | 7087.32 | 7053.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 6868.50 | 7087.32 | 7053.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 6868.50 | 7087.32 | 7053.90 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 6845.00 | 6999.04 | 7017.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 15:15:00 | 6810.00 | 6894.26 | 6955.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 14:15:00 | 6382.00 | 6368.75 | 6439.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 15:00:00 | 6382.00 | 6368.75 | 6439.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 6561.00 | 6412.12 | 6446.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:00:00 | 6561.00 | 6412.12 | 6446.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 6544.00 | 6438.50 | 6455.81 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 6523.50 | 6469.50 | 6467.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 6535.50 | 6487.58 | 6476.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 14:15:00 | 6500.00 | 6515.07 | 6506.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 14:15:00 | 6500.00 | 6515.07 | 6506.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 6500.00 | 6515.07 | 6506.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 15:00:00 | 6500.00 | 6515.07 | 6506.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 6497.50 | 6511.55 | 6505.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:15:00 | 6529.00 | 6511.55 | 6505.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 11:00:00 | 6527.50 | 6519.13 | 6510.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 13:45:00 | 6526.00 | 6548.05 | 6536.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 14:15:00 | 6529.50 | 6548.05 | 6536.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 6549.00 | 6548.24 | 6538.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 6477.00 | 6523.97 | 6528.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 6477.00 | 6523.97 | 6528.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 6477.00 | 6523.97 | 6528.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 6477.00 | 6523.97 | 6528.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 6477.00 | 6523.97 | 6528.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 6474.00 | 6513.98 | 6523.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 12:15:00 | 6514.00 | 6513.98 | 6522.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 12:15:00 | 6514.00 | 6513.98 | 6522.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 6514.00 | 6513.98 | 6522.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:45:00 | 6533.00 | 6513.98 | 6522.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 6383.00 | 6487.78 | 6510.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 09:15:00 | 6325.50 | 6455.62 | 6490.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 11:15:00 | 6378.00 | 6430.52 | 6472.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 6312.00 | 6226.11 | 6215.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 6312.00 | 6226.11 | 6215.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 6312.00 | 6226.11 | 6215.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 6396.00 | 6260.09 | 6231.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 6258.50 | 6281.02 | 6250.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 14:15:00 | 6258.50 | 6281.02 | 6250.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 6258.50 | 6281.02 | 6250.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 6258.50 | 6281.02 | 6250.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 6240.50 | 6272.92 | 6249.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 6339.00 | 6272.92 | 6249.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-02 09:15:00 | 6972.90 | 6772.43 | 6602.96 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 6713.00 | 6770.57 | 6774.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 15:15:00 | 6675.00 | 6741.45 | 6759.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 13:15:00 | 6704.50 | 6703.84 | 6730.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 14:00:00 | 6704.50 | 6703.84 | 6730.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 6664.00 | 6687.08 | 6715.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:45:00 | 6646.00 | 6677.57 | 6708.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 6729.00 | 6652.48 | 6658.06 | SL hit (close>static) qty=1.00 sl=6719.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 10:15:00 | 6744.00 | 6670.79 | 6665.87 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 6670.00 | 6678.08 | 6678.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 14:15:00 | 6630.00 | 6668.29 | 6673.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 6680.50 | 6663.57 | 6670.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 6680.50 | 6663.57 | 6670.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 6680.50 | 6663.57 | 6670.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 6680.50 | 6663.57 | 6670.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 6640.50 | 6658.95 | 6667.43 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 6816.00 | 6694.28 | 6680.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 6901.50 | 6751.04 | 6709.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 10:15:00 | 6849.50 | 6861.62 | 6805.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 11:00:00 | 6849.50 | 6861.62 | 6805.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 6902.00 | 6891.98 | 6848.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 13:45:00 | 6920.00 | 6900.79 | 6866.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 14:15:00 | 6919.00 | 6900.79 | 6866.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 6836.00 | 6864.34 | 6865.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 6836.00 | 6864.34 | 6865.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 6836.00 | 6864.34 | 6865.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 13:15:00 | 6754.00 | 6814.68 | 6838.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 10:15:00 | 6825.50 | 6807.24 | 6826.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 11:00:00 | 6825.50 | 6807.24 | 6826.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 6768.00 | 6799.39 | 6821.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:30:00 | 6838.50 | 6799.39 | 6821.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 6834.00 | 6782.66 | 6801.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 6834.00 | 6782.66 | 6801.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 6847.00 | 6795.53 | 6805.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:30:00 | 6846.00 | 6795.53 | 6805.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 6799.00 | 6802.01 | 6807.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:45:00 | 6795.00 | 6802.01 | 6807.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 14:15:00 | 6861.50 | 6813.91 | 6812.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 10:15:00 | 6898.00 | 6843.04 | 6826.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 6791.00 | 6856.95 | 6844.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 6791.00 | 6856.95 | 6844.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 6791.00 | 6856.95 | 6844.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:15:00 | 6755.00 | 6856.95 | 6844.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 6770.00 | 6839.56 | 6837.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 6761.00 | 6839.56 | 6837.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 6768.00 | 6825.25 | 6831.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 15:15:00 | 6725.00 | 6784.42 | 6808.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 09:15:00 | 5805.00 | 5803.52 | 5864.61 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 10:15:00 | 5765.00 | 5786.23 | 5825.16 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 10:45:00 | 5763.00 | 5779.39 | 5818.51 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 5804.50 | 5772.59 | 5801.50 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 5804.50 | 5772.59 | 5801.50 | SL hit (close>ema400) qty=1.00 sl=5801.50 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 5804.50 | 5772.59 | 5801.50 | SL hit (close>ema400) qty=1.00 sl=5801.50 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 5804.50 | 5772.59 | 5801.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 5815.50 | 5781.17 | 5802.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 5816.00 | 5781.17 | 5802.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 5820.00 | 5794.91 | 5805.64 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 5828.50 | 5811.68 | 5810.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 5877.00 | 5835.94 | 5823.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 10:15:00 | 5899.00 | 5912.77 | 5882.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 10:45:00 | 5898.00 | 5912.77 | 5882.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 5894.50 | 5902.46 | 5884.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 10:45:00 | 5915.00 | 5892.06 | 5884.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 11:15:00 | 5870.00 | 5887.65 | 5883.14 | SL hit (close<static) qty=1.00 sl=5882.50 alert=retest2 |

### Cycle 15 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 5855.50 | 5877.92 | 5879.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 15:15:00 | 5850.00 | 5870.67 | 5875.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 5889.50 | 5874.43 | 5876.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 5889.50 | 5874.43 | 5876.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 5889.50 | 5874.43 | 5876.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 5889.00 | 5874.43 | 5876.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 5878.00 | 5875.15 | 5877.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:00:00 | 5842.00 | 5866.73 | 5872.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:30:00 | 5831.00 | 5859.99 | 5869.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 14:30:00 | 5841.00 | 5854.39 | 5865.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 11:15:00 | 5893.50 | 5862.50 | 5865.46 | SL hit (close>static) qty=1.00 sl=5893.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-19 11:15:00 | 5893.50 | 5862.50 | 5865.46 | SL hit (close>static) qty=1.00 sl=5893.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-19 11:15:00 | 5893.50 | 5862.50 | 5865.46 | SL hit (close>static) qty=1.00 sl=5893.00 alert=retest2 |

### Cycle 16 — BUY (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 12:15:00 | 5914.00 | 5872.80 | 5869.87 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 12:15:00 | 5847.50 | 5868.70 | 5869.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 5840.00 | 5859.49 | 5865.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 5854.50 | 5854.33 | 5861.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 11:15:00 | 5868.50 | 5854.33 | 5861.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 5873.00 | 5858.07 | 5862.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 15:15:00 | 5842.00 | 5858.56 | 5861.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 5810.00 | 5825.55 | 5830.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 5549.90 | 5615.75 | 5686.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 11:15:00 | 5519.50 | 5587.68 | 5660.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 13:15:00 | 5628.50 | 5592.94 | 5650.31 | SL hit (close>ema200) qty=0.50 sl=5592.94 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 13:15:00 | 5628.50 | 5592.94 | 5650.31 | SL hit (close>ema200) qty=0.50 sl=5592.94 alert=retest2 |

### Cycle 18 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 5802.50 | 5684.03 | 5669.41 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 5672.50 | 5726.83 | 5733.56 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 5724.50 | 5720.40 | 5720.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 5735.50 | 5725.29 | 5722.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 14:15:00 | 5732.00 | 5749.09 | 5741.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 14:15:00 | 5732.00 | 5749.09 | 5741.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 5732.00 | 5749.09 | 5741.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:45:00 | 5735.00 | 5749.09 | 5741.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 5730.00 | 5745.27 | 5740.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 5775.00 | 5745.27 | 5740.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 5761.00 | 5766.85 | 5758.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 5710.50 | 5756.18 | 5755.78 | SL hit (close<static) qty=1.00 sl=5729.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 5710.50 | 5756.18 | 5755.78 | SL hit (close<static) qty=1.00 sl=5729.00 alert=retest2 |

### Cycle 21 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 5728.50 | 5750.65 | 5753.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 14:15:00 | 5698.50 | 5716.91 | 5730.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 5757.00 | 5722.06 | 5730.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 5757.00 | 5722.06 | 5730.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 5757.00 | 5722.06 | 5730.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 5750.00 | 5722.06 | 5730.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 5746.50 | 5726.95 | 5731.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:30:00 | 5753.00 | 5726.95 | 5731.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 5735.00 | 5730.89 | 5732.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:45:00 | 5724.00 | 5730.11 | 5732.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 12:15:00 | 5737.00 | 5733.72 | 5733.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 5737.00 | 5733.72 | 5733.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 14:15:00 | 5749.50 | 5737.08 | 5735.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 5817.00 | 5824.68 | 5793.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 09:45:00 | 5804.00 | 5824.68 | 5793.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 5778.50 | 5815.45 | 5792.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:45:00 | 5780.00 | 5815.45 | 5792.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 5781.00 | 5808.56 | 5791.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:00:00 | 5781.00 | 5808.56 | 5791.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 5756.00 | 5793.88 | 5787.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:00:00 | 5756.00 | 5793.88 | 5787.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 5761.00 | 5787.30 | 5784.84 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 15:15:00 | 5753.00 | 5780.44 | 5781.95 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 5822.00 | 5788.75 | 5785.59 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 5767.00 | 5786.38 | 5787.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 13:15:00 | 5727.00 | 5766.08 | 5776.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 5790.50 | 5767.42 | 5774.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 5790.50 | 5767.42 | 5774.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 5790.50 | 5767.42 | 5774.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 5790.50 | 5767.42 | 5774.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 5778.50 | 5769.64 | 5774.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:15:00 | 5752.50 | 5769.64 | 5774.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 12:00:00 | 5770.00 | 5769.71 | 5774.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:45:00 | 5770.00 | 5771.09 | 5774.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:30:00 | 5772.50 | 5772.07 | 5774.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 5770.50 | 5771.76 | 5774.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:15:00 | 5769.00 | 5771.76 | 5774.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 12:15:00 | 5800.00 | 5767.65 | 5770.42 | SL hit (close>static) qty=1.00 sl=5790.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-25 12:15:00 | 5800.00 | 5767.65 | 5770.42 | SL hit (close>static) qty=1.00 sl=5790.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-25 12:15:00 | 5800.00 | 5767.65 | 5770.42 | SL hit (close>static) qty=1.00 sl=5790.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-25 12:15:00 | 5800.00 | 5767.65 | 5770.42 | SL hit (close>static) qty=1.00 sl=5790.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-25 12:15:00 | 5800.00 | 5767.65 | 5770.42 | SL hit (close>static) qty=1.00 sl=5790.50 alert=retest2 |

### Cycle 26 — BUY (started 2025-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 13:15:00 | 5800.00 | 5774.12 | 5773.11 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 5742.00 | 5768.05 | 5771.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 5722.50 | 5758.94 | 5766.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 5723.50 | 5722.83 | 5743.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 5723.50 | 5722.83 | 5743.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 5723.50 | 5722.83 | 5743.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 5731.50 | 5722.83 | 5743.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 5714.50 | 5721.16 | 5740.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 5690.50 | 5714.53 | 5735.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 13:30:00 | 5651.50 | 5679.34 | 5716.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 5952.50 | 5708.82 | 5717.94 | SL hit (close>static) qty=1.00 sl=5750.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 5952.50 | 5708.82 | 5717.94 | SL hit (close>static) qty=1.00 sl=5750.00 alert=retest2 |

### Cycle 28 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 5914.00 | 5749.85 | 5735.76 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 10:15:00 | 5699.50 | 5742.22 | 5746.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 10:15:00 | 5646.50 | 5696.73 | 5718.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 14:15:00 | 5677.50 | 5674.29 | 5699.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-03 15:00:00 | 5677.50 | 5674.29 | 5699.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 5692.00 | 5677.83 | 5698.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 5624.50 | 5677.83 | 5698.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 11:00:00 | 5640.50 | 5663.03 | 5687.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 5634.00 | 5656.35 | 5672.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 15:15:00 | 5554.50 | 5531.18 | 5530.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 15:15:00 | 5554.50 | 5531.18 | 5530.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 15:15:00 | 5554.50 | 5531.18 | 5530.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 5554.50 | 5531.18 | 5530.78 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 5472.50 | 5519.44 | 5525.48 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 5540.00 | 5511.54 | 5508.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 14:15:00 | 5564.50 | 5530.19 | 5518.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 5522.00 | 5530.92 | 5521.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 5522.00 | 5530.92 | 5521.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 5522.00 | 5530.92 | 5521.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:00:00 | 5522.00 | 5530.92 | 5521.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 5568.00 | 5538.34 | 5525.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 5527.00 | 5538.34 | 5525.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 5536.00 | 5553.23 | 5542.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:00:00 | 5536.00 | 5553.23 | 5542.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 5538.50 | 5550.28 | 5541.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:45:00 | 5539.00 | 5550.28 | 5541.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 5548.00 | 5549.83 | 5542.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:00:00 | 5548.00 | 5549.83 | 5542.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 5531.50 | 5546.16 | 5541.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:30:00 | 5529.00 | 5546.16 | 5541.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 5546.00 | 5546.13 | 5541.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 5579.00 | 5552.76 | 5545.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:30:00 | 5565.50 | 5577.38 | 5570.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 5577.00 | 5571.91 | 5568.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 5534.00 | 5564.33 | 5565.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 5534.00 | 5564.33 | 5565.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 5534.00 | 5564.33 | 5565.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 09:15:00 | 5534.00 | 5564.33 | 5565.60 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 09:15:00 | 5572.00 | 5563.93 | 5563.58 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 5555.00 | 5562.14 | 5562.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 5532.50 | 5553.63 | 5558.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 15:15:00 | 5575.00 | 5553.66 | 5557.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 15:15:00 | 5575.00 | 5553.66 | 5557.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 5575.00 | 5553.66 | 5557.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 6007.50 | 5553.66 | 5557.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 6136.00 | 5670.13 | 5609.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 6319.00 | 5799.90 | 5674.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 15:15:00 | 6597.00 | 6620.97 | 6341.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 09:15:00 | 6698.50 | 6620.97 | 6341.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 6290.00 | 6524.71 | 6443.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 6290.00 | 6524.71 | 6443.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 6359.00 | 6491.57 | 6436.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 13:00:00 | 6375.00 | 6447.93 | 6424.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 6420.00 | 6416.99 | 6415.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 6391.50 | 6411.90 | 6412.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 6391.50 | 6411.90 | 6412.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 6391.50 | 6411.90 | 6412.99 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 6435.50 | 6416.62 | 6415.03 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 6393.00 | 6411.89 | 6413.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 12:15:00 | 6374.50 | 6404.41 | 6409.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 10:15:00 | 6403.50 | 6378.03 | 6392.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 10:15:00 | 6403.50 | 6378.03 | 6392.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 6403.50 | 6378.03 | 6392.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:00:00 | 6403.50 | 6378.03 | 6392.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 6415.50 | 6385.53 | 6394.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:00:00 | 6415.50 | 6385.53 | 6394.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 6378.00 | 6384.02 | 6393.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 14:15:00 | 6361.00 | 6384.82 | 6392.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 15:15:00 | 6042.95 | 6134.19 | 6183.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 13:15:00 | 6141.50 | 6121.40 | 6156.46 | SL hit (close>ema200) qty=0.50 sl=6121.40 alert=retest2 |

### Cycle 40 — BUY (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 15:15:00 | 5435.00 | 5410.21 | 5409.52 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 5399.00 | 5407.33 | 5408.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 5396.00 | 5404.13 | 5406.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 15:15:00 | 5296.50 | 5294.46 | 5325.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 09:15:00 | 5330.50 | 5294.46 | 5325.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 5357.00 | 5306.97 | 5328.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 5357.00 | 5306.97 | 5328.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 5360.00 | 5317.58 | 5331.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:45:00 | 5364.00 | 5317.58 | 5331.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 5374.00 | 5342.13 | 5340.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 5394.00 | 5352.51 | 5345.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 12:15:00 | 5457.50 | 5460.49 | 5436.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 12:45:00 | 5457.50 | 5460.49 | 5436.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 5459.00 | 5460.73 | 5442.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 5482.00 | 5460.73 | 5442.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 12:45:00 | 5485.50 | 5468.61 | 5452.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 5426.00 | 5446.07 | 5447.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 5426.00 | 5446.07 | 5447.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 5426.00 | 5446.07 | 5447.58 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 15:15:00 | 5470.00 | 5448.76 | 5447.58 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 5421.50 | 5443.30 | 5445.21 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 5499.00 | 5454.44 | 5450.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 12:15:00 | 5516.50 | 5473.34 | 5459.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 09:15:00 | 5513.00 | 5514.21 | 5486.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:45:00 | 5507.00 | 5514.21 | 5486.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 5509.50 | 5519.77 | 5500.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:30:00 | 5506.00 | 5519.77 | 5500.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 5591.00 | 5658.55 | 5609.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:30:00 | 5595.50 | 5658.55 | 5609.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 5592.00 | 5645.24 | 5607.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:45:00 | 5585.50 | 5645.24 | 5607.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 5601.00 | 5636.39 | 5607.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:30:00 | 5592.50 | 5636.39 | 5607.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 5593.00 | 5627.72 | 5605.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:00:00 | 5593.00 | 5627.72 | 5605.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 5600.50 | 5622.27 | 5605.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 15:00:00 | 5611.00 | 5620.02 | 5605.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 5554.00 | 5602.01 | 5599.80 | SL hit (close<static) qty=1.00 sl=5585.00 alert=retest2 |

### Cycle 47 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 5556.00 | 5592.81 | 5595.82 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 12:15:00 | 5659.00 | 5597.52 | 5596.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 14:15:00 | 5670.50 | 5621.23 | 5608.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 5596.00 | 5625.35 | 5613.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 5596.00 | 5625.35 | 5613.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 5596.00 | 5625.35 | 5613.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:15:00 | 5588.00 | 5625.35 | 5613.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 5575.50 | 5615.38 | 5609.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 5575.50 | 5615.38 | 5609.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 5565.00 | 5598.84 | 5602.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 14:15:00 | 5543.00 | 5580.66 | 5593.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 5372.00 | 5368.16 | 5397.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 5372.50 | 5368.16 | 5397.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 5360.00 | 5366.53 | 5394.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 5400.00 | 5366.53 | 5394.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 5436.50 | 5380.52 | 5398.10 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 5441.50 | 5409.94 | 5406.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 11:15:00 | 5442.00 | 5416.35 | 5409.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 11:15:00 | 5410.00 | 5425.27 | 5419.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 11:15:00 | 5410.00 | 5425.27 | 5419.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 5410.00 | 5425.27 | 5419.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:45:00 | 5414.50 | 5425.27 | 5419.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 5398.00 | 5419.81 | 5417.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:00:00 | 5398.00 | 5419.81 | 5417.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 5393.00 | 5414.45 | 5415.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 5381.00 | 5405.26 | 5410.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 5310.50 | 5293.26 | 5320.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 5310.50 | 5293.26 | 5320.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 5310.50 | 5293.26 | 5320.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 5334.00 | 5293.26 | 5320.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 5314.00 | 5297.41 | 5319.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:30:00 | 5296.00 | 5316.60 | 5321.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:30:00 | 5304.00 | 5313.28 | 5319.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 14:15:00 | 5339.50 | 5311.32 | 5310.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 14:15:00 | 5339.50 | 5311.32 | 5310.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 14:15:00 | 5339.50 | 5311.32 | 5310.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 15:15:00 | 5565.00 | 5362.05 | 5333.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 5440.00 | 5461.89 | 5412.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:00:00 | 5440.00 | 5461.89 | 5412.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 5413.50 | 5452.21 | 5412.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 5410.50 | 5452.21 | 5412.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 5425.50 | 5446.87 | 5413.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 5451.00 | 5432.12 | 5416.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:00:00 | 5455.00 | 5432.12 | 5416.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 10:00:00 | 5469.50 | 5480.56 | 5455.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:30:00 | 5453.00 | 5507.01 | 5485.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 5480.00 | 5501.61 | 5484.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 5480.00 | 5501.61 | 5484.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 5485.00 | 5498.29 | 5484.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:30:00 | 5499.00 | 5494.83 | 5485.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:30:00 | 5496.50 | 5495.56 | 5486.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 5569.50 | 5494.75 | 5487.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 5735.00 | 5811.76 | 5818.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 5735.00 | 5811.76 | 5818.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 5735.00 | 5811.76 | 5818.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 5735.00 | 5811.76 | 5818.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 5735.00 | 5811.76 | 5818.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 5735.00 | 5811.76 | 5818.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 5735.00 | 5811.76 | 5818.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 5735.00 | 5811.76 | 5818.63 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 5846.00 | 5801.25 | 5800.22 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 12:15:00 | 5784.50 | 5797.98 | 5799.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 15:15:00 | 5760.00 | 5785.78 | 5793.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 11:15:00 | 5772.00 | 5771.33 | 5783.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 12:00:00 | 5772.00 | 5771.33 | 5783.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 5773.50 | 5771.76 | 5782.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:00:00 | 5773.50 | 5771.76 | 5782.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 5798.50 | 5777.11 | 5784.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:00:00 | 5798.50 | 5777.11 | 5784.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 5825.00 | 5786.69 | 5787.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 5825.00 | 5786.69 | 5787.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 5825.00 | 5794.35 | 5791.28 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 5760.00 | 5787.06 | 5788.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 5737.50 | 5777.15 | 5784.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 15:15:00 | 5570.00 | 5569.03 | 5608.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 09:15:00 | 5545.00 | 5569.03 | 5608.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 5607.00 | 5576.62 | 5608.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 5607.00 | 5576.62 | 5608.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 5617.00 | 5584.70 | 5609.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:30:00 | 5620.50 | 5584.70 | 5609.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 5617.00 | 5591.16 | 5609.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:45:00 | 5615.50 | 5591.16 | 5609.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 5578.50 | 5588.63 | 5606.94 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 5639.50 | 5615.23 | 5613.89 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 12:15:00 | 5602.00 | 5612.15 | 5612.69 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 15:15:00 | 5630.50 | 5615.81 | 5614.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 09:15:00 | 5642.00 | 5621.05 | 5616.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 11:15:00 | 5611.00 | 5619.91 | 5617.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 11:15:00 | 5611.00 | 5619.91 | 5617.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 5611.00 | 5619.91 | 5617.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:30:00 | 5620.50 | 5619.91 | 5617.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 5620.50 | 5620.03 | 5617.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:30:00 | 5622.00 | 5620.03 | 5617.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 5637.00 | 5623.42 | 5619.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:45:00 | 5632.50 | 5623.42 | 5619.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 5581.00 | 5630.66 | 5624.97 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 5533.50 | 5611.22 | 5616.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 5512.00 | 5576.22 | 5598.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 10:15:00 | 5462.00 | 5456.59 | 5499.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 11:00:00 | 5462.00 | 5456.59 | 5499.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 5348.50 | 5333.87 | 5362.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:30:00 | 5365.00 | 5333.87 | 5362.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 5374.50 | 5342.00 | 5363.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:00:00 | 5374.50 | 5342.00 | 5363.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 5363.50 | 5346.30 | 5363.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 15:15:00 | 5374.50 | 5346.30 | 5363.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 5374.50 | 5351.94 | 5364.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 5394.50 | 5351.94 | 5364.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 5365.50 | 5354.65 | 5364.61 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 14:15:00 | 5406.00 | 5369.68 | 5367.44 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 13:15:00 | 5350.00 | 5369.36 | 5369.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 14:15:00 | 5301.00 | 5355.69 | 5363.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 5127.50 | 5126.41 | 5199.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 15:00:00 | 5127.50 | 5126.41 | 5199.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 5181.50 | 5137.36 | 5191.81 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 5248.50 | 5199.77 | 5194.39 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 5111.50 | 5184.22 | 5191.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 5100.00 | 5154.30 | 5175.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 5126.00 | 5114.49 | 5143.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 10:45:00 | 5116.00 | 5114.49 | 5143.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 5120.00 | 5115.59 | 5141.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 5131.50 | 5115.59 | 5141.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 4905.00 | 4959.86 | 5025.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 4863.00 | 4940.79 | 5010.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 12:15:00 | 5077.00 | 5014.82 | 5011.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 5077.00 | 5014.82 | 5011.26 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 4913.50 | 4995.42 | 5006.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 12:15:00 | 4879.50 | 4956.65 | 4985.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 4941.90 | 4804.76 | 4855.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 4941.90 | 4804.76 | 4855.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 4941.90 | 4804.76 | 4855.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 4919.90 | 4804.76 | 4855.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 4962.00 | 4836.21 | 4865.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 4962.00 | 4836.21 | 4865.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 4932.90 | 4881.92 | 4881.58 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 4825.00 | 4879.16 | 4881.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 10:15:00 | 4773.50 | 4858.03 | 4871.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 4848.20 | 4837.56 | 4856.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 14:00:00 | 4848.20 | 4837.56 | 4856.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 4909.90 | 4852.03 | 4861.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 4909.90 | 4852.03 | 4861.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 4894.00 | 4860.42 | 4864.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 4875.30 | 4860.42 | 4864.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 4913.90 | 4841.88 | 4850.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 15:00:00 | 4913.90 | 4841.88 | 4850.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2026-04-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 15:15:00 | 4937.50 | 4861.00 | 4858.03 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 10:15:00 | 4801.00 | 4846.44 | 4851.77 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2026-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 14:15:00 | 4928.70 | 4861.66 | 4856.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 5062.50 | 4912.61 | 4881.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 5009.00 | 5030.15 | 4983.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 5010.00 | 5030.15 | 4983.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 5058.00 | 5035.72 | 4990.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:45:00 | 5044.10 | 5035.72 | 4990.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 5137.60 | 5129.04 | 5073.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 5168.80 | 5129.04 | 5073.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:30:00 | 5145.00 | 5114.58 | 5092.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 12:15:00 | 5346.00 | 5420.16 | 5426.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 12:15:00 | 5346.00 | 5420.16 | 5426.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 12:15:00 | 5346.00 | 5420.16 | 5426.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 5340.80 | 5394.66 | 5413.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 5352.90 | 5342.49 | 5372.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 14:45:00 | 5350.00 | 5342.49 | 5372.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 5351.00 | 5344.19 | 5370.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 5400.20 | 5344.19 | 5370.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 5443.80 | 5364.11 | 5377.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 5477.50 | 5364.11 | 5377.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 5442.40 | 5379.77 | 5383.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 5460.30 | 5379.77 | 5383.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 5422.20 | 5388.26 | 5386.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 5469.90 | 5418.85 | 5403.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 5418.60 | 5430.89 | 5415.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 13:15:00 | 5418.60 | 5430.89 | 5415.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 5418.60 | 5430.89 | 5415.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:45:00 | 5407.20 | 5430.89 | 5415.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 5405.50 | 5425.81 | 5414.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:15:00 | 5401.00 | 5425.81 | 5414.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 5401.00 | 5420.85 | 5413.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 5433.80 | 5420.85 | 5413.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:00:00 | 5415.00 | 5419.68 | 5413.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-10 09:15:00 | 6529.00 | 2025-06-12 10:15:00 | 6477.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-06-10 11:00:00 | 6527.50 | 2025-06-12 10:15:00 | 6477.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-06-11 13:45:00 | 6526.00 | 2025-06-12 10:15:00 | 6477.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-06-11 14:15:00 | 6529.50 | 2025-06-12 10:15:00 | 6477.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-06-13 09:15:00 | 6325.50 | 2025-06-24 10:15:00 | 6312.00 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-06-13 11:15:00 | 6378.00 | 2025-06-24 10:15:00 | 6312.00 | STOP_HIT | 1.00 | 1.03% |
| BUY | retest2 | 2025-06-25 09:15:00 | 6339.00 | 2025-07-02 09:15:00 | 6972.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-08 10:45:00 | 6646.00 | 2025-07-10 09:15:00 | 6729.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-07-17 13:45:00 | 6920.00 | 2025-07-21 09:15:00 | 6836.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-07-17 14:15:00 | 6919.00 | 2025-07-21 09:15:00 | 6836.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest1 | 2025-08-07 10:15:00 | 5765.00 | 2025-08-07 14:15:00 | 5804.50 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest1 | 2025-08-07 10:45:00 | 5763.00 | 2025-08-07 14:15:00 | 5804.50 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-08-14 10:45:00 | 5915.00 | 2025-08-14 11:15:00 | 5870.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-08-18 13:00:00 | 5842.00 | 2025-08-19 11:15:00 | 5893.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-08-18 13:30:00 | 5831.00 | 2025-08-19 11:15:00 | 5893.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-08-18 14:30:00 | 5841.00 | 2025-08-19 11:15:00 | 5893.50 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-08-21 15:15:00 | 5842.00 | 2025-08-29 09:15:00 | 5549.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 09:15:00 | 5810.00 | 2025-08-29 11:15:00 | 5519.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 15:15:00 | 5842.00 | 2025-08-29 13:15:00 | 5628.50 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2025-08-26 09:15:00 | 5810.00 | 2025-08-29 13:15:00 | 5628.50 | STOP_HIT | 0.50 | 3.12% |
| BUY | retest2 | 2025-09-11 09:15:00 | 5775.00 | 2025-09-12 11:15:00 | 5710.50 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-09-12 09:15:00 | 5761.00 | 2025-09-12 11:15:00 | 5710.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-09-16 13:45:00 | 5724.00 | 2025-09-17 12:15:00 | 5737.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-09-24 11:15:00 | 5752.50 | 2025-09-25 12:15:00 | 5800.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-09-24 12:00:00 | 5770.00 | 2025-09-25 12:15:00 | 5800.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-09-24 13:45:00 | 5770.00 | 2025-09-25 12:15:00 | 5800.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-09-24 14:30:00 | 5772.50 | 2025-09-25 12:15:00 | 5800.00 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-09-25 09:15:00 | 5769.00 | 2025-09-25 12:15:00 | 5800.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-09-29 11:30:00 | 5690.50 | 2025-09-30 09:15:00 | 5952.50 | STOP_HIT | 1.00 | -4.60% |
| SELL | retest2 | 2025-09-29 13:30:00 | 5651.50 | 2025-09-30 09:15:00 | 5952.50 | STOP_HIT | 1.00 | -5.33% |
| SELL | retest2 | 2025-10-06 09:15:00 | 5624.50 | 2025-10-13 15:15:00 | 5554.50 | STOP_HIT | 1.00 | 1.24% |
| SELL | retest2 | 2025-10-06 11:00:00 | 5640.50 | 2025-10-13 15:15:00 | 5554.50 | STOP_HIT | 1.00 | 1.52% |
| SELL | retest2 | 2025-10-07 09:15:00 | 5634.00 | 2025-10-13 15:15:00 | 5554.50 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2025-10-21 13:45:00 | 5579.00 | 2025-10-27 09:15:00 | 5534.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-10-24 14:30:00 | 5565.50 | 2025-10-27 09:15:00 | 5534.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-10-27 09:15:00 | 5577.00 | 2025-10-27 09:15:00 | 5534.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-11-03 13:00:00 | 6375.00 | 2025-11-04 09:15:00 | 6391.50 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2025-11-04 09:15:00 | 6420.00 | 2025-11-04 09:15:00 | 6391.50 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-11-06 14:15:00 | 6361.00 | 2025-11-11 15:15:00 | 6042.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 14:15:00 | 6361.00 | 2025-11-12 13:15:00 | 6141.50 | STOP_HIT | 0.50 | 3.45% |
| BUY | retest2 | 2025-12-26 09:15:00 | 5482.00 | 2025-12-29 11:15:00 | 5426.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-12-26 12:45:00 | 5485.50 | 2025-12-29 11:15:00 | 5426.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-01-02 15:00:00 | 5611.00 | 2026-01-05 09:15:00 | 5554.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-01-23 09:30:00 | 5296.00 | 2026-01-27 14:15:00 | 5339.50 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-01-23 10:30:00 | 5304.00 | 2026-01-27 14:15:00 | 5339.50 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2026-01-30 09:30:00 | 5451.00 | 2026-02-13 09:15:00 | 5735.00 | STOP_HIT | 1.00 | 5.21% |
| BUY | retest2 | 2026-01-30 10:00:00 | 5455.00 | 2026-02-13 09:15:00 | 5735.00 | STOP_HIT | 1.00 | 5.13% |
| BUY | retest2 | 2026-02-01 10:00:00 | 5469.50 | 2026-02-13 09:15:00 | 5735.00 | STOP_HIT | 1.00 | 4.85% |
| BUY | retest2 | 2026-02-02 09:30:00 | 5453.00 | 2026-02-13 09:15:00 | 5735.00 | STOP_HIT | 1.00 | 5.17% |
| BUY | retest2 | 2026-02-02 13:30:00 | 5499.00 | 2026-02-13 09:15:00 | 5735.00 | STOP_HIT | 1.00 | 4.29% |
| BUY | retest2 | 2026-02-02 14:30:00 | 5496.50 | 2026-02-13 09:15:00 | 5735.00 | STOP_HIT | 1.00 | 4.34% |
| BUY | retest2 | 2026-02-03 09:15:00 | 5569.50 | 2026-02-13 09:15:00 | 5735.00 | STOP_HIT | 1.00 | 2.97% |
| SELL | retest2 | 2026-03-24 10:30:00 | 4863.00 | 2026-03-25 12:15:00 | 5077.00 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest2 | 2026-04-13 10:15:00 | 5168.80 | 2026-04-23 12:15:00 | 5346.00 | STOP_HIT | 1.00 | 3.43% |
| BUY | retest2 | 2026-04-15 09:30:00 | 5145.00 | 2026-04-23 12:15:00 | 5346.00 | STOP_HIT | 1.00 | 3.91% |
