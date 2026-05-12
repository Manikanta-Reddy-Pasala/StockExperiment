# DIVISLAB (DIVISLAB)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 6705.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 2 |
| ALERT3 | 50 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 34 |
| PARTIAL | 1 |
| TARGET_HIT | 2 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 37 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 33
- **Target hits / Stop hits / Partials:** 2 / 34 / 1
- **Avg / median % per leg:** -0.96% / -1.67%
- **Sum % (uncompounded):** -35.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 1 | 5.6% | 1 | 17 | 0 | -0.90% | -16.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 1 | 5.6% | 1 | 17 | 0 | -0.90% | -16.1% |
| SELL (all) | 19 | 3 | 15.8% | 1 | 17 | 1 | -1.03% | -19.5% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.32% | -4.6% |
| SELL @ 3rd Alert (retest2) | 17 | 3 | 17.6% | 1 | 15 | 1 | -0.87% | -14.8% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.32% | -4.6% |
| retest2 (combined) | 35 | 4 | 11.4% | 2 | 32 | 1 | -0.88% | -31.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 12:15:00 | 5666.35 | 5820.58 | 5820.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 09:15:00 | 5610.45 | 5814.96 | 5817.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 12:15:00 | 5805.15 | 5782.29 | 5800.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 12:15:00 | 5805.15 | 5782.29 | 5800.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 5805.15 | 5782.29 | 5800.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:00:00 | 5805.15 | 5782.29 | 5800.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 13:15:00 | 5848.00 | 5782.95 | 5800.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:45:00 | 5868.00 | 5782.95 | 5800.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 14:15:00 | 5866.25 | 5783.78 | 5801.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 15:15:00 | 5949.90 | 5783.78 | 5801.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 5949.90 | 5785.43 | 5801.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:15:00 | 6148.50 | 5785.43 | 5801.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 11:15:00 | 6150.00 | 5817.62 | 5817.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 09:15:00 | 6170.00 | 5833.33 | 5825.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 12:15:00 | 5863.35 | 5877.52 | 5850.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-11 13:00:00 | 5863.35 | 5877.52 | 5850.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 5836.50 | 5878.64 | 5851.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-12 09:30:00 | 5824.95 | 5878.64 | 5851.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 5925.55 | 5879.10 | 5851.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-12 11:30:00 | 5945.60 | 5879.88 | 5852.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-12 14:30:00 | 5946.65 | 5882.32 | 5853.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-14 13:15:00 | 5804.15 | 5892.60 | 5861.03 | SL hit (close<static) qty=1.00 sl=5819.20 alert=retest2 |

### Cycle 3 — SELL (started 2025-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 10:15:00 | 5435.00 | 5841.66 | 5842.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 11:15:00 | 5373.15 | 5837.00 | 5839.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 5737.25 | 5707.01 | 5763.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 10:00:00 | 5737.25 | 5707.01 | 5763.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 5768.00 | 5703.16 | 5756.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 15:00:00 | 5768.00 | 5703.16 | 5756.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 15:15:00 | 5783.00 | 5703.95 | 5756.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 09:30:00 | 5826.55 | 5705.26 | 5756.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 10:15:00 | 5838.20 | 5706.58 | 5757.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-19 11:45:00 | 5825.00 | 5707.73 | 5757.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 10:30:00 | 5824.05 | 5714.39 | 5759.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 12:30:00 | 5826.10 | 5716.47 | 5759.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 11:00:00 | 5821.30 | 5721.53 | 5761.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 5800.00 | 5724.50 | 5762.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 14:30:00 | 5799.50 | 5724.50 | 5762.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-24 13:15:00 | 5904.65 | 5730.94 | 5764.28 | SL hit (close>static) qty=1.00 sl=5850.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 11:15:00 | 6067.00 | 5751.83 | 5750.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 09:15:00 | 6116.00 | 5766.43 | 5758.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 10:15:00 | 5897.50 | 5905.42 | 5839.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-09 11:00:00 | 5897.50 | 5905.42 | 5839.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 6581.50 | 6694.16 | 6568.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:15:00 | 6509.00 | 6694.16 | 6568.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 6534.00 | 6692.56 | 6568.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:30:00 | 6489.50 | 6692.56 | 6568.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 6532.50 | 6690.97 | 6568.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:45:00 | 6532.00 | 6690.97 | 6568.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 6599.00 | 6686.99 | 6574.01 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 5980.00 | 6491.92 | 6494.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 14:15:00 | 5959.50 | 6462.34 | 6479.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 09:15:00 | 6260.00 | 6145.58 | 6248.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 6260.00 | 6145.58 | 6248.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 6260.00 | 6145.58 | 6248.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 6260.00 | 6145.58 | 6248.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 6240.00 | 6146.52 | 6248.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:30:00 | 6249.50 | 6146.52 | 6248.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 6213.00 | 6148.08 | 6247.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 13:45:00 | 6203.50 | 6148.66 | 6247.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 5893.32 | 6121.62 | 6220.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 11:15:00 | 6026.00 | 5993.88 | 6128.03 | SL hit (close>ema200) qty=0.50 sl=5993.88 alert=retest2 |

### Cycle 6 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 6593.00 | 6217.42 | 6216.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 6618.00 | 6228.92 | 6222.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 6475.00 | 6497.72 | 6400.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 10:00:00 | 6475.00 | 6497.72 | 6400.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 6405.50 | 6490.96 | 6407.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:30:00 | 6415.00 | 6490.96 | 6407.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 6407.50 | 6490.13 | 6407.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:30:00 | 6376.50 | 6490.13 | 6407.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 6412.00 | 6489.35 | 6407.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:45:00 | 6397.00 | 6489.35 | 6407.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 6400.00 | 6488.46 | 6407.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:00:00 | 6400.00 | 6488.46 | 6407.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 6375.00 | 6487.33 | 6407.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:45:00 | 6368.50 | 6487.33 | 6407.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 6377.00 | 6474.30 | 6404.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 13:00:00 | 6397.50 | 6473.53 | 6404.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 10:15:00 | 6381.00 | 6470.29 | 6413.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 6382.00 | 6465.05 | 6412.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 12:15:00 | 6342.00 | 6459.50 | 6415.50 | SL hit (close<static) qty=1.00 sl=6359.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 6072.50 | 6411.16 | 6412.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 6023.50 | 6407.30 | 6410.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 6241.50 | 6230.99 | 6307.35 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-04 09:15:00 | 6138.00 | 6230.75 | 6304.97 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-04 10:15:00 | 6149.50 | 6230.42 | 6304.44 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 6286.50 | 6189.93 | 6269.28 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-11 11:15:00 | 6286.50 | 6189.93 | 6269.28 | SL hit (close>ema400) qty=1.00 sl=6269.28 alert=retest1 |

### Cycle 8 — BUY (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 15:15:00 | 6330.00 | 6294.77 | 6294.64 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 6233.00 | 6294.15 | 6294.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 6169.00 | 6292.73 | 6293.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 11:15:00 | 6091.00 | 6061.97 | 6149.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-10 12:00:00 | 6091.00 | 6061.97 | 6149.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 6127.50 | 6064.95 | 6145.88 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 13:15:00 | 6444.50 | 6198.32 | 6198.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 6468.00 | 6205.80 | 6202.03 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-13 11:00:00 | 3812.90 | 2024-05-27 09:15:00 | 4194.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-12 11:30:00 | 5945.60 | 2025-02-14 13:15:00 | 5804.15 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2025-02-12 14:30:00 | 5946.65 | 2025-02-14 13:15:00 | 5804.15 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-02-20 09:30:00 | 5947.45 | 2025-02-21 09:15:00 | 5789.40 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2025-02-20 10:00:00 | 5945.10 | 2025-02-21 09:15:00 | 5789.40 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-03-19 11:45:00 | 5825.00 | 2025-03-24 13:15:00 | 5904.65 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-03-20 10:30:00 | 5824.05 | 2025-03-24 13:15:00 | 5904.65 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-03-20 12:30:00 | 5826.10 | 2025-03-24 13:15:00 | 5904.65 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-03-21 11:00:00 | 5821.30 | 2025-03-24 13:15:00 | 5904.65 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-04-01 11:15:00 | 5631.70 | 2025-04-03 09:15:00 | 5825.80 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2025-04-02 14:30:00 | 5653.10 | 2025-04-03 09:15:00 | 5825.80 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-04-04 09:30:00 | 5624.15 | 2025-04-07 09:15:00 | 5061.73 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-17 15:15:00 | 5622.00 | 2025-04-21 09:15:00 | 5822.50 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2025-09-19 13:45:00 | 6203.50 | 2025-09-25 14:15:00 | 5893.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 13:45:00 | 6203.50 | 2025-10-07 11:15:00 | 6026.00 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2025-10-08 12:00:00 | 6200.00 | 2025-10-10 11:15:00 | 6386.00 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-10-09 10:30:00 | 6191.00 | 2025-10-10 11:15:00 | 6386.00 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2025-11-25 13:00:00 | 6397.50 | 2025-12-08 12:15:00 | 6342.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-12-02 10:15:00 | 6381.00 | 2025-12-08 12:15:00 | 6342.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-12-03 10:30:00 | 6382.00 | 2025-12-08 12:15:00 | 6342.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-12-10 10:15:00 | 6381.00 | 2025-12-10 14:15:00 | 6293.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-12-18 10:30:00 | 6427.00 | 2025-12-30 09:15:00 | 6365.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-19 09:15:00 | 6589.00 | 2025-12-30 09:15:00 | 6365.00 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2025-12-26 14:45:00 | 6422.50 | 2025-12-30 09:15:00 | 6365.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-12-30 15:15:00 | 6480.50 | 2026-01-01 09:15:00 | 6355.00 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-12-31 14:45:00 | 6406.50 | 2026-01-01 09:15:00 | 6355.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-01-05 09:30:00 | 6407.50 | 2026-01-05 13:15:00 | 6360.50 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-01-05 12:45:00 | 6405.00 | 2026-01-05 13:15:00 | 6360.50 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2026-01-06 09:15:00 | 6456.00 | 2026-01-14 12:15:00 | 6373.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2026-01-12 14:45:00 | 6494.00 | 2026-01-14 12:15:00 | 6373.00 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest1 | 2026-02-04 09:15:00 | 6138.00 | 2026-02-11 11:15:00 | 6286.50 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest1 | 2026-02-04 10:15:00 | 6149.50 | 2026-02-11 11:15:00 | 6286.50 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2026-02-12 11:15:00 | 6231.00 | 2026-02-25 11:15:00 | 6376.00 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2026-02-12 11:45:00 | 6223.00 | 2026-02-25 11:15:00 | 6376.00 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-02-24 09:30:00 | 6234.00 | 2026-02-25 11:15:00 | 6376.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2026-02-24 13:15:00 | 6237.00 | 2026-02-25 11:15:00 | 6376.00 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2026-03-09 09:15:00 | 6200.50 | 2026-03-09 11:15:00 | 6304.00 | STOP_HIT | 1.00 | -1.67% |
