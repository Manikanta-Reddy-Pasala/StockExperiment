# UltraTech Cement Ltd. (ULTRACEMCO)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 11930.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 144 |
| ALERT1 | 110 |
| ALERT2 | 108 |
| ALERT2_SKIP | 54 |
| ALERT3 | 311 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 127 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 131 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 135 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 24 / 111
- **Target hits / Stop hits / Partials:** 1 / 131 / 3
- **Avg / median % per leg:** -0.38% / -0.76%
- **Sum % (uncompounded):** -51.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 11 | 18.0% | 0 | 61 | 0 | -0.35% | -21.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 61 | 11 | 18.0% | 0 | 61 | 0 | -0.35% | -21.3% |
| SELL (all) | 74 | 13 | 17.6% | 1 | 70 | 3 | -0.41% | -30.4% |
| SELL @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.77% | -3.9% |
| SELL @ 3rd Alert (retest2) | 69 | 13 | 18.8% | 1 | 65 | 3 | -0.39% | -26.6% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.77% | -3.9% |
| retest2 (combined) | 130 | 24 | 18.5% | 1 | 126 | 3 | -0.37% | -47.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 9559.00 | 9532.29 | 9529.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 10:15:00 | 9600.00 | 9545.83 | 9535.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 09:15:00 | 9611.00 | 9621.93 | 9586.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 10:00:00 | 9611.00 | 9621.93 | 9586.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 9614.80 | 9620.50 | 9588.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:45:00 | 9582.05 | 9620.50 | 9588.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 11:15:00 | 9618.95 | 9620.19 | 9591.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 11:45:00 | 9620.00 | 9620.19 | 9591.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 9610.05 | 9621.45 | 9599.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 15:00:00 | 9610.05 | 9621.45 | 9599.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 9575.70 | 9609.86 | 9597.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:45:00 | 9545.05 | 9609.86 | 9597.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 9562.50 | 9600.39 | 9594.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 10:45:00 | 9553.55 | 9600.39 | 9594.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 9550.90 | 9593.33 | 9592.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:00:00 | 9550.90 | 9593.33 | 9592.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 13:15:00 | 9535.60 | 9581.79 | 9587.44 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 14:15:00 | 9714.00 | 9608.23 | 9598.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 09:15:00 | 9773.40 | 9655.95 | 9623.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 9812.95 | 9836.18 | 9771.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 9812.95 | 9836.18 | 9771.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 9812.95 | 9836.18 | 9771.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 9874.70 | 9810.85 | 9784.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 09:15:00 | 10044.40 | 10188.64 | 10188.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 09:15:00 | 10044.40 | 10188.64 | 10188.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 10:15:00 | 9968.20 | 10144.55 | 10168.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 09:15:00 | 10014.00 | 9937.02 | 10000.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 09:15:00 | 10014.00 | 9937.02 | 10000.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 10014.00 | 9937.02 | 10000.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:00:00 | 10014.00 | 9937.02 | 10000.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 9940.80 | 9937.78 | 9994.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 15:00:00 | 9918.85 | 9951.31 | 9984.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 10431.25 | 10035.89 | 10016.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 10431.25 | 10035.89 | 10016.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 11:15:00 | 10447.95 | 10178.97 | 10088.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 10130.00 | 10322.49 | 10222.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 10130.00 | 10322.49 | 10222.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 10130.00 | 10322.49 | 10222.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 10130.00 | 10322.49 | 10222.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 9842.95 | 10226.58 | 10188.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 9842.95 | 10226.58 | 10188.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 10020.10 | 10185.28 | 10172.93 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 13:15:00 | 9999.10 | 10148.05 | 10157.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 9890.00 | 10050.34 | 10106.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 14:15:00 | 10054.00 | 10000.53 | 10054.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 14:15:00 | 10054.00 | 10000.53 | 10054.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 10054.00 | 10000.53 | 10054.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 10054.00 | 10000.53 | 10054.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 10016.00 | 10003.63 | 10050.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 09:15:00 | 9982.55 | 10003.63 | 10050.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-06 10:15:00 | 10145.75 | 10035.33 | 10057.05 | SL hit (close>static) qty=1.00 sl=10057.35 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 10:15:00 | 10220.90 | 10089.28 | 10071.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 11:15:00 | 10322.35 | 10135.89 | 10094.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 10:15:00 | 11014.05 | 11020.87 | 10899.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 11:00:00 | 11014.05 | 11020.87 | 10899.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 11096.05 | 11158.56 | 11105.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:30:00 | 11093.05 | 11158.56 | 11105.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 12:15:00 | 11077.10 | 11142.27 | 11103.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 13:00:00 | 11077.10 | 11142.27 | 11103.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 11105.85 | 11134.99 | 11103.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 13:45:00 | 11094.00 | 11134.99 | 11103.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 14:15:00 | 11118.40 | 11131.67 | 11104.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 14:30:00 | 11115.95 | 11131.67 | 11104.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 15:15:00 | 11120.10 | 11129.35 | 11106.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:15:00 | 11055.40 | 11129.35 | 11106.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 11028.55 | 11109.19 | 11099.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:00:00 | 11028.55 | 11109.19 | 11099.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 10991.45 | 11085.64 | 11089.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 14:15:00 | 10914.75 | 10989.56 | 11025.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 11:15:00 | 10747.65 | 10743.03 | 10832.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-24 12:00:00 | 10747.65 | 10743.03 | 10832.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 10780.55 | 10754.03 | 10822.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 14:00:00 | 10780.55 | 10754.03 | 10822.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 10955.50 | 10803.72 | 10829.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:45:00 | 10948.15 | 10803.72 | 10829.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 10972.80 | 10837.53 | 10842.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:15:00 | 10945.00 | 10837.53 | 10842.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 11:15:00 | 10926.40 | 10855.31 | 10849.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 11:15:00 | 10926.40 | 10855.31 | 10849.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 09:15:00 | 11210.05 | 10931.85 | 10887.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 11890.00 | 11893.06 | 11733.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 12:00:00 | 11890.00 | 11893.06 | 11733.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 11803.75 | 11860.55 | 11777.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:45:00 | 11775.00 | 11860.55 | 11777.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 11846.95 | 11857.83 | 11783.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:45:00 | 11793.05 | 11857.83 | 11783.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 11:15:00 | 11855.40 | 11857.34 | 11790.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 15:00:00 | 11874.60 | 11858.18 | 11806.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 10:15:00 | 11746.25 | 11846.45 | 11815.26 | SL hit (close<static) qty=1.00 sl=11790.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 14:15:00 | 11760.00 | 11794.57 | 11797.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 09:15:00 | 11700.00 | 11771.73 | 11786.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 13:15:00 | 11659.90 | 11575.59 | 11619.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 13:15:00 | 11659.90 | 11575.59 | 11619.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 11659.90 | 11575.59 | 11619.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:00:00 | 11659.90 | 11575.59 | 11619.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 11678.95 | 11596.26 | 11624.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 15:00:00 | 11678.95 | 11596.26 | 11624.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 11668.85 | 11610.78 | 11628.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:15:00 | 11708.25 | 11610.78 | 11628.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 11647.80 | 11581.16 | 11601.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:00:00 | 11647.80 | 11581.16 | 11601.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 11612.60 | 11587.45 | 11602.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 09:30:00 | 11590.25 | 11581.96 | 11598.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 09:45:00 | 11594.80 | 11559.42 | 11574.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 11:45:00 | 11547.40 | 11562.64 | 11573.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 14:45:00 | 11578.65 | 11573.97 | 11575.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 15:15:00 | 11660.10 | 11591.20 | 11583.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 15:15:00 | 11660.10 | 11591.20 | 11583.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 09:15:00 | 11728.00 | 11618.56 | 11596.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 10:15:00 | 11717.00 | 11762.32 | 11703.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 10:15:00 | 11717.00 | 11762.32 | 11703.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 11717.00 | 11762.32 | 11703.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 11:00:00 | 11717.00 | 11762.32 | 11703.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 11710.70 | 11751.99 | 11704.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 11:45:00 | 11710.00 | 11751.99 | 11704.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 11704.55 | 11742.50 | 11704.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:30:00 | 11685.05 | 11742.50 | 11704.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 11714.50 | 11736.90 | 11705.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:00:00 | 11714.50 | 11736.90 | 11705.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 11654.50 | 11720.42 | 11700.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 15:00:00 | 11654.50 | 11720.42 | 11700.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 11680.00 | 11712.34 | 11698.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 11550.00 | 11712.34 | 11698.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 11580.85 | 11686.04 | 11688.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 11465.00 | 11588.69 | 11626.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 11425.90 | 11414.00 | 11504.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 11425.90 | 11414.00 | 11504.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 11425.90 | 11414.00 | 11504.64 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 10:15:00 | 11663.20 | 11536.18 | 11525.06 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 09:15:00 | 11429.60 | 11510.08 | 11518.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 09:15:00 | 11365.60 | 11440.57 | 11471.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 13:15:00 | 11459.00 | 11424.89 | 11452.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 13:15:00 | 11459.00 | 11424.89 | 11452.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 11459.00 | 11424.89 | 11452.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:00:00 | 11459.00 | 11424.89 | 11452.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 11454.70 | 11430.85 | 11452.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 15:15:00 | 11457.00 | 11430.85 | 11452.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 11457.00 | 11436.08 | 11452.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:15:00 | 11457.05 | 11436.08 | 11452.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 11559.40 | 11460.75 | 11462.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 11559.40 | 11460.75 | 11462.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 11613.20 | 11491.24 | 11476.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 12:15:00 | 11709.60 | 11553.75 | 11508.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 11664.35 | 11779.21 | 11692.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 09:15:00 | 11664.35 | 11779.21 | 11692.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 11664.35 | 11779.21 | 11692.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:00:00 | 11664.35 | 11779.21 | 11692.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 11666.45 | 11756.66 | 11690.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:45:00 | 11664.65 | 11756.66 | 11690.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 11692.00 | 11743.73 | 11690.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 12:15:00 | 11712.40 | 11743.73 | 11690.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 15:15:00 | 11720.00 | 11809.61 | 11820.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 15:15:00 | 11720.00 | 11809.61 | 11820.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 11504.05 | 11748.50 | 11792.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 11524.40 | 11511.39 | 11622.33 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 10:30:00 | 11449.50 | 11502.64 | 11608.27 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 11:45:00 | 11438.90 | 11492.18 | 11593.91 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 12:30:00 | 11448.20 | 11471.26 | 11575.15 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 11546.35 | 11438.19 | 11520.71 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-07 09:15:00 | 11546.35 | 11438.19 | 11520.71 | SL hit (close>ema400) qty=1.00 sl=11520.71 alert=retest1 |

### Cycle 17 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 11322.60 | 11183.00 | 11169.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 11488.80 | 11331.87 | 11263.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 14:15:00 | 11390.50 | 11414.63 | 11338.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 15:00:00 | 11390.50 | 11414.63 | 11338.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 11299.95 | 11391.70 | 11334.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:15:00 | 11230.00 | 11391.70 | 11334.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 11196.70 | 11352.70 | 11322.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 10:00:00 | 11196.70 | 11352.70 | 11322.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 11:15:00 | 11215.50 | 11303.80 | 11303.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-21 12:15:00 | 11198.55 | 11282.75 | 11294.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 09:15:00 | 11246.20 | 11244.95 | 11269.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 09:15:00 | 11246.20 | 11244.95 | 11269.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 11246.20 | 11244.95 | 11269.45 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 14:15:00 | 11308.45 | 11284.59 | 11282.08 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 09:15:00 | 11215.30 | 11273.20 | 11277.49 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 13:15:00 | 11287.90 | 11279.70 | 11279.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-23 14:15:00 | 11355.95 | 11294.95 | 11286.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 09:15:00 | 11293.95 | 11307.15 | 11294.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 11293.95 | 11307.15 | 11294.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 11293.95 | 11307.15 | 11294.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:30:00 | 11314.95 | 11307.15 | 11294.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 11293.65 | 11304.45 | 11294.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:45:00 | 11293.20 | 11304.45 | 11294.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 11310.05 | 11305.57 | 11295.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 15:00:00 | 11345.90 | 11314.66 | 11302.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 09:15:00 | 11285.00 | 11319.36 | 11317.85 | SL hit (close<static) qty=1.00 sl=11288.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 10:15:00 | 11270.00 | 11309.49 | 11313.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 14:15:00 | 11229.10 | 11279.15 | 11296.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 12:15:00 | 11230.05 | 11220.04 | 11255.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 12:15:00 | 11230.05 | 11220.04 | 11255.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 12:15:00 | 11230.05 | 11220.04 | 11255.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:30:00 | 11238.00 | 11220.04 | 11255.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 11209.60 | 11205.15 | 11241.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 11209.60 | 11205.15 | 11241.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 11267.35 | 11218.37 | 11241.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:30:00 | 11292.60 | 11218.37 | 11241.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 11295.05 | 11233.70 | 11246.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:45:00 | 11280.80 | 11233.70 | 11246.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 12:15:00 | 11294.50 | 11257.87 | 11255.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 09:15:00 | 11364.95 | 11301.47 | 11278.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 13:15:00 | 11578.00 | 11587.21 | 11524.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-05 13:45:00 | 11578.70 | 11587.21 | 11524.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 11544.75 | 11578.72 | 11526.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 14:45:00 | 11535.65 | 11578.72 | 11526.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 11532.00 | 11569.38 | 11527.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 09:15:00 | 11470.00 | 11569.38 | 11527.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 11367.75 | 11529.05 | 11512.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 11367.75 | 11529.05 | 11512.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 11339.95 | 11491.23 | 11497.01 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 11524.70 | 11471.79 | 11464.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 12:15:00 | 11592.70 | 11506.13 | 11482.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 11515.10 | 11553.12 | 11527.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 13:15:00 | 11515.10 | 11553.12 | 11527.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 11515.10 | 11553.12 | 11527.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 11515.10 | 11553.12 | 11527.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 11484.80 | 11539.45 | 11523.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 11484.80 | 11539.45 | 11523.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 11474.10 | 11526.38 | 11519.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:15:00 | 11518.00 | 11526.38 | 11519.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 10:15:00 | 11458.05 | 11507.29 | 11511.51 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 12:15:00 | 11532.75 | 11514.13 | 11514.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 13:15:00 | 11630.20 | 11537.34 | 11524.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 15:15:00 | 11680.00 | 11685.07 | 11630.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 09:15:00 | 11746.70 | 11685.07 | 11630.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 11636.55 | 11682.99 | 11644.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:00:00 | 11636.55 | 11682.99 | 11644.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 11630.95 | 11672.58 | 11643.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 13:15:00 | 11616.30 | 11672.58 | 11643.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 13:15:00 | 11659.00 | 11669.87 | 11644.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 11:45:00 | 11699.10 | 11662.15 | 11648.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 13:30:00 | 11687.90 | 11665.46 | 11652.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 09:15:00 | 11713.05 | 11660.67 | 11652.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 12:15:00 | 11600.00 | 11669.00 | 11661.65 | SL hit (close<static) qty=1.00 sl=11604.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 13:15:00 | 11595.45 | 11654.29 | 11655.63 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 10:15:00 | 11690.25 | 11657.44 | 11655.54 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 11:15:00 | 11575.90 | 11641.13 | 11648.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 12:15:00 | 11556.25 | 11624.15 | 11639.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 11627.70 | 11611.67 | 11630.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 11627.70 | 11611.67 | 11630.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 11660.00 | 11621.34 | 11633.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 11615.00 | 11621.34 | 11633.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 11709.25 | 11638.92 | 11640.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:00:00 | 11709.25 | 11638.92 | 11640.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 11781.45 | 11667.43 | 11653.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 12:15:00 | 11798.45 | 11708.12 | 11674.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 11887.00 | 11913.10 | 11835.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 11:15:00 | 11779.25 | 11877.83 | 11832.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 11779.25 | 11877.83 | 11832.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 12:00:00 | 11779.25 | 11877.83 | 11832.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 11770.40 | 11856.35 | 11826.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 14:00:00 | 11791.90 | 11843.46 | 11823.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 09:15:00 | 11761.50 | 11805.07 | 11808.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 11761.50 | 11805.07 | 11808.95 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 11:15:00 | 11855.05 | 11805.00 | 11800.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 12:15:00 | 11893.00 | 11822.60 | 11808.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 14:15:00 | 11968.50 | 11976.40 | 11922.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-27 15:00:00 | 11968.50 | 11976.40 | 11922.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 11885.00 | 11953.90 | 11921.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:30:00 | 11872.15 | 11953.90 | 11921.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 11860.10 | 11935.14 | 11916.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:30:00 | 11872.10 | 11935.14 | 11916.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 13:15:00 | 11872.05 | 11903.95 | 11904.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 14:15:00 | 11788.65 | 11880.89 | 11894.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 12:15:00 | 11850.00 | 11827.41 | 11857.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 13:00:00 | 11850.00 | 11827.41 | 11857.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 11840.00 | 11829.93 | 11855.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:00:00 | 11840.00 | 11829.93 | 11855.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 11825.05 | 11828.95 | 11852.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 11825.05 | 11828.95 | 11852.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 11849.85 | 11833.13 | 11852.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:15:00 | 11817.65 | 11833.13 | 11852.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 11912.00 | 11848.91 | 11857.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 10:00:00 | 11912.00 | 11848.91 | 11857.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 11859.00 | 11850.93 | 11858.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 11:15:00 | 11827.30 | 11850.93 | 11858.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 13:15:00 | 11235.93 | 11398.99 | 11556.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 09:15:00 | 11399.95 | 11350.23 | 11490.23 | SL hit (close>ema200) qty=0.50 sl=11350.23 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 15:15:00 | 11425.00 | 11377.05 | 11373.40 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 09:15:00 | 11335.00 | 11368.64 | 11369.91 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 10:15:00 | 11410.45 | 11377.01 | 11373.59 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 11:15:00 | 11317.95 | 11365.19 | 11368.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 15:15:00 | 11300.00 | 11334.94 | 11351.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 10:15:00 | 11128.70 | 11102.40 | 11191.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 10:45:00 | 11104.15 | 11102.40 | 11191.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 10981.70 | 11062.98 | 11131.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:30:00 | 10863.20 | 11008.14 | 11093.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 10820.10 | 10955.59 | 11054.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:30:00 | 10906.60 | 10941.47 | 11038.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 09:30:00 | 10819.90 | 10900.58 | 11002.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 10892.35 | 10811.73 | 10863.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:00:00 | 10892.35 | 10811.73 | 10863.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 10990.00 | 10847.38 | 10874.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 11:00:00 | 10990.00 | 10847.38 | 10874.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-24 12:15:00 | 11024.45 | 10904.82 | 10897.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 12:15:00 | 11024.45 | 10904.82 | 10897.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-25 10:15:00 | 11100.00 | 11005.85 | 10956.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 13:15:00 | 10988.30 | 11016.16 | 10974.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-25 14:00:00 | 10988.30 | 11016.16 | 10974.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 14:15:00 | 10983.60 | 11009.65 | 10975.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 15:00:00 | 10983.60 | 11009.65 | 10975.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 11020.45 | 11007.22 | 10980.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:30:00 | 10907.20 | 11007.22 | 10980.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 10999.90 | 11064.75 | 11032.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:45:00 | 10954.00 | 11064.75 | 11032.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 10970.00 | 11045.80 | 11026.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:45:00 | 10959.45 | 11045.80 | 11026.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 11041.20 | 11040.10 | 11028.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 13:45:00 | 11043.85 | 11040.10 | 11028.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 11127.35 | 11057.55 | 11037.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-30 11:45:00 | 11146.60 | 11094.12 | 11062.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 09:45:00 | 11143.90 | 11159.43 | 11111.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 13:15:00 | 11135.00 | 11149.15 | 11118.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 13:45:00 | 11150.00 | 11140.07 | 11117.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 11068.10 | 11125.67 | 11113.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 15:00:00 | 11068.10 | 11125.67 | 11113.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 11142.05 | 11121.15 | 11112.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 11142.05 | 11121.15 | 11112.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 11111.00 | 11119.12 | 11112.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:45:00 | 11111.00 | 11119.12 | 11112.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 11035.90 | 11102.47 | 11105.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 11035.90 | 11102.47 | 11105.56 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 11175.00 | 11073.91 | 11071.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 15:15:00 | 11185.00 | 11096.13 | 11081.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 11014.50 | 11168.30 | 11141.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 11014.50 | 11168.30 | 11141.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 11014.50 | 11168.30 | 11141.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 11014.50 | 11168.30 | 11141.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 11050.00 | 11144.64 | 11132.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 11026.40 | 11144.64 | 11132.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 12:15:00 | 11117.30 | 11124.68 | 11125.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 10:15:00 | 11026.00 | 11085.80 | 11104.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 11076.85 | 11038.57 | 11063.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 11076.85 | 11038.57 | 11063.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 11076.85 | 11038.57 | 11063.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 10:30:00 | 11061.70 | 11038.57 | 11063.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 11010.00 | 11032.85 | 11058.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:45:00 | 11081.65 | 11032.85 | 11058.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 10957.30 | 10984.82 | 11022.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 13:15:00 | 10894.10 | 10969.62 | 11005.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 13:15:00 | 10804.50 | 10763.77 | 10762.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 13:15:00 | 10804.50 | 10763.77 | 10762.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 09:15:00 | 10845.35 | 10779.16 | 10769.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 11228.55 | 11401.17 | 11274.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 09:15:00 | 11228.55 | 11401.17 | 11274.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 11228.55 | 11401.17 | 11274.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 10:00:00 | 11228.55 | 11401.17 | 11274.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 11163.05 | 11353.55 | 11264.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 10:30:00 | 11167.20 | 11353.55 | 11264.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 14:15:00 | 11120.90 | 11222.42 | 11223.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 09:15:00 | 11043.25 | 11168.73 | 11197.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 13:15:00 | 11138.50 | 11104.50 | 11151.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 13:15:00 | 11138.50 | 11104.50 | 11151.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 11138.50 | 11104.50 | 11151.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:45:00 | 11165.65 | 11104.50 | 11151.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 11134.55 | 11110.51 | 11149.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 14:45:00 | 11171.40 | 11110.51 | 11149.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 11159.95 | 11120.40 | 11150.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:15:00 | 11127.00 | 11120.40 | 11150.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 11111.50 | 11118.62 | 11146.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 10:15:00 | 11084.25 | 11118.62 | 11146.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 11:45:00 | 11081.00 | 11106.52 | 11135.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 12:30:00 | 11034.05 | 11089.40 | 11125.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 14:15:00 | 11203.65 | 11128.29 | 11120.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 14:15:00 | 11203.65 | 11128.29 | 11120.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 11444.45 | 11201.08 | 11155.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 14:15:00 | 11768.40 | 11813.66 | 11700.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 15:00:00 | 11768.40 | 11813.66 | 11700.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 11739.85 | 11833.38 | 11812.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:30:00 | 11783.90 | 11833.38 | 11812.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 11777.00 | 11822.10 | 11809.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 11:15:00 | 11826.10 | 11822.10 | 11809.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 14:15:00 | 11805.00 | 11818.74 | 11811.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 09:15:00 | 11735.85 | 11798.70 | 11803.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 09:15:00 | 11735.85 | 11798.70 | 11803.85 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 09:15:00 | 11996.35 | 11807.71 | 11798.24 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 11807.45 | 11844.73 | 11848.39 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 11:15:00 | 12001.15 | 11876.01 | 11862.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 12:15:00 | 12058.00 | 11912.41 | 11880.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 10:15:00 | 11931.65 | 11988.52 | 11938.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 10:15:00 | 11931.65 | 11988.52 | 11938.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 11931.65 | 11988.52 | 11938.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:00:00 | 11931.65 | 11988.52 | 11938.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 11920.85 | 11974.99 | 11936.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:00:00 | 11920.85 | 11974.99 | 11936.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 11911.20 | 11962.23 | 11934.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:00:00 | 11911.20 | 11962.23 | 11934.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 11904.05 | 11950.59 | 11931.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:45:00 | 11909.70 | 11950.59 | 11931.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 11939.95 | 11946.60 | 11932.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:15:00 | 11877.10 | 11946.60 | 11932.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 11884.65 | 11934.21 | 11928.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:15:00 | 11874.90 | 11934.21 | 11928.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 11927.55 | 11932.88 | 11928.45 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 11895.15 | 11925.33 | 11925.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 12:15:00 | 11705.95 | 11881.46 | 11905.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 10:15:00 | 11519.10 | 11497.39 | 11582.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 11:00:00 | 11519.10 | 11497.39 | 11582.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 11454.15 | 11466.55 | 11526.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 14:15:00 | 11396.15 | 11444.10 | 11496.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 15:00:00 | 11389.15 | 11433.11 | 11486.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 10:00:00 | 11399.55 | 11416.11 | 11468.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 13:15:00 | 11399.45 | 11409.71 | 11452.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 11439.80 | 11418.63 | 11449.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 11439.80 | 11418.63 | 11449.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 11458.00 | 11426.50 | 11450.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 11521.35 | 11426.50 | 11450.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 11500.05 | 11441.21 | 11454.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:45:00 | 11507.50 | 11441.21 | 11454.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 11465.30 | 11446.03 | 11455.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:45:00 | 11477.80 | 11446.03 | 11455.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 11408.95 | 11439.44 | 11451.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 13:45:00 | 11390.05 | 11429.60 | 11441.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 15:00:00 | 11250.60 | 11393.80 | 11423.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 14:15:00 | 11399.05 | 11344.97 | 11377.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 09:15:00 | 11348.00 | 11372.63 | 11385.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 10:15:00 | 11497.15 | 11394.28 | 11392.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 11497.15 | 11394.28 | 11392.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 11515.00 | 11453.69 | 11427.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 11606.80 | 11730.42 | 11671.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 10:15:00 | 11606.80 | 11730.42 | 11671.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 11606.80 | 11730.42 | 11671.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 11606.80 | 11730.42 | 11671.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 11577.95 | 11699.93 | 11663.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 12:00:00 | 11577.95 | 11699.93 | 11663.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 11500.40 | 11636.86 | 11639.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 11:15:00 | 11461.95 | 11558.15 | 11585.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 09:15:00 | 10586.35 | 10571.87 | 10725.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 10:00:00 | 10586.35 | 10571.87 | 10725.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 10657.50 | 10580.52 | 10655.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:30:00 | 10701.55 | 10580.52 | 10655.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 10626.60 | 10589.73 | 10652.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 10:45:00 | 10541.85 | 10620.15 | 10646.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 09:15:00 | 10521.90 | 10590.51 | 10618.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 09:15:00 | 10768.65 | 10631.15 | 10619.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 09:15:00 | 10768.65 | 10631.15 | 10619.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 09:15:00 | 10889.70 | 10738.14 | 10709.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 10:15:00 | 11198.95 | 11260.17 | 11137.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-27 10:30:00 | 11193.60 | 11260.17 | 11137.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 11105.30 | 11222.25 | 11174.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-28 09:30:00 | 11100.00 | 11222.25 | 11174.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 10:15:00 | 11125.05 | 11202.81 | 11169.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-28 11:30:00 | 11189.45 | 11200.87 | 11171.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 13:15:00 | 11266.50 | 11479.79 | 11476.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 13:15:00 | 11274.65 | 11438.76 | 11457.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 13:15:00 | 11274.65 | 11438.76 | 11457.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 11048.65 | 11242.07 | 11345.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 14:15:00 | 11175.60 | 11169.71 | 11280.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-03 15:00:00 | 11175.60 | 11169.71 | 11280.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 11387.40 | 11220.99 | 11276.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:00:00 | 11387.40 | 11220.99 | 11276.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 11376.10 | 11252.01 | 11285.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:30:00 | 11392.10 | 11252.01 | 11285.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 13:15:00 | 11465.45 | 11326.66 | 11315.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 11:15:00 | 11519.75 | 11443.16 | 11385.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 11410.00 | 11495.88 | 11440.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 11410.00 | 11495.88 | 11440.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 11410.00 | 11495.88 | 11440.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:00:00 | 11410.00 | 11495.88 | 11440.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 11425.35 | 11481.77 | 11439.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:45:00 | 11430.00 | 11481.77 | 11439.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 11440.95 | 11473.61 | 11439.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:30:00 | 11460.10 | 11473.61 | 11439.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 11450.00 | 11468.89 | 11440.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:30:00 | 11453.15 | 11468.89 | 11440.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 11515.55 | 11595.05 | 11545.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 11515.55 | 11595.05 | 11545.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 11538.35 | 11583.71 | 11544.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 12:00:00 | 11568.80 | 11580.73 | 11546.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 12:30:00 | 11578.05 | 11574.24 | 11547.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 11425.25 | 11523.99 | 11530.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 11425.25 | 11523.99 | 11530.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 12:15:00 | 11271.65 | 11446.79 | 11491.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 11401.30 | 11334.74 | 11402.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 11:15:00 | 11401.30 | 11334.74 | 11402.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 11401.30 | 11334.74 | 11402.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 11401.30 | 11334.74 | 11402.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 11559.00 | 11379.60 | 11416.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 11559.00 | 11379.60 | 11416.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 11505.00 | 11404.68 | 11424.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 15:15:00 | 11466.00 | 11422.69 | 11431.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-13 09:15:00 | 11590.00 | 11463.08 | 11448.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 09:15:00 | 11590.00 | 11463.08 | 11448.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 10:15:00 | 11601.10 | 11490.69 | 11462.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-13 15:15:00 | 11506.30 | 11533.67 | 11499.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-14 09:15:00 | 11474.70 | 11533.67 | 11499.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 11384.70 | 11503.88 | 11489.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 10:00:00 | 11384.70 | 11503.88 | 11489.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 11287.50 | 11460.60 | 11470.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 11:15:00 | 11263.45 | 11421.17 | 11452.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 11:15:00 | 11330.00 | 11297.28 | 11356.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 11:15:00 | 11330.00 | 11297.28 | 11356.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 11330.00 | 11297.28 | 11356.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:00:00 | 11330.00 | 11297.28 | 11356.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 11324.20 | 11302.66 | 11353.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:45:00 | 11344.90 | 11302.66 | 11353.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 11398.90 | 11321.91 | 11357.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:45:00 | 11409.95 | 11321.91 | 11357.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 11497.00 | 11356.93 | 11370.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 11497.00 | 11356.93 | 11370.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 09:15:00 | 11403.05 | 11383.31 | 11381.14 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 10:15:00 | 11335.00 | 11373.65 | 11376.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 11:15:00 | 11310.00 | 11360.92 | 11370.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 14:15:00 | 11340.10 | 11338.59 | 11356.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 14:15:00 | 11340.10 | 11338.59 | 11356.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 11340.10 | 11338.59 | 11356.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 14:45:00 | 11331.00 | 11338.59 | 11356.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 11407.45 | 11343.47 | 11355.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 11407.45 | 11343.47 | 11355.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 11450.00 | 11364.78 | 11363.79 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 12:15:00 | 11334.00 | 11363.55 | 11363.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 13:15:00 | 11299.00 | 11350.64 | 11357.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 10:15:00 | 11325.60 | 11321.56 | 11339.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-20 11:00:00 | 11325.60 | 11321.56 | 11339.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 14:15:00 | 11279.50 | 11311.19 | 11328.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 15:15:00 | 11256.00 | 11311.19 | 11328.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 09:15:00 | 10693.20 | 10868.61 | 10998.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-28 13:15:00 | 10130.40 | 10342.96 | 10561.65 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 63 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 10470.00 | 10406.56 | 10403.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 13:15:00 | 10493.70 | 10439.50 | 10422.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 10:15:00 | 10544.60 | 10554.72 | 10512.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 11:00:00 | 10544.60 | 10554.72 | 10512.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 10488.40 | 10541.46 | 10510.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 10488.40 | 10541.46 | 10510.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 10544.60 | 10542.08 | 10513.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 13:15:00 | 10553.90 | 10542.08 | 10513.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 15:15:00 | 10478.50 | 10525.96 | 10513.24 | SL hit (close<static) qty=1.00 sl=10488.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 10:15:00 | 10465.90 | 10502.21 | 10503.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 09:15:00 | 10412.50 | 10456.43 | 10477.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 13:15:00 | 10487.50 | 10448.18 | 10465.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 13:15:00 | 10487.50 | 10448.18 | 10465.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 10487.50 | 10448.18 | 10465.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 14:00:00 | 10487.50 | 10448.18 | 10465.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 10520.00 | 10462.54 | 10470.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 10520.00 | 10462.54 | 10470.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 10409.15 | 10459.29 | 10467.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:15:00 | 10399.15 | 10459.29 | 10467.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 11:15:00 | 10487.10 | 10465.14 | 10464.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 10487.10 | 10465.14 | 10464.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 12:15:00 | 10530.25 | 10478.16 | 10470.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 10:15:00 | 10836.25 | 10841.28 | 10746.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:45:00 | 10851.25 | 10841.28 | 10746.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 10939.40 | 10960.14 | 10896.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:30:00 | 10904.75 | 10960.14 | 10896.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 11481.10 | 11513.38 | 11468.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 11543.95 | 11513.38 | 11468.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 11477.00 | 11506.10 | 11469.48 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 13:15:00 | 11397.00 | 11441.84 | 11446.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 09:15:00 | 11200.05 | 11377.38 | 11414.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 09:15:00 | 11404.00 | 11307.52 | 11347.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 11404.00 | 11307.52 | 11347.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 11404.00 | 11307.52 | 11347.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:00:00 | 11404.00 | 11307.52 | 11347.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 11460.85 | 11338.18 | 11357.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 11:00:00 | 11460.85 | 11338.18 | 11357.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 12:15:00 | 11484.00 | 11389.41 | 11378.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 11600.40 | 11450.24 | 11409.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 11:15:00 | 11443.10 | 11475.80 | 11438.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 11:15:00 | 11443.10 | 11475.80 | 11438.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 11:15:00 | 11443.10 | 11475.80 | 11438.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:00:00 | 11443.10 | 11475.80 | 11438.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 11453.60 | 11471.36 | 11439.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:30:00 | 11453.30 | 11471.36 | 11439.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 11040.50 | 11393.30 | 11414.90 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 13:15:00 | 11349.95 | 11313.38 | 11312.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 14:15:00 | 11394.30 | 11329.56 | 11320.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 11666.00 | 11715.78 | 11654.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 11666.00 | 11715.78 | 11654.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 11666.00 | 11715.78 | 11654.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:30:00 | 11676.00 | 11715.78 | 11654.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 11737.00 | 11720.03 | 11661.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 11758.00 | 11720.03 | 11661.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-29 10:15:00 | 11920.00 | 12081.90 | 12089.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 10:15:00 | 11920.00 | 12081.90 | 12089.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 11:15:00 | 11868.00 | 12039.12 | 12069.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 14:15:00 | 11650.00 | 11649.50 | 11750.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 15:00:00 | 11650.00 | 11649.50 | 11750.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 11631.00 | 11664.19 | 11704.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:00:00 | 11571.00 | 11645.55 | 11692.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:30:00 | 11558.00 | 11636.44 | 11683.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 10:00:00 | 11567.00 | 11633.32 | 11665.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 15:15:00 | 11539.00 | 11646.92 | 11653.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 11539.00 | 11625.33 | 11643.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 09:15:00 | 11446.00 | 11625.33 | 11643.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 13:15:00 | 11708.00 | 11580.11 | 11563.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 11708.00 | 11580.11 | 11563.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 11722.00 | 11608.49 | 11577.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 11630.00 | 11634.63 | 11596.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 11630.00 | 11634.63 | 11596.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 11630.00 | 11634.63 | 11596.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 09:45:00 | 11609.00 | 11634.63 | 11596.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 11665.00 | 11640.70 | 11602.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 10:30:00 | 11641.00 | 11640.70 | 11602.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 11682.00 | 11678.24 | 11648.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 11668.00 | 11678.24 | 11648.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 11650.00 | 11672.60 | 11648.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:00:00 | 11650.00 | 11672.60 | 11648.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 11674.00 | 11672.88 | 11651.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:30:00 | 11648.00 | 11672.88 | 11651.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 11698.00 | 11680.96 | 11658.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:30:00 | 11790.00 | 11687.37 | 11663.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 11831.00 | 11723.40 | 11684.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 11760.00 | 11861.69 | 11863.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 11760.00 | 11861.69 | 11863.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 11704.00 | 11830.16 | 11848.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 11824.00 | 11807.46 | 11833.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:00:00 | 11824.00 | 11807.46 | 11833.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 11884.00 | 11822.77 | 11838.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:45:00 | 11903.00 | 11822.77 | 11838.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 11781.00 | 11814.41 | 11833.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:15:00 | 11747.00 | 11814.41 | 11833.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:45:00 | 11750.00 | 11721.56 | 11731.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 13:15:00 | 11757.00 | 11739.76 | 11738.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 11757.00 | 11739.76 | 11738.45 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 11668.00 | 11730.35 | 11734.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 11458.00 | 11647.25 | 11689.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 11:15:00 | 11300.00 | 11299.74 | 11395.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 11:30:00 | 11321.00 | 11299.74 | 11395.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 11206.00 | 11146.09 | 11208.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 11206.00 | 11146.09 | 11208.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 11165.00 | 11149.87 | 11204.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 11155.00 | 11149.87 | 11204.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 11119.00 | 11143.70 | 11196.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:30:00 | 11067.00 | 11123.69 | 11178.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 11:00:00 | 11065.00 | 11052.50 | 11080.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 11:30:00 | 11085.00 | 11071.00 | 11086.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 14:15:00 | 11154.00 | 11103.31 | 11098.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 14:15:00 | 11154.00 | 11103.31 | 11098.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 11205.00 | 11142.44 | 11119.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 10:15:00 | 11211.00 | 11219.11 | 11177.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 11:00:00 | 11211.00 | 11219.11 | 11177.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 12:15:00 | 11174.00 | 11209.75 | 11180.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 12:45:00 | 11172.00 | 11209.75 | 11180.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 13:15:00 | 11234.00 | 11214.60 | 11184.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 15:15:00 | 11266.00 | 11223.48 | 11191.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 15:15:00 | 11299.00 | 11381.73 | 11382.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 11299.00 | 11381.73 | 11382.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 11143.00 | 11333.98 | 11360.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 11312.00 | 11250.14 | 11289.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 11312.00 | 11250.14 | 11289.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 11312.00 | 11250.14 | 11289.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:30:00 | 11331.00 | 11250.14 | 11289.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 11354.00 | 11270.91 | 11295.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:30:00 | 11355.00 | 11270.91 | 11295.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 11438.00 | 11330.90 | 11320.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 13:15:00 | 11493.00 | 11363.32 | 11335.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 11:15:00 | 11386.00 | 11406.14 | 11373.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 11:15:00 | 11386.00 | 11406.14 | 11373.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 11386.00 | 11406.14 | 11373.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:45:00 | 11387.00 | 11406.14 | 11373.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 11384.00 | 11401.71 | 11374.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:45:00 | 11378.00 | 11401.71 | 11374.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 11397.00 | 11400.77 | 11376.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 09:15:00 | 11432.00 | 11394.93 | 11377.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 12:00:00 | 11411.00 | 11402.13 | 11385.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 13:00:00 | 11417.00 | 11405.11 | 11388.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 14:30:00 | 11408.00 | 11402.43 | 11390.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 11461.00 | 11413.75 | 11397.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 11482.00 | 11413.75 | 11397.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 15:15:00 | 11359.00 | 11415.81 | 11408.06 | SL hit (close<static) qty=1.00 sl=11376.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 10:15:00 | 11359.00 | 11412.26 | 11416.74 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 11474.00 | 11420.08 | 11418.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 11731.00 | 11488.29 | 11450.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 11572.00 | 11626.24 | 11548.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 11572.00 | 11626.24 | 11548.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 12364.00 | 12408.52 | 12359.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 12359.00 | 12408.52 | 12359.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 12360.00 | 12398.82 | 12359.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 13:30:00 | 12417.00 | 12399.05 | 12363.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 11:15:00 | 12329.00 | 12398.66 | 12381.21 | SL hit (close<static) qty=1.00 sl=12344.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 13:15:00 | 12470.00 | 12512.11 | 12515.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 09:15:00 | 12448.00 | 12501.30 | 12509.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 12:15:00 | 12519.00 | 12496.95 | 12504.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 12:15:00 | 12519.00 | 12496.95 | 12504.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 12519.00 | 12496.95 | 12504.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:30:00 | 12524.00 | 12496.95 | 12504.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 12494.00 | 12496.36 | 12503.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 12434.00 | 12497.83 | 12503.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 10:30:00 | 12483.00 | 12482.16 | 12484.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 12:00:00 | 12468.00 | 12479.33 | 12482.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 12:15:00 | 12520.00 | 12487.47 | 12486.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 12:15:00 | 12520.00 | 12487.47 | 12486.28 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 12470.00 | 12486.02 | 12487.22 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 14:15:00 | 12509.00 | 12491.25 | 12489.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 09:15:00 | 12636.00 | 12521.28 | 12503.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 15:15:00 | 12561.00 | 12567.17 | 12540.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 09:15:00 | 12565.00 | 12567.17 | 12540.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 12456.00 | 12544.93 | 12532.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 12456.00 | 12544.93 | 12532.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 12462.00 | 12528.35 | 12526.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:15:00 | 12369.00 | 12528.35 | 12526.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 11:15:00 | 12386.00 | 12499.88 | 12513.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 10:15:00 | 12289.00 | 12366.21 | 12415.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 12:15:00 | 12323.00 | 12306.39 | 12346.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 12:30:00 | 12317.00 | 12306.39 | 12346.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 12317.00 | 12285.08 | 12321.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 12293.00 | 12285.08 | 12321.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 12262.00 | 12280.46 | 12316.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:30:00 | 12250.00 | 12268.66 | 12304.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 11:15:00 | 12355.00 | 12271.45 | 12268.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 12355.00 | 12271.45 | 12268.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 12:15:00 | 12361.00 | 12289.36 | 12276.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 12271.00 | 12286.11 | 12277.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 14:15:00 | 12271.00 | 12286.11 | 12277.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 12271.00 | 12286.11 | 12277.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 14:30:00 | 12247.00 | 12286.11 | 12277.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 12251.00 | 12279.09 | 12274.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 12250.00 | 12279.09 | 12274.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 12201.00 | 12263.47 | 12268.22 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 12328.00 | 12277.49 | 12272.50 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 12250.00 | 12266.95 | 12268.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 12215.00 | 12256.56 | 12263.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 12196.00 | 12174.85 | 12210.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 12196.00 | 12174.85 | 12210.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 12196.00 | 12174.85 | 12210.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 11:30:00 | 12138.00 | 12167.06 | 12200.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 12:00:00 | 12135.00 | 12167.06 | 12200.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 13:15:00 | 12289.00 | 12196.56 | 12208.57 | SL hit (close>static) qty=1.00 sl=12249.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 12327.00 | 12237.40 | 12225.50 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 12215.00 | 12260.20 | 12261.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 12196.00 | 12247.36 | 12255.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 12269.00 | 12218.45 | 12234.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 12269.00 | 12218.45 | 12234.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 12269.00 | 12218.45 | 12234.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 12269.00 | 12218.45 | 12234.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 12292.00 | 12233.16 | 12239.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 12285.00 | 12233.16 | 12239.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 12183.00 | 12224.46 | 12234.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:15:00 | 12159.00 | 12208.29 | 12224.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:00:00 | 12150.00 | 12196.63 | 12217.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 12246.00 | 12201.77 | 12216.32 | SL hit (close>static) qty=1.00 sl=12242.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 12322.00 | 12241.53 | 12232.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 12389.00 | 12284.38 | 12254.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 12446.00 | 12447.88 | 12393.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 12:00:00 | 12446.00 | 12447.88 | 12393.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 12393.00 | 12439.43 | 12403.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 12393.00 | 12439.43 | 12403.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 12411.00 | 12433.75 | 12404.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 12370.00 | 12433.75 | 12404.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 12317.00 | 12410.40 | 12396.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 12317.00 | 12410.40 | 12396.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 12282.00 | 12384.72 | 12386.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 11:15:00 | 12251.00 | 12357.97 | 12373.83 | Break + close below crossover candle low |

### Cycle 93 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 12874.00 | 12431.80 | 12396.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 10:15:00 | 12923.00 | 12818.87 | 12708.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 12829.00 | 12857.70 | 12781.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:30:00 | 12791.00 | 12857.70 | 12781.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 12770.00 | 12845.42 | 12813.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:45:00 | 12755.00 | 12845.42 | 12813.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 12714.00 | 12819.14 | 12804.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 12714.00 | 12819.14 | 12804.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 12687.00 | 12792.71 | 12793.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 12640.00 | 12762.17 | 12779.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 13:15:00 | 12663.00 | 12638.25 | 12688.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 14:00:00 | 12663.00 | 12638.25 | 12688.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 12663.00 | 12610.78 | 12652.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:00:00 | 12663.00 | 12610.78 | 12652.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 12698.00 | 12628.23 | 12656.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:00:00 | 12698.00 | 12628.23 | 12656.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 12604.00 | 12614.81 | 12639.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:45:00 | 12641.00 | 12614.81 | 12639.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 12644.00 | 12595.72 | 12618.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:45:00 | 12660.00 | 12595.72 | 12618.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 12678.00 | 12612.17 | 12624.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 12678.00 | 12612.17 | 12624.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 12:15:00 | 12654.00 | 12633.63 | 12632.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 12706.00 | 12650.82 | 12641.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 12728.00 | 12756.65 | 12721.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 12728.00 | 12756.65 | 12721.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 12728.00 | 12756.65 | 12721.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 12728.00 | 12756.65 | 12721.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 12733.00 | 12751.92 | 12722.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 11:00:00 | 12759.00 | 12745.07 | 12726.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 11:45:00 | 12750.00 | 12744.26 | 12727.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 12:15:00 | 12756.00 | 12744.26 | 12727.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 13:15:00 | 12684.00 | 12727.52 | 12722.62 | SL hit (close<static) qty=1.00 sl=12687.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 12650.00 | 12713.01 | 12719.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 12602.00 | 12680.09 | 12702.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 12620.00 | 12618.29 | 12661.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 14:00:00 | 12620.00 | 12618.29 | 12661.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 12695.00 | 12627.09 | 12654.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 12687.00 | 12627.09 | 12654.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 12634.00 | 12628.47 | 12652.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:00:00 | 12594.00 | 12642.03 | 12653.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 10:15:00 | 12515.00 | 12450.64 | 12442.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 10:15:00 | 12515.00 | 12450.64 | 12442.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 12558.00 | 12472.11 | 12453.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 12642.00 | 12672.74 | 12620.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 13:00:00 | 12642.00 | 12672.74 | 12620.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 12616.00 | 12661.39 | 12620.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:45:00 | 12593.00 | 12661.39 | 12620.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 12633.00 | 12655.71 | 12621.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:30:00 | 12572.00 | 12655.71 | 12621.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 12648.00 | 12654.17 | 12623.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 12542.00 | 12654.17 | 12623.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 12537.00 | 12630.74 | 12615.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 12537.00 | 12630.74 | 12615.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 12570.00 | 12618.59 | 12611.79 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 12567.00 | 12600.66 | 12604.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 12496.00 | 12573.54 | 12591.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 12590.00 | 12566.67 | 12584.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 12590.00 | 12566.67 | 12584.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 12590.00 | 12566.67 | 12584.22 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 12664.00 | 12602.19 | 12598.31 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 12423.00 | 12594.79 | 12601.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 12354.00 | 12447.34 | 12508.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 12117.00 | 12102.03 | 12178.61 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-29 11:30:00 | 12052.00 | 12079.90 | 12155.09 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 12170.00 | 12077.37 | 12122.02 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 12170.00 | 12077.37 | 12122.02 | SL hit (close>ema400) qty=1.00 sl=12122.02 alert=retest1 |

### Cycle 101 — BUY (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 13:15:00 | 12205.00 | 12141.13 | 12140.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 15:15:00 | 12240.00 | 12174.16 | 12156.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 09:15:00 | 12135.00 | 12166.33 | 12154.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 12135.00 | 12166.33 | 12154.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 12135.00 | 12166.33 | 12154.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:00:00 | 12135.00 | 12166.33 | 12154.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 12118.00 | 12156.67 | 12151.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 12118.00 | 12156.67 | 12151.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 12:15:00 | 12135.00 | 12145.99 | 12147.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 14:15:00 | 12090.00 | 12130.63 | 12139.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 10:15:00 | 12070.00 | 12040.74 | 12071.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 10:15:00 | 12070.00 | 12040.74 | 12071.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 12070.00 | 12040.74 | 12071.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 12070.00 | 12040.74 | 12071.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 12085.00 | 12049.59 | 12072.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:30:00 | 12086.00 | 12049.59 | 12072.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 12092.00 | 12058.07 | 12074.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:00:00 | 12092.00 | 12058.07 | 12074.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 12077.00 | 12061.86 | 12074.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:30:00 | 12094.00 | 12061.86 | 12074.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 12050.00 | 12059.49 | 12072.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 15:15:00 | 12073.00 | 12059.49 | 12072.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 12073.00 | 12062.19 | 12072.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:15:00 | 12162.00 | 12062.19 | 12072.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 09:15:00 | 12178.00 | 12085.35 | 12081.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 13:15:00 | 12185.00 | 12121.07 | 12100.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 12117.00 | 12136.81 | 12114.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 12117.00 | 12136.81 | 12114.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 12117.00 | 12136.81 | 12114.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:45:00 | 12113.00 | 12136.81 | 12114.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 12039.00 | 12117.25 | 12107.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 12039.00 | 12117.25 | 12107.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 11968.00 | 12087.40 | 12095.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 12:15:00 | 11958.00 | 12061.52 | 12082.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 12055.00 | 12030.21 | 12057.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 12055.00 | 12030.21 | 12057.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 12055.00 | 12030.21 | 12057.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 12055.00 | 12030.21 | 12057.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 12044.00 | 12032.97 | 12056.70 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 12177.00 | 12087.14 | 12075.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 12305.00 | 12142.85 | 12103.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 12217.00 | 12233.35 | 12180.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 12217.00 | 12233.35 | 12180.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 12217.00 | 12233.35 | 12180.50 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 12089.00 | 12163.32 | 12168.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 12036.00 | 12137.86 | 12156.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 12191.00 | 12114.81 | 12133.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 12191.00 | 12114.81 | 12133.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 12191.00 | 12114.81 | 12133.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 12191.00 | 12114.81 | 12133.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 12186.00 | 12129.05 | 12138.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 12186.00 | 12129.05 | 12138.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 12267.00 | 12168.63 | 12155.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 13:15:00 | 12288.00 | 12192.51 | 12167.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 12326.00 | 12333.19 | 12277.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 12326.00 | 12333.19 | 12277.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 12326.00 | 12333.19 | 12277.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 11:15:00 | 12358.00 | 12336.55 | 12284.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:00:00 | 12362.00 | 12338.41 | 12297.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 12373.00 | 12333.87 | 12316.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 09:15:00 | 12388.00 | 12330.49 | 12316.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 12295.00 | 12323.39 | 12314.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 12295.00 | 12323.39 | 12314.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 12316.00 | 12321.91 | 12314.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:30:00 | 12293.00 | 12321.91 | 12314.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 12305.00 | 12318.53 | 12313.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:45:00 | 12300.00 | 12318.53 | 12313.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-23 12:15:00 | 12268.00 | 12308.43 | 12309.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 12:15:00 | 12268.00 | 12308.43 | 12309.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 13:15:00 | 12190.00 | 12284.74 | 12298.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 12:15:00 | 11991.00 | 11986.76 | 12068.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 13:00:00 | 11991.00 | 11986.76 | 12068.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 11971.00 | 11989.09 | 12044.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:00:00 | 11950.00 | 11981.42 | 12031.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:15:00 | 11957.00 | 11977.93 | 12025.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 12:15:00 | 12051.00 | 11985.75 | 12000.80 | SL hit (close>static) qty=1.00 sl=12048.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 11:15:00 | 12038.00 | 12006.89 | 12005.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 09:15:00 | 12100.00 | 12041.69 | 12024.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 12030.00 | 12039.35 | 12025.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 10:15:00 | 12030.00 | 12039.35 | 12025.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 12030.00 | 12039.35 | 12025.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 12030.00 | 12039.35 | 12025.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 11991.00 | 12033.07 | 12026.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:45:00 | 11998.00 | 12033.07 | 12026.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 11947.00 | 12015.86 | 12018.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 15:15:00 | 11924.00 | 11997.48 | 12010.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 13:15:00 | 11957.00 | 11955.01 | 11981.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 14:00:00 | 11957.00 | 11955.01 | 11981.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 11952.00 | 11892.34 | 11918.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:45:00 | 11972.00 | 11892.34 | 11918.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 11946.00 | 11903.07 | 11921.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:45:00 | 11968.00 | 11903.07 | 11921.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 11911.00 | 11908.17 | 11920.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:30:00 | 11930.00 | 11908.17 | 11920.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 11942.00 | 11914.93 | 11922.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 11871.00 | 11914.93 | 11922.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 13:00:00 | 11895.00 | 11909.01 | 11916.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 14:00:00 | 11890.00 | 11905.21 | 11913.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 12:15:00 | 11890.00 | 11839.75 | 11834.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 11890.00 | 11839.75 | 11834.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 13:15:00 | 11899.00 | 11851.60 | 11840.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 11942.00 | 11943.44 | 11901.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 14:00:00 | 11942.00 | 11943.44 | 11901.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 11863.00 | 11925.05 | 11903.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 11863.00 | 11925.05 | 11903.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 11867.00 | 11913.44 | 11900.02 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 11819.00 | 11882.80 | 11887.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 09:15:00 | 11800.00 | 11859.58 | 11874.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 13:15:00 | 11763.00 | 11761.34 | 11797.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-18 13:45:00 | 11753.00 | 11761.34 | 11797.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 11698.00 | 11695.10 | 11731.34 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2025-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 13:15:00 | 11770.00 | 11741.41 | 11738.44 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 11717.00 | 11734.06 | 11735.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 11641.00 | 11715.45 | 11726.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 11:15:00 | 11635.00 | 11631.30 | 11666.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 12:00:00 | 11635.00 | 11631.30 | 11666.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 11628.00 | 11609.90 | 11641.19 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 11766.00 | 11675.54 | 11663.90 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 11614.00 | 11663.54 | 11666.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 11604.00 | 11639.48 | 11653.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 11601.00 | 11598.52 | 11620.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 11601.00 | 11598.52 | 11620.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 11601.00 | 11598.52 | 11620.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 10:15:00 | 11592.00 | 11598.52 | 11620.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:30:00 | 11584.00 | 11592.05 | 11613.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:00:00 | 11585.00 | 11592.05 | 11613.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 11661.00 | 11612.00 | 11617.95 | SL hit (close>static) qty=1.00 sl=11651.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 15:15:00 | 12013.00 | 11692.20 | 11653.86 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 11559.00 | 11643.75 | 11644.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 11507.00 | 11574.77 | 11588.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 11472.00 | 11469.21 | 11510.58 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 11:30:00 | 11424.00 | 11450.69 | 11494.80 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 11419.00 | 11391.25 | 11429.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 11435.00 | 11391.25 | 11429.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 11451.00 | 11403.20 | 11431.02 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 11451.00 | 11403.20 | 11431.02 | SL hit (close>ema400) qty=1.00 sl=11431.02 alert=retest1 |

### Cycle 119 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 11629.00 | 11465.20 | 11453.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 11673.00 | 11557.10 | 11503.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 11:15:00 | 11661.00 | 11670.91 | 11595.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 11:30:00 | 11683.00 | 11670.91 | 11595.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 11619.00 | 11681.01 | 11631.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 11619.00 | 11681.01 | 11631.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 11617.00 | 11668.21 | 11630.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 11617.00 | 11668.21 | 11630.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 11571.00 | 11648.77 | 11624.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:45:00 | 11572.00 | 11648.77 | 11624.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 11520.00 | 11608.84 | 11611.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 11478.00 | 11529.79 | 11559.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 11498.00 | 11488.63 | 11520.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 11498.00 | 11488.63 | 11520.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 11498.00 | 11488.63 | 11520.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 11496.00 | 11488.63 | 11520.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 11504.00 | 11493.44 | 11510.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 11475.00 | 11494.15 | 11509.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 14:15:00 | 11531.00 | 11503.93 | 11506.02 | SL hit (close>static) qty=1.00 sl=11529.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 15:15:00 | 11531.00 | 11509.35 | 11508.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 11675.00 | 11542.48 | 11523.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 15:15:00 | 11767.00 | 11792.41 | 11740.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 09:15:00 | 11766.00 | 11792.41 | 11740.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 11769.00 | 11787.73 | 11743.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 10:15:00 | 11804.00 | 11787.73 | 11743.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 11:00:00 | 11802.00 | 11790.58 | 11748.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 11:45:00 | 11800.00 | 11792.67 | 11753.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 12:30:00 | 11801.00 | 11791.33 | 11756.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 11754.00 | 11788.43 | 11766.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:30:00 | 11750.00 | 11788.43 | 11766.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 11702.00 | 11771.14 | 11760.38 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-30 10:15:00 | 11702.00 | 11771.14 | 11760.38 | SL hit (close<static) qty=1.00 sl=11731.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 11:15:00 | 11653.00 | 11747.51 | 11750.61 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 11766.00 | 11741.91 | 11740.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 11843.00 | 11780.22 | 11760.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 10:15:00 | 11834.00 | 11855.32 | 11820.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 11:00:00 | 11834.00 | 11855.32 | 11820.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 11888.00 | 11870.88 | 11837.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:45:00 | 11835.00 | 11870.88 | 11837.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 12039.00 | 11914.68 | 11866.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:30:00 | 12092.00 | 11973.32 | 11902.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:00:00 | 12098.00 | 11973.32 | 11902.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 13:00:00 | 12096.00 | 11997.85 | 11920.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 12121.00 | 12030.86 | 11956.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 12122.00 | 12151.16 | 12100.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:30:00 | 12096.00 | 12151.16 | 12100.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 12069.00 | 12143.80 | 12110.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 12069.00 | 12143.80 | 12110.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 12108.00 | 12136.64 | 12110.33 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-08 15:15:00 | 12047.00 | 12088.79 | 12094.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 12047.00 | 12088.79 | 12094.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 11996.00 | 12070.24 | 12085.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 11930.00 | 11910.58 | 11969.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 11959.00 | 11910.58 | 11969.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 12055.00 | 11939.46 | 11977.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:45:00 | 12056.00 | 11939.46 | 11977.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 12080.00 | 11967.57 | 11986.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 12102.00 | 11967.57 | 11986.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 09:15:00 | 12064.00 | 12011.24 | 12004.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 10:15:00 | 12148.00 | 12062.32 | 12039.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 12320.00 | 12324.61 | 12242.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 09:30:00 | 12313.00 | 12324.61 | 12242.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 12261.00 | 12315.11 | 12276.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 12366.00 | 12315.11 | 12276.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 12360.00 | 12324.09 | 12283.66 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2026-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 14:15:00 | 12045.00 | 12225.09 | 12249.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 15:15:00 | 12026.00 | 12185.27 | 12229.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 11:15:00 | 12210.00 | 12169.02 | 12208.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 11:15:00 | 12210.00 | 12169.02 | 12208.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 12210.00 | 12169.02 | 12208.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:00:00 | 12210.00 | 12169.02 | 12208.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 12239.00 | 12183.02 | 12211.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:45:00 | 12280.00 | 12183.02 | 12211.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 12152.00 | 12176.81 | 12206.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 14:15:00 | 12133.00 | 12176.81 | 12206.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 14:15:00 | 12243.00 | 12190.05 | 12209.58 | SL hit (close>static) qty=1.00 sl=12239.00 alert=retest2 |

### Cycle 127 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 12375.00 | 12225.43 | 12222.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 12450.00 | 12343.42 | 12292.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 13:15:00 | 12366.00 | 12392.70 | 12337.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 13:15:00 | 12366.00 | 12392.70 | 12337.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 12366.00 | 12392.70 | 12337.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:45:00 | 12306.00 | 12392.70 | 12337.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 12371.00 | 12388.36 | 12340.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 12371.00 | 12388.36 | 12340.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 12346.00 | 12379.89 | 12340.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 12624.00 | 12379.89 | 12340.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 12514.00 | 12637.43 | 12642.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 12514.00 | 12637.43 | 12642.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 12285.00 | 12551.99 | 12601.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 12429.00 | 12406.30 | 12497.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:00:00 | 12429.00 | 12406.30 | 12497.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 12449.00 | 12414.84 | 12493.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:30:00 | 12478.00 | 12414.84 | 12493.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 12531.00 | 12438.08 | 12496.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 12531.00 | 12438.08 | 12496.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 12543.00 | 12459.06 | 12500.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 12680.00 | 12459.06 | 12500.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 12745.00 | 12550.16 | 12537.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 12775.00 | 12635.62 | 12592.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 14:15:00 | 12767.00 | 12770.58 | 12717.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 14:45:00 | 12771.00 | 12770.58 | 12717.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 12706.00 | 12757.57 | 12721.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 12665.00 | 12757.57 | 12721.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 12718.00 | 12749.66 | 12720.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 12705.00 | 12749.66 | 12720.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 12695.00 | 12738.73 | 12718.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:00:00 | 12695.00 | 12738.73 | 12718.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 12643.00 | 12719.58 | 12711.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:45:00 | 12640.00 | 12719.58 | 12711.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 12733.00 | 12719.06 | 12712.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 12766.00 | 12719.06 | 12712.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:45:00 | 12821.00 | 12737.05 | 12721.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 10:15:00 | 12912.00 | 12950.32 | 12952.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 10:15:00 | 12912.00 | 12950.32 | 12952.39 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 12984.00 | 12947.07 | 12943.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 13025.00 | 12968.57 | 12954.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 13000.00 | 13016.74 | 12990.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 13000.00 | 13016.74 | 12990.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 13000.00 | 13016.74 | 12990.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 13000.00 | 13016.74 | 12990.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 12955.00 | 13004.39 | 12987.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 12955.00 | 13004.39 | 12987.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 12834.00 | 12970.31 | 12973.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 12707.00 | 12896.32 | 12937.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 12802.00 | 12784.55 | 12843.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 14:00:00 | 12802.00 | 12784.55 | 12843.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 12822.00 | 12783.82 | 12827.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 12826.00 | 12783.82 | 12827.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 12811.00 | 12789.25 | 12826.31 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 12964.00 | 12865.93 | 12854.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 13056.00 | 12975.63 | 12941.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 12980.00 | 12988.32 | 12957.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 10:15:00 | 12980.00 | 12988.32 | 12957.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 12980.00 | 12988.32 | 12957.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 12980.00 | 12988.32 | 12957.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 12963.00 | 12983.26 | 12957.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:00:00 | 12963.00 | 12983.26 | 12957.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 12923.00 | 12971.21 | 12954.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:45:00 | 12932.00 | 12971.21 | 12954.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 12873.00 | 12951.56 | 12947.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:00:00 | 12873.00 | 12951.56 | 12947.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 15:15:00 | 12933.00 | 12942.96 | 12943.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 12669.00 | 12888.17 | 12918.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 12541.00 | 12539.06 | 12663.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 12541.00 | 12539.06 | 12663.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 12303.00 | 12149.81 | 12257.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 12303.00 | 12149.81 | 12257.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 12260.00 | 12171.85 | 12257.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 12150.00 | 12171.85 | 12257.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 11542.50 | 11980.98 | 12112.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 11631.00 | 11550.52 | 11775.17 | SL hit (close>ema200) qty=0.50 sl=11550.52 alert=retest2 |

### Cycle 135 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 11119.00 | 11058.54 | 11054.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 11206.00 | 11088.03 | 11068.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 10960.00 | 11149.98 | 11124.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 10960.00 | 11149.98 | 11124.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 10960.00 | 11149.98 | 11124.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 10955.00 | 11149.98 | 11124.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 10906.00 | 11101.19 | 11104.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 10886.00 | 11029.32 | 11069.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 13:15:00 | 10929.00 | 10924.27 | 10976.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 14:15:00 | 10922.00 | 10923.81 | 10971.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 10922.00 | 10923.81 | 10971.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 10922.00 | 10923.81 | 10971.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 10549.00 | 10513.90 | 10681.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 10502.00 | 10516.32 | 10667.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 10823.00 | 10588.81 | 10674.79 | SL hit (close>static) qty=1.00 sl=10685.00 alert=retest2 |

### Cycle 137 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 11094.00 | 10764.84 | 10737.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 11158.00 | 10843.47 | 10775.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 11030.00 | 11069.64 | 10941.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:30:00 | 11006.00 | 11069.64 | 10941.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 10937.00 | 11046.44 | 10994.52 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 10789.00 | 10942.03 | 10954.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 10750.00 | 10903.62 | 10936.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 11:15:00 | 10867.00 | 10840.06 | 10886.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 12:00:00 | 10867.00 | 10840.06 | 10886.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 10848.00 | 10841.65 | 10882.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:15:00 | 10800.00 | 10841.65 | 10882.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 13:15:00 | 10928.00 | 10731.87 | 10727.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 10928.00 | 10731.87 | 10727.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 10963.00 | 10778.09 | 10748.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 10796.00 | 10816.82 | 10777.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 10:15:00 | 10796.00 | 10816.82 | 10777.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 10796.00 | 10816.82 | 10777.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:00:00 | 10796.00 | 10816.82 | 10777.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 11325.00 | 11503.10 | 11439.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:00:00 | 11427.00 | 11487.88 | 11438.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:30:00 | 11443.00 | 11482.70 | 11440.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 15:15:00 | 12035.00 | 12068.63 | 12069.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 15:15:00 | 12035.00 | 12068.63 | 12069.07 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 12096.00 | 12074.11 | 12071.51 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 15:15:00 | 12000.00 | 12079.01 | 12080.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 09:15:00 | 11909.00 | 12045.00 | 12065.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 11958.00 | 11898.85 | 11962.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 10:00:00 | 11958.00 | 11898.85 | 11962.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 11996.00 | 11918.28 | 11965.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 11:15:00 | 12016.00 | 11918.28 | 11965.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 12024.00 | 11939.42 | 11970.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:00:00 | 12024.00 | 11939.42 | 11970.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 11853.00 | 11927.03 | 11959.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:45:00 | 11838.00 | 11906.42 | 11947.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 11921.00 | 11775.13 | 11761.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 11921.00 | 11775.13 | 11761.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 13:15:00 | 11946.00 | 11809.31 | 11778.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 15:15:00 | 12121.00 | 12127.69 | 12036.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:15:00 | 12092.00 | 12127.69 | 12036.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 11964.00 | 12094.95 | 12029.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:00:00 | 11964.00 | 12094.95 | 12029.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 11955.00 | 12066.96 | 12022.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:15:00 | 11907.00 | 12066.96 | 12022.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 14:15:00 | 11956.00 | 11994.86 | 11997.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 15:15:00 | 11930.00 | 11981.89 | 11991.25 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-22 09:15:00 | 9874.70 | 2024-05-29 09:15:00 | 10044.40 | STOP_HIT | 1.00 | 1.72% |
| SELL | retest2 | 2024-05-31 15:00:00 | 9918.85 | 2024-06-03 09:15:00 | 10431.25 | STOP_HIT | 1.00 | -5.17% |
| SELL | retest2 | 2024-06-06 09:15:00 | 9982.55 | 2024-06-06 10:15:00 | 10145.75 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-06-25 11:15:00 | 10945.00 | 2024-06-25 11:15:00 | 10926.40 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2024-07-03 15:00:00 | 11874.60 | 2024-07-04 10:15:00 | 11746.25 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-07-11 09:30:00 | 11590.25 | 2024-07-12 15:15:00 | 11660.10 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-07-12 09:45:00 | 11594.80 | 2024-07-12 15:15:00 | 11660.10 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-07-12 11:45:00 | 11547.40 | 2024-07-12 15:15:00 | 11660.10 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-07-12 14:45:00 | 11578.65 | 2024-07-12 15:15:00 | 11660.10 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-07-30 12:15:00 | 11712.40 | 2024-08-02 15:15:00 | 11720.00 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest1 | 2024-08-06 10:30:00 | 11449.50 | 2024-08-07 09:15:00 | 11546.35 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest1 | 2024-08-06 11:45:00 | 11438.90 | 2024-08-07 09:15:00 | 11546.35 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest1 | 2024-08-06 12:30:00 | 11448.20 | 2024-08-07 09:15:00 | 11546.35 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-08-08 10:45:00 | 11349.20 | 2024-08-19 09:15:00 | 11322.60 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2024-08-08 12:45:00 | 11349.80 | 2024-08-19 09:15:00 | 11322.60 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2024-08-12 11:30:00 | 11349.30 | 2024-08-19 09:15:00 | 11322.60 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2024-08-26 15:00:00 | 11345.90 | 2024-08-28 09:15:00 | 11285.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-09-17 11:45:00 | 11699.10 | 2024-09-18 12:15:00 | 11600.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-09-17 13:30:00 | 11687.90 | 2024-09-18 12:15:00 | 11600.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-09-18 09:15:00 | 11713.05 | 2024-09-18 12:15:00 | 11600.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-09-24 14:00:00 | 11791.90 | 2024-09-25 09:15:00 | 11761.50 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2024-10-03 11:15:00 | 11827.30 | 2024-10-07 13:15:00 | 11235.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 11:15:00 | 11827.30 | 2024-10-08 09:15:00 | 11399.95 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2024-10-21 11:30:00 | 10863.20 | 2024-10-24 12:15:00 | 11024.45 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-10-21 14:00:00 | 10820.10 | 2024-10-24 12:15:00 | 11024.45 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-10-21 14:30:00 | 10906.60 | 2024-10-24 12:15:00 | 11024.45 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-10-22 09:30:00 | 10819.90 | 2024-10-24 12:15:00 | 11024.45 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-10-30 11:45:00 | 11146.60 | 2024-11-04 09:15:00 | 11035.90 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-10-31 09:45:00 | 11143.90 | 2024-11-04 09:15:00 | 11035.90 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-10-31 13:15:00 | 11135.00 | 2024-11-04 09:15:00 | 11035.90 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-10-31 13:45:00 | 11150.00 | 2024-11-04 09:15:00 | 11035.90 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-11-12 13:15:00 | 10894.10 | 2024-11-19 13:15:00 | 10804.50 | STOP_HIT | 1.00 | 0.82% |
| SELL | retest2 | 2024-11-28 10:15:00 | 11084.25 | 2024-11-29 14:15:00 | 11203.65 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-11-28 11:45:00 | 11081.00 | 2024-11-29 14:15:00 | 11203.65 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-11-28 12:30:00 | 11034.05 | 2024-11-29 14:15:00 | 11203.65 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-12-09 11:15:00 | 11826.10 | 2024-12-10 09:15:00 | 11735.85 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-12-09 14:15:00 | 11805.00 | 2024-12-10 09:15:00 | 11735.85 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-12-24 14:15:00 | 11396.15 | 2025-01-01 10:15:00 | 11497.15 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-12-24 15:00:00 | 11389.15 | 2025-01-01 10:15:00 | 11497.15 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-12-26 10:00:00 | 11399.55 | 2025-01-01 10:15:00 | 11497.15 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-12-26 13:15:00 | 11399.45 | 2025-01-01 10:15:00 | 11497.15 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-12-30 13:45:00 | 11390.05 | 2025-01-01 10:15:00 | 11497.15 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-12-30 15:00:00 | 11250.60 | 2025-01-01 10:15:00 | 11497.15 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-12-31 14:15:00 | 11399.05 | 2025-01-01 10:15:00 | 11497.15 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-01-01 09:15:00 | 11348.00 | 2025-01-01 10:15:00 | 11497.15 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-01-17 10:45:00 | 10541.85 | 2025-01-21 09:15:00 | 10768.65 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-01-20 09:15:00 | 10521.90 | 2025-01-21 09:15:00 | 10768.65 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-01-28 11:30:00 | 11189.45 | 2025-02-01 13:15:00 | 11274.65 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2025-02-01 13:15:00 | 11266.50 | 2025-02-01 13:15:00 | 11274.65 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2025-02-10 12:00:00 | 11568.80 | 2025-02-11 09:15:00 | 11425.25 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-02-10 12:30:00 | 11578.05 | 2025-02-11 09:15:00 | 11425.25 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-02-12 15:15:00 | 11466.00 | 2025-02-13 09:15:00 | 11590.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-02-20 15:15:00 | 11256.00 | 2025-02-27 09:15:00 | 10693.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-20 15:15:00 | 11256.00 | 2025-02-28 13:15:00 | 10130.40 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-10 13:15:00 | 10553.90 | 2025-03-10 15:15:00 | 10478.50 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-03-13 11:15:00 | 10399.15 | 2025-03-17 11:15:00 | 10487.10 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-04-17 11:15:00 | 11758.00 | 2025-04-29 10:15:00 | 11920.00 | STOP_HIT | 1.00 | 1.38% |
| SELL | retest2 | 2025-05-06 11:00:00 | 11571.00 | 2025-05-12 13:15:00 | 11708.00 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-05-06 11:30:00 | 11558.00 | 2025-05-12 13:15:00 | 11708.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-05-07 10:00:00 | 11567.00 | 2025-05-12 13:15:00 | 11708.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-05-08 15:15:00 | 11539.00 | 2025-05-12 13:15:00 | 11708.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-05-09 09:15:00 | 11446.00 | 2025-05-12 13:15:00 | 11708.00 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-05-15 10:30:00 | 11790.00 | 2025-05-20 13:15:00 | 11760.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-05-15 13:00:00 | 11831.00 | 2025-05-20 13:15:00 | 11760.00 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-05-21 12:15:00 | 11747.00 | 2025-05-23 13:15:00 | 11757.00 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-05-23 11:45:00 | 11750.00 | 2025-05-23 13:15:00 | 11757.00 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-06-03 11:30:00 | 11067.00 | 2025-06-05 14:15:00 | 11154.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-06-05 11:00:00 | 11065.00 | 2025-06-05 14:15:00 | 11154.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-06-05 11:30:00 | 11085.00 | 2025-06-05 14:15:00 | 11154.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-06-09 15:15:00 | 11266.00 | 2025-06-12 15:15:00 | 11299.00 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-06-18 09:15:00 | 11432.00 | 2025-06-19 15:15:00 | 11359.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-06-18 12:00:00 | 11411.00 | 2025-06-19 15:15:00 | 11359.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-06-18 13:00:00 | 11417.00 | 2025-06-19 15:15:00 | 11359.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-06-18 14:30:00 | 11408.00 | 2025-06-19 15:15:00 | 11359.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-06-19 10:15:00 | 11482.00 | 2025-06-23 09:15:00 | 11350.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-06-20 09:15:00 | 11491.00 | 2025-06-23 09:15:00 | 11350.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-06-20 10:15:00 | 11476.00 | 2025-06-23 09:15:00 | 11350.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-06-20 10:45:00 | 11478.00 | 2025-06-23 09:15:00 | 11350.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-06-20 15:15:00 | 11475.00 | 2025-06-23 09:15:00 | 11350.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-07-04 13:30:00 | 12417.00 | 2025-07-07 11:15:00 | 12329.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-07-07 12:30:00 | 12389.00 | 2025-07-07 14:15:00 | 12343.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-07-07 13:45:00 | 12370.00 | 2025-07-07 14:15:00 | 12343.00 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-07-08 09:15:00 | 12376.00 | 2025-07-14 13:15:00 | 12470.00 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2025-07-09 13:45:00 | 12524.00 | 2025-07-14 13:15:00 | 12470.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-07-11 09:30:00 | 12625.00 | 2025-07-14 13:15:00 | 12470.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-07-11 14:30:00 | 12496.00 | 2025-07-14 13:15:00 | 12470.00 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-07-14 09:30:00 | 12570.00 | 2025-07-14 13:15:00 | 12470.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-07-16 09:15:00 | 12434.00 | 2025-07-17 12:15:00 | 12520.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-07-17 10:30:00 | 12483.00 | 2025-07-17 12:15:00 | 12520.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-07-17 12:00:00 | 12468.00 | 2025-07-17 12:15:00 | 12520.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-07-28 12:30:00 | 12250.00 | 2025-07-30 11:15:00 | 12355.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-08-04 11:30:00 | 12138.00 | 2025-08-04 13:15:00 | 12289.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-08-04 12:00:00 | 12135.00 | 2025-08-04 13:15:00 | 12289.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-08-08 14:15:00 | 12159.00 | 2025-08-11 09:15:00 | 12246.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-08-08 15:00:00 | 12150.00 | 2025-08-11 09:15:00 | 12246.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-09-03 11:00:00 | 12759.00 | 2025-09-03 13:15:00 | 12684.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-09-03 11:45:00 | 12750.00 | 2025-09-03 13:15:00 | 12684.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-09-03 12:15:00 | 12756.00 | 2025-09-03 13:15:00 | 12684.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-09-04 09:15:00 | 12930.00 | 2025-09-04 14:15:00 | 12650.00 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-09-09 13:00:00 | 12594.00 | 2025-09-16 10:15:00 | 12515.00 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest1 | 2025-09-29 11:30:00 | 12052.00 | 2025-09-30 09:15:00 | 12170.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-10-17 11:15:00 | 12358.00 | 2025-10-23 12:15:00 | 12268.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-10-17 14:00:00 | 12362.00 | 2025-10-23 12:15:00 | 12268.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-10-21 13:45:00 | 12373.00 | 2025-10-23 12:15:00 | 12268.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-10-23 09:15:00 | 12388.00 | 2025-10-23 12:15:00 | 12268.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-10-28 12:00:00 | 11950.00 | 2025-10-29 12:15:00 | 12051.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-10-28 13:15:00 | 11957.00 | 2025-10-29 12:15:00 | 12051.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-10-30 09:30:00 | 11955.00 | 2025-10-30 11:15:00 | 12038.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-11-07 09:15:00 | 11871.00 | 2025-11-12 12:15:00 | 11890.00 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-11-07 13:00:00 | 11895.00 | 2025-11-12 12:15:00 | 11890.00 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-11-07 14:00:00 | 11890.00 | 2025-11-12 12:15:00 | 11890.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-12-01 10:15:00 | 11592.00 | 2025-12-01 14:15:00 | 11661.00 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-12-01 11:30:00 | 11584.00 | 2025-12-01 14:15:00 | 11661.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-12-01 12:00:00 | 11585.00 | 2025-12-01 14:15:00 | 11661.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest1 | 2025-12-10 11:30:00 | 11424.00 | 2025-12-11 13:15:00 | 11451.00 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-12-22 09:15:00 | 11475.00 | 2025-12-22 14:15:00 | 11531.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-12-29 10:15:00 | 11804.00 | 2025-12-30 10:15:00 | 11702.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-12-29 11:00:00 | 11802.00 | 2025-12-30 10:15:00 | 11702.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-12-29 11:45:00 | 11800.00 | 2025-12-30 10:15:00 | 11702.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-12-29 12:30:00 | 11801.00 | 2025-12-30 10:15:00 | 11702.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2026-01-05 11:30:00 | 12092.00 | 2026-01-08 15:15:00 | 12047.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2026-01-05 12:00:00 | 12098.00 | 2026-01-08 15:15:00 | 12047.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2026-01-05 13:00:00 | 12096.00 | 2026-01-08 15:15:00 | 12047.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2026-01-06 09:15:00 | 12121.00 | 2026-01-08 15:15:00 | 12047.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-01-21 14:15:00 | 12133.00 | 2026-01-21 14:15:00 | 12243.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-01-27 09:15:00 | 12624.00 | 2026-02-01 12:15:00 | 12514.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-02-09 09:15:00 | 12766.00 | 2026-02-16 10:15:00 | 12912.00 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2026-02-09 09:45:00 | 12821.00 | 2026-02-16 10:15:00 | 12912.00 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2026-03-06 09:15:00 | 12150.00 | 2026-03-09 09:15:00 | 11542.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 09:15:00 | 12150.00 | 2026-03-10 09:15:00 | 11631.00 | STOP_HIT | 0.50 | 4.27% |
| SELL | retest2 | 2026-03-24 10:30:00 | 10502.00 | 2026-03-24 12:15:00 | 10823.00 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2026-04-01 13:15:00 | 10800.00 | 2026-04-06 13:15:00 | 10928.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-04-13 11:00:00 | 11427.00 | 2026-04-24 15:15:00 | 12035.00 | STOP_HIT | 1.00 | 5.32% |
| BUY | retest2 | 2026-04-13 11:30:00 | 11443.00 | 2026-04-24 15:15:00 | 12035.00 | STOP_HIT | 1.00 | 5.17% |
| SELL | retest2 | 2026-04-29 14:45:00 | 11838.00 | 2026-05-05 12:15:00 | 11921.00 | STOP_HIT | 1.00 | -0.70% |
