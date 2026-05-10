# Gillette India Ltd. (GILLETTE)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 8188.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 81 |
| ALERT1 | 48 |
| ALERT2 | 46 |
| ALERT2_SKIP | 25 |
| ALERT3 | 94 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 47 |
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 46 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 28
- **Target hits / Stop hits / Partials:** 4 / 44 / 6
- **Avg / median % per leg:** 1.12% / -0.01%
- **Sum % (uncompounded):** 60.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 5 | 27.8% | 4 | 14 | 0 | 1.59% | 28.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 5 | 27.8% | 4 | 14 | 0 | 1.59% | 28.5% |
| SELL (all) | 36 | 21 | 58.3% | 0 | 30 | 6 | 0.88% | 31.7% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.68% | -4.7% |
| SELL @ 3rd Alert (retest2) | 35 | 21 | 60.0% | 0 | 29 | 6 | 1.04% | 36.4% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.68% | -4.7% |
| retest2 (combined) | 53 | 26 | 49.1% | 4 | 43 | 6 | 1.22% | 64.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 7938.00 | 7890.06 | 7887.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 8026.50 | 7931.65 | 7908.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 14:15:00 | 8687.00 | 8721.63 | 8611.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 15:00:00 | 8687.00 | 8721.63 | 8611.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 8772.00 | 8729.92 | 8634.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 11:45:00 | 8900.00 | 8773.99 | 8672.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 12:30:00 | 8875.00 | 8786.79 | 8687.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:15:00 | 8876.50 | 8802.73 | 8703.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 10:00:00 | 8885.50 | 8785.20 | 8758.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 9375.00 | 8927.62 | 8837.57 | EMA400 retest candle locked (from upside) |
| Target hit | 2025-05-26 14:15:00 | 9790.00 | 8927.62 | 8837.57 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-05-26 14:15:00 | 9762.50 | 8927.62 | 8837.57 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-05-26 14:15:00 | 9764.15 | 8927.62 | 8837.57 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-05-26 14:15:00 | 9774.05 | 8927.62 | 8837.57 | Target hit (10%) qty=1.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 13:15:00 | 9881.00 | 9715.94 | 9599.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 10062.00 | 9807.67 | 9675.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 09:15:00 | 9783.00 | 9885.55 | 9892.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-09 09:15:00 | 9783.00 | 9885.55 | 9892.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 09:15:00 | 9783.00 | 9885.55 | 9892.64 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 10085.50 | 9904.74 | 9885.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 14:15:00 | 10150.50 | 10029.16 | 9957.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 11:15:00 | 10084.00 | 10115.78 | 10027.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 12:00:00 | 10084.00 | 10115.78 | 10027.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 10257.00 | 10241.34 | 10172.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:30:00 | 10315.00 | 10254.37 | 10184.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 13:15:00 | 10284.50 | 10268.10 | 10203.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 14:30:00 | 10284.50 | 10273.18 | 10217.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 15:00:00 | 10288.00 | 10273.18 | 10217.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 10152.50 | 10252.54 | 10217.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 10098.50 | 10252.54 | 10217.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 10190.50 | 10240.13 | 10215.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:15:00 | 10164.00 | 10240.13 | 10215.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 10161.00 | 10201.55 | 10201.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 10161.00 | 10201.55 | 10201.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 10161.00 | 10201.55 | 10201.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 10161.00 | 10201.55 | 10201.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 13:15:00 | 10161.00 | 10201.55 | 10201.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 15:15:00 | 10119.50 | 10178.25 | 10190.80 | Break + close below crossover candle low |

### Cycle 5 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 10374.50 | 10217.50 | 10207.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 14:15:00 | 10448.50 | 10296.10 | 10252.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 10:15:00 | 10329.50 | 10330.86 | 10281.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 11:00:00 | 10329.50 | 10330.86 | 10281.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 10331.00 | 10330.89 | 10286.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 10324.50 | 10330.89 | 10286.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 10360.00 | 10336.71 | 10293.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:30:00 | 10313.50 | 10336.71 | 10293.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 10419.00 | 10419.79 | 10367.96 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 10:15:00 | 10196.50 | 10339.86 | 10345.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 10123.00 | 10271.23 | 10311.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 10:15:00 | 10119.00 | 10113.32 | 10205.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-23 10:45:00 | 10112.00 | 10113.32 | 10205.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 10107.50 | 10078.32 | 10144.54 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 13:15:00 | 10315.00 | 10188.71 | 10180.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 14:15:00 | 10345.50 | 10220.07 | 10195.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 13:15:00 | 10861.50 | 10863.51 | 10749.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 14:00:00 | 10861.50 | 10863.51 | 10749.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 10879.00 | 10875.77 | 10784.62 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 10550.00 | 10744.07 | 10756.44 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 10790.00 | 10733.19 | 10731.06 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 14:15:00 | 10697.00 | 10736.87 | 10738.50 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 10:15:00 | 10845.00 | 10750.28 | 10743.34 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 10690.00 | 10745.70 | 10748.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 11:15:00 | 10662.00 | 10721.33 | 10736.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 10705.00 | 10703.38 | 10723.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 15:00:00 | 10705.00 | 10703.38 | 10723.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 10796.00 | 10718.96 | 10726.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 10796.00 | 10718.96 | 10726.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 10739.00 | 10722.97 | 10727.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:30:00 | 10700.00 | 10716.78 | 10724.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 10825.00 | 10579.18 | 10615.90 | SL hit (close>static) qty=1.00 sl=10796.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 11:15:00 | 10768.00 | 10655.15 | 10646.43 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 13:15:00 | 10583.00 | 10672.63 | 10672.73 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 10725.00 | 10681.09 | 10676.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 11033.00 | 10751.47 | 10708.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 15:15:00 | 10950.00 | 10978.90 | 10912.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 15:15:00 | 10950.00 | 10978.90 | 10912.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 10950.00 | 10978.90 | 10912.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 11179.00 | 10978.90 | 10912.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 14:15:00 | 10935.00 | 11106.56 | 11112.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 14:15:00 | 10935.00 | 11106.56 | 11112.63 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 09:15:00 | 11250.00 | 11111.18 | 11105.94 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 11050.00 | 11130.38 | 11131.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 13:15:00 | 10991.00 | 11092.04 | 11112.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 10855.00 | 10791.85 | 10901.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 10:00:00 | 10855.00 | 10791.85 | 10901.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 10532.00 | 10500.77 | 10604.03 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 10989.00 | 10665.21 | 10630.51 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 10:15:00 | 10480.00 | 10666.22 | 10681.26 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 13:15:00 | 10680.00 | 10644.25 | 10643.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 09:15:00 | 10797.00 | 10696.01 | 10668.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 11:15:00 | 10691.00 | 10707.00 | 10679.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 11:15:00 | 10691.00 | 10707.00 | 10679.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 10691.00 | 10707.00 | 10679.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:00:00 | 10691.00 | 10707.00 | 10679.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 10612.00 | 10688.00 | 10673.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:00:00 | 10612.00 | 10688.00 | 10673.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 10633.00 | 10677.00 | 10669.54 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 10604.00 | 10652.96 | 10659.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 10538.00 | 10612.62 | 10637.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 10650.00 | 10611.68 | 10632.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 10650.00 | 10611.68 | 10632.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 10650.00 | 10611.68 | 10632.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 10650.00 | 10611.68 | 10632.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 10650.00 | 10619.34 | 10633.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 10551.00 | 10619.34 | 10633.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 10371.00 | 10345.21 | 10413.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:15:00 | 10424.00 | 10345.21 | 10413.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 10422.00 | 10360.57 | 10413.99 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 10509.00 | 10438.98 | 10433.01 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 10298.00 | 10419.30 | 10431.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 10279.00 | 10358.38 | 10395.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 10350.00 | 10344.16 | 10382.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 10350.00 | 10344.16 | 10382.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 10350.00 | 10344.16 | 10382.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:15:00 | 10447.00 | 10344.16 | 10382.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 10516.00 | 10378.53 | 10394.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 11:00:00 | 10516.00 | 10378.53 | 10394.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 11:15:00 | 10513.00 | 10405.42 | 10404.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 10546.00 | 10433.54 | 10417.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 10378.00 | 10453.76 | 10435.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 10378.00 | 10453.76 | 10435.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 10378.00 | 10453.76 | 10435.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 10378.00 | 10453.76 | 10435.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 10:15:00 | 10300.00 | 10423.01 | 10423.58 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 10445.00 | 10419.42 | 10416.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 10:15:00 | 10506.00 | 10436.74 | 10424.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 10513.00 | 10521.48 | 10486.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:00:00 | 10513.00 | 10521.48 | 10486.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 10460.00 | 10509.18 | 10484.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 10460.00 | 10509.18 | 10484.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 10425.00 | 10492.35 | 10478.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:45:00 | 10400.00 | 10492.35 | 10478.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 10400.00 | 10462.62 | 10467.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 10:15:00 | 10371.00 | 10437.48 | 10454.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 13:15:00 | 10417.00 | 10415.32 | 10438.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-22 13:45:00 | 10410.00 | 10415.32 | 10438.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 10399.00 | 10398.65 | 10424.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 15:00:00 | 10315.00 | 10375.33 | 10402.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 11:15:00 | 10112.00 | 10107.04 | 10106.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 10112.00 | 10107.04 | 10106.92 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 12:15:00 | 10099.00 | 10105.43 | 10106.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-01 13:15:00 | 10070.00 | 10098.35 | 10102.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 10172.00 | 10100.72 | 10101.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 10172.00 | 10100.72 | 10101.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 10172.00 | 10100.72 | 10101.48 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 10209.00 | 10122.38 | 10111.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 10552.00 | 10225.59 | 10178.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 09:15:00 | 10400.00 | 10471.65 | 10361.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 11:15:00 | 10382.00 | 10439.70 | 10365.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 10382.00 | 10439.70 | 10365.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:30:00 | 10425.00 | 10439.70 | 10365.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 10371.00 | 10425.96 | 10365.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:45:00 | 10360.00 | 10425.96 | 10365.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 10460.00 | 10432.77 | 10374.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:30:00 | 10383.00 | 10432.77 | 10374.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 10384.00 | 10427.98 | 10387.06 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 15:15:00 | 10305.00 | 10358.89 | 10366.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 09:15:00 | 10255.00 | 10338.11 | 10355.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 15:15:00 | 10305.00 | 10290.39 | 10317.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 15:15:00 | 10305.00 | 10290.39 | 10317.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 10305.00 | 10290.39 | 10317.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 10363.00 | 10290.39 | 10317.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 10281.00 | 10288.51 | 10314.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:30:00 | 10238.00 | 10272.41 | 10304.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:00:00 | 10208.00 | 10272.41 | 10304.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 10384.00 | 10064.51 | 10023.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 10384.00 | 10064.51 | 10023.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 14:15:00 | 10384.00 | 10064.51 | 10023.50 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 13:15:00 | 9993.00 | 10071.60 | 10078.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 12:15:00 | 9897.00 | 9984.54 | 10027.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 9713.00 | 9612.96 | 9709.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 9713.00 | 9612.96 | 9709.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 9713.00 | 9612.96 | 9709.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 9713.00 | 9612.96 | 9709.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 9610.00 | 9612.37 | 9700.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 9542.00 | 9597.10 | 9685.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 14:15:00 | 9575.00 | 9592.14 | 9667.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 11:15:00 | 9897.00 | 9597.85 | 9588.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 11:15:00 | 9897.00 | 9597.85 | 9588.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 9897.00 | 9597.85 | 9588.13 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 13:15:00 | 9463.50 | 9579.30 | 9594.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 11:15:00 | 9422.00 | 9501.44 | 9546.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 14:15:00 | 9479.00 | 9430.97 | 9469.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 14:15:00 | 9479.00 | 9430.97 | 9469.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 9479.00 | 9430.97 | 9469.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 9479.00 | 9430.97 | 9469.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 9493.00 | 9443.38 | 9471.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 9437.50 | 9443.38 | 9471.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 9504.00 | 9441.89 | 9453.70 | SL hit (close>static) qty=1.00 sl=9493.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 9548.00 | 9463.11 | 9462.27 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 9422.50 | 9479.38 | 9484.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 9405.00 | 9464.50 | 9477.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 12:15:00 | 9500.00 | 9465.60 | 9475.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 12:15:00 | 9500.00 | 9465.60 | 9475.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 9500.00 | 9465.60 | 9475.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:45:00 | 9505.00 | 9465.60 | 9475.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 9498.50 | 9472.18 | 9477.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:45:00 | 9496.00 | 9472.18 | 9477.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 9430.50 | 9463.85 | 9473.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 9372.00 | 9454.48 | 9468.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 10:15:00 | 9484.50 | 9378.20 | 9368.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 10:15:00 | 9484.50 | 9378.20 | 9368.53 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 9350.00 | 9368.10 | 9370.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 10:15:00 | 9303.00 | 9347.42 | 9357.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 15:15:00 | 9191.50 | 9184.05 | 9228.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 09:15:00 | 9158.50 | 9184.05 | 9228.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 9105.00 | 9088.02 | 9124.20 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 9239.50 | 9138.96 | 9136.31 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 9009.00 | 9113.64 | 9125.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 15:15:00 | 8995.00 | 9089.91 | 9113.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 8445.50 | 8351.24 | 8434.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 8445.50 | 8351.24 | 8434.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 8445.50 | 8351.24 | 8434.30 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 11:15:00 | 8497.50 | 8457.65 | 8456.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 12:15:00 | 8533.50 | 8472.82 | 8463.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 8400.00 | 8522.77 | 8508.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 8400.00 | 8522.77 | 8508.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 8400.00 | 8522.77 | 8508.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 8400.00 | 8522.77 | 8508.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 8394.00 | 8497.01 | 8497.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 11:15:00 | 8376.50 | 8472.91 | 8486.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 15:15:00 | 8371.00 | 8350.86 | 8390.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 09:15:00 | 8413.00 | 8350.86 | 8390.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 8410.50 | 8362.79 | 8392.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 8340.50 | 8398.48 | 8402.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 13:15:00 | 8415.50 | 8370.53 | 8365.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 13:15:00 | 8415.50 | 8370.53 | 8365.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 15:15:00 | 8464.00 | 8400.74 | 8380.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 09:15:00 | 8340.00 | 8388.59 | 8376.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 8340.00 | 8388.59 | 8376.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 8340.00 | 8388.59 | 8376.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:45:00 | 8365.50 | 8388.59 | 8376.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 8352.50 | 8381.37 | 8374.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 8336.00 | 8381.37 | 8374.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 12:15:00 | 8331.50 | 8367.02 | 8369.05 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 13:15:00 | 8401.50 | 8373.91 | 8372.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 8418.50 | 8396.59 | 8384.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 8647.00 | 8680.19 | 8586.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 09:30:00 | 8658.50 | 8680.19 | 8586.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 8639.50 | 8656.05 | 8603.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 14:15:00 | 8682.00 | 8656.05 | 8603.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 10:45:00 | 8655.50 | 8671.43 | 8629.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 8575.00 | 8649.69 | 8632.54 | SL hit (close<static) qty=1.00 sl=8600.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 8575.00 | 8649.69 | 8632.54 | SL hit (close<static) qty=1.00 sl=8600.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 8491.00 | 8601.84 | 8612.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 14:15:00 | 8386.00 | 8479.96 | 8541.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 15:15:00 | 8360.00 | 8327.37 | 8407.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 09:30:00 | 8350.00 | 8326.59 | 8400.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 8380.00 | 8341.65 | 8389.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:45:00 | 8385.00 | 8341.65 | 8389.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 8361.50 | 8345.62 | 8386.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 14:15:00 | 8352.50 | 8345.62 | 8386.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 14:15:00 | 8547.50 | 8385.99 | 8401.22 | SL hit (close>static) qty=1.00 sl=8392.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 13:45:00 | 8352.00 | 8378.98 | 8393.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 8324.50 | 8257.67 | 8268.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 10:15:00 | 8308.00 | 8280.51 | 8277.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-10 10:15:00 | 8308.00 | 8280.51 | 8277.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 8308.00 | 8280.51 | 8277.75 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 8192.50 | 8265.26 | 8271.91 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 8298.00 | 8267.77 | 8265.77 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 09:15:00 | 8239.00 | 8262.02 | 8263.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 10:15:00 | 8195.50 | 8248.71 | 8257.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 14:15:00 | 8115.00 | 8102.90 | 8151.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-15 14:30:00 | 8138.00 | 8102.90 | 8151.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 8150.00 | 8112.32 | 8151.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 8152.00 | 8112.32 | 8151.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 8170.00 | 8123.86 | 8153.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 8170.00 | 8123.86 | 8153.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 8119.50 | 8122.98 | 8150.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 12:30:00 | 8100.00 | 8112.51 | 8140.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 15:15:00 | 8025.00 | 8003.73 | 8001.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 15:15:00 | 8025.00 | 8003.73 | 8001.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 09:15:00 | 8127.50 | 8028.49 | 8012.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 10:15:00 | 8396.00 | 8405.98 | 8306.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 11:00:00 | 8396.00 | 8405.98 | 8306.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 8345.00 | 8390.94 | 8324.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 8345.00 | 8390.94 | 8324.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 8331.50 | 8379.05 | 8325.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:30:00 | 8324.50 | 8379.05 | 8325.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 8356.50 | 8374.54 | 8327.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 8271.00 | 8374.54 | 8327.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 8295.00 | 8358.63 | 8324.89 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 12:15:00 | 8235.50 | 8304.80 | 8305.93 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 8338.00 | 8311.44 | 8308.85 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 14:15:00 | 8184.00 | 8285.95 | 8297.50 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 8357.00 | 8301.86 | 8298.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 8380.00 | 8317.49 | 8305.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 8310.00 | 8318.95 | 8308.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 8310.00 | 8318.95 | 8308.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 8310.00 | 8318.95 | 8308.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:00:00 | 8310.00 | 8318.95 | 8308.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 8293.00 | 8313.76 | 8307.01 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 12:15:00 | 8258.00 | 8293.69 | 8298.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 13:15:00 | 8208.50 | 8276.65 | 8290.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 11:15:00 | 8256.50 | 8238.88 | 8261.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 11:15:00 | 8256.50 | 8238.88 | 8261.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 8256.50 | 8238.88 | 8261.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:00:00 | 8256.50 | 8238.88 | 8261.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 8220.00 | 8235.11 | 8258.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:30:00 | 8215.00 | 8235.11 | 8258.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 8174.50 | 8218.34 | 8242.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 8140.00 | 8193.80 | 8217.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 10:00:00 | 8144.00 | 8183.84 | 8210.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 13:00:00 | 8094.50 | 8149.38 | 8186.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 10:30:00 | 8139.00 | 8123.31 | 8156.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 8050.00 | 8100.73 | 8130.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:30:00 | 8030.00 | 8083.69 | 8120.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 8093.50 | 7939.02 | 7934.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 8093.50 | 7939.02 | 7934.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 8093.50 | 7939.02 | 7934.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 8093.50 | 7939.02 | 7934.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 8093.50 | 7939.02 | 7934.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 11:15:00 | 8093.50 | 7939.02 | 7934.68 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 7930.00 | 8019.03 | 8023.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 7905.00 | 7996.23 | 8012.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 7969.00 | 7939.66 | 7971.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 7969.00 | 7939.66 | 7971.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 7969.00 | 7939.66 | 7971.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:00:00 | 7969.00 | 7939.66 | 7971.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 7921.00 | 7935.93 | 7967.15 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 12:15:00 | 8124.00 | 7993.95 | 7989.37 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 7979.00 | 8005.20 | 8005.81 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 12:15:00 | 8019.00 | 8007.96 | 8007.01 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 7955.00 | 7998.42 | 8002.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 15:15:00 | 7916.00 | 7981.93 | 7995.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 7917.00 | 7911.52 | 7944.30 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 09:15:00 | 7855.00 | 7911.52 | 7944.30 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 7846.50 | 7875.04 | 7917.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:00:00 | 7846.50 | 7875.04 | 7917.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 7794.00 | 7851.48 | 7888.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-29 13:15:00 | 8223.00 | 7908.69 | 7900.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest1 |

### Cycle 65 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 8223.00 | 7908.69 | 7900.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 8610.00 | 8157.65 | 8026.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 8668.50 | 8676.20 | 8511.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:15:00 | 8746.50 | 8676.20 | 8511.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 8735.00 | 8771.80 | 8693.94 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 8610.50 | 8667.24 | 8672.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 15:15:00 | 8590.00 | 8642.90 | 8658.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 10:15:00 | 8652.00 | 8637.77 | 8653.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 10:15:00 | 8652.00 | 8637.77 | 8653.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 8652.00 | 8637.77 | 8653.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:15:00 | 8713.50 | 8637.77 | 8653.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 8674.00 | 8645.02 | 8654.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:30:00 | 8732.50 | 8645.02 | 8654.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 8712.00 | 8658.42 | 8660.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:45:00 | 8701.00 | 8658.42 | 8660.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 8738.50 | 8674.43 | 8667.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 8775.00 | 8702.04 | 8682.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 13:15:00 | 8715.00 | 8725.03 | 8701.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 13:15:00 | 8715.00 | 8725.03 | 8701.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 8715.00 | 8725.03 | 8701.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 8770.00 | 8730.30 | 8708.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 12:45:00 | 8779.00 | 8736.30 | 8716.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 15:15:00 | 8757.00 | 8767.05 | 8753.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 8694.50 | 8741.08 | 8744.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 8694.50 | 8741.08 | 8744.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 8694.50 | 8741.08 | 8744.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 8694.50 | 8741.08 | 8744.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 8679.00 | 8716.60 | 8730.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 8573.00 | 8526.07 | 8577.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:00:00 | 8573.00 | 8526.07 | 8577.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 8579.50 | 8536.76 | 8577.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:45:00 | 8574.50 | 8536.76 | 8577.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 8674.00 | 8564.20 | 8586.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:00:00 | 8674.00 | 8564.20 | 8586.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 8628.50 | 8577.06 | 8590.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:30:00 | 8664.00 | 8577.06 | 8590.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 8629.00 | 8601.52 | 8599.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 15:15:00 | 8675.00 | 8616.22 | 8606.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 8596.00 | 8612.17 | 8605.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 8596.00 | 8612.17 | 8605.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 8596.00 | 8612.17 | 8605.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 8596.00 | 8612.17 | 8605.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 8623.50 | 8614.44 | 8607.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 15:15:00 | 8650.00 | 8619.85 | 8611.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 11:15:00 | 8558.50 | 8604.90 | 8607.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 8558.50 | 8604.90 | 8607.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 8534.00 | 8590.72 | 8601.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 8482.00 | 8475.11 | 8514.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 8482.00 | 8475.11 | 8514.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 8482.00 | 8475.11 | 8514.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 8513.50 | 8475.11 | 8514.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 8565.50 | 8493.18 | 8519.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 8565.50 | 8493.18 | 8519.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 8578.00 | 8510.15 | 8524.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:30:00 | 8616.50 | 8510.15 | 8524.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 12:15:00 | 8667.00 | 8541.52 | 8537.69 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 8443.00 | 8531.97 | 8540.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 13:15:00 | 8438.00 | 8502.22 | 8524.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 11:15:00 | 8474.50 | 8471.80 | 8497.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 12:00:00 | 8474.50 | 8471.80 | 8497.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 8437.50 | 8437.42 | 8468.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 10:45:00 | 8402.00 | 8431.33 | 8462.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 8404.00 | 8424.57 | 8456.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 15:00:00 | 8354.50 | 8406.50 | 8439.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 8360.00 | 8408.60 | 8437.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 8427.00 | 8413.71 | 8435.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:30:00 | 8409.00 | 8413.71 | 8435.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 8416.00 | 8414.16 | 8433.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 13:30:00 | 8395.00 | 8410.53 | 8428.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 7981.90 | 8186.79 | 8276.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 7983.80 | 8186.79 | 8276.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 7975.25 | 8186.79 | 8276.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 8069.00 | 8054.71 | 8152.02 | SL hit (close>ema200) qty=0.50 sl=8054.71 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 8069.00 | 8054.71 | 8152.02 | SL hit (close>ema200) qty=0.50 sl=8054.71 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 8069.00 | 8054.71 | 8152.02 | SL hit (close>ema200) qty=0.50 sl=8054.71 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 7936.77 | 8073.29 | 8114.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 7942.00 | 8073.29 | 8114.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 8038.50 | 7991.02 | 8048.90 | SL hit (close>ema200) qty=0.50 sl=7991.02 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 8038.50 | 7991.02 | 8048.90 | SL hit (close>ema200) qty=0.50 sl=7991.02 alert=retest2 |

### Cycle 73 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 8189.00 | 8074.79 | 8072.60 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 7997.00 | 8081.43 | 8086.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 7985.00 | 8050.24 | 8070.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 8001.00 | 7986.29 | 8022.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 13:45:00 | 7994.50 | 7986.29 | 8022.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 8012.50 | 7969.28 | 7998.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:45:00 | 8018.50 | 7969.28 | 7998.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 8020.00 | 7979.42 | 8000.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 13:30:00 | 7934.50 | 7960.94 | 7990.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 14:45:00 | 7894.50 | 7947.95 | 7981.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 09:15:00 | 7917.50 | 7961.36 | 7984.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 11:15:00 | 8051.00 | 7961.63 | 7959.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 11:15:00 | 8051.00 | 7961.63 | 7959.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 11:15:00 | 8051.00 | 7961.63 | 7959.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 8051.00 | 7961.63 | 7959.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 8118.00 | 7997.80 | 7979.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 14:15:00 | 7940.00 | 8014.62 | 7996.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 14:15:00 | 7940.00 | 8014.62 | 7996.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 7940.00 | 8014.62 | 7996.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 15:00:00 | 7940.00 | 8014.62 | 7996.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 7900.00 | 7991.70 | 7987.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 7893.50 | 7991.70 | 7987.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 7916.00 | 7976.56 | 7981.44 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 8134.00 | 7997.22 | 7980.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 12:15:00 | 8147.50 | 8048.20 | 8007.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 14:15:00 | 8004.00 | 8039.57 | 8011.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 14:15:00 | 8004.00 | 8039.57 | 8011.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 8004.00 | 8039.57 | 8011.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 8004.00 | 8039.57 | 8011.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 7968.00 | 8025.26 | 8007.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 7830.00 | 8025.26 | 8007.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 7781.00 | 7976.40 | 7986.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 15:15:00 | 7751.00 | 7841.75 | 7903.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 7817.00 | 7810.80 | 7867.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:30:00 | 7813.00 | 7810.80 | 7867.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 7835.00 | 7792.18 | 7837.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 14:15:00 | 7736.50 | 7801.29 | 7828.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 7349.67 | 7508.13 | 7629.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 7464.50 | 7376.12 | 7486.18 | SL hit (close>ema200) qty=0.50 sl=7376.12 alert=retest2 |

### Cycle 79 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 7580.00 | 7487.68 | 7483.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 7686.00 | 7565.12 | 7535.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 7620.00 | 7648.47 | 7603.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 7620.00 | 7648.47 | 7603.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 7620.00 | 7648.47 | 7603.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 7620.00 | 7648.47 | 7603.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 7602.50 | 7639.45 | 7614.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 7650.00 | 7629.69 | 7613.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 13:15:00 | 8061.00 | 8138.83 | 8138.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 8061.00 | 8138.83 | 8138.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 7982.00 | 8084.99 | 8112.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 8060.50 | 8018.50 | 8056.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 8060.50 | 8018.50 | 8056.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 8060.50 | 8018.50 | 8056.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:30:00 | 8085.50 | 8018.50 | 8056.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 8079.00 | 8030.60 | 8058.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 8018.00 | 8028.08 | 8054.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 10:15:00 | 8019.00 | 7949.82 | 7947.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 10:15:00 | 8019.00 | 7949.82 | 7947.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 11:15:00 | 8065.50 | 7972.96 | 7958.54 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 10:45:00 | 7976.50 | 2025-05-12 12:15:00 | 7938.00 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-05-21 11:45:00 | 8900.00 | 2025-05-26 14:15:00 | 9790.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-21 12:30:00 | 8875.00 | 2025-05-26 14:15:00 | 9762.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-21 14:15:00 | 8876.50 | 2025-05-26 14:15:00 | 9764.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-26 10:00:00 | 8885.50 | 2025-05-26 14:15:00 | 9774.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-04 13:15:00 | 9881.00 | 2025-06-09 09:15:00 | 9783.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-06-05 09:15:00 | 10062.00 | 2025-06-09 09:15:00 | 9783.00 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2025-06-13 10:30:00 | 10315.00 | 2025-06-16 13:15:00 | 10161.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-06-13 13:15:00 | 10284.50 | 2025-06-16 13:15:00 | 10161.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-06-13 14:30:00 | 10284.50 | 2025-06-16 13:15:00 | 10161.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-06-13 15:00:00 | 10288.00 | 2025-06-16 13:15:00 | 10161.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-07-09 11:30:00 | 10700.00 | 2025-07-11 09:15:00 | 10825.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-07-17 09:15:00 | 11179.00 | 2025-07-21 14:15:00 | 10935.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-08-25 15:00:00 | 10315.00 | 2025-09-01 11:15:00 | 10112.00 | STOP_HIT | 1.00 | 1.97% |
| SELL | retest2 | 2025-09-10 10:30:00 | 10238.00 | 2025-09-19 14:15:00 | 10384.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-09-10 11:00:00 | 10208.00 | 2025-09-19 14:15:00 | 10384.00 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-09-29 11:30:00 | 9542.00 | 2025-10-01 11:15:00 | 9897.00 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2025-09-29 14:15:00 | 9575.00 | 2025-10-01 11:15:00 | 9897.00 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2025-10-08 09:15:00 | 9437.50 | 2025-10-09 09:15:00 | 9504.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-10-15 09:15:00 | 9372.00 | 2025-10-17 10:15:00 | 9484.50 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-11-21 09:15:00 | 8340.50 | 2025-11-24 13:15:00 | 8415.50 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-11-28 14:15:00 | 8682.00 | 2025-12-01 14:15:00 | 8575.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-01 10:45:00 | 8655.50 | 2025-12-01 14:15:00 | 8575.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-12-04 14:15:00 | 8352.50 | 2025-12-04 14:15:00 | 8547.50 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-12-05 13:45:00 | 8352.00 | 2025-12-10 10:15:00 | 8308.00 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest2 | 2025-12-10 09:15:00 | 8324.50 | 2025-12-10 10:15:00 | 8308.00 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-12-16 12:30:00 | 8100.00 | 2025-12-23 15:15:00 | 8025.00 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2026-01-06 09:15:00 | 8140.00 | 2026-01-13 11:15:00 | 8093.50 | STOP_HIT | 1.00 | 0.57% |
| SELL | retest2 | 2026-01-06 10:00:00 | 8144.00 | 2026-01-13 11:15:00 | 8093.50 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2026-01-06 13:00:00 | 8094.50 | 2026-01-13 11:15:00 | 8093.50 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2026-01-07 10:30:00 | 8139.00 | 2026-01-13 11:15:00 | 8093.50 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2026-01-08 10:30:00 | 8030.00 | 2026-01-13 11:15:00 | 8093.50 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest1 | 2026-01-28 09:15:00 | 7855.00 | 2026-01-29 13:15:00 | 8223.00 | STOP_HIT | 1.00 | -4.68% |
| BUY | retest2 | 2026-02-10 09:15:00 | 8770.00 | 2026-02-12 10:15:00 | 8694.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-02-10 12:45:00 | 8779.00 | 2026-02-12 10:15:00 | 8694.50 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2026-02-11 15:15:00 | 8757.00 | 2026-02-12 10:15:00 | 8694.50 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2026-02-18 15:15:00 | 8650.00 | 2026-02-19 11:15:00 | 8558.50 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-02-26 10:45:00 | 8402.00 | 2026-03-04 09:15:00 | 7981.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 11:30:00 | 8404.00 | 2026-03-04 09:15:00 | 7983.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 15:00:00 | 8354.50 | 2026-03-04 09:15:00 | 7975.25 | PARTIAL | 0.50 | 4.54% |
| SELL | retest2 | 2026-02-26 10:45:00 | 8402.00 | 2026-03-05 09:15:00 | 8069.00 | STOP_HIT | 0.50 | 3.96% |
| SELL | retest2 | 2026-02-26 11:30:00 | 8404.00 | 2026-03-05 09:15:00 | 8069.00 | STOP_HIT | 0.50 | 3.99% |
| SELL | retest2 | 2026-02-26 15:00:00 | 8354.50 | 2026-03-05 09:15:00 | 8069.00 | STOP_HIT | 0.50 | 3.42% |
| SELL | retest2 | 2026-02-27 09:15:00 | 8360.00 | 2026-03-09 09:15:00 | 7936.77 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2026-02-27 13:30:00 | 8395.00 | 2026-03-09 09:15:00 | 7942.00 | PARTIAL | 0.50 | 5.40% |
| SELL | retest2 | 2026-02-27 09:15:00 | 8360.00 | 2026-03-09 14:15:00 | 8038.50 | STOP_HIT | 0.50 | 3.85% |
| SELL | retest2 | 2026-02-27 13:30:00 | 8395.00 | 2026-03-09 14:15:00 | 8038.50 | STOP_HIT | 0.50 | 4.25% |
| SELL | retest2 | 2026-03-13 13:30:00 | 7934.50 | 2026-03-17 11:15:00 | 8051.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-03-13 14:45:00 | 7894.50 | 2026-03-17 11:15:00 | 8051.00 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2026-03-16 09:15:00 | 7917.50 | 2026-03-17 11:15:00 | 8051.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2026-03-25 14:15:00 | 7736.50 | 2026-03-30 09:15:00 | 7349.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 14:15:00 | 7736.50 | 2026-04-01 09:15:00 | 7464.50 | STOP_HIT | 0.50 | 3.52% |
| BUY | retest2 | 2026-04-10 09:15:00 | 7650.00 | 2026-04-29 13:15:00 | 8061.00 | STOP_HIT | 1.00 | 5.37% |
| SELL | retest2 | 2026-05-04 12:00:00 | 8018.00 | 2026-05-08 10:15:00 | 8019.00 | STOP_HIT | 1.00 | -0.01% |
