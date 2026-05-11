# Gillette India Ltd. (GILLETTE)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1528 bars)
- **Last close:** 8188.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 57 |
| ALERT1 | 32 |
| ALERT2 | 30 |
| ALERT2_SKIP | 21 |
| ALERT3 | 56 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 29 |
| PARTIAL | 4 |
| TARGET_HIT | 8 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 16
- **Target hits / Stop hits / Partials:** 8 / 21 / 4
- **Avg / median % per leg:** 2.39% / 0.20%
- **Sum % (uncompounded):** 78.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 5 | 50.0% | 4 | 6 | 0 | 3.67% | 36.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 5 | 50.0% | 4 | 6 | 0 | 3.67% | 36.7% |
| SELL (all) | 23 | 12 | 52.2% | 4 | 15 | 4 | 1.83% | 42.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 23 | 12 | 52.2% | 4 | 15 | 4 | 1.83% | 42.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 33 | 17 | 51.5% | 8 | 21 | 4 | 2.39% | 78.8% |

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
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 10257.00 | 10241.34 | 10172.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 10257.00 | 10241.34 | 10172.40 | EMA400 retest candle locked (from upside) |

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
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 11:15:00 | 10331.00 | 10330.89 | 10286.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 10331.00 | 10330.89 | 10286.36 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 10:15:00 | 10196.50 | 10339.86 | 10345.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 10123.00 | 10271.23 | 10311.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 10:15:00 | 10119.00 | 10113.32 | 10205.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 10107.50 | 10078.32 | 10144.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 10107.50 | 10078.32 | 10144.54 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 13:15:00 | 10315.00 | 10188.71 | 10180.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 14:15:00 | 10345.50 | 10220.07 | 10195.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 13:15:00 | 10861.50 | 10863.51 | 10749.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 09:15:00 | 10879.00 | 10875.77 | 10784.62 | EMA400 touched before retest1 break — omit ENTRY1 |
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
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 10796.00 | 10718.96 | 10726.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 10796.00 | 10718.96 | 10726.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:45:00 | 10600.00 | 10653.76 | 10685.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 10825.00 | 10579.18 | 10615.90 | SL hit (close>static) qty=1.00 sl=10823.00 alert=retest2 |

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
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 13:00:00 | 8094.50 | 7976.26 | 7995.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 14:00:00 | 8100.50 | 7976.26 | 7995.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:15:00 | 8096.00 | 7976.26 | 7995.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 09:15:00 | 7695.00 | 7884.31 | 7951.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 09:15:00 | 7689.77 | 7884.31 | 7951.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 09:15:00 | 7695.47 | 7884.31 | 7951.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 09:15:00 | 7691.20 | 7884.31 | 7951.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-30 09:15:00 | 7290.00 | 7545.15 | 7719.29 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 53 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 7592.50 | 7508.99 | 7505.21 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 12:15:00 | 7498.00 | 7502.71 | 7502.89 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 09:15:00 | 7550.50 | 7510.23 | 7505.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 12:15:00 | 7586.00 | 7535.99 | 7519.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 7500.50 | 7540.65 | 7525.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 14:15:00 | 7500.50 | 7540.65 | 7525.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 7500.50 | 7540.65 | 7525.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 15:00:00 | 7500.50 | 7540.65 | 7525.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 7512.00 | 7534.92 | 7524.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 7693.00 | 7534.92 | 7524.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 13:15:00 | 8061.00 | 8138.83 | 8138.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-04-29 13:15:00)

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

### Cycle 57 — BUY (started 2026-05-08 10:15:00)

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
| SELL | retest2 | 2025-07-10 10:45:00 | 10600.00 | 2025-07-11 09:15:00 | 10825.00 | STOP_HIT | 1.00 | -2.12% |
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
| SELL | retest2 | 2025-12-16 12:30:00 | 8100.00 | 2026-03-27 09:15:00 | 7695.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 13:00:00 | 8094.50 | 2026-03-27 09:15:00 | 7689.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 14:00:00 | 8100.50 | 2026-03-27 09:15:00 | 7695.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 12:15:00 | 8096.00 | 2026-03-27 09:15:00 | 7691.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-16 12:30:00 | 8100.00 | 2026-03-30 09:15:00 | 7290.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-06 13:00:00 | 8094.50 | 2026-03-30 09:15:00 | 7285.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-06 14:00:00 | 8100.50 | 2026-03-30 09:15:00 | 7290.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-07 12:15:00 | 8096.00 | 2026-03-30 09:15:00 | 7286.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-02 09:15:00 | 7329.50 | 2026-04-06 09:15:00 | 7592.50 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2026-04-08 09:15:00 | 7693.00 | 2026-04-29 13:15:00 | 8061.00 | STOP_HIT | 1.00 | 4.78% |
| SELL | retest2 | 2026-05-04 12:00:00 | 8018.00 | 2026-05-08 10:15:00 | 8019.00 | STOP_HIT | 1.00 | -0.01% |
