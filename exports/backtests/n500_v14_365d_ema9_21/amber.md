# Amber Enterprises India Ltd. (AMBER)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 8851.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 73 |
| ALERT1 | 50 |
| ALERT2 | 49 |
| ALERT2_SKIP | 19 |
| ALERT3 | 142 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 63 |
| PARTIAL | 7 |
| TARGET_HIT | 6 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 74 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 56
- **Target hits / Stop hits / Partials:** 6 / 61 / 7
- **Avg / median % per leg:** 0.56% / -0.98%
- **Sum % (uncompounded):** 41.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 37 | 2 | 5.4% | 0 | 37 | 0 | -1.19% | -44.1% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.95% | -3.8% |
| BUY @ 3rd Alert (retest2) | 33 | 2 | 6.1% | 0 | 33 | 0 | -1.22% | -40.3% |
| SELL (all) | 37 | 16 | 43.2% | 6 | 24 | 7 | 2.31% | 85.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 16 | 43.2% | 6 | 24 | 7 | 2.31% | 85.5% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.95% | -3.8% |
| retest2 (combined) | 70 | 18 | 25.7% | 6 | 57 | 7 | 0.65% | 45.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 6210.00 | 6029.60 | 6017.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 6256.00 | 6140.10 | 6079.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 6203.00 | 6218.94 | 6155.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 15:00:00 | 6203.00 | 6218.94 | 6155.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 6266.00 | 6227.72 | 6170.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 10:15:00 | 6284.50 | 6227.72 | 6170.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 11:00:00 | 6285.00 | 6364.65 | 6349.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 11:45:00 | 6288.50 | 6348.12 | 6343.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 12:15:00 | 6286.50 | 6335.80 | 6338.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-19 12:15:00 | 6286.50 | 6335.80 | 6338.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-19 12:15:00 | 6286.50 | 6335.80 | 6338.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 12:15:00 | 6286.50 | 6335.80 | 6338.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 14:15:00 | 6251.00 | 6312.23 | 6326.53 | Break + close below crossover candle low |

### Cycle 3 — BUY (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 09:15:00 | 6502.50 | 6340.01 | 6336.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 10:15:00 | 6510.50 | 6374.11 | 6351.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 10:15:00 | 6579.00 | 6607.40 | 6539.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 11:00:00 | 6579.00 | 6607.40 | 6539.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 6564.50 | 6598.82 | 6542.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:30:00 | 6539.00 | 6598.82 | 6542.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 6529.00 | 6584.85 | 6540.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:45:00 | 6534.00 | 6584.85 | 6540.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 6548.50 | 6577.58 | 6541.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:15:00 | 6505.00 | 6577.58 | 6541.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 6578.50 | 6577.77 | 6544.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:30:00 | 6523.00 | 6577.77 | 6544.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 6539.50 | 6570.11 | 6544.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:15:00 | 6605.50 | 6570.11 | 6544.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:15:00 | 6590.50 | 6575.31 | 6551.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 13:15:00 | 6408.50 | 6547.08 | 6545.29 | SL hit (close<static) qty=1.00 sl=6525.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-23 13:15:00 | 6408.50 | 6547.08 | 6545.29 | SL hit (close<static) qty=1.00 sl=6525.50 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 14:15:00 | 6413.50 | 6520.36 | 6533.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 09:15:00 | 6333.00 | 6463.39 | 6503.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 11:15:00 | 6454.50 | 6442.67 | 6486.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-26 12:00:00 | 6454.50 | 6442.67 | 6486.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 6580.00 | 6456.37 | 6473.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 6580.00 | 6456.37 | 6473.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 6590.00 | 6483.10 | 6484.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:45:00 | 6610.50 | 6483.10 | 6484.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 6538.50 | 6494.18 | 6489.18 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 6434.50 | 6478.98 | 6483.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 6390.50 | 6429.38 | 6451.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 6425.00 | 6411.56 | 6436.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 6425.00 | 6411.56 | 6436.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 6425.00 | 6411.56 | 6436.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:30:00 | 6446.00 | 6411.56 | 6436.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 6436.00 | 6416.45 | 6436.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 6436.00 | 6416.45 | 6436.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 6482.50 | 6429.66 | 6440.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:45:00 | 6482.00 | 6429.66 | 6440.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 6490.00 | 6441.73 | 6444.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:30:00 | 6478.00 | 6441.73 | 6444.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 13:15:00 | 6490.00 | 6451.38 | 6449.08 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 10:15:00 | 6424.00 | 6450.28 | 6450.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 11:15:00 | 6398.00 | 6439.82 | 6445.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 6321.00 | 6286.47 | 6326.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 6321.00 | 6286.47 | 6326.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 6321.00 | 6286.47 | 6326.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:30:00 | 6320.50 | 6286.47 | 6326.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 6382.00 | 6305.57 | 6331.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 14:15:00 | 6304.50 | 6329.94 | 6338.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 11:15:00 | 6304.50 | 6314.48 | 6327.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 12:30:00 | 6306.00 | 6315.36 | 6325.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 14:15:00 | 6403.00 | 6338.11 | 6334.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-06 14:15:00 | 6403.00 | 6338.11 | 6334.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-06 14:15:00 | 6403.00 | 6338.11 | 6334.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 14:15:00 | 6403.00 | 6338.11 | 6334.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 11:15:00 | 6660.00 | 6413.93 | 6371.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 13:15:00 | 6557.50 | 6561.20 | 6499.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 14:00:00 | 6557.50 | 6561.20 | 6499.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 6567.50 | 6597.05 | 6552.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 6567.50 | 6597.05 | 6552.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 6647.00 | 6607.04 | 6560.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:30:00 | 6573.50 | 6607.04 | 6560.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 6575.00 | 6599.03 | 6571.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 6575.00 | 6599.03 | 6571.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 6592.00 | 6597.63 | 6573.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:15:00 | 6575.00 | 6597.63 | 6573.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 6493.50 | 6576.80 | 6566.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 6493.50 | 6576.80 | 6566.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 6520.00 | 6565.44 | 6562.21 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 6516.00 | 6555.55 | 6558.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 6471.50 | 6538.74 | 6550.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 6445.00 | 6430.03 | 6468.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 6445.00 | 6430.03 | 6468.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 6469.00 | 6437.82 | 6468.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 6469.00 | 6437.82 | 6468.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 6450.50 | 6440.36 | 6467.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 6499.00 | 6440.36 | 6467.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 6483.00 | 6448.89 | 6468.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 6511.50 | 6448.89 | 6468.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 6548.50 | 6468.81 | 6475.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:00:00 | 6548.50 | 6468.81 | 6475.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 12:15:00 | 6535.50 | 6485.22 | 6482.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 09:15:00 | 6582.50 | 6520.78 | 6501.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 6549.00 | 6616.94 | 6574.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 10:15:00 | 6549.00 | 6616.94 | 6574.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 6549.00 | 6616.94 | 6574.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:00:00 | 6549.00 | 6616.94 | 6574.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 6525.00 | 6598.55 | 6569.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 6525.00 | 6598.55 | 6569.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 6444.00 | 6567.64 | 6558.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 6444.00 | 6567.64 | 6558.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 13:15:00 | 6462.00 | 6546.51 | 6549.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 6419.50 | 6503.27 | 6528.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 6486.00 | 6483.93 | 6514.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 11:00:00 | 6486.00 | 6483.93 | 6514.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 6511.50 | 6482.79 | 6503.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 6511.50 | 6482.79 | 6503.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 6519.00 | 6490.03 | 6504.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:30:00 | 6575.00 | 6504.32 | 6509.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 6586.50 | 6520.76 | 6516.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 6710.50 | 6558.71 | 6534.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 6790.00 | 6792.82 | 6735.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:00:00 | 6790.00 | 6792.82 | 6735.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 6743.00 | 6783.39 | 6745.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:00:00 | 6743.00 | 6783.39 | 6745.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 6772.00 | 6781.11 | 6748.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 6916.50 | 6783.91 | 6755.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 12:15:00 | 6815.00 | 6842.18 | 6816.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 14:00:00 | 6830.00 | 6831.88 | 6815.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 6778.50 | 6803.08 | 6804.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 6778.50 | 6803.08 | 6804.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 6778.50 | 6803.08 | 6804.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 6778.50 | 6803.08 | 6804.90 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 10:15:00 | 6847.00 | 6811.86 | 6808.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 11:15:00 | 6883.50 | 6826.19 | 6815.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 7311.50 | 7317.86 | 7219.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 12:00:00 | 7311.50 | 7317.86 | 7219.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 7452.00 | 7347.92 | 7270.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 10:15:00 | 7480.00 | 7408.39 | 7342.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 10:45:00 | 7478.00 | 7416.31 | 7352.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:15:00 | 7472.00 | 7416.31 | 7352.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:45:00 | 7482.00 | 7428.25 | 7363.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 7470.00 | 7673.41 | 7616.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 7470.00 | 7673.41 | 7616.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 7471.00 | 7632.93 | 7602.87 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-11 12:15:00 | 7399.00 | 7555.27 | 7570.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 12:15:00 | 7399.00 | 7555.27 | 7570.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 12:15:00 | 7399.00 | 7555.27 | 7570.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 12:15:00 | 7399.00 | 7555.27 | 7570.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 7399.00 | 7555.27 | 7570.88 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 7620.00 | 7557.59 | 7555.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 15:15:00 | 7644.00 | 7574.87 | 7563.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 7636.00 | 7748.91 | 7686.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 7636.00 | 7748.91 | 7686.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 7636.00 | 7748.91 | 7686.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:00:00 | 7636.00 | 7748.91 | 7686.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 7644.50 | 7728.03 | 7682.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 7644.50 | 7728.03 | 7682.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 7602.00 | 7702.82 | 7675.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:30:00 | 7595.00 | 7702.82 | 7675.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 15:15:00 | 7590.50 | 7651.30 | 7657.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 09:15:00 | 7570.50 | 7635.14 | 7649.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 09:15:00 | 7646.00 | 7612.49 | 7626.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 09:15:00 | 7646.00 | 7612.49 | 7626.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 7646.00 | 7612.49 | 7626.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:45:00 | 7680.00 | 7612.49 | 7626.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 7644.50 | 7618.89 | 7628.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:00:00 | 7644.50 | 7618.89 | 7628.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 7612.00 | 7617.51 | 7626.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:30:00 | 7637.50 | 7617.51 | 7626.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 7610.00 | 7600.76 | 7615.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:45:00 | 7624.50 | 7600.76 | 7615.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 7488.50 | 7492.19 | 7536.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 7530.50 | 7492.19 | 7536.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 7358.00 | 7311.88 | 7356.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:45:00 | 7342.50 | 7311.88 | 7356.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 7378.00 | 7325.11 | 7358.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 10:45:00 | 7306.00 | 7330.53 | 7353.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 11:15:00 | 7444.00 | 7347.16 | 7340.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 11:15:00 | 7444.00 | 7347.16 | 7340.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 12:15:00 | 7645.50 | 7464.02 | 7413.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 7850.00 | 7924.47 | 7767.71 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 10:45:00 | 7969.00 | 7930.97 | 7784.92 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 11:30:00 | 7993.50 | 7940.48 | 7802.52 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 14:30:00 | 7975.00 | 7955.31 | 7844.66 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 09:15:00 | 7979.50 | 7956.15 | 7855.10 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 7903.50 | 7978.14 | 7910.16 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 7903.50 | 7978.14 | 7910.16 | SL hit (close<ema400) qty=1.00 sl=7910.16 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 7903.50 | 7978.14 | 7910.16 | SL hit (close<ema400) qty=1.00 sl=7910.16 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 7903.50 | 7978.14 | 7910.16 | SL hit (close<ema400) qty=1.00 sl=7910.16 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 7903.50 | 7978.14 | 7910.16 | SL hit (close<ema400) qty=1.00 sl=7910.16 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-08-01 14:00:00 | 7903.50 | 7978.14 | 7910.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 7858.50 | 7954.21 | 7905.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 7858.50 | 7954.21 | 7905.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 7855.00 | 7934.37 | 7900.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 7880.00 | 7934.37 | 7900.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 7881.00 | 7923.70 | 7899.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 7849.00 | 7923.70 | 7899.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 7910.00 | 7920.96 | 7900.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 12:45:00 | 7950.00 | 7921.79 | 7903.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 13:30:00 | 7945.00 | 7940.83 | 7914.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 11:30:00 | 7946.00 | 7935.00 | 7921.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 12:15:00 | 7943.50 | 7935.00 | 7921.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 7930.00 | 7934.00 | 7922.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:00:00 | 7930.00 | 7934.00 | 7922.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 7845.00 | 7916.20 | 7915.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-05 13:15:00 | 7845.00 | 7916.20 | 7915.24 | SL hit (close<static) qty=1.00 sl=7872.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 13:15:00 | 7845.00 | 7916.20 | 7915.24 | SL hit (close<static) qty=1.00 sl=7872.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 13:15:00 | 7845.00 | 7916.20 | 7915.24 | SL hit (close<static) qty=1.00 sl=7872.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 13:15:00 | 7845.00 | 7916.20 | 7915.24 | SL hit (close<static) qty=1.00 sl=7872.50 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-08-05 14:00:00 | 7845.00 | 7916.20 | 7915.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 7910.00 | 7914.96 | 7914.76 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 15:15:00 | 7899.00 | 7911.77 | 7913.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 7672.00 | 7863.82 | 7891.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 7705.50 | 7669.79 | 7724.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 7705.50 | 7669.79 | 7724.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 7765.00 | 7688.83 | 7727.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 7677.00 | 7688.83 | 7727.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 7628.50 | 7676.76 | 7718.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:00:00 | 7606.00 | 7655.45 | 7701.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:15:00 | 7610.00 | 7647.06 | 7693.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-11 09:15:00 | 6845.40 | 7434.59 | 7573.23 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-11 09:15:00 | 6849.00 | 7434.59 | 7573.23 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 7420.00 | 7010.18 | 7001.35 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 7296.00 | 7331.35 | 7333.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 7251.00 | 7315.28 | 7325.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 7310.00 | 7305.22 | 7318.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 7310.00 | 7305.22 | 7318.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 7310.00 | 7305.22 | 7318.97 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 15:15:00 | 7377.50 | 7333.53 | 7328.12 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 7270.00 | 7317.45 | 7321.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 7197.00 | 7293.36 | 7310.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 10:15:00 | 7340.00 | 7280.13 | 7295.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 10:15:00 | 7340.00 | 7280.13 | 7295.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 7340.00 | 7280.13 | 7295.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 7340.00 | 7280.13 | 7295.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 7308.00 | 7285.71 | 7297.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:15:00 | 7293.00 | 7285.71 | 7297.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 10:15:00 | 7295.00 | 7279.55 | 7289.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 11:15:00 | 7321.50 | 7295.85 | 7295.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 11:15:00 | 7321.50 | 7295.85 | 7295.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 11:15:00 | 7321.50 | 7295.85 | 7295.48 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 14:15:00 | 7271.50 | 7296.47 | 7296.51 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 7425.00 | 7318.42 | 7306.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 7573.00 | 7369.33 | 7330.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 7604.00 | 7634.20 | 7571.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 7604.00 | 7634.20 | 7571.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 7530.00 | 7717.26 | 7698.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 7530.00 | 7717.26 | 7698.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 10:15:00 | 7529.50 | 7679.71 | 7682.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 7458.00 | 7563.53 | 7620.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 10:15:00 | 7540.50 | 7528.85 | 7587.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 10:45:00 | 7565.50 | 7528.85 | 7587.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 7640.50 | 7551.18 | 7592.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:45:00 | 7640.50 | 7551.18 | 7592.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 7690.00 | 7578.95 | 7601.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:00:00 | 7690.00 | 7578.95 | 7601.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 13:15:00 | 7781.50 | 7619.46 | 7617.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 7823.00 | 7660.17 | 7636.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 15:15:00 | 7917.50 | 7918.71 | 7872.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 09:15:00 | 8007.00 | 7918.71 | 7872.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 8205.00 | 8241.23 | 8190.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:30:00 | 8215.00 | 8241.23 | 8190.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 8199.00 | 8232.78 | 8190.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:45:00 | 8199.00 | 8232.78 | 8190.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 8199.00 | 8226.02 | 8191.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 8199.00 | 8226.02 | 8191.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 8200.00 | 8220.82 | 8192.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:15:00 | 8265.00 | 8220.82 | 8192.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 11:15:00 | 8193.00 | 8328.97 | 8347.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 8193.00 | 8328.97 | 8347.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 09:15:00 | 8017.00 | 8186.73 | 8263.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 12:15:00 | 8135.50 | 8131.79 | 8215.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 13:00:00 | 8135.50 | 8131.79 | 8215.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 8270.00 | 8159.44 | 8220.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:45:00 | 8275.00 | 8159.44 | 8220.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 8230.00 | 8173.55 | 8221.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 8244.00 | 8173.55 | 8221.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 8200.00 | 8178.84 | 8219.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 8122.50 | 8178.84 | 8219.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:45:00 | 8165.00 | 8181.64 | 8213.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:00:00 | 8187.00 | 8150.70 | 8173.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 8232.00 | 8192.60 | 8187.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 8232.00 | 8192.60 | 8187.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 8232.00 | 8192.60 | 8187.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 8232.00 | 8192.60 | 8187.57 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 09:15:00 | 8136.00 | 8193.90 | 8194.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 10:15:00 | 8110.50 | 8177.22 | 8187.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 14:15:00 | 8175.00 | 8166.07 | 8177.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 14:15:00 | 8175.00 | 8166.07 | 8177.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 8175.00 | 8166.07 | 8177.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:45:00 | 8175.50 | 8166.07 | 8177.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 8175.00 | 8167.85 | 8177.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:15:00 | 8210.00 | 8167.85 | 8177.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 09:15:00 | 8255.00 | 8185.28 | 8184.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 10:15:00 | 8324.00 | 8213.03 | 8196.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 8337.00 | 8347.86 | 8285.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 09:30:00 | 8359.00 | 8347.86 | 8285.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 8274.00 | 8333.09 | 8284.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 8274.00 | 8333.09 | 8284.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 8330.00 | 8332.47 | 8288.56 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 8120.50 | 8249.29 | 8264.27 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 8329.00 | 8269.46 | 8262.00 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 14:15:00 | 8186.50 | 8273.97 | 8284.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 15:15:00 | 8170.00 | 8253.18 | 8274.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 8268.50 | 8256.24 | 8273.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 8268.50 | 8256.24 | 8273.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 8268.50 | 8256.24 | 8273.82 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 8355.50 | 8266.19 | 8260.72 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 8255.00 | 8270.34 | 8271.15 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 8278.00 | 8271.87 | 8271.78 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 12:15:00 | 8263.00 | 8270.10 | 8270.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-21 14:15:00 | 8223.50 | 8251.02 | 8260.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 12:15:00 | 8308.00 | 8252.32 | 8256.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 12:15:00 | 8308.00 | 8252.32 | 8256.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 8308.00 | 8252.32 | 8256.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:45:00 | 8297.50 | 8252.32 | 8256.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 13:15:00 | 8312.00 | 8264.26 | 8261.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 8387.00 | 8298.52 | 8278.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 8393.00 | 8421.31 | 8375.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 8393.00 | 8421.31 | 8375.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 8393.00 | 8421.31 | 8375.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 15:00:00 | 8536.50 | 8444.25 | 8403.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 14:15:00 | 8308.50 | 8421.58 | 8418.66 | SL hit (close<static) qty=1.00 sl=8353.50 alert=retest2 |

### Cycle 42 — SELL (started 2025-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 15:15:00 | 8310.00 | 8399.26 | 8408.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 8142.50 | 8347.91 | 8384.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 09:15:00 | 7974.00 | 7877.36 | 7950.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 7974.00 | 7877.36 | 7950.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 7974.00 | 7877.36 | 7950.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 7974.00 | 7877.36 | 7950.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 7950.50 | 7891.99 | 7950.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:00:00 | 7950.50 | 7891.99 | 7950.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 7803.00 | 7874.19 | 7937.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:30:00 | 7959.00 | 7874.19 | 7937.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 7268.00 | 7160.25 | 7265.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:30:00 | 7281.00 | 7160.25 | 7265.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 7195.00 | 7167.20 | 7258.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 14:45:00 | 7147.00 | 7161.96 | 7248.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 09:30:00 | 7158.00 | 7153.45 | 7229.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 13:30:00 | 7175.00 | 7158.08 | 7207.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 10:00:00 | 7168.00 | 7170.47 | 7201.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 7254.00 | 7170.05 | 7183.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:45:00 | 7295.00 | 7170.05 | 7183.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 7294.00 | 7194.84 | 7193.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 7294.00 | 7194.84 | 7193.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 7294.00 | 7194.84 | 7193.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 7294.00 | 7194.84 | 7193.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 10:15:00 | 7294.00 | 7194.84 | 7193.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 11:15:00 | 7327.50 | 7221.37 | 7205.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 7360.50 | 7403.87 | 7347.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 7360.50 | 7403.87 | 7347.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 7360.50 | 7403.87 | 7347.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 7360.50 | 7403.87 | 7347.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 7356.00 | 7394.29 | 7348.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 11:00:00 | 7356.00 | 7394.29 | 7348.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 7361.00 | 7387.64 | 7349.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:15:00 | 7348.50 | 7387.64 | 7349.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 7355.50 | 7381.21 | 7350.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 14:00:00 | 7374.00 | 7379.77 | 7352.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 7382.50 | 7369.71 | 7352.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:30:00 | 7370.00 | 7367.61 | 7354.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:15:00 | 7376.00 | 7366.69 | 7354.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 7354.50 | 7382.15 | 7369.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 7354.50 | 7382.15 | 7369.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 7356.50 | 7377.02 | 7367.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 7356.50 | 7377.02 | 7367.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 7345.00 | 7370.61 | 7365.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:00:00 | 7345.00 | 7370.61 | 7365.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-20 12:15:00 | 7288.50 | 7354.19 | 7358.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 12:15:00 | 7288.50 | 7354.19 | 7358.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 12:15:00 | 7288.50 | 7354.19 | 7358.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 12:15:00 | 7288.50 | 7354.19 | 7358.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 12:15:00 | 7288.50 | 7354.19 | 7358.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 13:15:00 | 7264.00 | 7336.15 | 7350.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 12:15:00 | 7088.00 | 7081.19 | 7141.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 12:45:00 | 7076.00 | 7081.19 | 7141.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 7146.50 | 7099.50 | 7139.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:45:00 | 7198.50 | 7099.50 | 7139.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 7140.00 | 7107.60 | 7139.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 7170.50 | 7107.60 | 7139.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 7172.00 | 7120.48 | 7142.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 7181.50 | 7120.48 | 7142.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 7182.00 | 7132.78 | 7145.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 7182.00 | 7132.78 | 7145.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 7157.50 | 7137.73 | 7146.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:30:00 | 7155.00 | 7137.73 | 7146.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 7202.00 | 7150.58 | 7151.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:00:00 | 7202.00 | 7150.58 | 7151.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 7291.00 | 7178.67 | 7164.62 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 7106.00 | 7160.21 | 7167.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 7068.50 | 7127.19 | 7149.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 14:15:00 | 7164.00 | 7107.38 | 7127.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 14:15:00 | 7164.00 | 7107.38 | 7127.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 7164.00 | 7107.38 | 7127.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 7164.00 | 7107.38 | 7127.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 7220.00 | 7129.90 | 7136.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 7133.50 | 7129.90 | 7136.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 7062.00 | 7076.90 | 7098.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 12:15:00 | 7027.00 | 7073.12 | 7092.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 14:15:00 | 7034.00 | 7041.24 | 7057.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 11:30:00 | 7036.00 | 7027.97 | 7044.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 6675.65 | 6826.07 | 6932.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 6682.30 | 6826.07 | 6932.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 6684.20 | 6826.07 | 6932.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 6503.00 | 6482.89 | 6581.09 | SL hit (close>ema200) qty=0.50 sl=6482.89 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 6503.00 | 6482.89 | 6581.09 | SL hit (close>ema200) qty=0.50 sl=6482.89 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 6503.00 | 6482.89 | 6581.09 | SL hit (close>ema200) qty=0.50 sl=6482.89 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 6720.50 | 6631.16 | 6620.66 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 14:15:00 | 6560.00 | 6619.00 | 6620.04 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 6625.00 | 6590.92 | 6586.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 6721.50 | 6617.04 | 6599.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 13:15:00 | 6767.50 | 6772.97 | 6722.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 14:00:00 | 6767.50 | 6772.97 | 6722.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 6725.50 | 6764.81 | 6731.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 6725.50 | 6764.81 | 6731.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 6715.00 | 6754.85 | 6729.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:30:00 | 6717.00 | 6754.85 | 6729.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 6648.00 | 6733.48 | 6722.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:00:00 | 6648.00 | 6733.48 | 6722.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 6583.50 | 6703.48 | 6709.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 13:15:00 | 6556.50 | 6600.80 | 6641.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 6621.50 | 6601.48 | 6631.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 6621.50 | 6601.48 | 6631.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 6621.50 | 6601.48 | 6631.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 6656.50 | 6601.48 | 6631.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 6635.00 | 6608.18 | 6631.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 6635.00 | 6608.18 | 6631.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 6645.50 | 6615.65 | 6632.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 6645.50 | 6615.65 | 6632.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 6626.50 | 6617.82 | 6632.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 6682.50 | 6617.82 | 6632.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 6651.00 | 6624.45 | 6633.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 6651.00 | 6624.45 | 6633.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 6678.00 | 6635.16 | 6637.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 6678.00 | 6635.16 | 6637.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 6685.00 | 6645.13 | 6642.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 6715.00 | 6659.10 | 6648.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 6692.00 | 6698.29 | 6677.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 6663.00 | 6698.29 | 6677.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 6698.50 | 6698.33 | 6679.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 6636.00 | 6698.33 | 6679.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 6697.50 | 6697.39 | 6682.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 6675.00 | 6697.39 | 6682.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 6659.50 | 6689.81 | 6679.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 6659.50 | 6689.81 | 6679.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 6678.50 | 6687.55 | 6679.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 6755.00 | 6675.73 | 6675.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 14:45:00 | 6689.50 | 6696.84 | 6689.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 6724.50 | 6695.27 | 6689.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 10:15:00 | 6668.50 | 6686.03 | 6686.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-26 10:15:00 | 6668.50 | 6686.03 | 6686.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-26 10:15:00 | 6668.50 | 6686.03 | 6686.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 6668.50 | 6686.03 | 6686.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 15:15:00 | 6629.00 | 6668.22 | 6677.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 12:15:00 | 6358.00 | 6343.14 | 6428.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 12:45:00 | 6353.50 | 6343.14 | 6428.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 6392.50 | 6358.83 | 6421.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 6392.50 | 6358.83 | 6421.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 6401.50 | 6371.55 | 6416.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:45:00 | 6433.50 | 6371.55 | 6416.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 6412.00 | 6379.64 | 6415.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:45:00 | 6410.00 | 6379.64 | 6415.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 6390.00 | 6381.71 | 6413.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:30:00 | 6411.00 | 6381.71 | 6413.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 6408.50 | 6389.76 | 6411.87 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 6514.00 | 6430.52 | 6426.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 6668.50 | 6509.65 | 6471.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 6642.50 | 6659.26 | 6590.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 11:00:00 | 6642.50 | 6659.26 | 6590.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 6650.00 | 6666.86 | 6640.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:45:00 | 6667.50 | 6665.89 | 6642.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:45:00 | 6664.00 | 6658.53 | 6643.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 6569.50 | 6640.72 | 6636.32 | SL hit (close<static) qty=1.00 sl=6620.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 6569.50 | 6640.72 | 6636.32 | SL hit (close<static) qty=1.00 sl=6620.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 6552.00 | 6622.98 | 6628.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 6454.00 | 6543.84 | 6583.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 6154.50 | 6127.68 | 6223.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 6154.50 | 6127.68 | 6223.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 6223.00 | 6141.20 | 6173.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:45:00 | 6135.00 | 6159.11 | 6173.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 6129.00 | 6152.13 | 6167.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 12:00:00 | 6135.00 | 6159.79 | 6168.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 13:15:00 | 5828.25 | 5933.71 | 6017.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 13:15:00 | 5822.55 | 5933.71 | 6017.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 13:15:00 | 5828.25 | 5933.71 | 6017.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-27 09:15:00 | 5521.50 | 5595.60 | 5709.41 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-27 09:15:00 | 5516.10 | 5595.60 | 5709.41 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-27 09:15:00 | 5521.50 | 5595.60 | 5709.41 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 5832.00 | 5593.38 | 5582.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 5882.00 | 5738.42 | 5676.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 10:15:00 | 5831.50 | 5881.83 | 5798.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 11:00:00 | 5831.50 | 5881.83 | 5798.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 7873.00 | 7781.31 | 7716.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 10:15:00 | 7898.00 | 7781.31 | 7716.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 13:15:00 | 7880.00 | 7805.72 | 7773.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 15:15:00 | 7698.00 | 7776.94 | 7787.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-19 15:15:00 | 7698.00 | 7776.94 | 7787.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 7698.00 | 7776.94 | 7787.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 12:15:00 | 7628.50 | 7716.64 | 7753.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 7807.00 | 7711.64 | 7736.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 7807.00 | 7711.64 | 7736.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 7807.00 | 7711.64 | 7736.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 7846.00 | 7711.64 | 7736.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 7845.00 | 7738.31 | 7746.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 7845.00 | 7738.31 | 7746.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 7823.00 | 7755.25 | 7753.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 11:15:00 | 7850.00 | 7796.64 | 7776.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 7997.00 | 7997.37 | 7933.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 10:45:00 | 7984.00 | 7997.37 | 7933.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 7966.50 | 7991.19 | 7936.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:30:00 | 7957.00 | 7991.19 | 7936.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 7966.50 | 7980.14 | 7940.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 8041.00 | 7961.25 | 7938.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 10:30:00 | 7981.50 | 7966.16 | 7944.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:00:00 | 7996.00 | 7966.16 | 7944.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:30:00 | 7986.00 | 7977.42 | 7955.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 7972.50 | 7976.43 | 7957.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 7972.50 | 7976.43 | 7957.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 7930.00 | 7966.12 | 7955.72 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 7930.00 | 7966.12 | 7955.72 | SL hit (close<static) qty=1.00 sl=7936.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 7930.00 | 7966.12 | 7955.72 | SL hit (close<static) qty=1.00 sl=7936.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 7930.00 | 7966.12 | 7955.72 | SL hit (close<static) qty=1.00 sl=7936.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 7930.00 | 7966.12 | 7955.72 | SL hit (close<static) qty=1.00 sl=7936.50 alert=retest2 |

### Cycle 58 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 7851.00 | 7943.09 | 7946.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 7791.00 | 7912.68 | 7932.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 7897.00 | 7881.51 | 7909.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 7897.00 | 7881.51 | 7909.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 7890.00 | 7883.20 | 7908.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 7647.50 | 7883.20 | 7908.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:45:00 | 7848.50 | 7732.15 | 7740.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 11:15:00 | 7849.00 | 7764.94 | 7754.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-06 11:15:00 | 7849.00 | 7764.94 | 7754.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 7849.00 | 7764.94 | 7754.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 12:15:00 | 7926.50 | 7797.25 | 7770.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 7650.00 | 7801.40 | 7785.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 7650.00 | 7801.40 | 7785.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 7650.00 | 7801.40 | 7785.13 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 7312.50 | 7687.40 | 7735.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-10 09:15:00 | 7238.50 | 7444.29 | 7583.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 13:15:00 | 7445.00 | 7428.28 | 7529.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 13:45:00 | 7441.50 | 7428.28 | 7529.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 7505.00 | 7443.63 | 7527.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:45:00 | 7472.00 | 7443.63 | 7527.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 7530.00 | 7460.90 | 7527.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 09:15:00 | 7434.50 | 7460.90 | 7527.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 7062.77 | 7295.77 | 7404.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-13 09:15:00 | 6691.05 | 6980.30 | 7174.21 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 6867.50 | 6717.61 | 6714.24 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 6677.00 | 6743.52 | 6750.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 6369.50 | 6633.18 | 6685.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 6325.00 | 6304.98 | 6437.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 6325.00 | 6304.98 | 6437.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 6465.00 | 6336.99 | 6439.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 6479.50 | 6336.99 | 6439.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 6485.50 | 6366.69 | 6444.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 6498.50 | 6366.69 | 6444.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 6450.00 | 6398.28 | 6445.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 6708.50 | 6398.28 | 6445.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 6738.00 | 6466.22 | 6472.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 6738.00 | 6466.22 | 6472.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 6750.00 | 6522.98 | 6497.73 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 6503.00 | 6573.15 | 6579.17 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 12:15:00 | 6622.50 | 6583.80 | 6583.01 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 6575.50 | 6582.14 | 6582.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 15:15:00 | 6535.00 | 6568.05 | 6575.60 | Break + close below crossover candle low |

### Cycle 67 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 6711.00 | 6596.64 | 6587.91 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 6247.50 | 6528.39 | 6564.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 09:15:00 | 6219.00 | 6321.22 | 6421.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 11:15:00 | 6324.50 | 6316.08 | 6401.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 11:45:00 | 6345.00 | 6316.08 | 6401.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 6398.00 | 6341.87 | 6392.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 15:00:00 | 6398.00 | 6341.87 | 6392.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 15:15:00 | 6430.00 | 6359.49 | 6395.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:15:00 | 6348.50 | 6359.49 | 6395.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 11:00:00 | 6380.00 | 6364.95 | 6392.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 15:15:00 | 6435.00 | 6406.27 | 6404.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-07 15:15:00 | 6435.00 | 6406.27 | 6404.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 15:15:00 | 6435.00 | 6406.27 | 6404.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 6793.00 | 6483.61 | 6439.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 6789.50 | 6829.10 | 6681.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 6789.50 | 6829.10 | 6681.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 7863.50 | 7924.67 | 7863.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:45:00 | 7865.00 | 7924.67 | 7863.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 7843.50 | 7908.43 | 7862.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 7890.00 | 7908.43 | 7862.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 7848.00 | 7896.35 | 7860.78 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 15:15:00 | 7803.00 | 7852.30 | 7852.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 09:15:00 | 7758.50 | 7833.54 | 7844.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 09:15:00 | 7834.50 | 7811.73 | 7824.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 09:15:00 | 7834.50 | 7811.73 | 7824.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 7834.50 | 7811.73 | 7824.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 10:45:00 | 7782.00 | 7802.68 | 7819.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 11:45:00 | 7785.00 | 7806.95 | 7820.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 12:30:00 | 7784.50 | 7800.06 | 7815.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 13:00:00 | 7772.50 | 7800.06 | 7815.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 7972.00 | 7816.36 | 7815.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 7972.00 | 7816.36 | 7815.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 7972.00 | 7816.36 | 7815.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 7972.00 | 7816.36 | 7815.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 7972.00 | 7816.36 | 7815.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 11:15:00 | 8171.00 | 7914.91 | 7862.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 11:15:00 | 8138.00 | 8140.72 | 8070.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 11:45:00 | 8130.50 | 8140.72 | 8070.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 8081.00 | 8127.62 | 8076.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 8081.00 | 8127.62 | 8076.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 8077.00 | 8117.50 | 8076.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:30:00 | 8096.50 | 8117.50 | 8076.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 8081.00 | 8110.20 | 8077.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 8018.00 | 8110.20 | 8077.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 7878.50 | 8063.86 | 8059.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 7878.50 | 8063.86 | 8059.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 7939.00 | 8038.89 | 8048.14 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 09:15:00 | 8122.50 | 8032.03 | 8027.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 12:15:00 | 8245.00 | 8090.26 | 8056.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 13:15:00 | 8726.00 | 8768.51 | 8638.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:45:00 | 8721.50 | 8768.51 | 8638.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 10:15:00 | 6284.50 | 2025-05-19 12:15:00 | 6286.50 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-05-19 11:00:00 | 6285.00 | 2025-05-19 12:15:00 | 6286.50 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-05-19 11:45:00 | 6288.50 | 2025-05-19 12:15:00 | 6286.50 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-05-23 09:15:00 | 6605.50 | 2025-05-23 13:15:00 | 6408.50 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2025-05-23 11:15:00 | 6590.50 | 2025-05-23 13:15:00 | 6408.50 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-06-05 14:15:00 | 6304.50 | 2025-06-06 14:15:00 | 6403.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-06-06 11:15:00 | 6304.50 | 2025-06-06 14:15:00 | 6403.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-06-06 12:30:00 | 6306.00 | 2025-06-06 14:15:00 | 6403.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-06-27 09:15:00 | 6916.50 | 2025-07-01 09:15:00 | 6778.50 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-06-30 12:15:00 | 6815.00 | 2025-07-01 09:15:00 | 6778.50 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-06-30 14:00:00 | 6830.00 | 2025-07-01 09:15:00 | 6778.50 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-07-08 10:15:00 | 7480.00 | 2025-07-11 12:15:00 | 7399.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-07-08 10:45:00 | 7478.00 | 2025-07-11 12:15:00 | 7399.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-07-08 11:15:00 | 7472.00 | 2025-07-11 12:15:00 | 7399.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-07-08 11:45:00 | 7482.00 | 2025-07-11 12:15:00 | 7399.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-07-25 10:45:00 | 7306.00 | 2025-07-28 11:15:00 | 7444.00 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest1 | 2025-07-31 10:45:00 | 7969.00 | 2025-08-01 13:15:00 | 7903.50 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest1 | 2025-07-31 11:30:00 | 7993.50 | 2025-08-01 13:15:00 | 7903.50 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest1 | 2025-07-31 14:30:00 | 7975.00 | 2025-08-01 13:15:00 | 7903.50 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest1 | 2025-08-01 09:15:00 | 7979.50 | 2025-08-01 13:15:00 | 7903.50 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-08-04 12:45:00 | 7950.00 | 2025-08-05 13:15:00 | 7845.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-08-04 13:30:00 | 7945.00 | 2025-08-05 13:15:00 | 7845.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-08-05 11:30:00 | 7946.00 | 2025-08-05 13:15:00 | 7845.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-08-05 12:15:00 | 7943.50 | 2025-08-05 13:15:00 | 7845.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-08-08 12:00:00 | 7606.00 | 2025-08-11 09:15:00 | 6845.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-08 13:15:00 | 7610.00 | 2025-08-11 09:15:00 | 6849.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-28 12:15:00 | 7293.00 | 2025-08-29 11:15:00 | 7321.50 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-08-29 10:15:00 | 7295.00 | 2025-08-29 11:15:00 | 7321.50 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-09-18 14:15:00 | 8265.00 | 2025-09-26 11:15:00 | 8193.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-09-30 09:15:00 | 8122.50 | 2025-10-01 15:15:00 | 8232.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-09-30 10:45:00 | 8165.00 | 2025-10-01 15:15:00 | 8232.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-10-01 12:00:00 | 8187.00 | 2025-10-01 15:15:00 | 8232.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-10-28 15:00:00 | 8536.50 | 2025-10-29 14:15:00 | 8308.50 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-11-11 14:45:00 | 7147.00 | 2025-11-14 10:15:00 | 7294.00 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-11-12 09:30:00 | 7158.00 | 2025-11-14 10:15:00 | 7294.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-11-12 13:30:00 | 7175.00 | 2025-11-14 10:15:00 | 7294.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-11-13 10:00:00 | 7168.00 | 2025-11-14 10:15:00 | 7294.00 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-11-18 14:00:00 | 7374.00 | 2025-11-20 12:15:00 | 7288.50 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-11-19 09:15:00 | 7382.50 | 2025-11-20 12:15:00 | 7288.50 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-11-19 10:30:00 | 7370.00 | 2025-11-20 12:15:00 | 7288.50 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-11-19 12:15:00 | 7376.00 | 2025-11-20 12:15:00 | 7288.50 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-12-02 12:15:00 | 7027.00 | 2025-12-05 09:15:00 | 6675.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 14:15:00 | 7034.00 | 2025-12-05 09:15:00 | 6682.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 11:30:00 | 7036.00 | 2025-12-05 09:15:00 | 6684.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 12:15:00 | 7027.00 | 2025-12-09 11:15:00 | 6503.00 | STOP_HIT | 0.50 | 7.46% |
| SELL | retest2 | 2025-12-03 14:15:00 | 7034.00 | 2025-12-09 11:15:00 | 6503.00 | STOP_HIT | 0.50 | 7.55% |
| SELL | retest2 | 2025-12-04 11:30:00 | 7036.00 | 2025-12-09 11:15:00 | 6503.00 | STOP_HIT | 0.50 | 7.58% |
| BUY | retest2 | 2025-12-24 09:15:00 | 6755.00 | 2025-12-26 10:15:00 | 6668.50 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-12-24 14:45:00 | 6689.50 | 2025-12-26 10:15:00 | 6668.50 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-12-26 09:15:00 | 6724.50 | 2025-12-26 10:15:00 | 6668.50 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2026-01-07 14:45:00 | 6667.50 | 2026-01-08 10:15:00 | 6569.50 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-01-08 09:45:00 | 6664.00 | 2026-01-08 10:15:00 | 6569.50 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-01-16 13:45:00 | 6135.00 | 2026-01-21 13:15:00 | 5828.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 6129.00 | 2026-01-21 13:15:00 | 5822.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 12:00:00 | 6135.00 | 2026-01-21 13:15:00 | 5828.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 13:45:00 | 6135.00 | 2026-01-27 09:15:00 | 5521.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 6129.00 | 2026-01-27 09:15:00 | 5516.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-19 12:00:00 | 6135.00 | 2026-01-27 09:15:00 | 5521.50 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-17 10:15:00 | 7898.00 | 2026-02-19 15:15:00 | 7698.00 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2026-02-18 13:15:00 | 7880.00 | 2026-02-19 15:15:00 | 7698.00 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2026-02-27 09:15:00 | 8041.00 | 2026-03-02 09:15:00 | 7930.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-02-27 10:30:00 | 7981.50 | 2026-03-02 09:15:00 | 7930.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2026-02-27 11:00:00 | 7996.00 | 2026-03-02 09:15:00 | 7930.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2026-02-27 13:30:00 | 7986.00 | 2026-03-02 09:15:00 | 7930.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2026-03-04 09:15:00 | 7647.50 | 2026-03-06 11:15:00 | 7849.00 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2026-03-06 09:45:00 | 7848.50 | 2026-03-06 11:15:00 | 7849.00 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2026-03-11 09:15:00 | 7434.50 | 2026-03-12 09:15:00 | 7062.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 09:15:00 | 7434.50 | 2026-03-13 09:15:00 | 6691.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-07 09:15:00 | 6348.50 | 2026-04-07 15:15:00 | 6435.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-04-07 11:00:00 | 6380.00 | 2026-04-07 15:15:00 | 6435.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-04-24 10:45:00 | 7782.00 | 2026-04-27 09:15:00 | 7972.00 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2026-04-24 11:45:00 | 7785.00 | 2026-04-27 09:15:00 | 7972.00 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2026-04-24 12:30:00 | 7784.50 | 2026-04-27 09:15:00 | 7972.00 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2026-04-24 13:00:00 | 7772.50 | 2026-04-27 09:15:00 | 7972.00 | STOP_HIT | 1.00 | -2.57% |
