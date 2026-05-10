# Tata Elxsi Ltd. (TATAELXSI)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 4313.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 11 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 8
- **Target hits / Stop hits / Partials:** 0 / 8 / 0
- **Avg / median % per leg:** -1.77% / -1.52%
- **Sum % (uncompounded):** -14.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.77% | -14.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.77% | -14.1% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.77% | -14.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 6234.00 | 5720.65 | 5718.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 6283.00 | 5830.94 | 5777.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 6295.00 | 6295.97 | 6111.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 13:00:00 | 6295.00 | 6295.97 | 6111.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 6162.00 | 6290.33 | 6152.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:00:00 | 6162.00 | 6290.33 | 6152.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 6148.00 | 6285.54 | 6153.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:45:00 | 6148.00 | 6285.54 | 6153.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 6096.50 | 6283.66 | 6153.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:00:00 | 6096.50 | 6283.66 | 6153.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 6189.00 | 6279.99 | 6153.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 6202.50 | 6278.01 | 6153.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 10:15:00 | 6207.50 | 6273.52 | 6156.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 12:15:00 | 6195.00 | 6272.33 | 6156.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 6122.00 | 6261.93 | 6158.36 | SL hit (close<static) qty=1.00 sl=6130.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 6122.00 | 6261.93 | 6158.36 | SL hit (close<static) qty=1.00 sl=6130.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 6122.00 | 6261.93 | 6158.36 | SL hit (close<static) qty=1.00 sl=6130.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 6350.00 | 6217.39 | 6150.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 6170.00 | 6234.50 | 6166.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 6170.00 | 6234.50 | 6166.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 6143.00 | 6233.59 | 6166.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:00:00 | 6143.00 | 6233.59 | 6166.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 6175.50 | 6233.01 | 6166.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 14:00:00 | 6198.50 | 6231.82 | 6166.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:15:00 | 6216.00 | 6230.20 | 6166.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 15:00:00 | 6200.00 | 6228.21 | 6167.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 10:30:00 | 6201.00 | 6227.35 | 6167.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 6201.00 | 6224.72 | 6172.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 6186.00 | 6224.72 | 6172.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 6161.00 | 6223.77 | 6172.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:00:00 | 6161.00 | 6223.77 | 6172.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 6104.50 | 6222.59 | 6172.09 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 6104.50 | 6222.59 | 6172.09 | SL hit (close<static) qty=1.00 sl=6130.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 6104.50 | 6222.59 | 6172.09 | SL hit (close<static) qty=1.00 sl=6141.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 6104.50 | 6222.59 | 6172.09 | SL hit (close<static) qty=1.00 sl=6141.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 6104.50 | 6222.59 | 6172.09 | SL hit (close<static) qty=1.00 sl=6141.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 6104.50 | 6222.59 | 6172.09 | SL hit (close<static) qty=1.00 sl=6141.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-07-25 14:00:00 | 6104.50 | 6222.59 | 6172.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 6092.50 | 6196.33 | 6162.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:30:00 | 6090.00 | 6196.33 | 6162.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| CROSSOVER_SKIP | 2025-08-06 14:15:00 | 5828.00 | 6134.85 | 6135.62 | min_gap filter: gap=0.013% < 0.030% |
| TREND_RESET | 2025-08-06 14:15:00 | 5828.00 | 6134.85 | 6135.62 | EMA inversion without crossover edge (EMA200=6134.85 EMA400=6135.62) — end cycle |
| CROSSOVER_SKIP | 2026-01-12 10:15:00 | 5661.00 | 5346.01 | 5345.46 | min_gap filter: gap=0.010% < 0.030% |
| CROSSOVER_SKIP | 2026-02-12 09:15:00 | 5097.50 | 5373.91 | 5375.24 | min_gap filter: gap=0.026% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-03 09:15:00 | 6202.50 | 2025-07-08 09:15:00 | 6122.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-07-04 10:15:00 | 6207.50 | 2025-07-08 09:15:00 | 6122.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-07-04 12:15:00 | 6195.00 | 2025-07-08 09:15:00 | 6122.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-07-15 09:15:00 | 6350.00 | 2025-07-25 13:15:00 | 6104.50 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2025-07-18 14:00:00 | 6198.50 | 2025-07-25 13:15:00 | 6104.50 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-07-21 10:15:00 | 6216.00 | 2025-07-25 13:15:00 | 6104.50 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-07-21 15:00:00 | 6200.00 | 2025-07-25 13:15:00 | 6104.50 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-07-22 10:30:00 | 6201.00 | 2025-07-25 13:15:00 | 6104.50 | STOP_HIT | 1.00 | -1.56% |
