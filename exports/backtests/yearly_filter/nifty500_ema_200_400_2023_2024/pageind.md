# Page Industries Ltd. (PAGEIND)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 37365.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 56 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 56 |
| PARTIAL | 11 |
| TARGET_HIT | 10 |
| STOP_HIT | 46 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 67 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 24 / 43
- **Target hits / Stop hits / Partials:** 10 / 46 / 11
- **Avg / median % per leg:** 1.04% / -0.90%
- **Sum % (uncompounded):** 69.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 1 | 7.1% | 1 | 13 | 0 | -0.93% | -13.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 1 | 7.1% | 1 | 13 | 0 | -0.93% | -13.1% |
| SELL (all) | 53 | 23 | 43.4% | 9 | 33 | 11 | 1.56% | 82.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 53 | 23 | 43.4% | 9 | 33 | 11 | 1.56% | 82.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 67 | 24 | 35.8% | 10 | 46 | 11 | 1.04% | 69.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 10:15:00 | 41099.90 | 38667.61 | 38659.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 09:15:00 | 41536.40 | 39544.23 | 39217.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-13 13:15:00 | 40060.00 | 40070.92 | 39554.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-13 13:30:00 | 40074.00 | 40070.92 | 39554.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 13:15:00 | 39623.80 | 40132.84 | 39671.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 13:30:00 | 39674.10 | 40132.84 | 39671.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 14:15:00 | 39540.00 | 40126.94 | 39671.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 14:45:00 | 39505.40 | 40126.94 | 39671.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 39501.20 | 40107.12 | 39667.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-22 10:30:00 | 39442.10 | 40107.12 | 39667.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 12:15:00 | 39529.90 | 40096.13 | 39666.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-25 09:45:00 | 39740.40 | 40069.35 | 39661.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-25 12:30:00 | 39649.60 | 40054.12 | 39660.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-25 13:30:00 | 39652.70 | 40049.85 | 39659.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-26 09:15:00 | 39379.10 | 40030.49 | 39655.94 | SL hit (close<static) qty=1.00 sl=39384.80 alert=retest2 |

### Cycle 2 — SELL (started 2023-10-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 13:15:00 | 38510.00 | 39432.65 | 39437.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 10:15:00 | 38348.90 | 39399.14 | 39420.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 13:15:00 | 38261.20 | 38230.95 | 38676.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-10 14:00:00 | 38261.20 | 38230.95 | 38676.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 09:15:00 | 37899.80 | 37605.11 | 38008.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-15 09:30:00 | 37960.00 | 37605.11 | 38008.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 13:15:00 | 37900.00 | 37617.14 | 38006.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-15 13:30:00 | 37980.80 | 37617.14 | 38006.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 38187.50 | 37626.95 | 38005.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-18 10:00:00 | 38187.50 | 37626.95 | 38005.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 10:15:00 | 38450.00 | 37635.14 | 38007.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-18 11:00:00 | 38450.00 | 37635.14 | 38007.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 14:15:00 | 37994.10 | 37651.52 | 38008.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-18 15:15:00 | 37910.00 | 37651.52 | 38008.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-19 10:00:00 | 37871.60 | 37656.26 | 38007.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-19 11:15:00 | 37883.10 | 37659.28 | 38007.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-20 13:15:00 | 37840.80 | 37683.80 | 38004.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 37500.40 | 37681.98 | 38001.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-20 14:30:00 | 37260.30 | 37675.76 | 37997.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 09:30:00 | 37223.20 | 37636.70 | 37963.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-28 09:15:00 | 38252.00 | 37635.11 | 37929.31 | SL hit (close>static) qty=1.00 sl=38150.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 14:15:00 | 38135.45 | 35561.41 | 35559.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 14:15:00 | 38562.15 | 35900.95 | 35738.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 15:15:00 | 40301.00 | 40616.23 | 39452.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 09:15:00 | 39900.05 | 40616.23 | 39452.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 40441.60 | 41273.23 | 40430.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:00:00 | 40441.60 | 41273.23 | 40430.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 40570.00 | 41266.23 | 40431.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 12:15:00 | 40614.15 | 41266.23 | 40431.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 13:30:00 | 40612.20 | 41251.95 | 40432.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 14:15:00 | 40302.45 | 41242.51 | 40432.07 | SL hit (close<static) qty=1.00 sl=40420.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 15:15:00 | 42850.00 | 45907.27 | 45914.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 42316.90 | 45702.80 | 45810.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 10:15:00 | 41600.00 | 41531.62 | 42851.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:30:00 | 41572.60 | 41531.62 | 42851.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 12:15:00 | 42848.80 | 41634.28 | 42802.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 12:30:00 | 42782.60 | 41634.28 | 42802.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 13:15:00 | 42962.00 | 41647.49 | 42803.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 14:00:00 | 42962.00 | 41647.49 | 42803.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 43109.45 | 41662.04 | 42805.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 15:00:00 | 43109.45 | 41662.04 | 42805.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 43066.30 | 41744.85 | 42813.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:45:00 | 43002.55 | 41744.85 | 42813.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 43019.20 | 41757.53 | 42814.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 14:45:00 | 43028.95 | 41757.53 | 42814.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 43066.40 | 41780.68 | 42815.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:00:00 | 43066.40 | 41780.68 | 42815.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 42799.80 | 41790.82 | 42815.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 11:45:00 | 42751.10 | 41801.41 | 42815.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 12:45:00 | 42570.20 | 41809.35 | 42814.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 14:15:00 | 44109.80 | 41869.55 | 42800.95 | SL hit (close>static) qty=1.00 sl=43099.25 alert=retest2 |

### Cycle 5 — BUY (started 2025-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 13:15:00 | 45970.00 | 43164.18 | 43158.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 47015.00 | 44156.26 | 43722.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 46080.00 | 46086.16 | 45103.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 15:00:00 | 46080.00 | 46086.16 | 45103.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 45530.00 | 46122.03 | 45514.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:30:00 | 45585.00 | 46122.03 | 45514.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 45200.00 | 46112.85 | 45512.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:30:00 | 45270.00 | 46112.85 | 45512.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 44960.00 | 46101.38 | 45510.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 44960.00 | 46101.38 | 45510.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 45500.00 | 45861.92 | 45452.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 46510.00 | 45861.92 | 45452.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 15:00:00 | 45820.00 | 47202.29 | 46824.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 13:30:00 | 45695.00 | 47050.47 | 46770.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 15:00:00 | 45830.00 | 47038.33 | 46765.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 44110.00 | 46996.35 | 46747.21 | SL hit (close<static) qty=1.00 sl=45425.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 10:15:00 | 43840.00 | 46509.07 | 46516.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 11:15:00 | 43630.00 | 46480.42 | 46502.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 46215.00 | 45977.12 | 46221.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 46215.00 | 45977.12 | 46221.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 46215.00 | 45977.12 | 46221.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:45:00 | 46405.00 | 45977.12 | 46221.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 46190.00 | 45979.24 | 46220.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 46285.00 | 45979.24 | 46220.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 46190.00 | 45981.34 | 46220.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:15:00 | 46015.00 | 45981.34 | 46220.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 15:00:00 | 45955.00 | 45985.31 | 46219.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 10:45:00 | 45950.00 | 45980.33 | 46213.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 46540.00 | 45970.21 | 46201.14 | SL hit (close>static) qty=1.00 sl=46285.00 alert=retest2 |

### Cycle 7 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 10:15:00 | 38120.00 | 33873.65 | 33859.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 11:15:00 | 38230.00 | 33917.00 | 33881.54 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-09-25 09:45:00 | 39740.40 | 2023-09-26 09:15:00 | 39379.10 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2023-09-25 12:30:00 | 39649.60 | 2023-09-26 09:15:00 | 39379.10 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2023-09-25 13:30:00 | 39652.70 | 2023-09-26 09:15:00 | 39379.10 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2023-10-11 09:15:00 | 39667.80 | 2023-10-11 13:15:00 | 39324.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2023-10-12 11:30:00 | 39699.40 | 2023-10-13 10:15:00 | 39352.90 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2023-12-18 15:15:00 | 37910.00 | 2023-12-28 09:15:00 | 38252.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2023-12-19 10:00:00 | 37871.60 | 2023-12-28 09:15:00 | 38252.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2023-12-19 11:15:00 | 37883.10 | 2023-12-28 09:15:00 | 38252.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2023-12-20 13:15:00 | 37840.80 | 2023-12-28 09:15:00 | 38252.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2023-12-20 14:30:00 | 37260.30 | 2023-12-28 09:15:00 | 38252.00 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2023-12-22 09:30:00 | 37223.20 | 2023-12-28 09:15:00 | 38252.00 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2024-01-09 10:30:00 | 37047.40 | 2024-01-16 09:15:00 | 38600.10 | STOP_HIT | 1.00 | -4.19% |
| SELL | retest2 | 2024-01-15 11:30:00 | 37255.50 | 2024-01-16 09:15:00 | 38600.10 | STOP_HIT | 1.00 | -3.61% |
| SELL | retest2 | 2024-01-17 11:15:00 | 37746.80 | 2024-02-09 09:15:00 | 35892.04 | PARTIAL | 0.50 | 4.91% |
| SELL | retest2 | 2024-01-18 12:15:00 | 37781.10 | 2024-02-09 09:15:00 | 35869.53 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2024-01-19 10:45:00 | 37725.20 | 2024-02-12 11:15:00 | 35859.46 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2024-01-20 09:30:00 | 37757.40 | 2024-02-12 11:15:00 | 35838.94 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2024-01-30 11:00:00 | 37389.90 | 2024-02-27 14:15:00 | 35538.36 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2024-01-30 11:45:00 | 37388.60 | 2024-02-27 15:15:00 | 35520.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-30 14:00:00 | 37408.80 | 2024-02-27 15:15:00 | 35519.17 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2024-01-17 11:15:00 | 37746.80 | 2024-02-29 09:15:00 | 33972.12 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-01-18 12:15:00 | 37781.10 | 2024-02-29 09:15:00 | 34002.99 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-01-19 10:45:00 | 37725.20 | 2024-02-29 09:15:00 | 33981.66 | TARGET_HIT | 0.50 | 9.92% |
| SELL | retest2 | 2024-01-20 09:30:00 | 37757.40 | 2024-03-13 14:15:00 | 33952.68 | TARGET_HIT | 0.50 | 10.08% |
| SELL | retest2 | 2024-01-30 11:00:00 | 37389.90 | 2024-03-14 09:15:00 | 33650.91 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-01-30 11:45:00 | 37388.60 | 2024-03-14 09:15:00 | 33649.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-01-30 14:00:00 | 37408.80 | 2024-03-14 09:15:00 | 33667.92 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-04-12 09:15:00 | 36527.80 | 2024-04-23 12:15:00 | 35900.00 | STOP_HIT | 1.00 | 1.72% |
| SELL | retest2 | 2024-04-16 14:45:00 | 35505.95 | 2024-04-23 12:15:00 | 35900.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-04-18 09:30:00 | 35398.20 | 2024-04-23 12:15:00 | 35900.00 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-04-19 09:15:00 | 34945.30 | 2024-04-23 12:15:00 | 35900.00 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2024-04-22 12:45:00 | 35513.95 | 2024-04-30 15:15:00 | 34701.41 | PARTIAL | 0.50 | 2.29% |
| SELL | retest2 | 2024-04-22 12:45:00 | 35513.95 | 2024-05-14 09:15:00 | 35070.00 | STOP_HIT | 0.50 | 1.25% |
| SELL | retest2 | 2024-04-26 12:45:00 | 35367.25 | 2024-05-16 11:15:00 | 35520.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2024-04-30 10:45:00 | 35299.55 | 2024-05-17 11:15:00 | 35865.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-04-30 11:30:00 | 35292.00 | 2024-05-17 11:15:00 | 35865.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-04-30 13:45:00 | 35341.15 | 2024-05-17 11:15:00 | 35865.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-05-15 10:15:00 | 34924.00 | 2024-05-17 11:15:00 | 35865.00 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2024-05-21 13:00:00 | 35082.35 | 2024-05-22 11:15:00 | 35544.40 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-05-22 09:15:00 | 34974.35 | 2024-05-22 11:15:00 | 35544.40 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-05-24 09:45:00 | 35016.25 | 2024-05-24 10:15:00 | 35656.55 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2024-09-06 12:15:00 | 40614.15 | 2024-09-06 14:15:00 | 40302.45 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-09-06 13:30:00 | 40612.20 | 2024-09-06 14:15:00 | 40302.45 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-09-10 09:15:00 | 40689.20 | 2024-09-10 14:15:00 | 40363.45 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-09-10 10:15:00 | 40614.40 | 2024-09-10 14:15:00 | 40363.45 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2024-10-08 13:00:00 | 42025.05 | 2024-10-15 10:15:00 | 46227.56 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-26 11:45:00 | 42751.10 | 2025-03-27 14:15:00 | 44109.80 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-03-26 12:45:00 | 42570.20 | 2025-03-27 14:15:00 | 44109.80 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2025-03-27 15:15:00 | 41000.00 | 2025-04-02 15:15:00 | 43150.00 | STOP_HIT | 1.00 | -5.24% |
| SELL | retest2 | 2025-03-28 15:00:00 | 42699.95 | 2025-04-02 15:15:00 | 43150.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-04-01 11:00:00 | 42379.95 | 2025-04-02 15:15:00 | 43150.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-04-02 11:45:00 | 42449.30 | 2025-04-02 15:15:00 | 43150.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-04-04 11:15:00 | 42346.00 | 2025-04-11 13:15:00 | 43889.95 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2025-04-07 09:15:00 | 41635.90 | 2025-04-11 13:15:00 | 43889.95 | STOP_HIT | 1.00 | -5.41% |
| BUY | retest2 | 2025-06-24 09:15:00 | 46510.00 | 2025-08-08 09:15:00 | 44110.00 | STOP_HIT | 1.00 | -5.16% |
| BUY | retest2 | 2025-08-05 15:00:00 | 45820.00 | 2025-08-08 09:15:00 | 44110.00 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2025-08-07 13:30:00 | 45695.00 | 2025-08-08 09:15:00 | 44110.00 | STOP_HIT | 1.00 | -3.47% |
| BUY | retest2 | 2025-08-07 15:00:00 | 45830.00 | 2025-08-08 09:15:00 | 44110.00 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2025-08-21 12:15:00 | 46015.00 | 2025-08-25 09:15:00 | 46540.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-08-21 15:00:00 | 45955.00 | 2025-08-25 09:15:00 | 46540.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-08-22 10:45:00 | 45950.00 | 2025-08-25 09:15:00 | 46540.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-08-26 11:30:00 | 46005.00 | 2025-08-29 09:15:00 | 43704.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 11:30:00 | 46005.00 | 2025-09-16 09:15:00 | 45100.00 | STOP_HIT | 0.50 | 1.97% |
| SELL | retest2 | 2025-08-26 13:15:00 | 45905.00 | 2025-09-19 12:15:00 | 43609.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 13:45:00 | 45905.00 | 2025-09-19 12:15:00 | 43609.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 13:15:00 | 45905.00 | 2025-09-26 14:15:00 | 41314.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-26 13:45:00 | 45905.00 | 2025-09-26 14:15:00 | 41314.50 | TARGET_HIT | 0.50 | 10.00% |
