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
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 26 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 23 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 19
- **Target hits / Stop hits / Partials:** 3 / 20 / 3
- **Avg / median % per leg:** -0.05% / -1.14%
- **Sum % (uncompounded):** -1.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 1 | 11.1% | 1 | 8 | 0 | -1.01% | -9.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 1 | 11.1% | 1 | 8 | 0 | -1.01% | -9.1% |
| SELL (all) | 17 | 6 | 35.3% | 2 | 12 | 3 | 0.45% | 7.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 17 | 6 | 35.3% | 2 | 12 | 3 | 0.45% | 7.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 26 | 7 | 26.9% | 3 | 20 | 3 | -0.05% | -1.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-30 14:15:00)

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

### Cycle 2 — SELL (started 2025-02-07 15:15:00)

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

### Cycle 3 — BUY (started 2025-04-23 13:15:00)

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

### Cycle 4 — SELL (started 2025-08-13 10:15:00)

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

### Cycle 5 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 10:15:00 | 38120.00 | 33873.65 | 33859.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 11:15:00 | 38230.00 | 33917.00 | 33881.54 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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
