# APL Apollo Tubes Ltd. (APLAPOLLO)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1950.00
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
| ALERT2_SKIP | 3 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 8
- **Target hits / Stop hits / Partials:** 1 / 8 / 0
- **Avg / median % per leg:** -0.15% / -1.38%
- **Sum % (uncompounded):** -1.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| SELL (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.42% | -11.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.42% | -11.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 1 | 11.1% | 1 | 8 | 0 | -0.15% | -1.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 15:15:00 | 1551.60 | 1716.09 | 1716.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 09:15:00 | 1497.50 | 1713.92 | 1715.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 1638.10 | 1630.49 | 1660.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 09:45:00 | 1641.20 | 1630.49 | 1660.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1665.00 | 1630.71 | 1659.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 1665.00 | 1630.71 | 1659.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1664.90 | 1631.05 | 1659.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:30:00 | 1659.90 | 1631.30 | 1659.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 11:30:00 | 1659.10 | 1631.58 | 1653.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 12:00:00 | 1659.30 | 1631.58 | 1653.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:15:00 | 1655.00 | 1631.89 | 1653.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 1655.00 | 1632.58 | 1653.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 1674.10 | 1632.58 | 1653.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1668.00 | 1632.94 | 1653.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 11:15:00 | 1647.20 | 1633.14 | 1653.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 14:15:00 | 1677.80 | 1634.35 | 1654.05 | SL hit (close>static) qty=1.00 sl=1669.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 14:15:00 | 1677.80 | 1634.35 | 1654.05 | SL hit (close>static) qty=1.00 sl=1669.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 14:15:00 | 1677.80 | 1634.35 | 1654.05 | SL hit (close>static) qty=1.00 sl=1669.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 14:15:00 | 1677.80 | 1634.35 | 1654.05 | SL hit (close>static) qty=1.00 sl=1669.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 12:00:00 | 1653.30 | 1638.01 | 1654.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 13:00:00 | 1653.60 | 1638.17 | 1654.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 1679.90 | 1639.79 | 1655.29 | SL hit (close>static) qty=1.00 sl=1679.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 1679.90 | 1639.79 | 1655.29 | SL hit (close>static) qty=1.00 sl=1679.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 1679.90 | 1639.79 | 1655.29 | SL hit (close>static) qty=1.00 sl=1679.30 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 09:15:00 | 1669.70 | 1666.01 | 1666.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 1689.40 | 1667.86 | 1666.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 1667.90 | 1669.40 | 1667.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 13:15:00 | 1667.90 | 1669.40 | 1667.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 1667.90 | 1669.40 | 1667.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:15:00 | 1657.90 | 1669.40 | 1667.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 1664.00 | 1669.35 | 1667.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 1682.60 | 1669.22 | 1667.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-22 10:15:00 | 1850.86 | 1751.24 | 1740.43 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 12:15:00 | 1910.30 | 2018.31 | 2018.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 09:15:00 | 1880.00 | 2013.93 | 2016.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 12:15:00 | 2033.90 | 2006.43 | 2012.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 12:15:00 | 2033.90 | 2006.43 | 2012.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 12:15:00 | 2033.90 | 2006.43 | 2012.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 13:00:00 | 2033.90 | 2006.43 | 2012.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 13:15:00 | 2034.40 | 2006.71 | 2012.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 13:30:00 | 2037.00 | 2006.71 | 2012.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 2011.90 | 2012.20 | 2015.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:00:00 | 2011.90 | 2012.20 | 2015.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 11:15:00 | 2012.00 | 2012.20 | 2015.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:30:00 | 2015.20 | 2012.20 | 2015.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 12:15:00 | 2011.50 | 2012.19 | 2015.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 13:15:00 | 2002.50 | 2012.19 | 2015.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 11:15:00 | 2031.70 | 2011.61 | 2014.64 | SL hit (close>static) qty=1.00 sl=2027.70 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 13:15:00 | 2106.90 | 2017.77 | 2017.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 10:15:00 | 2121.80 | 2021.48 | 2019.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 14:15:00 | 2023.10 | 2039.66 | 2029.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 14:15:00 | 2023.10 | 2039.66 | 2029.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 2023.10 | 2039.66 | 2029.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 15:00:00 | 2023.10 | 2039.66 | 2029.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 2028.80 | 2039.55 | 2029.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:30:00 | 2013.00 | 2039.25 | 2029.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 1986.90 | 2038.73 | 2029.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 1986.90 | 2038.73 | 2029.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 2012.50 | 2036.68 | 2028.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 2012.50 | 2036.68 | 2028.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 13:15:00 | 1902.30 | 2021.13 | 2021.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 09:15:00 | 1878.20 | 2017.46 | 2019.38 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-21 12:30:00 | 1659.90 | 2025-09-03 14:15:00 | 1677.80 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-09-02 11:30:00 | 1659.10 | 2025-09-03 14:15:00 | 1677.80 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-09-02 12:00:00 | 1659.30 | 2025-09-03 14:15:00 | 1677.80 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-09-02 13:15:00 | 1655.00 | 2025-09-03 14:15:00 | 1677.80 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-09-03 11:15:00 | 1647.20 | 2025-09-08 10:15:00 | 1679.90 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-09-05 12:00:00 | 1653.30 | 2025-09-08 10:15:00 | 1679.90 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-09-05 13:00:00 | 1653.60 | 2025-09-08 10:15:00 | 1679.90 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-09-29 09:15:00 | 1682.60 | 2025-12-22 10:15:00 | 1850.86 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-13 13:15:00 | 2002.50 | 2026-04-15 11:15:00 | 2031.70 | STOP_HIT | 1.00 | -1.46% |
