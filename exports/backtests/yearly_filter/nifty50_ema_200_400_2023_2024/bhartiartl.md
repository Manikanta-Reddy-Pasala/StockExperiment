# BHARTIARTL (BHARTIARTL)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 1834.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 6 |
| ALERT2_SKIP | 4 |
| ALERT3 | 46 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 45 |
| PARTIAL | 4 |
| TARGET_HIT | 12 |
| STOP_HIT | 37 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 35
- **Target hits / Stop hits / Partials:** 12 / 37 / 4
- **Avg / median % per leg:** 1.57% / -0.89%
- **Sum % (uncompounded):** 83.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 10 | 26.3% | 8 | 30 | 0 | 0.77% | 29.3% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.51% | -10.0% |
| BUY @ 3rd Alert (retest2) | 34 | 10 | 29.4% | 8 | 26 | 0 | 1.16% | 39.3% |
| SELL (all) | 15 | 8 | 53.3% | 4 | 7 | 4 | 3.60% | 54.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 8 | 53.3% | 4 | 7 | 4 | 3.60% | 54.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.51% | -10.0% |
| retest2 (combined) | 49 | 18 | 36.7% | 12 | 33 | 4 | 1.90% | 93.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-03 11:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-03 11:30:00 | 879.35 | 874.52 | 853.75 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 10:15:00 | 881.30 | 874.44 | 854.22 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 11:00:00 | 880.85 | 874.51 | 854.35 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-10 10:00:00 | 880.80 | 877.81 | 858.66 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 10:15:00 | 859.75 | 876.83 | 859.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 11:00:00 | 859.75 | 876.83 | 859.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 11:15:00 | 860.15 | 876.67 | 859.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 11:45:00 | 859.00 | 876.67 | 859.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 858.50 | 876.03 | 859.65 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-08-16 09:15:00 | 858.50 | 876.03 | 859.65 | SL hit (close<ema400) qty=1.00 sl=859.65 alert=retest1 |

### Cycle 2 — SELL (started 2024-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 15:15:00 | 1583.45 | 1598.60 | 1598.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 11:15:00 | 1571.80 | 1597.93 | 1598.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 1613.65 | 1593.40 | 1595.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 10:15:00 | 1613.65 | 1593.40 | 1595.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 1613.65 | 1593.40 | 1595.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:00:00 | 1613.65 | 1593.40 | 1595.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 1624.60 | 1593.71 | 1596.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:00:00 | 1624.60 | 1593.71 | 1596.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 15:15:00 | 1642.00 | 1598.22 | 1598.19 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 09:15:00 | 1577.45 | 1598.35 | 1598.38 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 10:15:00 | 1633.80 | 1598.55 | 1598.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 11:15:00 | 1671.40 | 1599.28 | 1598.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 10:15:00 | 1603.55 | 1607.40 | 1603.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 10:15:00 | 1603.55 | 1607.40 | 1603.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 1603.55 | 1607.40 | 1603.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 11:00:00 | 1603.55 | 1607.40 | 1603.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 1602.30 | 1607.35 | 1603.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 11:30:00 | 1602.25 | 1607.35 | 1603.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 1600.75 | 1607.28 | 1603.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 12:45:00 | 1600.40 | 1607.28 | 1603.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 13:15:00 | 1603.15 | 1607.24 | 1603.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 13:30:00 | 1599.80 | 1607.24 | 1603.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 1601.50 | 1607.18 | 1603.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 14:45:00 | 1601.60 | 1607.18 | 1603.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 15:15:00 | 1601.40 | 1607.13 | 1603.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 09:15:00 | 1592.60 | 1607.13 | 1603.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 10:15:00 | 1597.85 | 1606.96 | 1603.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:00:00 | 1597.85 | 1606.96 | 1603.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 1605.85 | 1606.95 | 1603.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:30:00 | 1598.05 | 1606.95 | 1603.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 1606.10 | 1606.95 | 1603.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 12:30:00 | 1603.80 | 1606.95 | 1603.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 1601.05 | 1606.89 | 1603.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 13:30:00 | 1600.75 | 1606.89 | 1603.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 1602.15 | 1606.84 | 1603.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:30:00 | 1601.25 | 1606.84 | 1603.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 1609.45 | 1606.77 | 1603.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 09:30:00 | 1619.80 | 1603.30 | 1601.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 10:30:00 | 1617.55 | 1603.42 | 1601.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 13:15:00 | 1598.50 | 1603.52 | 1601.88 | SL hit (close<static) qty=1.00 sl=1602.20 alert=retest2 |

### Cycle 6 — SELL (started 2025-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 11:15:00 | 1590.30 | 1600.55 | 1600.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 09:15:00 | 1585.95 | 1600.09 | 1600.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 1603.60 | 1599.76 | 1600.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 13:15:00 | 1603.60 | 1599.76 | 1600.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 1603.60 | 1599.76 | 1600.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:00:00 | 1603.60 | 1599.76 | 1600.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 1599.30 | 1599.75 | 1600.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:45:00 | 1595.45 | 1600.16 | 1600.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-10 12:15:00 | 1617.25 | 1600.57 | 1600.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 12:15:00 | 1617.25 | 1600.57 | 1600.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 11:15:00 | 1622.30 | 1601.44 | 1601.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-14 13:15:00 | 1586.80 | 1601.31 | 1600.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 13:15:00 | 1586.80 | 1601.31 | 1600.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 1586.80 | 1601.31 | 1600.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-14 13:45:00 | 1581.00 | 1601.31 | 1600.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 1600.10 | 1601.30 | 1600.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-15 09:45:00 | 1603.45 | 1601.24 | 1600.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-15 11:15:00 | 1604.70 | 1601.24 | 1600.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-27 14:00:00 | 1603.80 | 1614.99 | 1608.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-28 09:45:00 | 1608.15 | 1614.68 | 1608.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 10:15:00 | 1611.80 | 1614.65 | 1608.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-28 11:30:00 | 1613.45 | 1614.65 | 1608.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-28 12:00:00 | 1613.95 | 1614.65 | 1608.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-28 13:00:00 | 1615.40 | 1614.65 | 1608.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-29 09:45:00 | 1614.95 | 1614.77 | 1609.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 1600.10 | 1614.84 | 1609.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-29 14:15:00 | 1600.10 | 1614.84 | 1609.18 | SL hit (close<static) qty=1.00 sl=1605.45 alert=retest2 |

### Cycle 8 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 2002.80 | 2056.72 | 2056.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 1997.40 | 2055.12 | 2055.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 10:15:00 | 2022.20 | 2019.76 | 2035.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-04 10:45:00 | 2021.40 | 2019.76 | 2035.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 2034.10 | 2018.71 | 2033.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:00:00 | 2034.10 | 2018.71 | 2033.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 2029.00 | 2018.82 | 2033.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 13:15:00 | 2016.90 | 2021.13 | 2034.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 13:45:00 | 2014.10 | 2021.03 | 2033.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 12:15:00 | 2016.10 | 2020.47 | 2033.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:45:00 | 2014.50 | 2020.21 | 2032.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 2030.00 | 2019.30 | 2031.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:30:00 | 2026.30 | 2019.36 | 2031.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 10:30:00 | 2026.70 | 2019.44 | 2031.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 11:15:00 | 2034.80 | 2019.59 | 2031.18 | SL hit (close>static) qty=1.00 sl=2034.30 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-03 11:30:00 | 879.35 | 2023-08-16 09:15:00 | 858.50 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest1 | 2023-08-04 10:15:00 | 881.30 | 2023-08-16 09:15:00 | 858.50 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest1 | 2023-08-04 11:00:00 | 880.85 | 2023-08-16 09:15:00 | 858.50 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest1 | 2023-08-10 10:00:00 | 880.80 | 2023-08-16 09:15:00 | 858.50 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2023-08-17 15:00:00 | 857.10 | 2023-09-15 14:15:00 | 942.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-18 11:30:00 | 859.35 | 2023-09-15 14:15:00 | 945.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-18 13:15:00 | 856.75 | 2023-09-15 14:15:00 | 942.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-18 13:45:00 | 857.05 | 2023-09-15 14:15:00 | 942.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-06 09:15:00 | 870.25 | 2023-10-16 09:15:00 | 957.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-27 09:30:00 | 1619.80 | 2024-12-27 13:15:00 | 1598.50 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-12-27 10:30:00 | 1617.55 | 2024-12-27 13:15:00 | 1598.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-01-10 09:45:00 | 1595.45 | 2025-01-10 12:15:00 | 1617.25 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-01-15 09:45:00 | 1603.45 | 2025-01-29 14:15:00 | 1600.10 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-01-15 11:15:00 | 1604.70 | 2025-01-29 14:15:00 | 1600.10 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-01-27 14:00:00 | 1603.80 | 2025-01-29 14:15:00 | 1600.10 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-01-28 09:45:00 | 1608.15 | 2025-01-29 14:15:00 | 1600.10 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-01-28 11:30:00 | 1613.45 | 2025-02-24 09:15:00 | 1616.75 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-01-28 12:00:00 | 1613.95 | 2025-02-24 09:15:00 | 1616.75 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-01-28 13:00:00 | 1615.40 | 2025-02-24 11:15:00 | 1604.05 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-01-29 09:45:00 | 1614.95 | 2025-02-24 11:15:00 | 1604.05 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-01-30 13:30:00 | 1633.45 | 2025-02-24 11:15:00 | 1604.05 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-01-31 14:00:00 | 1626.50 | 2025-02-24 11:15:00 | 1604.05 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-02-01 12:30:00 | 1626.55 | 2025-02-24 11:15:00 | 1604.05 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-02-03 09:15:00 | 1632.80 | 2025-02-24 11:15:00 | 1604.05 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-02-03 10:15:00 | 1646.55 | 2025-02-24 11:15:00 | 1604.05 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-02-03 10:45:00 | 1647.30 | 2025-02-24 11:15:00 | 1604.05 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2025-02-04 09:30:00 | 1645.55 | 2025-02-28 09:15:00 | 1615.80 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-02-04 11:30:00 | 1643.50 | 2025-02-28 09:15:00 | 1615.80 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-02-21 09:15:00 | 1655.90 | 2025-02-28 13:15:00 | 1581.90 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest2 | 2025-02-21 10:30:00 | 1648.75 | 2025-02-28 13:15:00 | 1581.90 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2025-02-27 09:45:00 | 1649.65 | 2025-02-28 13:15:00 | 1581.90 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2025-02-27 11:00:00 | 1648.85 | 2025-02-28 13:15:00 | 1581.90 | STOP_HIT | 1.00 | -4.06% |
| BUY | retest2 | 2025-03-10 09:30:00 | 1650.15 | 2025-03-18 13:15:00 | 1627.35 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-03-11 10:00:00 | 1643.20 | 2025-03-18 13:15:00 | 1627.35 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-03-12 14:30:00 | 1642.65 | 2025-03-18 13:15:00 | 1627.35 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-03-13 12:45:00 | 1643.00 | 2025-03-18 13:15:00 | 1627.35 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-03-18 09:30:00 | 1649.05 | 2025-04-15 09:15:00 | 1813.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-18 10:00:00 | 1650.65 | 2025-04-15 09:15:00 | 1815.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-20 09:15:00 | 1659.90 | 2025-04-16 14:15:00 | 1825.89 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-10 13:15:00 | 2016.90 | 2026-02-17 11:15:00 | 2034.80 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2026-02-10 13:45:00 | 2014.10 | 2026-02-17 11:15:00 | 2034.80 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-02-11 12:15:00 | 2016.10 | 2026-02-17 11:15:00 | 2034.80 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-02-12 14:45:00 | 2014.50 | 2026-02-17 11:15:00 | 2034.80 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-02-17 09:30:00 | 2026.30 | 2026-02-17 11:15:00 | 2034.80 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-02-17 10:30:00 | 2026.70 | 2026-02-17 11:15:00 | 2034.80 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2026-02-17 13:45:00 | 2024.20 | 2026-02-24 09:15:00 | 1922.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 09:45:00 | 2021.50 | 2026-02-25 12:15:00 | 1920.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 10:30:00 | 2011.80 | 2026-02-25 13:15:00 | 1911.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 11:15:00 | 2011.50 | 2026-02-25 13:15:00 | 1910.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 13:45:00 | 2024.20 | 2026-03-04 09:15:00 | 1821.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-18 09:45:00 | 2021.50 | 2026-03-04 09:15:00 | 1819.35 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-19 10:30:00 | 2011.80 | 2026-03-11 11:15:00 | 1810.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-19 11:15:00 | 2011.50 | 2026-03-11 11:15:00 | 1810.35 | TARGET_HIT | 0.50 | 10.00% |
