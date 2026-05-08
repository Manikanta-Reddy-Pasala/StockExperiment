# BHARTIARTL (BHARTIARTL)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 1834.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 4 |
| ALERT3 | 6 |
| PENDING | 15 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 1 |
| ENTRY2 | 9 |
| PARTIAL | 0 |
| TARGET_HIT | 5 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 5
- **Target hits / Stop hits / Partials:** 5 / 5 / 0
- **Avg / median % per leg:** 3.74% / 9.15%
- **Sum % (uncompounded):** 37.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 3 | 4 | 0 | 2.65% | 18.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 3 | 42.9% | 3 | 4 | 0 | 2.65% | 18.6% |
| SELL (all) | 3 | 2 | 66.7% | 2 | 1 | 0 | 6.29% | 18.9% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.11% | -1.1% |
| SELL @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.11% | -1.1% |
| retest2 (combined) | 9 | 5 | 55.6% | 5 | 4 | 0 | 4.28% | 38.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 11:15:00 | 1572.00 | 1598.17 | 1598.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 12:15:00 | 1568.80 | 1597.87 | 1598.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 1614.00 | 1593.56 | 1595.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 10:15:00 | 1614.00 | 1593.56 | 1595.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 1614.00 | 1593.56 | 1595.80 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 15:15:00 | 1642.00 | 1598.35 | 1598.14 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 11:15:00 | 1578.20 | 1598.05 | 1598.14 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 09:15:00 | 1635.10 | 1598.27 | 1598.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 11:15:00 | 1671.00 | 1599.34 | 1598.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 10:15:00 | 1602.90 | 1607.43 | 1603.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 10:15:00 | 1602.90 | 1607.43 | 1603.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 1602.90 | 1607.43 | 1603.15 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-12-27 09:15:00 | 1618.30 | 1603.35 | 1601.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-27 10:15:00 | 1615.00 | 1603.46 | 1601.78 | ENTRY2 sustain failed after 60m |

### Cycle 5 — SELL (started 2025-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 11:15:00 | 1590.25 | 1600.48 | 1600.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 09:15:00 | 1585.95 | 1600.03 | 1600.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 1603.60 | 1599.70 | 1600.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 13:15:00 | 1603.60 | 1599.70 | 1600.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 1603.60 | 1599.70 | 1600.10 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 12:15:00 | 1617.25 | 1600.50 | 1600.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 11:15:00 | 1622.20 | 1601.43 | 1600.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-14 13:15:00 | 1586.80 | 1601.30 | 1600.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 13:15:00 | 1586.80 | 1601.30 | 1600.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 1586.80 | 1601.30 | 1600.91 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-01-15 11:15:00 | 1610.60 | 1601.38 | 1600.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-15 12:15:00 | 1606.10 | 1601.43 | 1600.98 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-27 15:15:00 | 1604.15 | 1614.75 | 1608.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-28 09:15:00 | 1607.20 | 1614.68 | 1608.73 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-01-30 09:15:00 | 1619.30 | 1614.75 | 1609.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 10:15:00 | 1622.40 | 1614.83 | 1609.24 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-25 09:15:00 | 1627.70 | 1648.30 | 1634.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 10:15:00 | 1629.20 | 1648.11 | 1634.08 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 1615.45 | 1647.54 | 1634.67 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-28 14:15:00 | 1569.90 | 1644.71 | 1633.56 | SL hit (close<static) qty=1.00 sl=1578.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-28 14:15:00 | 1569.90 | 1644.71 | 1633.56 | SL hit (close<static) qty=1.00 sl=1578.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-28 14:15:00 | 1569.90 | 1644.71 | 1633.56 | SL hit (close<static) qty=1.00 sl=1578.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-28 14:15:00 | 1569.90 | 1644.71 | 1633.56 | SL hit (close<static) qty=1.00 sl=1578.75 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-10 09:15:00 | 1650.55 | 1633.47 | 1629.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-03-10 10:15:00 | 1646.00 | 1633.60 | 1629.22 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-03-11 11:15:00 | 1653.25 | 1634.06 | 1629.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 12:15:00 | 1661.55 | 1634.34 | 1629.79 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-13 09:15:00 | 1652.85 | 1636.20 | 1630.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 10:15:00 | 1648.65 | 1636.33 | 1631.08 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-18 09:15:00 | 1650.50 | 1636.73 | 1631.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-03-18 10:15:00 | 1639.95 | 1636.76 | 1631.66 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-03-20 09:15:00 | 1664.85 | 1636.60 | 1631.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 10:15:00 | 1671.40 | 1636.94 | 1632.09 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Target hit | 2025-04-15 09:15:00 | 1813.52 | 1698.01 | 1672.04 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-04-17 09:15:00 | 1827.71 | 1712.41 | 1681.23 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-04-17 09:15:00 | 1838.54 | 1712.41 | 1681.23 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 2002.50 | 2056.66 | 2056.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 1997.40 | 2055.06 | 2055.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 12:15:00 | 2024.20 | 2024.05 | 2038.16 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-05 09:15:00 | 2009.30 | 2023.97 | 2037.84 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 10:15:00 | 2020.10 | 2023.93 | 2037.75 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 2033.40 | 2022.62 | 2036.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-06 14:15:00 | 2042.60 | 2022.82 | 2036.43 | SL hit (close>ema400) qty=1.00 sl=2036.43 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-06 15:15:00 | 2023.00 | 2022.82 | 2036.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-09 09:15:00 | 2048.10 | 2023.07 | 2036.42 | ENTRY2 sustain failed after 3960m |
| Cross detected — sustain check pending | 2026-02-10 11:15:00 | 2023.70 | 2024.32 | 2036.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-10 12:15:00 | 2025.60 | 2024.34 | 2036.42 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-10 13:15:00 | 2011.10 | 2024.20 | 2036.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 14:15:00 | 2012.20 | 2024.08 | 2036.17 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-17 13:15:00 | 2021.10 | 2021.92 | 2033.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:15:00 | 2020.70 | 2021.90 | 2033.01 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Target hit | 2026-03-04 09:15:00 | 1818.63 | 1983.36 | 2008.81 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-11 11:15:00 | 1810.98 | 1948.96 | 1985.95 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-01-15 12:15:00 | 1606.10 | 2025-02-28 14:15:00 | 1569.90 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-01-28 09:15:00 | 1607.20 | 2025-02-28 14:15:00 | 1569.90 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-01-30 10:15:00 | 1622.40 | 2025-02-28 14:15:00 | 1569.90 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2025-02-25 10:15:00 | 1629.20 | 2025-02-28 14:15:00 | 1569.90 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest2 | 2025-03-11 12:15:00 | 1661.55 | 2025-04-15 09:15:00 | 1813.52 | TARGET_HIT | 1.00 | 9.15% |
| BUY | retest2 | 2025-03-13 10:15:00 | 1648.65 | 2025-04-17 09:15:00 | 1827.71 | TARGET_HIT | 1.00 | 10.86% |
| BUY | retest2 | 2025-03-20 10:15:00 | 1671.40 | 2025-04-17 09:15:00 | 1838.54 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2026-02-05 10:15:00 | 2020.10 | 2026-02-06 14:15:00 | 2042.60 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2026-02-10 14:15:00 | 2012.20 | 2026-03-04 09:15:00 | 1818.63 | TARGET_HIT | 1.00 | 9.62% |
| SELL | retest2 | 2026-02-17 14:15:00 | 2020.70 | 2026-03-11 11:15:00 | 1810.98 | TARGET_HIT | 1.00 | 10.38% |
