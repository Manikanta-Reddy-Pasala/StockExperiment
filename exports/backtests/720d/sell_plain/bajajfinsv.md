# BAJAJFINSV (BAJAJFINSV)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1829.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 5 |
| PENDING | 13 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 2 |
| ENTRY2 | 6 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 0 / 7 / 0
- **Avg / median % per leg:** -0.20% / 2.52%
- **Sum % (uncompounded):** -1.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 4 | 57.1% | 0 | 7 | 0 | -0.20% | -1.4% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.96% | -9.9% |
| SELL @ 3rd Alert (retest2) | 5 | 4 | 80.0% | 0 | 5 | 0 | 1.71% | 8.5% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.96% | -9.9% |
| retest2 (combined) | 5 | 4 | 80.0% | 0 | 5 | 0 | 1.71% | 8.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 1533.65 | 1587.83 | 1588.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-16 10:15:00 | 1524.70 | 1583.95 | 1586.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 10:15:00 | 1581.00 | 1579.91 | 1583.82 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 11:15:00 | 1590.55 | 1580.01 | 1583.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 1590.55 | 1580.01 | 1583.85 | EMA400 retest candle locked |

### Cycle 2 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 1679.60 | 1770.44 | 1770.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 11:15:00 | 1675.50 | 1769.50 | 1770.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 1661.90 | 1661.20 | 1699.75 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-12-18 09:15:00 | 1626.05 | 1662.72 | 1693.37 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 11:15:00 | 1630.85 | 1662.08 | 1692.74 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 1657.65 | 1620.86 | 1659.57 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-01-02 12:15:00 | 1711.90 | 1621.76 | 1659.83 | SL hit (close>ema400) qty=1.00 sl=1659.83 alert=retest1 |

### Cycle 3 — SELL (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 15:15:00 | 1942.00 | 1992.10 | 1992.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 1928.90 | 1991.47 | 1992.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 2010.90 | 1963.58 | 1976.17 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 2010.90 | 1963.58 | 1976.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 2010.90 | 1963.58 | 1976.17 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-26 09:15:00 | 1939.90 | 1966.14 | 1975.31 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:15:00 | 1939.50 | 1965.65 | 1974.97 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-09-02 13:15:00 | 1949.20 | 1957.10 | 1969.03 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-09-02 14:15:00 | 1957.70 | 1957.11 | 1968.97 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-09-04 12:15:00 | 2016.20 | 1959.65 | 1969.57 | SL hit (close>static) qty=1.00 sl=2015.90 alert=retest2 |

### Cycle 4 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 2009.50 | 2053.54 | 2053.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1988.60 | 2051.66 | 2052.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 2073.20 | 2046.18 | 2049.44 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 2073.20 | 2046.18 | 2049.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 2073.20 | 2046.18 | 2049.44 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-07 09:15:00 | 2030.50 | 2046.22 | 2049.35 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 2032.90 | 2045.94 | 2049.18 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-02-11 11:15:00 | 2032.40 | 1997.65 | 2012.59 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 13:15:00 | 2028.70 | 1998.27 | 2012.76 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-02-19 13:15:00 | 2040.30 | 2013.25 | 2018.18 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 15:15:00 | 2040.00 | 2013.72 | 2018.37 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-02-24 09:15:00 | 2032.90 | 2018.98 | 2020.76 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-02-24 10:15:00 | 2042.00 | 2019.20 | 2020.87 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-24 15:15:00 | 2038.20 | 2020.31 | 2021.39 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-02-25 09:15:00 | 2071.40 | 2020.82 | 2021.64 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-02-26 13:15:00 | 2034.50 | 2023.81 | 2023.14 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-02-26 14:15:00 | 2042.70 | 2024.00 | 2023.23 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-27 09:15:00 | 2016.60 | 2024.11 | 2023.29 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 11:15:00 | 2017.00 | 2024.00 | 2023.25 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 1966.20 | 2022.51 | 2022.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 1966.20 | 2022.51 | 2022.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 1966.20 | 2022.51 | 2022.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 1966.20 | 2022.51 | 2022.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1966.20 | 2022.51 | 2022.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 1946.20 | 2021.16 | 2021.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1805.00 | 1783.43 | 1865.95 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-04-09 12:15:00 | 1775.00 | 1783.29 | 1862.25 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 14:15:00 | 1769.70 | 1782.96 | 1861.30 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1769.60 | 1784.18 | 1858.47 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-13 11:15:00 | 1775.50 | 1783.99 | 1857.64 | ENTRY1 sustain failed after 120m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 1852.40 | 1797.68 | 1852.87 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-22 12:15:00 | 1857.20 | 1801.13 | 1852.73 | SL hit (close>ema400) qty=1.00 sl=1852.73 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-23 09:15:00 | 1818.00 | 1802.64 | 1852.47 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 11:15:00 | 1798.00 | 1802.53 | 1851.92 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-12-18 11:15:00 | 1630.85 | 2025-01-02 12:15:00 | 1711.90 | STOP_HIT | 1.00 | -4.97% |
| SELL | retest2 | 2025-08-26 11:15:00 | 1939.50 | 2025-09-04 12:15:00 | 2016.20 | STOP_HIT | 1.00 | -3.95% |
| SELL | retest2 | 2026-01-07 11:15:00 | 2032.90 | 2026-03-02 09:15:00 | 1966.20 | STOP_HIT | 1.00 | 3.28% |
| SELL | retest2 | 2026-02-11 13:15:00 | 2028.70 | 2026-03-02 09:15:00 | 1966.20 | STOP_HIT | 1.00 | 3.08% |
| SELL | retest2 | 2026-02-19 15:15:00 | 2040.00 | 2026-03-02 09:15:00 | 1966.20 | STOP_HIT | 1.00 | 3.62% |
| SELL | retest2 | 2026-02-27 11:15:00 | 2017.00 | 2026-03-02 09:15:00 | 1966.20 | STOP_HIT | 1.00 | 2.52% |
| SELL | retest1 | 2026-04-09 14:15:00 | 1769.70 | 2026-04-22 12:15:00 | 1857.20 | STOP_HIT | 1.00 | -4.94% |
