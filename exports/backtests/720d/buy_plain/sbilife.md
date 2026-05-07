# SBILIFE (SBILIFE)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1875.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 3 |
| PENDING | 14 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 7 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 4
- **Target hits / Stop hits / Partials:** 0 / 10 / 3
- **Avg / median % per leg:** 6.78% / 0.87%
- **Sum % (uncompounded):** 88.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 9 | 69.2% | 0 | 10 | 3 | 6.78% | 88.2% |
| BUY @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 3 | 0 | 0.73% | 2.2% |
| BUY @ 3rd Alert (retest2) | 10 | 6 | 60.0% | 0 | 7 | 3 | 8.60% | 86.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 3 | 100.0% | 0 | 3 | 0 | 0.73% | 2.2% |
| retest2 (combined) | 10 | 6 | 60.0% | 0 | 7 | 3 | 8.60% | 86.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 09:15:00 | 1538.50 | 1473.16 | 1472.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 13:15:00 | 1551.00 | 1475.89 | 1474.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1464.95 | 1496.14 | 1485.79 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 1464.95 | 1496.14 | 1485.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 1464.95 | 1496.14 | 1485.79 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-08 14:15:00 | 1487.90 | 1492.65 | 1484.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 15:15:00 | 1493.50 | 1492.66 | 1484.63 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-09 11:15:00 | 1490.00 | 1492.52 | 1484.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 12:15:00 | 1488.45 | 1492.47 | 1484.69 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-11 09:15:00 | 1523.45 | 1492.55 | 1484.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 10:15:00 | 1523.90 | 1492.86 | 1485.08 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-04-25 09:15:00 | 1717.52 | 1536.98 | 1511.48 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-04-25 09:15:00 | 1711.72 | 1536.98 | 1511.48 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-04-25 09:15:00 | 1752.48 | 1536.98 | 1511.48 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 1742.70 | 1756.72 | 1694.23 | SL hit (close<ema200) qty=0.50 sl=1756.72 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 1742.70 | 1756.72 | 1694.23 | SL hit (close<ema200) qty=0.50 sl=1756.72 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 1742.70 | 1756.72 | 1694.23 | SL hit (close<ema200) qty=0.50 sl=1756.72 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 1850.00 | 1809.08 | 1809.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1898.00 | 1821.18 | 1815.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 14:15:00 | 1963.70 | 1964.61 | 1916.05 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-12-01 15:15:00 | 1974.00 | 1964.58 | 1917.94 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-02 09:15:00 | 1960.80 | 1964.54 | 1918.15 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-12-02 13:15:00 | 1975.40 | 1964.63 | 1919.12 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 14:15:00 | 1980.30 | 1964.79 | 1919.42 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-03 11:15:00 | 1976.80 | 1965.31 | 1920.58 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 12:15:00 | 1974.50 | 1965.40 | 1920.85 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-03 15:15:00 | 1974.00 | 1965.57 | 1921.60 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 09:15:00 | 1976.60 | 1965.68 | 1921.88 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 2023.80 | 2046.62 | 2010.05 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 1991.60 | 2045.86 | 2010.03 | SL hit (close<ema400) qty=1.00 sl=2010.03 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 1991.60 | 2045.86 | 2010.03 | SL hit (close<ema400) qty=1.00 sl=2010.03 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 1991.60 | 2045.86 | 2010.03 | SL hit (close<ema400) qty=1.00 sl=2010.03 alert=retest1 |
| Cross detected — sustain check pending | 2026-01-27 09:15:00 | 2030.30 | 2043.21 | 2009.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-27 10:15:00 | 2025.50 | 2043.03 | 2009.99 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-27 11:15:00 | 2029.80 | 2042.90 | 2010.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 12:15:00 | 2034.00 | 2042.81 | 2010.21 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-29 10:15:00 | 1999.90 | 2042.69 | 2012.05 | SL hit (close<static) qty=1.00 sl=2009.30 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-03 09:15:00 | 2051.70 | 2030.33 | 2009.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 2049.70 | 2030.52 | 2009.50 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-03 14:15:00 | 2000.50 | 2030.16 | 2009.73 | SL hit (close<static) qty=1.00 sl=2009.30 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-04 12:15:00 | 2040.80 | 2029.78 | 2010.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 13:15:00 | 2042.50 | 2029.90 | 2010.20 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 2007.90 | 2029.06 | 2010.74 | SL hit (close<static) qty=1.00 sl=2009.30 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-09 10:15:00 | 2033.40 | 2026.68 | 2010.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-09 11:15:00 | 2025.40 | 2026.66 | 2010.31 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-13 09:15:00 | 2031.30 | 2024.80 | 2011.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-13 10:15:00 | 2020.70 | 2024.76 | 2011.35 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-13 13:15:00 | 2031.90 | 2024.73 | 2011.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 14:15:00 | 2032.60 | 2024.81 | 2011.65 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 2010.90 | 2046.48 | 2028.16 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 2005.00 | 2045.82 | 2028.01 | SL hit (close<static) qty=1.00 sl=2009.30 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-04-08 15:15:00 | 1493.50 | 2025-04-25 09:15:00 | 1717.52 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-09 12:15:00 | 1488.45 | 2025-04-25 09:15:00 | 1711.72 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-11 10:15:00 | 1523.90 | 2025-04-25 09:15:00 | 1752.48 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-08 15:15:00 | 1493.50 | 2025-06-13 09:15:00 | 1742.70 | STOP_HIT | 0.50 | 16.69% |
| BUY | retest2 | 2025-04-09 12:15:00 | 1488.45 | 2025-06-13 09:15:00 | 1742.70 | STOP_HIT | 0.50 | 17.08% |
| BUY | retest2 | 2025-04-11 10:15:00 | 1523.90 | 2025-06-13 09:15:00 | 1742.70 | STOP_HIT | 0.50 | 14.36% |
| BUY | retest1 | 2025-12-02 14:15:00 | 1980.30 | 2026-01-23 09:15:00 | 1991.60 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest1 | 2025-12-03 12:15:00 | 1974.50 | 2026-01-23 09:15:00 | 1991.60 | STOP_HIT | 1.00 | 0.87% |
| BUY | retest1 | 2025-12-04 09:15:00 | 1976.60 | 2026-01-23 09:15:00 | 1991.60 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2026-01-27 12:15:00 | 2034.00 | 2026-01-29 10:15:00 | 1999.90 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2026-02-03 10:15:00 | 2049.70 | 2026-02-03 14:15:00 | 2000.50 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2026-02-04 13:15:00 | 2042.50 | 2026-02-06 09:15:00 | 2007.90 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2026-02-13 14:15:00 | 2032.60 | 2026-03-02 11:15:00 | 2005.00 | STOP_HIT | 1.00 | -1.36% |
