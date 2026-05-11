# Aditya Birla Real Estate Ltd. (ABREL)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 1479.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 17 |
| PARTIAL | 10 |
| TARGET_HIT | 5 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 7
- **Target hits / Stop hits / Partials:** 5 / 12 / 10
- **Avg / median % per leg:** 3.58% / 4.86%
- **Sum % (uncompounded):** 96.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 27 | 20 | 74.1% | 5 | 12 | 10 | 3.58% | 96.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 20 | 74.1% | 5 | 12 | 10 | 3.58% | 96.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 27 | 20 | 74.1% | 5 | 12 | 10 | 3.58% | 96.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 1898.50 | 2156.62 | 2156.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 1884.50 | 2146.55 | 2151.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 14:15:00 | 1865.90 | 1852.10 | 1933.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 14:45:00 | 1871.30 | 1852.10 | 1933.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 1929.00 | 1856.50 | 1932.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 1941.40 | 1856.50 | 1932.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1913.90 | 1857.07 | 1932.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 14:00:00 | 1909.30 | 1864.53 | 1932.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 15:15:00 | 1900.00 | 1865.03 | 1932.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:00:00 | 1908.00 | 1867.84 | 1931.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:30:00 | 1907.40 | 1870.02 | 1930.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 10:15:00 | 1813.83 | 1866.87 | 1924.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 10:15:00 | 1812.60 | 1866.87 | 1924.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 10:15:00 | 1812.03 | 1866.87 | 1924.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 11:15:00 | 1805.00 | 1866.14 | 1923.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-29 11:15:00 | 1718.37 | 1851.77 | 1912.11 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 13:15:00 | 1546.40 | 1344.76 | 1344.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 14:15:00 | 1584.80 | 1396.96 | 1372.80 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-18 14:00:00 | 1909.30 | 2025-09-25 10:15:00 | 1813.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 15:15:00 | 1900.00 | 2025-09-25 10:15:00 | 1812.60 | PARTIAL | 0.50 | 4.60% |
| SELL | retest2 | 2025-09-19 14:00:00 | 1908.00 | 2025-09-25 10:15:00 | 1812.03 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2025-09-22 13:30:00 | 1907.40 | 2025-09-25 11:15:00 | 1805.00 | PARTIAL | 0.50 | 5.37% |
| SELL | retest2 | 2025-09-18 14:00:00 | 1909.30 | 2025-09-29 11:15:00 | 1718.37 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-18 15:15:00 | 1900.00 | 2025-09-29 11:15:00 | 1710.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-19 14:00:00 | 1908.00 | 2025-09-29 11:15:00 | 1717.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-22 13:30:00 | 1907.40 | 2025-09-29 11:15:00 | 1716.66 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-07 14:45:00 | 1771.50 | 2025-11-13 09:15:00 | 1802.20 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-11-10 10:00:00 | 1766.40 | 2025-11-13 10:15:00 | 1814.30 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-11-10 10:30:00 | 1768.60 | 2025-11-13 10:15:00 | 1814.30 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-11-10 11:45:00 | 1767.30 | 2025-11-13 10:15:00 | 1814.30 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-11-12 13:30:00 | 1775.10 | 2025-11-13 10:15:00 | 1814.30 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-11-13 12:30:00 | 1778.60 | 2025-11-19 10:15:00 | 1808.60 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-11-19 11:15:00 | 1775.00 | 2025-11-19 12:15:00 | 1800.90 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-11-20 10:00:00 | 1780.70 | 2025-12-05 10:15:00 | 1694.23 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2025-11-20 10:00:00 | 1780.70 | 2025-12-05 13:15:00 | 1758.90 | STOP_HIT | 0.50 | 1.22% |
| SELL | retest2 | 2025-11-20 12:30:00 | 1780.00 | 2025-12-08 09:15:00 | 1691.66 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2025-12-02 10:30:00 | 1783.40 | 2025-12-08 09:15:00 | 1691.00 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2025-12-02 12:00:00 | 1759.20 | 2025-12-08 09:15:00 | 1671.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 13:00:00 | 1780.70 | 2025-12-08 09:15:00 | 1691.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 12:30:00 | 1780.00 | 2025-12-15 11:15:00 | 1729.00 | STOP_HIT | 0.50 | 2.87% |
| SELL | retest2 | 2025-12-02 10:30:00 | 1783.40 | 2025-12-15 11:15:00 | 1729.00 | STOP_HIT | 0.50 | 3.05% |
| SELL | retest2 | 2025-12-02 12:00:00 | 1759.20 | 2025-12-15 11:15:00 | 1729.00 | STOP_HIT | 0.50 | 1.72% |
| SELL | retest2 | 2025-12-04 13:00:00 | 1780.70 | 2025-12-15 11:15:00 | 1729.00 | STOP_HIT | 0.50 | 2.90% |
| SELL | retest2 | 2026-01-06 09:15:00 | 1687.70 | 2026-01-09 09:15:00 | 1603.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 09:15:00 | 1687.70 | 2026-01-19 09:15:00 | 1518.93 | TARGET_HIT | 0.50 | 10.00% |
