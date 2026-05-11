# ICICI Lombard General Insurance Company Ltd. (ICICIGI)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1820.00
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
| ALERT2 | 5 |
| ALERT2_SKIP | 4 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 0 |
| TARGET_HIT | 4 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 10
- **Target hits / Stop hits / Partials:** 4 / 10 / 0
- **Avg / median % per leg:** 2.01% / -0.66%
- **Sum % (uncompounded):** 28.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 4 | 4 | 0 | 4.54% | 36.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 4 | 50.0% | 4 | 4 | 0 | 4.54% | 36.3% |
| SELL (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.35% | -8.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.35% | -8.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 4 | 28.6% | 4 | 10 | 0 | 2.01% | 28.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 10:15:00 | 1847.80 | 1923.02 | 1923.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-01 10:15:00 | 1832.50 | 1917.64 | 1920.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 12:15:00 | 1898.10 | 1883.97 | 1899.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 12:15:00 | 1898.10 | 1883.97 | 1899.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 1898.10 | 1883.97 | 1899.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:00:00 | 1898.10 | 1883.97 | 1899.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 1893.00 | 1884.18 | 1899.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:30:00 | 1898.60 | 1884.18 | 1899.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 1900.10 | 1884.34 | 1899.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 09:45:00 | 1889.20 | 1884.35 | 1899.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 11:15:00 | 1908.40 | 1885.76 | 1899.81 | SL hit (close>static) qty=1.00 sl=1907.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 2007.20 | 1900.19 | 1900.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 2027.80 | 1908.38 | 1904.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 13:15:00 | 1995.70 | 1996.81 | 1966.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 1970.70 | 1996.28 | 1969.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1970.70 | 1996.28 | 1969.70 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 1950.00 | 1960.04 | 1960.07 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1979.90 | 1960.23 | 1960.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 1988.40 | 1960.51 | 1960.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 1946.50 | 1965.51 | 1962.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 1946.50 | 1965.51 | 1962.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1946.50 | 1965.51 | 1962.98 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 1895.40 | 1960.57 | 1960.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 12:15:00 | 1874.00 | 1951.38 | 1955.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 13:15:00 | 1875.50 | 1873.77 | 1904.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 10:15:00 | 1899.90 | 1873.24 | 1901.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 1899.90 | 1873.24 | 1901.78 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-28 13:30:00 | 1858.40 | 2025-06-30 10:15:00 | 2040.94 | TARGET_HIT | 1.00 | 9.82% |
| BUY | retest2 | 2025-05-28 14:00:00 | 1859.50 | 2025-06-30 11:15:00 | 2044.24 | TARGET_HIT | 1.00 | 9.93% |
| BUY | retest2 | 2025-05-29 11:45:00 | 1855.40 | 2025-06-30 11:15:00 | 2045.45 | TARGET_HIT | 1.00 | 10.24% |
| BUY | retest2 | 2025-05-30 09:15:00 | 1859.30 | 2025-06-30 11:15:00 | 2045.23 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-18 09:15:00 | 1966.10 | 2025-08-19 09:15:00 | 1928.20 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-08-19 12:00:00 | 1950.10 | 2025-08-22 11:15:00 | 1937.10 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-08-19 14:00:00 | 1946.20 | 2025-08-22 11:15:00 | 1937.10 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-08-19 14:30:00 | 1949.90 | 2025-08-22 11:15:00 | 1937.10 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-09-15 09:45:00 | 1889.20 | 2025-09-16 11:15:00 | 1908.40 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-09-17 12:45:00 | 1885.60 | 2025-09-26 09:15:00 | 1900.10 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-09-23 11:15:00 | 1889.20 | 2025-10-01 11:15:00 | 1900.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-09-23 12:45:00 | 1889.10 | 2025-10-03 09:15:00 | 1908.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-25 09:15:00 | 1861.00 | 2025-10-03 09:15:00 | 1908.00 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-10-01 09:45:00 | 1866.20 | 2025-10-03 09:15:00 | 1908.00 | STOP_HIT | 1.00 | -2.24% |
