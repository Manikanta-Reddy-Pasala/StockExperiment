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
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 4 |
| PENDING | 8 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 0 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -3.06% / -2.05%
- **Sum % (uncompounded):** -12.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.06% | -12.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.06% | -12.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.06% | -12.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 14:15:00 | 1604.95 | 1742.09 | 1742.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 1593.40 | 1716.00 | 1728.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 11:15:00 | 1448.95 | 1446.36 | 1514.41 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-01-06 14:15:00 | 1429.95 | 1445.87 | 1510.85 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-06 15:15:00 | 1437.50 | 1445.79 | 1510.48 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 14:15:00 | 1499.70 | 1455.18 | 1503.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 1499.70 | 1455.18 | 1503.58 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-14 15:15:00 | 1492.15 | 1455.55 | 1503.53 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:15:00 | 1472.90 | 1455.72 | 1503.37 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 1538.60 | 1457.75 | 1502.76 | SL hit (close>static) qty=1.00 sl=1504.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-21 09:15:00 | 1482.15 | 1468.30 | 1503.86 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 11:15:00 | 1478.95 | 1468.55 | 1503.63 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-02-21 15:15:00 | 1480.55 | 1463.96 | 1479.15 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 09:15:00 | 1488.40 | 1464.20 | 1479.20 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2025-03-21 09:15:00 | 1509.20 | 1446.79 | 1460.61 | SL hit (close>static) qty=1.00 sl=1504.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-21 09:15:00 | 1509.20 | 1446.79 | 1460.61 | SL hit (close>static) qty=1.00 sl=1504.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-07 09:15:00 | 1464.95 | 1496.14 | 1485.79 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 11:15:00 | 1459.90 | 1495.52 | 1485.58 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1462.00 | 1493.69 | 1484.89 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 1523.45 | 1492.55 | 1484.88 | SL hit (close>static) qty=1.00 sl=1504.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 1786.70 | 1809.18 | 1809.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 1778.00 | 1808.87 | 1809.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 1808.20 | 1805.73 | 1807.44 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 13:15:00 | 1808.20 | 1805.73 | 1807.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 1808.20 | 1805.73 | 1807.44 | EMA400 retest candle locked |

### Cycle 3 — SELL (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 15:15:00 | 1963.00 | 2013.11 | 2013.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 09:15:00 | 1940.10 | 2012.38 | 2012.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1914.90 | 1898.14 | 1942.45 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 09:15:00 | 1942.30 | 1899.91 | 1940.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1942.30 | 1899.91 | 1940.58 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1909.70 | 1901.92 | 1940.20 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-13 10:15:00 | 1918.60 | 1902.08 | 1940.10 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-13 15:15:00 | 1914.10 | 1902.98 | 1939.61 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-15 09:15:00 | 1960.00 | 1903.55 | 1939.71 | ENTRY2 sustain failed after 2520m |
| Cross detected — sustain check pending | 2026-04-21 09:15:00 | 1900.00 | 1919.94 | 1943.76 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 11:15:00 | 1899.40 | 1919.64 | 1943.38 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-01-15 09:15:00 | 1472.90 | 2025-01-16 09:15:00 | 1538.60 | STOP_HIT | 1.00 | -4.46% |
| SELL | retest2 | 2025-01-21 11:15:00 | 1478.95 | 2025-03-21 09:15:00 | 1509.20 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-02-24 09:15:00 | 1488.40 | 2025-03-21 09:15:00 | 1509.20 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-04-07 11:15:00 | 1459.90 | 2025-04-11 09:15:00 | 1523.45 | STOP_HIT | 1.00 | -4.35% |
