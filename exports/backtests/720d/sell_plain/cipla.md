# CIPLA (CIPLA)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1363.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 4 |
| ALERT2_SKIP | 4 |
| ALERT3 | 4 |
| PENDING | 11 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 2
- **Target hits / Stop hits / Partials:** 0 / 6 / 3
- **Avg / median % per leg:** 8.15% / 10.65%
- **Sum % (uncompounded):** 73.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 7 | 77.8% | 0 | 6 | 3 | 8.15% | 73.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 7 | 77.8% | 0 | 6 | 3 | 8.15% | 73.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 7 | 77.8% | 0 | 6 | 3 | 8.15% | 73.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 1480.20 | 1586.62 | 1586.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 1457.85 | 1574.08 | 1580.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 11:15:00 | 1559.80 | 1557.80 | 1571.30 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 1569.05 | 1557.88 | 1571.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 1569.05 | 1557.88 | 1571.01 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-11-11 11:15:00 | 1553.00 | 1566.31 | 1573.35 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-11-11 13:15:00 | 1562.15 | 1566.10 | 1573.17 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2024-11-11 14:15:00 | 1553.80 | 1565.98 | 1573.08 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 09:15:00 | 1539.50 | 1565.56 | 1572.79 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1140m) |
| Stop hit — per-position SL triggered | 2025-04-02 11:15:00 | 1443.90 | 1471.98 | 1472.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 11:15:00 | 1443.90 | 1471.98 | 1472.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 1403.60 | 1472.20 | 1472.11 | Break + close below crossover candle low |

### Cycle 3 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 1410.00 | 1471.58 | 1471.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1381.20 | 1467.95 | 1469.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 09:15:00 | 1468.00 | 1457.10 | 1463.99 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 09:15:00 | 1468.00 | 1457.10 | 1463.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 1468.00 | 1457.10 | 1463.99 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-11 10:15:00 | 1454.90 | 1457.08 | 1463.95 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-11 11:15:00 | 1456.75 | 1457.07 | 1463.91 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-20 14:15:00 | 1456.10 | 1498.01 | 1490.19 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-21 09:15:00 | 1480.40 | 1497.40 | 1489.97 | ENTRY2 sustain failed after 1140m |

### Cycle 4 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 1475.30 | 1493.10 | 1493.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 12:15:00 | 1472.20 | 1492.23 | 1492.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 13:15:00 | 1541.60 | 1488.39 | 1490.49 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 13:15:00 | 1541.60 | 1488.39 | 1490.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 1541.60 | 1488.39 | 1490.49 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-06 09:15:00 | 1481.30 | 1506.64 | 1500.74 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 11:15:00 | 1479.20 | 1506.06 | 1500.50 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-08-06 13:15:00 | 1480.50 | 1505.63 | 1500.34 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 15:15:00 | 1485.00 | 1505.22 | 1500.19 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-08-08 09:15:00 | 1480.70 | 1503.15 | 1499.33 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-08 10:15:00 | 1490.00 | 1503.01 | 1499.28 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-08 14:15:00 | 1485.00 | 1502.45 | 1499.07 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-08 15:15:00 | 1487.30 | 1502.30 | 1499.01 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-08-13 11:15:00 | 1559.50 | 1504.31 | 1500.30 | SL hit (close>static) qty=1.00 sl=1549.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-13 11:15:00 | 1559.50 | 1504.31 | 1500.30 | SL hit (close>static) qty=1.00 sl=1549.80 alert=retest2 |

### Cycle 5 — SELL (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 12:15:00 | 1513.00 | 1531.14 | 1531.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 1506.80 | 1559.52 | 1548.56 | Break + close below crossover candle low |

### Cycle 6 — SELL (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 10:15:00 | 1524.90 | 1539.68 | 1539.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 13:15:00 | 1518.00 | 1539.12 | 1539.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 12:15:00 | 1530.00 | 1529.74 | 1533.78 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 14:15:00 | 1532.20 | 1529.77 | 1533.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 1532.20 | 1529.77 | 1533.75 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-11-28 15:15:00 | 1529.00 | 1529.76 | 1533.73 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 1524.60 | 1529.71 | 1533.69 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 3960m) |
| Cross detected — sustain check pending | 2025-12-02 12:15:00 | 1517.00 | 1529.19 | 1533.23 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 14:15:00 | 1517.70 | 1528.99 | 1533.08 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-01-07 09:15:00 | 1503.10 | 1511.09 | 1517.90 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 1460.30 | 1510.27 | 1517.43 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-01-27 09:15:00 | 1295.91 | 1450.77 | 1480.85 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-01-27 09:15:00 | 1290.05 | 1450.77 | 1480.85 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 1362.20 | 1357.40 | 1398.98 | SL hit (close>ema200) qty=0.50 sl=1357.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 1362.20 | 1357.40 | 1398.98 | SL hit (close>ema200) qty=0.50 sl=1357.40 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-19 13:15:00 | 1241.25 | 1327.09 | 1364.80 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1289.30 | 1250.23 | 1293.63 | SL hit (close>ema200) qty=0.50 sl=1250.23 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-12 09:15:00 | 1539.50 | 2025-04-02 11:15:00 | 1443.90 | STOP_HIT | 1.00 | 6.21% |
| SELL | retest2 | 2025-08-06 11:15:00 | 1479.20 | 2025-08-13 11:15:00 | 1559.50 | STOP_HIT | 1.00 | -5.43% |
| SELL | retest2 | 2025-08-06 15:15:00 | 1485.00 | 2025-08-13 11:15:00 | 1559.50 | STOP_HIT | 1.00 | -5.02% |
| SELL | retest2 | 2025-12-01 09:15:00 | 1524.60 | 2026-01-27 09:15:00 | 1295.91 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-12-02 14:15:00 | 1517.70 | 2026-01-27 09:15:00 | 1290.05 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-12-01 09:15:00 | 1524.60 | 2026-02-26 09:15:00 | 1362.20 | STOP_HIT | 0.50 | 10.65% |
| SELL | retest2 | 2025-12-02 14:15:00 | 1517.70 | 2026-02-26 09:15:00 | 1362.20 | STOP_HIT | 0.50 | 10.25% |
| SELL | retest2 | 2026-01-07 11:15:00 | 1460.30 | 2026-03-19 13:15:00 | 1241.25 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-01-07 11:15:00 | 1460.30 | 2026-04-23 09:15:00 | 1289.30 | STOP_HIT | 0.50 | 11.71% |
