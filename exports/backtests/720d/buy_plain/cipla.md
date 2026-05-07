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
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 5 |
| PENDING | 23 |
| PENDING_CANCEL | 8 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 13
- **Target hits / Stop hits / Partials:** 0 / 15 / 0
- **Avg / median % per leg:** -1.12% / -1.48%
- **Sum % (uncompounded):** -16.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 2 | 13.3% | 0 | 15 | 0 | -1.12% | -16.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 2 | 13.3% | 0 | 15 | 0 | -1.12% | -16.9% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 2 | 13.3% | 0 | 15 | 0 | -1.12% | -16.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 13:15:00 | 1523.55 | 1471.63 | 1471.52 | EMA200 above EMA400 |

### Cycle 2 — BUY (started 2025-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 12:15:00 | 1499.95 | 1472.12 | 1472.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 10:15:00 | 1503.50 | 1461.54 | 1465.57 | Break + close above crossover candle high |

### Cycle 3 — BUY (started 2025-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 12:15:00 | 1526.90 | 1469.82 | 1469.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 1550.90 | 1475.92 | 1472.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 1501.30 | 1501.94 | 1488.48 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 09:15:00 | 1490.00 | 1502.27 | 1489.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1490.00 | 1502.27 | 1489.11 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-12 14:15:00 | 1511.10 | 1499.88 | 1489.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-12 15:15:00 | 1507.30 | 1499.95 | 1489.12 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-13 09:15:00 | 1529.60 | 1500.25 | 1489.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 10:15:00 | 1512.90 | 1500.37 | 1489.44 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-15 09:15:00 | 1477.60 | 1500.90 | 1490.42 | SL hit (close<static) qty=1.00 sl=1489.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-10 11:15:00 | 1508.80 | 1488.54 | 1486.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 12:15:00 | 1510.60 | 1488.76 | 1487.11 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-16 09:15:00 | 1535.50 | 1494.00 | 1490.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:15:00 | 1530.80 | 1494.37 | 1490.31 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-17 13:15:00 | 1510.30 | 1496.70 | 1491.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-17 14:15:00 | 1503.70 | 1496.77 | 1491.76 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-06-19 12:15:00 | 1488.20 | 1496.97 | 1492.16 | SL hit (close<static) qty=1.00 sl=1489.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 12:15:00 | 1488.20 | 1496.97 | 1492.16 | SL hit (close<static) qty=1.00 sl=1489.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-24 09:15:00 | 1512.60 | 1496.85 | 1492.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 10:15:00 | 1511.00 | 1496.99 | 1492.60 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1499.10 | 1499.01 | 1494.08 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-30 12:15:00 | 1507.20 | 1499.31 | 1494.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-30 13:15:00 | 1506.10 | 1499.38 | 1494.52 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-01 12:15:00 | 1507.90 | 1499.66 | 1494.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 1514.80 | 1499.81 | 1494.91 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-02 12:15:00 | 1507.70 | 1500.44 | 1495.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-02 13:15:00 | 1499.10 | 1500.43 | 1495.39 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-03 11:15:00 | 1510.00 | 1500.52 | 1495.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 12:15:00 | 1508.50 | 1500.60 | 1495.63 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 1481.00 | 1502.04 | 1496.84 | SL hit (close<static) qty=1.00 sl=1489.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 1481.00 | 1502.04 | 1496.84 | SL hit (close<static) qty=1.00 sl=1492.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 1481.00 | 1502.04 | 1496.84 | SL hit (close<static) qty=1.00 sl=1492.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-25 13:15:00 | 1541.60 | 1488.39 | 1490.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 14:15:00 | 1532.00 | 1488.82 | 1490.69 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-28 13:15:00 | 1567.20 | 1492.84 | 1492.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 13:15:00 | 1567.20 | 1492.84 | 1492.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 14:15:00 | 1573.30 | 1493.64 | 1493.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 11:15:00 | 1506.70 | 1507.04 | 1500.33 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 13:15:00 | 1504.00 | 1506.97 | 1500.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 1504.00 | 1506.97 | 1500.36 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-04 09:15:00 | 1512.30 | 1506.89 | 1500.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 10:15:00 | 1511.10 | 1506.93 | 1500.47 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-05 10:15:00 | 1506.70 | 1507.31 | 1500.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-05 11:15:00 | 1504.60 | 1507.29 | 1500.91 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-08-05 12:15:00 | 1494.80 | 1507.16 | 1500.88 | SL hit (close<static) qty=1.00 sl=1499.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-12 09:15:00 | 1513.20 | 1502.29 | 1499.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 10:15:00 | 1507.60 | 1502.34 | 1499.18 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 1497.20 | 1550.59 | 1539.55 | SL hit (close<static) qty=1.00 sl=1499.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-29 09:15:00 | 1520.90 | 1547.22 | 1538.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 10:15:00 | 1519.00 | 1546.94 | 1538.11 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-29 13:15:00 | 1496.40 | 1545.60 | 1537.57 | SL hit (close<static) qty=1.00 sl=1499.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-01 11:15:00 | 1505.60 | 1540.23 | 1535.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-01 12:15:00 | 1503.10 | 1539.86 | 1535.11 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-01 13:15:00 | 1515.40 | 1539.62 | 1535.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 14:15:00 | 1513.70 | 1539.36 | 1534.90 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 1511.60 | 1539.09 | 1534.79 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-03 09:15:00 | 1518.30 | 1538.88 | 1534.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-03 10:15:00 | 1510.20 | 1538.59 | 1534.58 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-03 11:15:00 | 1518.70 | 1538.39 | 1534.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 12:15:00 | 1518.60 | 1538.20 | 1534.42 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-03 14:15:00 | 1519.00 | 1537.76 | 1534.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 15:15:00 | 1517.50 | 1537.56 | 1534.16 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1503.00 | 1537.22 | 1534.00 | SL hit (close<static) qty=1.00 sl=1510.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1503.00 | 1537.22 | 1534.00 | SL hit (close<static) qty=1.00 sl=1510.80 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-07 09:15:00 | 1517.50 | 1535.59 | 1533.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-07 10:15:00 | 1516.40 | 1535.40 | 1533.20 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-07 12:15:00 | 1518.80 | 1534.98 | 1533.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:15:00 | 1517.50 | 1534.81 | 1532.94 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 1500.70 | 1534.03 | 1532.57 | SL hit (close<static) qty=1.00 sl=1510.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 1495.00 | 1532.31 | 1531.74 | SL hit (close<static) qty=1.00 sl=1499.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-10 09:15:00 | 1520.20 | 1530.50 | 1530.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 1539.00 | 1530.59 | 1530.88 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 1555.20 | 1531.34 | 1531.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 1555.20 | 1531.34 | 1531.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 15:15:00 | 1568.00 | 1531.98 | 1531.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 13:15:00 | 1546.50 | 1561.42 | 1549.22 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 13:15:00 | 1546.50 | 1561.42 | 1549.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 1546.50 | 1561.42 | 1549.22 | EMA400 retest candle locked |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-13 10:15:00 | 1512.90 | 2025-05-15 09:15:00 | 1477.60 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-06-10 12:15:00 | 1510.60 | 2025-06-19 12:15:00 | 1488.20 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-06-16 10:15:00 | 1530.80 | 2025-06-19 12:15:00 | 1488.20 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-06-24 10:15:00 | 1511.00 | 2025-07-08 10:15:00 | 1481.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-07-01 13:15:00 | 1514.80 | 2025-07-08 10:15:00 | 1481.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-07-03 12:15:00 | 1508.50 | 2025-07-08 10:15:00 | 1481.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-07-25 14:15:00 | 1532.00 | 2025-07-28 13:15:00 | 1567.20 | STOP_HIT | 1.00 | 2.30% |
| BUY | retest2 | 2025-08-04 10:15:00 | 1511.10 | 2025-08-05 12:15:00 | 1494.80 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-08-12 10:15:00 | 1507.60 | 2025-09-26 09:15:00 | 1497.20 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-09-29 10:15:00 | 1519.00 | 2025-09-29 13:15:00 | 1496.40 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-10-01 14:15:00 | 1513.70 | 2025-10-06 09:15:00 | 1503.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-10-03 12:15:00 | 1518.60 | 2025-10-06 09:15:00 | 1503.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-10-03 15:15:00 | 1517.50 | 2025-10-08 09:15:00 | 1500.70 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-10-07 13:15:00 | 1517.50 | 2025-10-08 14:15:00 | 1495.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-10-10 10:15:00 | 1539.00 | 2025-10-10 13:15:00 | 1555.20 | STOP_HIT | 1.00 | 1.05% |
