# CIPLA (CIPLA)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 1348.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 23 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 4 |
| TARGET_HIT | 4 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 7
- **Target hits / Stop hits / Partials:** 4 / 8 / 4
- **Avg / median % per leg:** 3.18% / 4.97%
- **Sum % (uncompounded):** 50.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 1 | 12.5% | 0 | 8 | 0 | -1.13% | -9.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 1 | 12.5% | 0 | 8 | 0 | -1.13% | -9.1% |
| SELL (all) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 16 | 9 | 56.2% | 4 | 8 | 4 | 3.18% | 50.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 12:15:00 | 1513.10 | 1492.43 | 1492.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 09:15:00 | 1535.50 | 1493.77 | 1493.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 15:15:00 | 1496.00 | 1496.96 | 1494.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 09:15:00 | 1501.80 | 1496.96 | 1494.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1491.40 | 1496.93 | 1494.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:00:00 | 1491.40 | 1496.93 | 1494.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 1491.00 | 1496.87 | 1494.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 1491.00 | 1496.87 | 1494.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1493.00 | 1496.61 | 1494.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:30:00 | 1490.80 | 1496.51 | 1494.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1490.60 | 1496.45 | 1494.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:15:00 | 1493.70 | 1496.45 | 1494.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 1492.70 | 1502.17 | 1498.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 1481.00 | 1501.96 | 1498.53 | SL hit (close<static) qty=1.00 sl=1484.20 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 11:15:00 | 1478.50 | 1495.65 | 1495.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 14:15:00 | 1476.10 | 1495.15 | 1495.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 13:15:00 | 1541.60 | 1488.36 | 1491.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 13:15:00 | 1541.60 | 1488.36 | 1491.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 1541.60 | 1488.36 | 1491.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:00:00 | 1541.60 | 1488.36 | 1491.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 1532.00 | 1488.79 | 1491.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:30:00 | 1532.10 | 1488.79 | 1491.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 09:15:00 | 1567.70 | 1495.10 | 1494.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 11:15:00 | 1573.30 | 1496.61 | 1495.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 11:15:00 | 1506.70 | 1507.02 | 1501.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 12:00:00 | 1506.70 | 1507.02 | 1501.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 1503.50 | 1506.98 | 1501.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:30:00 | 1505.00 | 1506.98 | 1501.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 1504.00 | 1506.95 | 1501.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:15:00 | 1500.10 | 1506.95 | 1501.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 1495.20 | 1506.84 | 1501.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 1495.20 | 1506.84 | 1501.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 1505.00 | 1506.82 | 1501.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 1511.60 | 1506.82 | 1501.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 1481.30 | 1506.62 | 1501.55 | SL hit (close<static) qty=1.00 sl=1494.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 10:15:00 | 1512.30 | 1531.50 | 1531.52 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 14:15:00 | 1559.00 | 1531.62 | 1531.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 15:15:00 | 1568.00 | 1531.98 | 1531.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 13:15:00 | 1546.50 | 1561.42 | 1549.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 13:15:00 | 1546.50 | 1561.42 | 1549.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 1546.50 | 1561.42 | 1549.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:45:00 | 1543.50 | 1561.42 | 1549.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 1542.70 | 1561.23 | 1549.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:45:00 | 1537.70 | 1561.23 | 1549.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 1541.00 | 1561.03 | 1549.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 1518.20 | 1561.03 | 1549.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 09:15:00 | 1519.10 | 1539.83 | 1539.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 1510.20 | 1535.76 | 1537.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 12:15:00 | 1530.00 | 1529.74 | 1533.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 13:00:00 | 1530.00 | 1529.74 | 1533.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 1532.20 | 1529.77 | 1533.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:30:00 | 1535.20 | 1529.77 | 1533.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1524.60 | 1529.71 | 1533.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 13:45:00 | 1520.80 | 1529.54 | 1533.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:30:00 | 1520.10 | 1529.47 | 1533.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:45:00 | 1520.60 | 1529.32 | 1533.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 12:45:00 | 1520.00 | 1529.19 | 1533.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 1513.60 | 1508.94 | 1517.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:45:00 | 1516.90 | 1508.94 | 1517.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1518.40 | 1509.08 | 1517.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:45:00 | 1521.50 | 1509.08 | 1517.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1521.50 | 1509.20 | 1517.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:45:00 | 1522.90 | 1509.20 | 1517.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 1520.60 | 1509.65 | 1517.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:45:00 | 1520.40 | 1509.65 | 1517.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 1520.10 | 1509.75 | 1517.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 1527.10 | 1509.75 | 1517.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1444.76 | 1502.17 | 1512.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1444.57 | 1502.17 | 1512.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 09:15:00 | 1444.09 | 1499.25 | 1510.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 09:15:00 | 1444.00 | 1499.25 | 1510.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-16 09:15:00 | 1368.72 | 1490.86 | 1505.62 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-23 11:15:00 | 1493.70 | 2025-07-08 10:15:00 | 1481.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-07-08 09:30:00 | 1492.70 | 2025-07-08 10:15:00 | 1481.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-07-08 13:30:00 | 1492.40 | 2025-07-10 09:15:00 | 1470.10 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-07-09 09:15:00 | 1496.00 | 2025-07-10 09:15:00 | 1470.10 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-08-04 09:15:00 | 1511.60 | 2025-08-06 09:15:00 | 1481.30 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-08-12 09:15:00 | 1508.10 | 2025-09-26 10:15:00 | 1493.20 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-09-29 09:15:00 | 1510.10 | 2025-09-29 14:15:00 | 1487.40 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-10-01 12:15:00 | 1507.80 | 2025-10-09 10:15:00 | 1512.30 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2025-12-01 13:45:00 | 1520.80 | 2026-01-12 09:15:00 | 1444.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 14:30:00 | 1520.10 | 2026-01-12 09:15:00 | 1444.57 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2025-12-02 09:45:00 | 1520.60 | 2026-01-13 09:15:00 | 1444.09 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2025-12-02 12:45:00 | 1520.00 | 2026-01-13 09:15:00 | 1444.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 13:45:00 | 1520.80 | 2026-01-16 09:15:00 | 1368.72 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-01 14:30:00 | 1520.10 | 2026-01-16 09:15:00 | 1368.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-02 09:45:00 | 1520.60 | 2026-01-16 09:15:00 | 1368.54 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-02 12:45:00 | 1520.00 | 2026-01-16 09:15:00 | 1368.00 | TARGET_HIT | 0.50 | 10.00% |
