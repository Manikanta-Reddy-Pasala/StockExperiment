# BHARTIARTL (BHARTIARTL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 1833.70
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 2 |
| ALERT3 | 4 |
| PENDING | 13 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 10
- **Target hits / Stop hits / Partials:** 1 / 10 / 1
- **Avg / median % per leg:** 2.44% / -1.07%
- **Sum % (uncompounded):** 29.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 2 | 16.7% | 1 | 10 | 1 | 2.44% | 29.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 2 | 16.7% | 1 | 10 | 1 | 2.44% | 29.3% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 2 | 16.7% | 1 | 10 | 1 | 2.44% | 29.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 15:15:00 | 1642.00 | 1598.35 | 1598.14 | EMA200 above EMA400 |
| CROSSOVER_SKIP | 2024-12-11 11:15:00 | 1578.20 | 1598.05 | 1598.14 | HTF filter: close above htf_sma |

### Cycle 2 — BUY (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 09:15:00 | 1635.10 | 1598.27 | 1598.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 11:15:00 | 1671.00 | 1599.34 | 1598.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 10:15:00 | 1602.90 | 1607.43 | 1603.15 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 10:15:00 | 1602.90 | 1607.43 | 1603.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 1602.90 | 1607.43 | 1603.15 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-27 09:15:00 | 1618.30 | 1603.35 | 1601.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-27 10:15:00 | 1615.00 | 1603.46 | 1601.78 | ENTRY2 sustain failed after 60m |
| CROSSOVER_SKIP | 2025-01-07 11:15:00 | 1590.25 | 1600.48 | 1600.50 | HTF filter: close above htf_sma |

### Cycle 3 — BUY (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 12:15:00 | 1617.25 | 1600.50 | 1600.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 11:15:00 | 1622.20 | 1601.43 | 1600.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-14 13:15:00 | 1586.80 | 1601.30 | 1600.91 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 13:15:00 | 1586.80 | 1601.30 | 1600.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 1586.80 | 1601.30 | 1600.91 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-15 11:15:00 | 1610.60 | 1601.38 | 1600.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-15 12:15:00 | 1606.10 | 1601.43 | 1600.98 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-27 15:15:00 | 1604.15 | 1614.75 | 1608.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-28 09:15:00 | 1607.20 | 1614.68 | 1608.73 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-01-30 09:15:00 | 1619.30 | 1614.75 | 1609.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 10:15:00 | 1622.40 | 1614.83 | 1609.24 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-31 09:15:00 | 1578.75 | 1615.54 | 1609.76 | SL hit qty=1.00 sl=1578.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-31 09:15:00 | 1578.75 | 1615.54 | 1609.76 | SL hit qty=1.00 sl=1578.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-31 09:15:00 | 1578.75 | 1615.54 | 1609.76 | SL hit qty=1.00 sl=1578.75 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-25 09:15:00 | 1627.70 | 1648.30 | 1634.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 10:15:00 | 1629.20 | 1648.11 | 1634.08 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 1638.90 | 1648.02 | 1634.10 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-25 13:15:00 | 1645.00 | 1647.90 | 1634.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 14:15:00 | 1642.85 | 1647.85 | 1634.23 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-27 09:15:00 | 1646.50 | 1647.78 | 1634.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 10:15:00 | 1648.85 | 1647.79 | 1634.40 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-28 09:15:00 | 1628.40 | 1647.54 | 1634.67 | SL hit qty=1.00 sl=1628.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-28 09:15:00 | 1628.40 | 1647.54 | 1634.67 | SL hit qty=1.00 sl=1628.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-28 14:15:00 | 1578.75 | 1644.71 | 1633.56 | SL hit qty=1.00 sl=1578.75 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-10 09:15:00 | 1650.55 | 1633.47 | 1629.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 10:15:00 | 1646.00 | 1633.60 | 1629.22 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-10 14:15:00 | 1628.40 | 1633.67 | 1629.34 | SL hit qty=1.00 sl=1628.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-11 09:15:00 | 1643.20 | 1633.74 | 1629.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 10:15:00 | 1646.95 | 1633.87 | 1629.51 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 1632.90 | 1636.39 | 1631.24 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-03-17 09:15:00 | 1637.50 | 1636.40 | 1631.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 1628.40 | 1636.40 | 1631.28 | SL hit qty=1.00 sl=1628.40 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 10:15:00 | 1645.70 | 1636.50 | 1631.35 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-17 15:15:00 | 1641.20 | 1636.59 | 1631.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 09:15:00 | 1650.50 | 1636.73 | 1631.62 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-03-18 12:15:00 | 1629.95 | 1636.66 | 1631.66 | SL hit qty=1.00 sl=1629.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-18 12:15:00 | 1629.95 | 1636.66 | 1631.66 | SL hit qty=1.00 sl=1629.95 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-19 14:15:00 | 1637.75 | 1636.30 | 1631.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-03-19 15:15:00 | 1637.15 | 1636.31 | 1631.72 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-03-20 09:15:00 | 1664.85 | 1636.60 | 1631.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 10:15:00 | 1671.40 | 1636.94 | 1632.09 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-06-20 10:15:00 | 1922.11 | 1850.39 | 1817.63 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2025-11-21 14:15:00 | 2172.82 | 2061.30 | 2011.38 | Target hit (30%) qty=0.50 alert=retest2 |
| CROSSOVER_SKIP | 2026-01-23 09:15:00 | 2002.50 | 2056.66 | 2056.70 | HTF filter: close above htf_sma |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-01-15 12:15:00 | 1606.10 | 2025-01-31 09:15:00 | 1578.75 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-01-28 09:15:00 | 1607.20 | 2025-01-31 09:15:00 | 1578.75 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-01-30 10:15:00 | 1622.40 | 2025-01-31 09:15:00 | 1578.75 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-02-25 10:15:00 | 1629.20 | 2025-02-28 09:15:00 | 1628.40 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-02-25 14:15:00 | 1642.85 | 2025-02-28 09:15:00 | 1628.40 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-02-27 10:15:00 | 1648.85 | 2025-02-28 14:15:00 | 1578.75 | STOP_HIT | 1.00 | -4.25% |
| BUY | retest2 | 2025-03-10 10:15:00 | 1646.00 | 2025-03-10 14:15:00 | 1628.40 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-03-11 10:15:00 | 1646.95 | 2025-03-17 09:15:00 | 1628.40 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-03-17 10:15:00 | 1645.70 | 2025-03-18 12:15:00 | 1629.95 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-03-18 09:15:00 | 1650.50 | 2025-03-18 12:15:00 | 1629.95 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-03-20 10:15:00 | 1671.40 | 2025-06-20 10:15:00 | 1922.11 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-03-20 10:15:00 | 1671.40 | 2025-11-21 14:15:00 | 2172.82 | TARGET_HIT | 0.50 | 30.00% |
