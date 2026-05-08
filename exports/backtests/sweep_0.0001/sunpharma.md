# SUNPHARMA (SUNPHARMA)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 1845.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 3 |
| PENDING | 7 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 1 |
| ENTRY2 | 4 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 0
- **Avg / median % per leg:** -1.76% / -1.22%
- **Sum % (uncompounded):** -8.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.76% | -8.8% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.99% | -1.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.96% | -7.8% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.99% | -1.0% |
| retest2 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.96% | -7.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 15:15:00 | 1638.40 | 1685.67 | 1685.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 1628.40 | 1685.10 | 1685.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 14:15:00 | 1656.70 | 1649.46 | 1662.64 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-08-26 09:15:00 | 1619.80 | 1649.25 | 1662.40 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 10:15:00 | 1619.10 | 1648.95 | 1662.19 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1635.20 | 1613.22 | 1632.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 1635.20 | 1613.22 | 1632.97 | SL hit (close>ema400) qty=1.00 sl=1632.97 alert=retest1 |
| Cross detected — sustain check pending | 2025-09-24 12:15:00 | 1627.50 | 1620.97 | 1634.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-24 13:15:00 | 1628.90 | 1621.05 | 1634.33 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-24 14:15:00 | 1627.40 | 1621.11 | 1634.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 15:15:00 | 1626.80 | 1621.17 | 1634.26 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-25 12:15:00 | 1626.50 | 1621.67 | 1634.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:15:00 | 1625.00 | 1621.71 | 1634.20 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-25 15:15:00 | 1625.00 | 1621.80 | 1634.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1591.00 | 1621.49 | 1633.91 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-10-01 10:15:00 | 1624.50 | 1616.84 | 1630.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-01 11:15:00 | 1630.20 | 1616.98 | 1630.14 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-03 09:15:00 | 1626.20 | 1617.80 | 1630.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:15:00 | 1611.00 | 1617.73 | 1630.13 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 1632.30 | 1617.86 | 1629.95 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-06 11:15:00 | 1644.90 | 1618.64 | 1630.11 | SL hit (close>static) qty=1.00 sl=1644.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 11:15:00 | 1644.90 | 1618.64 | 1630.11 | SL hit (close>static) qty=1.00 sl=1644.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 11:15:00 | 1644.90 | 1618.64 | 1630.11 | SL hit (close>static) qty=1.00 sl=1644.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 11:15:00 | 1644.90 | 1618.64 | 1630.11 | SL hit (close>static) qty=1.00 sl=1644.50 alert=retest2 |
| CROSSOVER_SKIP | 2025-10-17 15:15:00 | 1678.10 | 1638.10 | 1638.04 | min_gap filter: gap=0.004% < 0.010% |
| TREND_RESET | 2025-10-17 15:15:00 | 1678.10 | 1638.10 | 1638.04 | EMA inversion without crossover edge (EMA200=1638.10 EMA400=1638.04) — end cycle |

### Cycle 2 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 1621.30 | 1728.59 | 1728.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 09:15:00 | 1587.80 | 1697.02 | 1711.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1683.00 | 1675.58 | 1698.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1683.00 | 1675.58 | 1698.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1683.00 | 1675.58 | 1698.16 | EMA400 retest candle locked (from downside) |
| CROSSOVER_SKIP | 2026-02-26 10:15:00 | 1780.10 | 1707.48 | 1707.37 | min_gap filter: gap=0.006% < 0.010% |
| TREND_RESET | 2026-02-26 10:15:00 | 1780.10 | 1707.48 | 1707.37 | EMA inversion without crossover edge (EMA200=1707.48 EMA400=1707.37) — end cycle |
| CROSSOVER_SKIP | 2026-04-15 09:15:00 | 1670.00 | 1733.20 | 1733.35 | min_gap filter: gap=0.009% < 0.010% |

### Cycle 3 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 1829.70 | 1728.28 | 1727.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 1842.70 | 1734.74 | 1731.25 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-08-26 10:15:00 | 1619.10 | 2025-09-18 09:15:00 | 1635.20 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-09-24 15:15:00 | 1626.80 | 2025-10-06 11:15:00 | 1644.90 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-09-25 13:15:00 | 1625.00 | 2025-10-06 11:15:00 | 1644.90 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1591.00 | 2025-10-06 11:15:00 | 1644.90 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2025-10-03 10:15:00 | 1611.00 | 2025-10-06 11:15:00 | 1644.90 | STOP_HIT | 1.00 | -2.10% |
