# INFY (INFY)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1165.90
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
| ALERT3 | 2 |
| PENDING | 8 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 4 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / Stop hits / Partials:** 0 / 7 / 0
- **Avg / median % per leg:** -5.59% / -5.52%
- **Sum % (uncompounded):** -39.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -5.59% | -39.1% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -6.33% | -19.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.03% | -20.1% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -6.33% | -19.0% |
| retest2 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.03% | -20.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 11:15:00 | 1605.50 | 1591.48 | 1591.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 09:15:00 | 1641.90 | 1592.54 | 1592.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 1594.20 | 1605.23 | 1599.26 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 10:15:00 | 1594.20 | 1605.23 | 1599.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1594.20 | 1605.23 | 1599.26 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-16 12:15:00 | 1608.40 | 1601.09 | 1597.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:15:00 | 1609.50 | 1601.17 | 1597.77 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-16 15:15:00 | 1609.00 | 1601.30 | 1597.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-17 09:15:00 | 1596.00 | 1601.24 | 1597.85 | ENTRY2 sustain failed after 1080m |
| Stop hit — per-position SL triggered | 2025-07-17 12:15:00 | 1588.50 | 1601.01 | 1597.79 | SL hit (close<static) qty=1.00 sl=1590.60 alert=retest2 |

### Cycle 2 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 1540.30 | 1500.97 | 1500.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 09:15:00 | 1571.80 | 1506.56 | 1503.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 1591.10 | 1606.85 | 1574.47 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-01-07 13:15:00 | 1639.50 | 1608.34 | 1578.02 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 14:15:00 | 1641.10 | 1608.67 | 1578.33 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 09:15:00 | 1673.30 | 1608.30 | 1583.31 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:15:00 | 1684.80 | 1609.06 | 1583.81 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-30 14:15:00 | 1641.00 | 1634.77 | 1606.24 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 15:15:00 | 1641.20 | 1634.83 | 1606.42 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1616.70 | 1634.65 | 1606.47 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-01 11:15:00 | 1638.30 | 1634.55 | 1606.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 1660.10 | 1634.81 | 1606.97 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-02 10:15:00 | 1633.00 | 1635.24 | 1607.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 11:15:00 | 1631.20 | 1635.20 | 1607.99 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-03 09:15:00 | 1662.20 | 1635.14 | 1608.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 1672.10 | 1635.51 | 1608.95 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1550.60 | 1635.99 | 1609.98 | SL hit (close<ema400) qty=1.00 sl=1609.98 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1550.60 | 1635.99 | 1609.98 | SL hit (close<ema400) qty=1.00 sl=1609.98 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1550.60 | 1635.99 | 1609.98 | SL hit (close<ema400) qty=1.00 sl=1609.98 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1550.60 | 1635.99 | 1609.98 | SL hit (close<static) qty=1.00 sl=1604.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1550.60 | 1635.99 | 1609.98 | SL hit (close<static) qty=1.00 sl=1604.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1550.60 | 1635.99 | 1609.98 | SL hit (close<static) qty=1.00 sl=1604.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-16 13:15:00 | 1609.50 | 2025-07-17 12:15:00 | 1588.50 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest1 | 2026-01-07 14:15:00 | 1641.10 | 2026-02-04 09:15:00 | 1550.60 | STOP_HIT | 1.00 | -5.51% |
| BUY | retest1 | 2026-01-16 10:15:00 | 1684.80 | 2026-02-04 09:15:00 | 1550.60 | STOP_HIT | 1.00 | -7.97% |
| BUY | retest1 | 2026-01-30 15:15:00 | 1641.20 | 2026-02-04 09:15:00 | 1550.60 | STOP_HIT | 1.00 | -5.52% |
| BUY | retest2 | 2026-02-01 12:15:00 | 1660.10 | 2026-02-04 09:15:00 | 1550.60 | STOP_HIT | 1.00 | -6.60% |
| BUY | retest2 | 2026-02-02 11:15:00 | 1631.20 | 2026-02-04 09:15:00 | 1550.60 | STOP_HIT | 1.00 | -4.94% |
| BUY | retest2 | 2026-02-03 10:15:00 | 1672.10 | 2026-02-04 09:15:00 | 1550.60 | STOP_HIT | 1.00 | -7.27% |
