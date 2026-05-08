# INFY (INFY)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 1179.50
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
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 1 |
| PENDING | 8 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 4
- **Avg / median % per leg:** -0.96% / -4.94%
- **Sum % (uncompounded):** -10.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 0 | 6 | 2 | -3.82% | -30.5% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 3 | 2 | -2.34% | -11.7% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -6.27% | -18.8% |
| SELL (all) | 3 | 3 | 100.0% | 1 | 0 | 2 | 6.67% | 20.0% |
| SELL @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 1 | 0 | 2 | 6.67% | 20.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 5 | 62.5% | 1 | 3 | 4 | 1.04% | 8.3% |
| retest2 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -6.27% | -18.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 1540.30 | 1500.97 | 1500.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 09:15:00 | 1571.80 | 1506.56 | 1503.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 1591.10 | 1606.85 | 1574.46 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-07 13:15:00 | 1639.50 | 1608.34 | 1578.00 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 14:15:00 | 1641.10 | 1608.67 | 1578.31 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 09:15:00 | 1673.30 | 1608.30 | 1583.29 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:15:00 | 1684.80 | 1609.06 | 1583.80 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-30 14:15:00 | 1641.00 | 1634.77 | 1606.23 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 15:15:00 | 1641.20 | 1634.83 | 1606.41 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1616.70 | 1634.65 | 1606.46 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-02-01 11:15:00 | 1638.30 | 1634.55 | 1606.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 1660.10 | 1634.81 | 1606.96 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-02 10:15:00 | 1633.00 | 1635.24 | 1607.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 11:15:00 | 1631.20 | 1635.20 | 1607.98 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-03 09:15:00 | 1662.20 | 1635.14 | 1608.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-03 09:15:00 | 1723.15 | 1635.14 | 1608.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-03 09:15:00 | 1723.26 | 1635.14 | 1608.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 1672.10 | 1635.51 | 1608.94 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1550.60 | 1635.99 | 1609.97 | SL hit (close<ema200) qty=0.50 sl=1635.99 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1550.60 | 1635.99 | 1609.97 | SL hit (close<ema400) qty=1.00 sl=1609.97 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1550.60 | 1635.99 | 1609.97 | SL hit (close<ema200) qty=0.50 sl=1635.99 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1550.60 | 1635.99 | 1609.97 | SL hit (close<static) qty=1.00 sl=1604.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1550.60 | 1635.99 | 1609.97 | SL hit (close<static) qty=1.00 sl=1604.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1550.60 | 1635.99 | 1609.97 | SL hit (close<static) qty=1.00 sl=1604.00 alert=retest2 |

### Cycle 2 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 1405.10 | 1587.88 | 1588.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 1399.50 | 1586.00 | 1587.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 09:15:00 | 1313.10 | 1310.98 | 1382.90 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 1289.20 | 1314.93 | 1375.60 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:15:00 | 1287.70 | 1314.66 | 1375.16 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-22 09:15:00 | 1271.60 | 1311.16 | 1360.16 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:15:00 | 1264.30 | 1310.70 | 1359.68 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 1223.32 | 1303.24 | 1352.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 1201.08 | 1303.24 | 1352.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-04-24 13:15:00 | 1158.93 | 1297.91 | 1349.10 | Target hit (10%) qty=0.50 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-01-07 14:15:00 | 1641.10 | 2026-02-03 09:15:00 | 1723.15 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-01-16 10:15:00 | 1684.80 | 2026-02-03 09:15:00 | 1723.26 | PARTIAL | 0.50 | 2.28% |
| BUY | retest1 | 2026-01-07 14:15:00 | 1641.10 | 2026-02-04 09:15:00 | 1550.60 | STOP_HIT | 0.50 | -5.51% |
| BUY | retest1 | 2026-01-16 10:15:00 | 1684.80 | 2026-02-04 09:15:00 | 1550.60 | STOP_HIT | 0.50 | -7.97% |
| BUY | retest1 | 2026-01-30 15:15:00 | 1641.20 | 2026-02-04 09:15:00 | 1550.60 | STOP_HIT | 1.00 | -5.52% |
| BUY | retest2 | 2026-02-01 12:15:00 | 1660.10 | 2026-02-04 09:15:00 | 1550.60 | STOP_HIT | 1.00 | -6.60% |
| BUY | retest2 | 2026-02-02 11:15:00 | 1631.20 | 2026-02-04 09:15:00 | 1550.60 | STOP_HIT | 1.00 | -4.94% |
| BUY | retest2 | 2026-02-03 10:15:00 | 1672.10 | 2026-02-04 09:15:00 | 1550.60 | STOP_HIT | 1.00 | -7.27% |
| SELL | retest1 | 2026-04-10 10:15:00 | 1287.70 | 2026-04-24 09:15:00 | 1223.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-22 10:15:00 | 1264.30 | 2026-04-24 09:15:00 | 1201.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-10 10:15:00 | 1287.70 | 2026-04-24 13:15:00 | 1158.93 | TARGET_HIT | 0.50 | 10.00% |
