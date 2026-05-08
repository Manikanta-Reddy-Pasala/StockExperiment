# M&M (M&M)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 3330.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 4 |
| ALERT3 | 9 |
| PENDING | 22 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 2 |
| ENTRY2 | 17 |
| PARTIAL | 2 |
| TARGET_HIT | 13 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 5
- **Target hits / Stop hits / Partials:** 13 / 6 / 2
- **Avg / median % per leg:** 6.46% / 10.00%
- **Sum % (uncompounded):** 135.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 13 | 72.2% | 10 | 6 | 2 | 5.87% | 105.6% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 2 | 2 | 3.00% | 12.0% |
| BUY @ 3rd Alert (retest2) | 14 | 10 | 71.4% | 10 | 4 | 0 | 6.68% | 93.6% |
| SELL (all) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 2 | 2 | 3.00% | 12.0% |
| retest2 (combined) | 17 | 13 | 76.5% | 13 | 4 | 0 | 7.27% | 123.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 10:15:00 | 1476.10 | 1533.95 | 1534.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 11:15:00 | 1469.90 | 1533.31 | 1533.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-09 09:15:00 | 1527.55 | 1521.78 | 1527.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 09:15:00 | 1527.55 | 1521.78 | 1527.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 1527.55 | 1521.78 | 1527.67 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 11:15:00 | 1548.80 | 1531.97 | 1531.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-20 13:15:00 | 1554.05 | 1532.34 | 1532.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-21 11:15:00 | 1631.80 | 1632.75 | 1596.17 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-12-26 09:15:00 | 1664.65 | 1633.42 | 1598.64 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 10:15:00 | 1662.05 | 1633.71 | 1598.96 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 10:15:00 | 1745.15 | 1645.16 | 1608.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-01-03 14:15:00 | 1653.25 | 1653.94 | 1617.47 | SL hit (close<ema200) qty=0.50 sl=1653.94 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 12:15:00 | 1620.90 | 1652.31 | 1619.95 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-01-09 10:15:00 | 1628.65 | 1650.68 | 1619.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 11:15:00 | 1632.30 | 1650.50 | 1619.98 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-12 09:15:00 | 1610.30 | 1646.92 | 1620.88 | SL hit (close<static) qty=1.00 sl=1616.95 alert=retest2 |
| Cross detected — sustain check pending | 2024-01-12 13:15:00 | 1624.35 | 1645.70 | 1620.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-12 14:15:00 | 1622.55 | 1645.47 | 1620.78 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-12 15:15:00 | 1624.45 | 1645.26 | 1620.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-15 09:15:00 | 1621.95 | 1645.03 | 1620.81 | ENTRY2 sustain failed after 3960m |
| Cross detected — sustain check pending | 2024-01-15 10:15:00 | 1625.00 | 1644.83 | 1620.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 11:15:00 | 1627.90 | 1644.66 | 1620.86 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-17 09:15:00 | 1600.05 | 1642.55 | 1621.17 | SL hit (close<static) qty=1.00 sl=1616.95 alert=retest2 |
| Cross detected — sustain check pending | 2024-01-19 09:15:00 | 1633.70 | 1638.15 | 1620.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 10:15:00 | 1633.20 | 1638.10 | 1620.39 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-23 12:15:00 | 1599.45 | 1638.01 | 1621.13 | SL hit (close<static) qty=1.00 sl=1616.95 alert=retest2 |
| Cross detected — sustain check pending | 2024-01-24 13:15:00 | 1624.00 | 1635.46 | 1620.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 14:15:00 | 1626.70 | 1635.37 | 1620.52 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 1624.55 | 1635.19 | 1620.57 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-25 10:15:00 | 1605.60 | 1634.90 | 1620.50 | SL hit (close<static) qty=1.00 sl=1616.95 alert=retest2 |
| Cross detected — sustain check pending | 2024-01-25 15:15:00 | 1635.50 | 1634.60 | 1620.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 09:15:00 | 1638.50 | 1634.64 | 1620.79 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 5400m) |
| Cross detected — sustain check pending | 2024-01-29 11:15:00 | 1638.65 | 1634.65 | 1620.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 12:15:00 | 1638.80 | 1634.69 | 1621.02 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-30 11:15:00 | 1643.20 | 1634.99 | 1621.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-30 12:15:00 | 1636.40 | 1635.01 | 1621.66 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-31 09:15:00 | 1647.70 | 1634.79 | 1621.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 10:15:00 | 1649.00 | 1634.93 | 1621.95 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 1641.00 | 1656.92 | 1636.88 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-02-09 11:15:00 | 1648.95 | 1656.84 | 1636.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-02-09 12:15:00 | 1644.85 | 1656.72 | 1636.98 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-02-09 13:15:00 | 1650.25 | 1656.65 | 1637.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 14:15:00 | 1646.95 | 1656.56 | 1637.10 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-12 09:15:00 | 1648.15 | 1656.35 | 1637.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-12 10:15:00 | 1651.35 | 1656.30 | 1637.26 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-13 15:15:00 | 1650.00 | 1656.17 | 1638.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 09:15:00 | 1650.90 | 1656.11 | 1638.37 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2024-02-14 11:15:00 | 1650.35 | 1655.96 | 1638.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 12:15:00 | 1653.00 | 1655.93 | 1638.54 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 13:15:00 | 1647.50 | 1655.84 | 1638.58 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-02-15 09:15:00 | 1724.35 | 1656.55 | 1639.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-15 10:15:00 | 1719.80 | 1657.18 | 1639.60 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2024-02-16 09:15:00 | 1802.35 | 1663.81 | 1643.48 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2024-02-16 09:15:00 | 1802.68 | 1663.81 | 1643.48 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2024-02-16 09:15:00 | 1800.04 | 1663.81 | 1643.48 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2024-02-16 09:15:00 | 1811.64 | 1663.81 | 1643.48 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2024-02-16 10:15:00 | 1813.90 | 1665.29 | 1644.32 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2024-02-16 10:15:00 | 1816.48 | 1665.29 | 1644.32 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2024-02-16 10:15:00 | 1815.99 | 1665.29 | 1644.32 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2024-02-16 11:15:00 | 1818.30 | 1667.11 | 1645.34 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2024-02-22 14:15:00 | 1891.78 | 1716.96 | 1675.24 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 2782.70 | 2975.60 | 2975.69 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 14:15:00 | 3174.00 | 2973.74 | 2973.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 09:15:00 | 3189.00 | 2977.81 | 2975.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-12 09:15:00 | 2978.55 | 3037.60 | 3009.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 09:15:00 | 2978.55 | 3037.60 | 3009.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 2978.55 | 3037.60 | 3009.41 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 12:15:00 | 2756.00 | 2987.75 | 2987.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 09:15:00 | 2695.00 | 2967.89 | 2977.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 13:15:00 | 2788.40 | 2782.02 | 2855.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 09:15:00 | 2859.20 | 2787.51 | 2852.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 2859.20 | 2787.51 | 2852.59 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-03-21 14:15:00 | 2800.80 | 2790.33 | 2852.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 15:15:00 | 2786.00 | 2790.29 | 2852.09 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2025-04-07 09:15:00 | 2507.40 | 2734.33 | 2804.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 14:15:00 | 3071.40 | 2808.08 | 2806.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 3097.30 | 2816.09 | 2811.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 2977.50 | 2983.44 | 2922.72 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-06-02 10:15:00 | 3018.40 | 2983.46 | 2925.10 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 11:15:00 | 3017.40 | 2983.80 | 2925.56 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2989.30 | 3020.92 | 2962.15 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-06-13 13:15:00 | 3003.30 | 3019.86 | 2962.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 14:15:00 | 3006.00 | 3019.72 | 2963.00 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 10:15:00 | 3168.27 | 3027.96 | 2975.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 3093.90 | 3121.51 | 3055.92 | SL hit (close<ema200) qty=0.50 sl=3121.51 alert=retest1 |
| Target hit | 2025-08-18 09:15:00 | 3306.60 | 3197.30 | 3144.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 3380.00 | 3618.44 | 3619.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 10:15:00 | 3349.00 | 3615.76 | 3618.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 3592.10 | 3575.65 | 3596.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 3592.10 | 3575.65 | 3596.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 3592.10 | 3575.65 | 3596.00 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-16 09:15:00 | 3529.40 | 3590.36 | 3599.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 10:15:00 | 3510.60 | 3589.57 | 3598.81 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-19 09:15:00 | 3502.40 | 3572.32 | 3588.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:15:00 | 3478.70 | 3571.39 | 3588.30 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Target hit | 2026-03-09 10:15:00 | 3159.54 | 3467.13 | 3522.92 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-12 09:15:00 | 3130.83 | 3419.89 | 3492.85 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-12-26 10:15:00 | 1662.05 | 2023-12-29 10:15:00 | 1745.15 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-12-26 10:15:00 | 1662.05 | 2024-01-03 14:15:00 | 1653.25 | STOP_HIT | 0.50 | -0.53% |
| BUY | retest2 | 2024-01-09 11:15:00 | 1632.30 | 2024-01-12 09:15:00 | 1610.30 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-01-15 11:15:00 | 1627.90 | 2024-01-17 09:15:00 | 1600.05 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-01-19 10:15:00 | 1633.20 | 2024-01-23 12:15:00 | 1599.45 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-01-24 14:15:00 | 1626.70 | 2024-01-25 10:15:00 | 1605.60 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-01-29 09:15:00 | 1638.50 | 2024-02-16 09:15:00 | 1802.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-29 12:15:00 | 1638.80 | 2024-02-16 09:15:00 | 1802.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-30 12:15:00 | 1636.40 | 2024-02-16 09:15:00 | 1800.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-31 10:15:00 | 1649.00 | 2024-02-16 09:15:00 | 1811.64 | TARGET_HIT | 1.00 | 9.86% |
| BUY | retest2 | 2024-02-09 14:15:00 | 1646.95 | 2024-02-16 10:15:00 | 1813.90 | TARGET_HIT | 1.00 | 10.14% |
| BUY | retest2 | 2024-02-12 10:15:00 | 1651.35 | 2024-02-16 10:15:00 | 1816.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-14 09:15:00 | 1650.90 | 2024-02-16 10:15:00 | 1815.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-14 12:15:00 | 1653.00 | 2024-02-16 11:15:00 | 1818.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-15 10:15:00 | 1719.80 | 2024-02-22 14:15:00 | 1891.78 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-21 15:15:00 | 2786.00 | 2025-04-07 09:15:00 | 2507.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2025-06-02 11:15:00 | 3017.40 | 2025-06-20 10:15:00 | 3168.27 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-02 11:15:00 | 3017.40 | 2025-07-11 09:15:00 | 3093.90 | STOP_HIT | 0.50 | 2.54% |
| BUY | retest2 | 2025-06-13 14:15:00 | 3006.00 | 2025-08-18 09:15:00 | 3306.60 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-16 10:15:00 | 3510.60 | 2026-03-09 10:15:00 | 3159.54 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-19 10:15:00 | 3478.70 | 2026-03-12 09:15:00 | 3130.83 | TARGET_HIT | 1.00 | 10.00% |
