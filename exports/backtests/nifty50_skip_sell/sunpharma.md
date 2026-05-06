# SUNPHARMA (SUNPHARMA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 1850.20
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 4 |
| PENDING | 13 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 1 |
| ENTRY2 | 8 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 8
- **Target hits / Stop hits / Partials:** 0 / 9 / 0
- **Avg / median % per leg:** -1.21% / -1.60%
- **Sum % (uncompounded):** -10.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 1 | 11.1% | 0 | 9 | 0 | -1.21% | -10.8% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.71% | -3.7% |
| BUY @ 3rd Alert (retest2) | 8 | 1 | 12.5% | 0 | 8 | 0 | -0.89% | -7.1% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.71% | -3.7% |
| retest2 (combined) | 8 | 1 | 12.5% | 0 | 8 | 0 | -0.89% | -7.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 12:15:00 | 1525.40 | 1505.54 | 1505.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 14:15:00 | 1535.65 | 1506.05 | 1505.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-23 09:15:00 | 1873.30 | 1880.50 | 1824.28 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-10-28 13:15:00 | 1896.95 | 1876.18 | 1828.56 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 14:15:00 | 1901.30 | 1876.43 | 1828.92 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 1837.70 | 1876.01 | 1830.80 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-10-30 09:15:00 | 1830.80 | 1876.01 | 1830.80 | SL hit qty=1.00 sl=1830.80 alert=retest1 |
| Cross detected — sustain check pending | 2024-10-30 12:15:00 | 1868.80 | 1875.52 | 1831.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-30 13:15:00 | 1851.75 | 1875.29 | 1831.33 | ENTRY2 sustain failed after 60m |

### Cycle 2 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 1850.20 | 1809.01 | 1808.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 1859.15 | 1810.34 | 1809.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 15:15:00 | 1831.95 | 1834.06 | 1823.66 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 09:15:00 | 1820.10 | 1833.92 | 1823.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1820.10 | 1833.92 | 1823.64 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-23 13:15:00 | 1838.50 | 1806.54 | 1810.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-23 14:15:00 | 1833.65 | 1806.81 | 1810.51 | ENTRY2 sustain failed after 60m |

### Cycle 3 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 1818.00 | 1726.01 | 1725.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 10:15:00 | 1829.00 | 1727.04 | 1726.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 1753.30 | 1762.34 | 1746.74 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 1753.30 | 1762.34 | 1746.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 1753.30 | 1762.34 | 1746.74 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2025-10-20 12:15:00 | 1691.90 | 1640.29 | 1640.06 | HTF filter: close below htf_sma |
| Cross detected — sustain check pending | 2025-11-19 10:15:00 | 1768.10 | 1697.81 | 1676.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 11:15:00 | 1782.90 | 1698.65 | 1676.65 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-18 11:15:00 | 1743.80 | 1771.29 | 1737.55 | SL hit qty=1.00 sl=1743.80 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-22 14:15:00 | 1770.80 | 1768.10 | 1738.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 15:15:00 | 1769.80 | 1768.12 | 1738.78 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 1743.80 | 1767.44 | 1739.59 | SL hit qty=1.00 sl=1743.80 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-07 09:15:00 | 1798.00 | 1749.62 | 1736.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:15:00 | 1774.60 | 1749.87 | 1736.77 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 1743.80 | 1752.18 | 1738.80 | SL hit qty=1.00 sl=1743.80 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-26 09:15:00 | 1788.70 | 1708.09 | 1708.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 10:15:00 | 1780.20 | 1708.80 | 1709.04 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-26 11:15:00 | 1777.80 | 1709.49 | 1709.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 11:15:00 | 1777.80 | 1709.49 | 1709.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 15:15:00 | 1787.40 | 1712.26 | 1710.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 10:15:00 | 1759.70 | 1760.58 | 1740.50 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 14:15:00 | 1739.70 | 1760.37 | 1740.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 1739.70 | 1760.37 | 1740.79 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-20 09:15:00 | 1761.20 | 1760.28 | 1740.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 10:15:00 | 1766.60 | 1760.34 | 1741.07 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-24 09:15:00 | 1761.80 | 1761.33 | 1742.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 1761.70 | 1761.33 | 1742.89 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-25 09:15:00 | 1775.60 | 1761.44 | 1743.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 1783.10 | 1761.65 | 1743.69 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-01 09:15:00 | 1777.80 | 1766.36 | 1747.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-01 10:15:00 | 1757.90 | 1766.28 | 1747.95 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-01 11:15:00 | 1761.50 | 1766.23 | 1748.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-01 12:15:00 | 1740.10 | 1765.97 | 1747.98 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 1738.30 | 1765.97 | 1747.98 | SL hit qty=1.00 sl=1738.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 1738.30 | 1765.97 | 1747.98 | SL hit qty=1.00 sl=1738.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 1738.30 | 1765.97 | 1747.98 | SL hit qty=1.00 sl=1738.30 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-29 12:15:00 | 1762.00 | 1713.89 | 1721.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 13:15:00 | 1762.10 | 1714.37 | 1722.01 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-05 10:15:00 | 1829.70 | 1729.44 | 1729.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 1829.70 | 1729.44 | 1729.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 1842.70 | 1735.85 | 1732.46 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-10-28 14:15:00 | 1901.30 | 2024-10-30 09:15:00 | 1830.80 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2025-11-19 11:15:00 | 1782.90 | 2025-12-18 11:15:00 | 1743.80 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-12-22 15:15:00 | 1769.80 | 2025-12-24 09:15:00 | 1743.80 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-01-07 10:15:00 | 1774.60 | 2026-01-09 09:15:00 | 1743.80 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-02-26 10:15:00 | 1780.20 | 2026-02-26 11:15:00 | 1777.80 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2026-03-20 10:15:00 | 1766.60 | 2026-04-01 12:15:00 | 1738.30 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-03-24 10:15:00 | 1761.70 | 2026-04-01 12:15:00 | 1738.30 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-03-25 10:15:00 | 1783.10 | 2026-04-01 12:15:00 | 1738.30 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2026-04-29 13:15:00 | 1762.10 | 2026-05-05 10:15:00 | 1829.70 | STOP_HIT | 1.00 | 3.84% |
