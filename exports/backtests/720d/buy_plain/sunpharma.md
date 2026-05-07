# SUNPHARMA (SUNPHARMA)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1832.40
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
| ALERT2_SKIP | 4 |
| ALERT3 | 5 |
| PENDING | 9 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / Stop hits / Partials:** 0 / 8 / 0
- **Avg / median % per leg:** -1.30% / -1.91%
- **Sum % (uncompounded):** -10.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 0 | 8 | 0 | -1.30% | -10.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 2 | 25.0% | 0 | 8 | 0 | -1.30% | -10.4% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 2 | 25.0% | 0 | 8 | 0 | -1.30% | -10.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 1850.20 | 1809.03 | 1808.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 1859.15 | 1810.35 | 1809.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 15:15:00 | 1832.80 | 1834.09 | 1823.77 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 09:15:00 | 1820.05 | 1833.95 | 1823.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1820.05 | 1833.95 | 1823.75 | EMA400 retest candle locked |

### Cycle 2 — BUY (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 15:15:00 | 1781.80 | 1725.07 | 1724.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 09:15:00 | 1817.90 | 1726.00 | 1725.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 15:15:00 | 1755.00 | 1762.28 | 1746.45 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 1753.30 | 1762.19 | 1746.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 1753.30 | 1762.19 | 1746.48 | EMA400 retest candle locked |

### Cycle 3 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 1691.90 | 1640.23 | 1640.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 1701.40 | 1643.22 | 1641.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 09:15:00 | 1754.20 | 1771.79 | 1737.47 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 1743.50 | 1771.27 | 1737.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1743.50 | 1771.27 | 1737.55 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-22 10:15:00 | 1755.70 | 1768.24 | 1738.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 11:15:00 | 1758.00 | 1768.14 | 1738.21 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 1735.00 | 1767.12 | 1739.58 | SL hit (close<static) qty=1.00 sl=1737.20 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-06 13:15:00 | 1754.50 | 1748.84 | 1736.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 14:15:00 | 1761.70 | 1748.97 | 1736.13 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-09 10:15:00 | 1750.10 | 1752.12 | 1738.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-09 11:15:00 | 1744.70 | 1752.04 | 1738.86 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-01-09 13:15:00 | 1728.00 | 1751.67 | 1738.81 | SL hit (close<static) qty=1.00 sl=1737.20 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-25 09:15:00 | 1752.80 | 1702.43 | 1704.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 10:15:00 | 1761.00 | 1703.02 | 1705.22 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-26 10:15:00 | 1780.10 | 1707.48 | 1707.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 1780.10 | 1707.48 | 1707.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 11:15:00 | 1804.10 | 1724.48 | 1716.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 10:15:00 | 1759.70 | 1760.04 | 1739.48 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 14:15:00 | 1739.70 | 1759.85 | 1739.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 1739.70 | 1759.85 | 1739.79 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-20 09:15:00 | 1761.40 | 1759.76 | 1739.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 10:15:00 | 1766.70 | 1759.83 | 1740.08 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-24 09:15:00 | 1761.80 | 1760.71 | 1741.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 1761.70 | 1760.72 | 1741.88 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-25 09:15:00 | 1776.60 | 1760.87 | 1742.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 1782.40 | 1761.08 | 1742.71 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-30 15:15:00 | 1762.00 | 1765.86 | 1746.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 1777.80 | 1765.97 | 1747.06 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 2520m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 1740.10 | 1765.60 | 1747.15 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-01 13:15:00 | 1721.50 | 1765.16 | 1747.03 | SL hit (close<static) qty=1.00 sl=1737.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 13:15:00 | 1721.50 | 1765.16 | 1747.03 | SL hit (close<static) qty=1.00 sl=1737.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 13:15:00 | 1721.50 | 1765.16 | 1747.03 | SL hit (close<static) qty=1.00 sl=1737.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 13:15:00 | 1721.50 | 1765.16 | 1747.03 | SL hit (close<static) qty=1.00 sl=1737.90 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-29 14:15:00 | 1780.10 | 1713.68 | 1720.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 15:15:00 | 1773.00 | 1714.27 | 1721.23 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-05 10:15:00 | 1829.70 | 1728.28 | 1727.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 1829.70 | 1728.28 | 1727.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 1842.70 | 1734.74 | 1731.25 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-12-22 11:15:00 | 1758.00 | 2025-12-24 10:15:00 | 1735.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-01-06 14:15:00 | 1761.70 | 2026-01-09 13:15:00 | 1728.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-02-25 10:15:00 | 1761.00 | 2026-02-26 10:15:00 | 1780.10 | STOP_HIT | 1.00 | 1.08% |
| BUY | retest2 | 2026-03-20 10:15:00 | 1766.70 | 2026-04-01 13:15:00 | 1721.50 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2026-03-24 10:15:00 | 1761.70 | 2026-04-01 13:15:00 | 1721.50 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2026-03-25 10:15:00 | 1782.40 | 2026-04-01 13:15:00 | 1721.50 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2026-04-01 09:15:00 | 1777.80 | 2026-04-01 13:15:00 | 1721.50 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2026-04-29 15:15:00 | 1773.00 | 2026-05-05 10:15:00 | 1829.70 | STOP_HIT | 1.00 | 3.20% |
