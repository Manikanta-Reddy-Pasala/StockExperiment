# Concord Biotech Ltd. (CONCORDBIO)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1168.40
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
| ALERT2_SKIP | 1 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 7
- **Target hits / Stop hits / Partials:** 1 / 8 / 1
- **Avg / median % per leg:** -0.22% / -1.66%
- **Sum % (uncompounded):** -2.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 1 | 12.5% | 1 | 7 | 0 | -1.30% | -10.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 1 | 12.5% | 1 | 7 | 0 | -1.30% | -10.4% |
| SELL (all) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.10% | 8.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.10% | 8.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 10 | 3 | 30.0% | 1 | 8 | 1 | -0.22% | -2.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 15:15:00 | 2009.00 | 1684.35 | 1682.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 2052.50 | 1688.01 | 1684.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 1833.50 | 1852.35 | 1781.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 15:00:00 | 1833.50 | 1852.35 | 1781.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1825.00 | 1851.57 | 1781.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 10:45:00 | 1837.20 | 1851.43 | 1782.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 14:00:00 | 1836.70 | 1850.69 | 1782.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 10:45:00 | 1839.80 | 1845.28 | 1783.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:15:00 | 1836.10 | 1841.93 | 1785.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 1790.10 | 1837.68 | 1788.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 1788.00 | 1837.68 | 1788.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1770.00 | 1837.00 | 1788.81 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 1770.00 | 1837.00 | 1788.81 | SL hit (close<static) qty=1.00 sl=1773.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 1770.00 | 1837.00 | 1788.81 | SL hit (close<static) qty=1.00 sl=1773.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 1770.00 | 1837.00 | 1788.81 | SL hit (close<static) qty=1.00 sl=1773.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 1770.00 | 1837.00 | 1788.81 | SL hit (close<static) qty=1.00 sl=1773.60 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 1770.00 | 1837.00 | 1788.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 1775.00 | 1836.38 | 1788.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:15:00 | 1771.30 | 1836.38 | 1788.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1774.90 | 1820.88 | 1785.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 14:30:00 | 1810.30 | 1814.52 | 1783.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 15:15:00 | 1800.00 | 1814.52 | 1783.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 11:30:00 | 1801.90 | 1815.27 | 1786.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 13:45:00 | 1799.80 | 1813.93 | 1786.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-24 10:15:00 | 1979.78 | 1848.71 | 1812.78 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 1843.40 | 1856.60 | 1821.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 1843.40 | 1856.60 | 1821.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 1823.90 | 1854.88 | 1821.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 1816.30 | 1854.88 | 1821.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1795.20 | 1854.28 | 1821.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-31 14:15:00 | 1770.10 | 1850.85 | 1820.77 | SL hit (close<static) qty=1.00 sl=1770.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 14:15:00 | 1770.10 | 1850.85 | 1820.77 | SL hit (close<static) qty=1.00 sl=1770.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 14:15:00 | 1770.10 | 1850.85 | 1820.77 | SL hit (close<static) qty=1.00 sl=1770.50 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 1611.40 | 1795.14 | 1795.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 1600.80 | 1789.62 | 1793.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 11:15:00 | 1763.00 | 1731.43 | 1758.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 11:15:00 | 1763.00 | 1731.43 | 1758.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1763.00 | 1731.43 | 1758.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 1763.00 | 1731.43 | 1758.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 1741.40 | 1731.53 | 1758.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:45:00 | 1790.30 | 1731.53 | 1758.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1765.80 | 1732.43 | 1758.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 1765.80 | 1732.43 | 1758.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1764.40 | 1732.75 | 1758.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:30:00 | 1770.30 | 1732.75 | 1758.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1743.00 | 1734.68 | 1758.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 10:15:00 | 1734.90 | 1734.68 | 1758.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 10:15:00 | 1648.15 | 1711.45 | 1739.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 1679.30 | 1677.43 | 1712.13 | SL hit (close>ema200) qty=0.50 sl=1677.43 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-23 10:45:00 | 1837.20 | 2025-07-02 09:15:00 | 1770.00 | STOP_HIT | 1.00 | -3.66% |
| BUY | retest2 | 2025-06-23 14:00:00 | 1836.70 | 2025-07-02 09:15:00 | 1770.00 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest2 | 2025-06-25 10:45:00 | 1839.80 | 2025-07-02 09:15:00 | 1770.00 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2025-06-27 10:15:00 | 1836.10 | 2025-07-02 09:15:00 | 1770.00 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2025-07-08 14:30:00 | 1810.30 | 2025-07-24 10:15:00 | 1979.78 | TARGET_HIT | 1.00 | 9.36% |
| BUY | retest2 | 2025-07-08 15:15:00 | 1800.00 | 2025-07-31 14:15:00 | 1770.10 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-07-10 11:30:00 | 1801.90 | 2025-07-31 14:15:00 | 1770.10 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-07-11 13:45:00 | 1799.80 | 2025-07-31 14:15:00 | 1770.10 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-08-25 10:15:00 | 1734.90 | 2025-09-08 10:15:00 | 1648.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 10:15:00 | 1734.90 | 2025-09-19 09:15:00 | 1679.30 | STOP_HIT | 0.50 | 3.20% |
