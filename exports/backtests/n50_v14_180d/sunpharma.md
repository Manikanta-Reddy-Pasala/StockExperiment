# SUNPHARMA (SUNPHARMA)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 1845.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 13 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 11
- **Target hits / Stop hits / Partials:** 0 / 11 / 0
- **Avg / median % per leg:** -1.73% / -1.50%
- **Sum % (uncompounded):** -19.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 0 | 0.0% | 0 | 11 | 0 | -1.73% | -19.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 0 | 0.0% | 0 | 11 | 0 | -1.73% | -19.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 0 | 0.0% | 0 | 11 | 0 | -1.73% | -19.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 1621.30 | 1728.59 | 1728.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 09:15:00 | 1587.80 | 1697.02 | 1711.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1683.00 | 1675.58 | 1698.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1683.00 | 1675.58 | 1698.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1683.00 | 1675.58 | 1698.24 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 1780.10 | 1707.48 | 1707.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 11:15:00 | 1804.10 | 1724.48 | 1716.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 10:15:00 | 1759.70 | 1760.04 | 1739.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 11:00:00 | 1759.70 | 1760.04 | 1739.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 1739.70 | 1759.85 | 1739.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 1739.70 | 1759.85 | 1739.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 1749.60 | 1759.75 | 1739.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 1764.40 | 1759.75 | 1739.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 1775.50 | 1760.70 | 1741.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 09:15:00 | 1770.00 | 1760.71 | 1742.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 15:15:00 | 1762.00 | 1765.89 | 1746.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 1740.10 | 1765.60 | 1747.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:00:00 | 1740.10 | 1765.60 | 1747.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 1721.50 | 1765.16 | 1747.03 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-01 13:15:00 | 1721.50 | 1765.16 | 1747.03 | SL hit (close<static) qty=1.00 sl=1738.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 13:15:00 | 1721.50 | 1765.16 | 1747.03 | SL hit (close<static) qty=1.00 sl=1738.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 13:15:00 | 1721.50 | 1765.16 | 1747.03 | SL hit (close<static) qty=1.00 sl=1738.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 13:15:00 | 1721.50 | 1765.16 | 1747.03 | SL hit (close<static) qty=1.00 sl=1738.90 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-04-01 13:30:00 | 1721.10 | 1765.16 | 1747.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 1694.70 | 1750.07 | 1741.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:30:00 | 1690.60 | 1750.07 | 1741.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 09:15:00 | 1670.00 | 1733.20 | 1733.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 1640.00 | 1711.32 | 1721.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1737.50 | 1706.49 | 1718.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1737.50 | 1706.49 | 1718.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1737.50 | 1706.49 | 1718.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 1737.50 | 1706.49 | 1718.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1722.30 | 1706.64 | 1718.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 1741.40 | 1706.64 | 1718.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 1757.80 | 1708.76 | 1719.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:00:00 | 1757.80 | 1708.76 | 1719.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 1829.70 | 1728.28 | 1727.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 1842.70 | 1734.74 | 1731.25 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-12-19 09:15:00 | 1754.40 | 2025-12-24 10:15:00 | 1735.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-22 09:15:00 | 1757.80 | 2025-12-24 10:15:00 | 1735.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-12-22 10:15:00 | 1754.90 | 2025-12-24 10:15:00 | 1735.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2026-01-06 13:30:00 | 1754.30 | 2026-01-09 13:15:00 | 1728.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-01-09 15:15:00 | 1736.10 | 2026-01-12 09:15:00 | 1719.70 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-01-12 13:30:00 | 1734.30 | 2026-01-13 09:15:00 | 1719.10 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-01-13 14:45:00 | 1732.40 | 2026-01-14 09:15:00 | 1704.00 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2026-03-20 09:15:00 | 1764.40 | 2026-04-01 13:15:00 | 1721.50 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2026-03-24 09:15:00 | 1775.50 | 2026-04-01 13:15:00 | 1721.50 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2026-03-25 09:15:00 | 1770.00 | 2026-04-01 13:15:00 | 1721.50 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2026-03-30 15:15:00 | 1762.00 | 2026-04-01 13:15:00 | 1721.50 | STOP_HIT | 1.00 | -2.30% |
