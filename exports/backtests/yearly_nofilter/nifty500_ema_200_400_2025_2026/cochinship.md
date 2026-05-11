# Cochin Shipyard Ltd. (COCHINSHIP)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1769.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 4 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 3 |
| PARTIAL | 5 |
| TARGET_HIT | 5 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 0
- **Target hits / Stop hits / Partials:** 3 / 2 / 5
- **Avg / median % per leg:** 5.61% / 5.00%
- **Sum % (uncompounded):** 56.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 10 | 100.0% | 3 | 2 | 5 | 5.61% | 56.1% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.78% | 11.1% |
| SELL @ 3rd Alert (retest2) | 6 | 6 | 100.0% | 3 | 0 | 3 | 7.50% | 45.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.78% | 11.1% |
| retest2 (combined) | 6 | 6 | 100.0% | 3 | 0 | 3 | 7.50% | 45.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 1734.00 | 1883.43 | 1884.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 15:15:00 | 1731.00 | 1879.00 | 1881.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 12:15:00 | 1748.50 | 1735.68 | 1788.47 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-03 13:15:00 | 1729.20 | 1736.41 | 1787.02 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-04 09:15:00 | 1718.00 | 1736.45 | 1786.29 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 09:15:00 | 1642.74 | 1722.20 | 1773.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 11:15:00 | 1632.10 | 1720.47 | 1772.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 1713.90 | 1708.58 | 1760.78 | SL hit (close>ema200) qty=0.50 sl=1708.58 alert=retest1 |

### Cycle 2 — BUY (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 10:15:00 | 1891.80 | 1795.48 | 1795.26 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 1768.10 | 1800.02 | 1800.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 1719.00 | 1798.93 | 1799.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 1796.70 | 1784.71 | 1791.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 1796.70 | 1784.71 | 1791.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 1796.70 | 1784.71 | 1791.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:45:00 | 1796.80 | 1784.71 | 1791.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 1791.80 | 1784.78 | 1791.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 11:15:00 | 1788.50 | 1784.78 | 1791.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 14:15:00 | 1786.00 | 1784.96 | 1791.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 1741.60 | 1785.11 | 1791.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 09:15:00 | 1699.07 | 1784.54 | 1791.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 09:15:00 | 1696.70 | 1784.54 | 1791.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 09:15:00 | 1654.52 | 1784.54 | 1791.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-08 14:15:00 | 1609.65 | 1701.61 | 1736.93 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 1678.80 | 1481.89 | 1481.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 1686.30 | 1483.92 | 1482.43 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-09-03 13:15:00 | 1729.20 | 2025-09-09 09:15:00 | 1642.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-09-04 09:15:00 | 1718.00 | 2025-09-09 11:15:00 | 1632.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-09-03 13:15:00 | 1729.20 | 2025-09-12 11:15:00 | 1713.90 | STOP_HIT | 0.50 | 0.88% |
| SELL | retest1 | 2025-09-04 09:15:00 | 1718.00 | 2025-09-12 11:15:00 | 1713.90 | STOP_HIT | 0.50 | 0.24% |
| SELL | retest2 | 2025-11-12 11:15:00 | 1788.50 | 2025-11-13 09:15:00 | 1699.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 14:15:00 | 1786.00 | 2025-11-13 09:15:00 | 1696.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-13 09:15:00 | 1741.60 | 2025-11-13 09:15:00 | 1654.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 11:15:00 | 1788.50 | 2025-12-08 14:15:00 | 1609.65 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-12 14:15:00 | 1786.00 | 2025-12-08 14:15:00 | 1607.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-13 09:15:00 | 1741.60 | 2025-12-09 09:15:00 | 1567.44 | TARGET_HIT | 0.50 | 10.00% |
