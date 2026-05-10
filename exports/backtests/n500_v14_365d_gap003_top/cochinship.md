# Cochin Shipyard Ltd. (COCHINSHIP)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1769.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 2 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 0
- **Target hits / Stop hits / Partials:** 0 / 2 / 2
- **Avg / median % per leg:** 2.78% / 5.00%
- **Sum % (uncompounded):** 11.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.78% | 11.1% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.78% | 11.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.78% | 11.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

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
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 1713.90 | 1708.58 | 1760.78 | SL hit (close>ema200) qty=0.50 sl=1708.58 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1780.50 | 1710.31 | 1760.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:30:00 | 1807.10 | 1710.31 | 1760.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 1785.50 | 1711.06 | 1760.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:30:00 | 1795.60 | 1711.06 | 1760.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| CROSSOVER_SKIP | 2025-09-26 10:15:00 | 1891.80 | 1795.48 | 1795.26 | min_gap filter: gap=0.011% < 0.030% |
| TREND_RESET | 2025-09-26 10:15:00 | 1891.80 | 1795.48 | 1795.26 | EMA inversion without crossover edge (EMA200=1795.48 EMA400=1795.26) — end cycle |
| CROSSOVER_SKIP | 2025-11-04 14:15:00 | 1768.10 | 1800.02 | 1800.12 | min_gap filter: gap=0.006% < 0.030% |
| CROSSOVER_SKIP | 2026-04-27 15:15:00 | 1678.80 | 1481.89 | 1481.41 | min_gap filter: gap=0.029% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-09-03 13:15:00 | 1729.20 | 2025-09-09 09:15:00 | 1642.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-09-04 09:15:00 | 1718.00 | 2025-09-09 11:15:00 | 1632.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-09-03 13:15:00 | 1729.20 | 2025-09-12 11:15:00 | 1713.90 | STOP_HIT | 0.50 | 0.88% |
| SELL | retest1 | 2025-09-04 09:15:00 | 1718.00 | 2025-09-12 11:15:00 | 1713.90 | STOP_HIT | 0.50 | 0.24% |
