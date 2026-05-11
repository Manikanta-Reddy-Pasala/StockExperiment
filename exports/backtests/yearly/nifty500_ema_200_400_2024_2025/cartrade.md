# Cartrade Tech Ltd. (CARTRADE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1949.90
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
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 50 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 0 |
| TARGET_HIT | 9 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 16
- **Target hits / Stop hits / Partials:** 9 / 16 / 0
- **Avg / median % per leg:** 1.41% / -0.76%
- **Sum % (uncompounded):** 35.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 9 | 36.0% | 9 | 16 | 0 | 1.41% | 35.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 25 | 9 | 36.0% | 9 | 16 | 0 | 1.41% | 35.3% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 9 | 36.0% | 9 | 16 | 0 | 1.41% | 35.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 11:15:00 | 1571.50 | 1602.55 | 1602.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 13:15:00 | 1559.80 | 1601.85 | 1602.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 12:15:00 | 1602.00 | 1597.75 | 1600.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 12:15:00 | 1602.00 | 1597.75 | 1600.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 12:15:00 | 1602.00 | 1597.75 | 1600.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 13:00:00 | 1602.00 | 1597.75 | 1600.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 13:15:00 | 1602.90 | 1597.80 | 1600.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:00:00 | 1602.90 | 1597.80 | 1600.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 1605.00 | 1597.88 | 1600.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:45:00 | 1605.30 | 1597.88 | 1600.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 1605.00 | 1597.95 | 1600.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:15:00 | 1622.30 | 1597.95 | 1600.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 10:15:00 | 1657.20 | 1602.46 | 1602.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 15:15:00 | 1685.00 | 1613.69 | 1608.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 1628.80 | 1628.96 | 1617.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 15:00:00 | 1628.80 | 1628.96 | 1617.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 12:15:00 | 1617.30 | 1629.07 | 1618.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:00:00 | 1617.30 | 1629.07 | 1618.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 1612.30 | 1628.90 | 1617.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:45:00 | 1613.80 | 1628.90 | 1617.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 1619.30 | 1628.80 | 1617.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:15:00 | 1606.50 | 1628.80 | 1617.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 1606.50 | 1628.58 | 1617.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 1641.20 | 1628.58 | 1617.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 14:00:00 | 1620.20 | 1628.33 | 1618.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 15:15:00 | 1625.00 | 1628.24 | 1618.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 1592.80 | 1627.85 | 1617.98 | SL hit (close<static) qty=1.00 sl=1604.30 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 12:15:00 | 2519.00 | 2765.23 | 2765.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 2443.40 | 2730.98 | 2748.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1866.00 | 1813.09 | 2015.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:00:00 | 1866.00 | 1813.09 | 2015.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1908.00 | 1755.58 | 1883.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:00:00 | 1908.00 | 1755.58 | 1883.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 2097.00 | 1758.98 | 1884.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 12:00:00 | 2097.00 | 1758.98 | 1884.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-05 11:15:00 | 788.05 | 2024-06-26 14:15:00 | 786.70 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2024-06-06 09:15:00 | 786.90 | 2024-06-26 14:15:00 | 786.70 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2024-06-07 09:45:00 | 792.75 | 2024-06-26 14:15:00 | 786.70 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-06-11 09:45:00 | 788.75 | 2024-06-26 14:15:00 | 786.70 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2024-06-12 09:45:00 | 807.10 | 2024-06-27 15:15:00 | 759.55 | STOP_HIT | 1.00 | -5.89% |
| BUY | retest2 | 2024-06-12 10:45:00 | 804.45 | 2024-06-27 15:15:00 | 759.55 | STOP_HIT | 1.00 | -5.58% |
| BUY | retest2 | 2024-06-13 15:00:00 | 803.45 | 2024-06-27 15:15:00 | 759.55 | STOP_HIT | 1.00 | -5.46% |
| BUY | retest2 | 2024-06-14 15:00:00 | 805.50 | 2024-06-27 15:15:00 | 759.55 | STOP_HIT | 1.00 | -5.70% |
| BUY | retest2 | 2024-08-13 14:15:00 | 847.95 | 2024-08-14 09:15:00 | 820.50 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2024-08-16 09:30:00 | 861.90 | 2024-08-23 09:15:00 | 948.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-30 09:30:00 | 849.05 | 2024-09-05 10:15:00 | 933.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-30 15:00:00 | 848.60 | 2024-09-05 10:15:00 | 933.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-03 12:15:00 | 890.65 | 2024-09-12 09:15:00 | 977.08 | TARGET_HIT | 1.00 | 9.70% |
| BUY | retest2 | 2024-09-03 13:30:00 | 888.25 | 2024-09-13 11:15:00 | 979.72 | TARGET_HIT | 1.00 | 10.30% |
| BUY | retest2 | 2024-09-03 14:15:00 | 890.45 | 2024-09-13 11:15:00 | 979.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-03 15:15:00 | 890.00 | 2024-09-13 11:15:00 | 979.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-11 10:45:00 | 921.00 | 2024-10-17 12:15:00 | 1013.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-25 09:15:00 | 1641.20 | 2025-06-26 09:15:00 | 1592.80 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2025-06-25 14:00:00 | 1620.20 | 2025-06-26 09:15:00 | 1592.80 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-06-25 15:15:00 | 1625.00 | 2025-06-26 09:15:00 | 1592.80 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-06-30 09:15:00 | 1631.60 | 2025-07-08 12:15:00 | 1794.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-22 14:45:00 | 2907.00 | 2025-12-23 13:15:00 | 2776.90 | STOP_HIT | 1.00 | -4.48% |
| BUY | retest2 | 2026-01-02 10:00:00 | 2945.00 | 2026-01-08 10:15:00 | 2750.20 | STOP_HIT | 1.00 | -6.61% |
| BUY | retest2 | 2026-01-02 12:15:00 | 2890.70 | 2026-01-08 10:15:00 | 2750.20 | STOP_HIT | 1.00 | -4.86% |
| BUY | retest2 | 2026-01-02 13:00:00 | 2895.40 | 2026-01-08 10:15:00 | 2750.20 | STOP_HIT | 1.00 | -5.01% |
