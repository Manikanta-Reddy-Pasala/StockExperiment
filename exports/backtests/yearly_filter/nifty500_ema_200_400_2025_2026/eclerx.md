# eClerx Services Ltd. (ECLERX)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1669.00
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
| ALERT3 | 13 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 12 |
| PARTIAL | 1 |
| TARGET_HIT | 5 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 8
- **Target hits / Stop hits / Partials:** 5 / 8 / 1
- **Avg / median % per leg:** 2.30% / -1.20%
- **Sum % (uncompounded):** 32.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 5 | 41.7% | 5 | 7 | 0 | 2.44% | 29.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 5 | 41.7% | 5 | 7 | 0 | 2.44% | 29.3% |
| SELL (all) | 2 | 1 | 50.0% | 0 | 1 | 1 | 1.43% | 2.9% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | 1.43% | 2.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 1 | 1 | 1.43% | 2.9% |
| retest2 (combined) | 12 | 5 | 41.7% | 5 | 7 | 0 | 2.44% | 29.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 1701.00 | 1394.37 | 1393.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 1736.40 | 1468.64 | 1433.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 1687.50 | 1691.99 | 1594.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 15:00:00 | 1687.50 | 1691.99 | 1594.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 2040.25 | 2117.78 | 2025.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 13:45:00 | 2073.00 | 2065.87 | 2021.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 13:15:00 | 2020.95 | 2064.92 | 2022.17 | SL hit (close<static) qty=1.00 sl=2021.05 alert=retest2 |

### Cycle 2 — SELL (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 13:15:00 | 1999.05 | 2254.17 | 2254.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 1958.75 | 2246.06 | 2250.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 09:15:00 | 1555.10 | 1553.74 | 1710.40 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:45:00 | 1499.60 | 1563.54 | 1694.57 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 15:15:00 | 1424.62 | 1531.65 | 1653.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 1531.80 | 1511.86 | 1629.14 | SL hit (close>ema200) qty=0.50 sl=1511.86 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-10-13 13:45:00 | 2073.00 | 2025-10-14 13:15:00 | 2020.95 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-10-15 11:00:00 | 2074.50 | 2025-10-27 09:15:00 | 2281.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-17 09:30:00 | 2078.95 | 2025-10-27 09:15:00 | 2286.84 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-20 09:15:00 | 2133.10 | 2025-10-27 09:15:00 | 2346.41 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-12 09:15:00 | 2237.90 | 2026-01-06 09:15:00 | 2461.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-18 09:15:00 | 2246.70 | 2026-01-06 09:15:00 | 2471.37 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-22 09:45:00 | 2237.00 | 2026-01-22 11:15:00 | 2199.00 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2026-01-22 10:30:00 | 2225.70 | 2026-01-22 11:15:00 | 2199.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-02-04 12:30:00 | 2261.00 | 2026-02-06 09:15:00 | 2173.60 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2026-02-04 13:15:00 | 2252.50 | 2026-02-06 09:15:00 | 2173.60 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest2 | 2026-02-04 14:30:00 | 2251.85 | 2026-02-06 09:15:00 | 2173.60 | STOP_HIT | 1.00 | -3.47% |
| BUY | retest2 | 2026-02-04 15:00:00 | 2274.40 | 2026-02-06 09:15:00 | 2173.60 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest1 | 2026-04-22 10:45:00 | 1499.60 | 2026-04-29 15:15:00 | 1424.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-22 10:45:00 | 1499.60 | 2026-05-06 11:15:00 | 1531.80 | STOP_HIT | 0.50 | -2.15% |
