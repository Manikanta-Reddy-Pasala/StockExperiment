# Ashok Leyland Ltd. (ASHOKLEY)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 168.77
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 15 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 13
- **Target hits / Stop hits / Partials:** 1 / 18 / 5
- **Avg / median % per leg:** 1.28% / -0.50%
- **Sum % (uncompounded):** 30.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 9 | 52.9% | 1 | 12 | 4 | 2.19% | 37.3% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 0 | 4 | 4 | 4.99% | 39.9% |
| BUY @ 3rd Alert (retest2) | 9 | 1 | 11.1% | 1 | 8 | 0 | -0.29% | -2.6% |
| SELL (all) | 7 | 2 | 28.6% | 0 | 6 | 1 | -0.94% | -6.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 2 | 28.6% | 0 | 6 | 1 | -0.94% | -6.6% |
| retest1 (combined) | 8 | 8 | 100.0% | 0 | 4 | 4 | 4.99% | 39.9% |
| retest2 (combined) | 16 | 3 | 18.8% | 1 | 14 | 1 | -0.57% | -9.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 10:15:00 | 172.24 | 185.63 | 185.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-25 14:15:00 | 171.07 | 185.08 | 185.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 09:15:00 | 175.71 | 173.61 | 178.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-10 10:00:00 | 175.71 | 173.61 | 178.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 177.43 | 173.76 | 178.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 15:00:00 | 175.38 | 173.91 | 178.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:45:00 | 175.01 | 173.95 | 178.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 09:30:00 | 175.30 | 174.11 | 178.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 11:00:00 | 175.20 | 174.14 | 178.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 178.36 | 174.27 | 178.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:00:00 | 178.36 | 174.27 | 178.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 178.45 | 174.32 | 178.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:00:00 | 178.45 | 174.32 | 178.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 176.78 | 174.34 | 178.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 15:00:00 | 176.61 | 174.41 | 178.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 11:15:00 | 179.88 | 174.54 | 178.05 | SL hit (close>static) qty=1.00 sl=178.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-22 11:15:00 | 179.88 | 174.54 | 178.05 | SL hit (close>static) qty=1.00 sl=178.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-22 11:15:00 | 179.88 | 174.54 | 178.05 | SL hit (close>static) qty=1.00 sl=178.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-22 11:15:00 | 179.88 | 174.54 | 178.05 | SL hit (close>static) qty=1.00 sl=178.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-22 11:15:00 | 179.88 | 174.54 | 178.05 | SL hit (close>static) qty=1.00 sl=178.58 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 174.40 | 174.74 | 178.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:15:00 | 165.68 | 173.35 | 176.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 12:15:00 | 172.80 | 170.28 | 174.59 | SL hit (close>ema200) qty=0.50 sl=170.28 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-13 10:15:00 | 116.51 | 2025-06-25 14:15:00 | 122.34 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-16 10:15:00 | 116.94 | 2025-06-25 14:15:00 | 122.39 | PARTIAL | 0.50 | 4.66% |
| BUY | retest1 | 2025-06-19 12:00:00 | 116.56 | 2025-06-26 09:15:00 | 122.79 | PARTIAL | 0.50 | 5.34% |
| BUY | retest1 | 2025-06-20 09:15:00 | 116.79 | 2025-06-26 09:15:00 | 122.63 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-13 10:15:00 | 116.51 | 2025-07-18 10:15:00 | 122.50 | STOP_HIT | 0.50 | 5.14% |
| BUY | retest1 | 2025-06-16 10:15:00 | 116.94 | 2025-07-18 10:15:00 | 122.50 | STOP_HIT | 0.50 | 4.75% |
| BUY | retest1 | 2025-06-19 12:00:00 | 116.56 | 2025-07-18 10:15:00 | 122.50 | STOP_HIT | 0.50 | 5.10% |
| BUY | retest1 | 2025-06-20 09:15:00 | 116.79 | 2025-07-18 10:15:00 | 122.50 | STOP_HIT | 0.50 | 4.89% |
| BUY | retest2 | 2025-07-31 11:30:00 | 121.30 | 2025-08-01 09:15:00 | 118.80 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-07-31 12:15:00 | 121.35 | 2025-08-01 09:15:00 | 118.80 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-07-31 13:15:00 | 121.30 | 2025-08-01 09:15:00 | 118.80 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-07-31 14:00:00 | 121.50 | 2025-08-01 09:15:00 | 118.80 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-08-04 09:15:00 | 122.25 | 2025-08-07 11:15:00 | 119.70 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-08-07 09:45:00 | 120.93 | 2025-08-07 11:15:00 | 119.70 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-08-13 10:00:00 | 120.59 | 2025-08-13 13:15:00 | 119.99 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-08-13 11:00:00 | 120.69 | 2025-08-13 13:15:00 | 119.99 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-08-18 09:15:00 | 129.60 | 2025-09-19 10:15:00 | 142.56 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-15 15:00:00 | 175.38 | 2026-04-22 11:15:00 | 179.88 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2026-04-16 09:45:00 | 175.01 | 2026-04-22 11:15:00 | 179.88 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2026-04-17 09:30:00 | 175.30 | 2026-04-22 11:15:00 | 179.88 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2026-04-20 11:00:00 | 175.20 | 2026-04-22 11:15:00 | 179.88 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-04-21 15:00:00 | 176.61 | 2026-04-22 11:15:00 | 179.88 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2026-04-23 09:15:00 | 174.40 | 2026-04-29 11:15:00 | 165.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 09:15:00 | 174.40 | 2026-05-07 12:15:00 | 172.80 | STOP_HIT | 0.50 | 0.92% |
