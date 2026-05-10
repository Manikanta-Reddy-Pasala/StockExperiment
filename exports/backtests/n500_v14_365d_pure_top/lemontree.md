# Lemon Tree Hotels Ltd. (LEMONTREE)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 120.45
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
| ALERT2_SKIP | 1 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 27 |
| PARTIAL | 4 |
| TARGET_HIT | 6 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 31 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 21
- **Target hits / Stop hits / Partials:** 6 / 21 / 4
- **Avg / median % per leg:** 1.38% / -1.12%
- **Sum % (uncompounded):** 42.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 2 | 14.3% | 2 | 12 | 0 | 0.12% | 1.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 2 | 14.3% | 2 | 12 | 0 | 0.12% | 1.7% |
| SELL (all) | 17 | 8 | 47.1% | 4 | 9 | 4 | 2.41% | 40.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 17 | 8 | 47.1% | 4 | 9 | 4 | 2.41% | 40.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 31 | 10 | 32.3% | 6 | 21 | 4 | 1.38% | 42.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 15:15:00 | 151.51 | 162.46 | 162.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 150.07 | 160.98 | 161.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 161.28 | 158.87 | 160.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 161.28 | 158.87 | 160.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 161.28 | 158.87 | 160.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 14:45:00 | 158.66 | 160.18 | 160.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 09:45:00 | 158.71 | 160.18 | 160.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 164.00 | 160.24 | 160.90 | SL hit (close>static) qty=1.00 sl=162.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 164.00 | 160.24 | 160.90 | SL hit (close>static) qty=1.00 sl=162.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 13:15:00 | 158.70 | 160.84 | 161.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:30:00 | 157.20 | 160.75 | 161.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 160.72 | 160.59 | 160.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:15:00 | 162.86 | 160.59 | 160.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 162.00 | 160.60 | 160.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 163.26 | 160.63 | 160.98 | SL hit (close>static) qty=1.00 sl=162.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 163.26 | 160.63 | 160.98 | SL hit (close>static) qty=1.00 sl=162.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:30:00 | 160.70 | 161.09 | 161.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:00:00 | 160.38 | 161.09 | 161.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:00:00 | 160.60 | 161.09 | 161.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:00:00 | 160.32 | 161.01 | 161.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 09:15:00 | 152.66 | 160.31 | 160.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 09:15:00 | 152.36 | 160.31 | 160.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 09:15:00 | 152.57 | 160.31 | 160.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 09:15:00 | 152.30 | 160.31 | 160.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-16 13:15:00 | 144.63 | 155.89 | 158.12 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-16 13:15:00 | 144.34 | 155.89 | 158.12 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-16 13:15:00 | 144.54 | 155.89 | 158.12 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-16 13:15:00 | 144.29 | 155.89 | 158.12 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 124.36 | 112.17 | 118.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:00:00 | 124.36 | 112.17 | 118.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 125.15 | 112.30 | 118.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:45:00 | 124.82 | 112.30 | 118.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 120.02 | 114.29 | 118.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 120.47 | 114.29 | 118.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 120.30 | 114.35 | 118.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 120.52 | 114.35 | 118.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 118.79 | 114.72 | 118.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:30:00 | 119.05 | 114.72 | 118.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 118.42 | 114.76 | 118.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 12:30:00 | 118.19 | 114.79 | 118.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 13:15:00 | 118.03 | 114.79 | 118.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 14:00:00 | 118.17 | 114.82 | 118.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 15:00:00 | 118.19 | 114.86 | 118.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 119.20 | 114.90 | 118.83 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-28 15:15:00 | 119.20 | 114.90 | 118.83 | SL hit (close>static) qty=1.00 sl=118.89 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-28 15:15:00 | 119.20 | 114.90 | 118.83 | SL hit (close>static) qty=1.00 sl=118.89 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-28 15:15:00 | 119.20 | 114.90 | 118.83 | SL hit (close>static) qty=1.00 sl=118.89 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-28 15:15:00 | 119.20 | 114.90 | 118.83 | SL hit (close>static) qty=1.00 sl=118.89 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 120.44 | 114.90 | 118.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 119.80 | 114.95 | 118.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:30:00 | 119.10 | 115.12 | 118.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 14:15:00 | 121.55 | 115.66 | 118.85 | SL hit (close>static) qty=1.00 sl=121.45 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-13 10:30:00 | 139.38 | 2025-05-20 13:15:00 | 136.57 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-05-13 14:15:00 | 138.70 | 2025-05-20 13:15:00 | 136.57 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-05-15 13:00:00 | 138.87 | 2025-05-20 13:15:00 | 136.57 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-05-23 10:30:00 | 138.60 | 2025-06-16 09:15:00 | 136.96 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-05-27 11:30:00 | 139.30 | 2025-06-16 09:15:00 | 136.96 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-05-27 12:00:00 | 139.44 | 2025-06-16 09:15:00 | 136.96 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-05-27 13:15:00 | 139.28 | 2025-06-16 09:15:00 | 136.96 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-05-27 14:30:00 | 139.37 | 2025-06-18 09:15:00 | 137.81 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-06-17 09:15:00 | 139.06 | 2025-06-18 09:15:00 | 137.81 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-06-17 11:30:00 | 140.00 | 2025-06-18 12:15:00 | 136.70 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-06-26 09:15:00 | 139.17 | 2025-06-26 09:15:00 | 137.53 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-06-27 09:15:00 | 139.59 | 2025-07-03 14:15:00 | 137.91 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-07-04 09:15:00 | 139.10 | 2025-07-10 13:15:00 | 152.16 | TARGET_HIT | 1.00 | 9.39% |
| BUY | retest2 | 2025-07-04 14:30:00 | 138.33 | 2025-07-10 14:15:00 | 153.01 | TARGET_HIT | 1.00 | 10.61% |
| SELL | retest2 | 2025-12-08 14:45:00 | 158.66 | 2025-12-09 11:15:00 | 164.00 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2025-12-09 09:45:00 | 158.71 | 2025-12-09 11:15:00 | 164.00 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2025-12-17 13:15:00 | 158.70 | 2025-12-22 11:15:00 | 163.26 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-12-18 09:30:00 | 157.20 | 2025-12-22 11:15:00 | 163.26 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2025-12-29 10:30:00 | 160.70 | 2026-01-05 09:15:00 | 152.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 11:00:00 | 160.38 | 2026-01-05 09:15:00 | 152.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 11:00:00 | 160.60 | 2026-01-05 09:15:00 | 152.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 10:00:00 | 160.32 | 2026-01-05 09:15:00 | 152.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 10:30:00 | 160.70 | 2026-01-16 13:15:00 | 144.63 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-29 11:00:00 | 160.38 | 2026-01-16 13:15:00 | 144.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-31 11:00:00 | 160.60 | 2026-01-16 13:15:00 | 144.54 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-01 10:00:00 | 160.32 | 2026-01-16 13:15:00 | 144.29 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-28 12:30:00 | 118.19 | 2026-04-28 15:15:00 | 119.20 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-04-28 13:15:00 | 118.03 | 2026-04-28 15:15:00 | 119.20 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-04-28 14:00:00 | 118.17 | 2026-04-28 15:15:00 | 119.20 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-04-28 15:00:00 | 118.19 | 2026-04-28 15:15:00 | 119.20 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-04-29 13:30:00 | 119.10 | 2026-05-04 14:15:00 | 121.55 | STOP_HIT | 1.00 | -2.06% |
