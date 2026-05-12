# Lemon Tree Hotels Ltd. (LEMONTREE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 120.45
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 58 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 43 |
| PARTIAL | 7 |
| TARGET_HIT | 6 |
| STOP_HIT | 37 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 50 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 34
- **Target hits / Stop hits / Partials:** 6 / 37 / 7
- **Avg / median % per leg:** 0.55% / -1.20%
- **Sum % (uncompounded):** 27.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 2 | 7.7% | 2 | 24 | 0 | -1.13% | -29.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 26 | 2 | 7.7% | 2 | 24 | 0 | -1.13% | -29.3% |
| SELL (all) | 24 | 14 | 58.3% | 4 | 13 | 7 | 2.37% | 56.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 24 | 14 | 58.3% | 4 | 13 | 7 | 2.37% | 56.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 50 | 16 | 32.0% | 6 | 37 | 7 | 0.55% | 27.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 15:15:00 | 126.24 | 143.95 | 143.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 09:15:00 | 122.08 | 143.73 | 143.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 09:15:00 | 135.50 | 133.77 | 137.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 09:15:00 | 135.50 | 133.77 | 137.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 135.50 | 133.77 | 137.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:45:00 | 136.75 | 133.77 | 137.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 124.20 | 120.08 | 124.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 10:00:00 | 124.20 | 120.08 | 124.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 123.93 | 120.11 | 124.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 15:15:00 | 123.20 | 120.28 | 124.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 10:45:00 | 123.43 | 120.38 | 124.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 11:30:00 | 123.18 | 120.40 | 124.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 117.04 | 120.40 | 123.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 117.26 | 120.40 | 123.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 117.02 | 120.40 | 123.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 14:15:00 | 122.00 | 120.20 | 123.65 | SL hit (close>ema200) qty=0.50 sl=120.20 alert=retest2 |

### Cycle 2 — BUY (started 2024-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 15:15:00 | 135.45 | 125.39 | 125.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 09:15:00 | 135.59 | 125.49 | 125.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 11:15:00 | 145.61 | 146.20 | 139.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-08 12:00:00 | 145.61 | 146.20 | 139.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 139.16 | 146.07 | 139.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 139.16 | 146.07 | 139.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 140.85 | 146.02 | 139.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 11:15:00 | 140.97 | 146.02 | 139.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 11:45:00 | 141.79 | 145.97 | 139.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 15:15:00 | 141.04 | 145.79 | 139.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-13 09:15:00 | 138.42 | 145.67 | 139.68 | SL hit (close<static) qty=1.00 sl=139.09 alert=retest2 |

### Cycle 3 — SELL (started 2025-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 12:15:00 | 127.96 | 138.36 | 138.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 09:15:00 | 125.74 | 137.95 | 138.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 131.23 | 130.49 | 133.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 10:00:00 | 131.23 | 130.49 | 133.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 133.84 | 130.57 | 133.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 14:00:00 | 133.84 | 130.57 | 133.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 134.59 | 130.61 | 133.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 14:30:00 | 134.85 | 130.61 | 133.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 133.00 | 132.54 | 133.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:45:00 | 133.80 | 132.54 | 133.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 12:15:00 | 134.47 | 132.56 | 133.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 12:45:00 | 134.60 | 132.56 | 133.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 135.36 | 132.59 | 133.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 14:00:00 | 135.36 | 132.59 | 133.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 133.59 | 133.87 | 134.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 14:30:00 | 134.00 | 133.87 | 134.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 134.00 | 133.87 | 134.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 137.76 | 133.87 | 134.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 135.73 | 133.89 | 134.27 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 14:15:00 | 139.70 | 134.66 | 134.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 141.51 | 134.78 | 134.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 10:15:00 | 137.66 | 138.58 | 136.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 10:15:00 | 137.66 | 138.58 | 136.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 137.66 | 138.58 | 136.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:30:00 | 137.93 | 138.58 | 136.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 138.36 | 138.58 | 136.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 12:15:00 | 138.55 | 138.58 | 136.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 09:30:00 | 138.88 | 138.75 | 137.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 15:15:00 | 135.01 | 138.64 | 137.09 | SL hit (close<static) qty=1.00 sl=136.61 alert=retest2 |

### Cycle 5 — SELL (started 2025-11-19 15:15:00)

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


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-11 15:15:00 | 123.20 | 2024-11-13 09:15:00 | 117.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 10:45:00 | 123.43 | 2024-11-13 09:15:00 | 117.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 11:30:00 | 123.18 | 2024-11-13 09:15:00 | 117.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 15:15:00 | 123.20 | 2024-11-14 14:15:00 | 122.00 | STOP_HIT | 0.50 | 0.97% |
| SELL | retest2 | 2024-11-12 10:45:00 | 123.43 | 2024-11-14 14:15:00 | 122.00 | STOP_HIT | 0.50 | 1.16% |
| SELL | retest2 | 2024-11-12 11:30:00 | 123.18 | 2024-11-14 14:15:00 | 122.00 | STOP_HIT | 0.50 | 0.96% |
| SELL | retest2 | 2024-11-21 14:45:00 | 123.49 | 2024-11-22 10:15:00 | 126.10 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-01-10 11:15:00 | 140.97 | 2025-01-13 09:15:00 | 138.42 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-01-10 11:45:00 | 141.79 | 2025-01-13 09:15:00 | 138.42 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2025-01-10 15:15:00 | 141.04 | 2025-01-13 09:15:00 | 138.42 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-01-14 11:00:00 | 141.44 | 2025-01-22 09:15:00 | 139.14 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-01-20 09:45:00 | 145.09 | 2025-01-22 09:15:00 | 139.14 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2025-01-20 11:15:00 | 143.88 | 2025-01-22 09:15:00 | 139.14 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2025-01-21 11:45:00 | 143.39 | 2025-01-22 10:15:00 | 138.68 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2025-02-03 10:00:00 | 143.60 | 2025-02-07 13:15:00 | 138.85 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2025-04-25 12:15:00 | 138.55 | 2025-04-30 15:15:00 | 135.01 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-04-30 09:30:00 | 138.88 | 2025-04-30 15:15:00 | 135.01 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-05-05 09:15:00 | 139.14 | 2025-05-06 12:15:00 | 136.05 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-05-05 10:30:00 | 138.55 | 2025-05-06 12:15:00 | 136.05 | STOP_HIT | 1.00 | -1.80% |
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
