# Ashok Leyland Ltd. (ASHOKLEY)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 168.77
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 33 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 22 |
| PARTIAL | 8 |
| TARGET_HIT | 4 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 17
- **Target hits / Stop hits / Partials:** 4 / 22 / 8
- **Avg / median % per leg:** 2.09% / 0.92%
- **Sum % (uncompounded):** 71.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 9 | 42.9% | 1 | 16 | 4 | 1.56% | 32.7% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 0 | 4 | 4 | 4.99% | 39.9% |
| BUY @ 3rd Alert (retest2) | 13 | 1 | 7.7% | 1 | 12 | 0 | -0.55% | -7.2% |
| SELL (all) | 13 | 8 | 61.5% | 3 | 6 | 4 | 2.96% | 38.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 8 | 61.5% | 3 | 6 | 4 | 2.96% | 38.4% |
| retest1 (combined) | 8 | 8 | 100.0% | 0 | 4 | 4 | 4.99% | 39.9% |
| retest2 (combined) | 26 | 9 | 34.6% | 4 | 18 | 4 | 1.20% | 31.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 10:15:00 | 111.50 | 119.83 | 119.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-08 14:15:00 | 111.24 | 119.49 | 119.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 14:15:00 | 111.00 | 110.03 | 113.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-08 15:00:00 | 111.00 | 110.03 | 113.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 114.35 | 110.09 | 113.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:30:00 | 114.68 | 110.09 | 113.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 114.99 | 110.13 | 113.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 114.99 | 110.13 | 113.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 112.28 | 110.10 | 112.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 14:00:00 | 112.28 | 110.10 | 112.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 114.70 | 110.20 | 112.60 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 12:15:00 | 116.00 | 114.03 | 114.03 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 11:15:00 | 111.49 | 114.02 | 114.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 12:15:00 | 111.25 | 113.99 | 114.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 09:15:00 | 114.54 | 112.22 | 112.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 114.54 | 112.22 | 112.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 114.54 | 112.22 | 112.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:00:00 | 114.54 | 112.22 | 112.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 115.48 | 112.25 | 112.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 11:00:00 | 114.44 | 112.84 | 113.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 11:45:00 | 114.03 | 112.85 | 113.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 13:00:00 | 114.15 | 112.87 | 113.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 11:15:00 | 108.72 | 112.65 | 113.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 12:15:00 | 108.33 | 112.61 | 113.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 12:15:00 | 108.44 | 112.61 | 113.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 12:15:00 | 103.00 | 111.62 | 112.53 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 12:15:00 | 115.31 | 106.40 | 106.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 115.60 | 109.60 | 108.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 116.44 | 117.38 | 114.40 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-13 10:15:00 | 116.51 | 117.38 | 114.40 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 10:15:00 | 116.94 | 117.36 | 114.49 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 12:00:00 | 116.56 | 117.36 | 114.81 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 09:15:00 | 116.79 | 117.32 | 114.83 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 14:15:00 | 122.34 | 117.84 | 115.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 14:15:00 | 122.39 | 117.84 | 115.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 09:15:00 | 122.79 | 117.94 | 115.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 09:15:00 | 122.63 | 117.94 | 115.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 122.50 | 122.62 | 119.55 | SL hit (close<ema200) qty=0.50 sl=122.62 alert=retest1 |

### Cycle 5 — SELL (started 2026-03-25 10:15:00)

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


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-09-17 11:15:00 | 119.55 | 2024-09-18 09:15:00 | 118.73 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-09-17 12:30:00 | 119.58 | 2024-09-18 09:15:00 | 118.73 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-09-26 10:00:00 | 119.85 | 2024-09-30 14:15:00 | 117.93 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-09-26 11:30:00 | 119.83 | 2024-09-30 14:15:00 | 117.93 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-01-06 11:00:00 | 114.44 | 2025-01-09 11:15:00 | 108.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 11:45:00 | 114.03 | 2025-01-09 12:15:00 | 108.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 13:00:00 | 114.15 | 2025-01-09 12:15:00 | 108.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 11:00:00 | 114.44 | 2025-01-13 12:15:00 | 103.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-06 11:45:00 | 114.03 | 2025-01-13 12:15:00 | 102.74 | TARGET_HIT | 0.50 | 9.91% |
| SELL | retest2 | 2025-01-06 13:00:00 | 114.15 | 2025-01-13 13:15:00 | 102.63 | TARGET_HIT | 0.50 | 10.09% |
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
