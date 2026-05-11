# Zee Entertainment Enterprises Ltd. (ZEEL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 95.22
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
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 26 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 10 |
| TARGET_HIT | 4 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 35 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 20
- **Target hits / Stop hits / Partials:** 4 / 21 / 10
- **Avg / median % per leg:** 0.98% / -0.85%
- **Sum % (uncompounded):** 34.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 35 | 15 | 42.9% | 4 | 21 | 10 | 0.98% | 34.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 35 | 15 | 42.9% | 4 | 21 | 10 | 0.98% | 34.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 35 | 15 | 42.9% | 4 | 21 | 10 | 0.98% | 34.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 15:15:00 | 137.80 | 129.92 | 129.89 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 13:15:00 | 124.87 | 130.09 | 130.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 15:15:00 | 124.44 | 129.66 | 129.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 09:15:00 | 128.25 | 128.03 | 128.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 09:15:00 | 128.25 | 128.03 | 128.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 128.25 | 128.03 | 128.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:30:00 | 129.65 | 128.03 | 128.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 129.34 | 128.05 | 128.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 11:00:00 | 129.34 | 128.05 | 128.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 127.27 | 128.04 | 128.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 13:00:00 | 126.35 | 128.02 | 128.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 14:00:00 | 126.71 | 128.01 | 128.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 14:45:00 | 126.00 | 127.99 | 128.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-07 14:15:00 | 129.61 | 127.55 | 128.64 | SL hit (close>static) qty=1.00 sl=129.40 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 15:15:00 | 115.40 | 107.98 | 107.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 118.59 | 108.09 | 108.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 127.30 | 127.93 | 121.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 13:00:00 | 127.30 | 127.93 | 121.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 133.80 | 139.58 | 133.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 133.80 | 139.58 | 133.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 133.00 | 139.52 | 133.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 131.19 | 139.52 | 133.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 130.80 | 139.43 | 133.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:30:00 | 131.25 | 139.43 | 133.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 130.70 | 139.34 | 133.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 130.70 | 139.34 | 133.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 113.20 | 129.38 | 129.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 112.41 | 128.89 | 129.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 10:15:00 | 123.43 | 122.25 | 125.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-22 11:00:00 | 123.43 | 122.25 | 125.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 119.70 | 117.94 | 120.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:45:00 | 119.50 | 117.94 | 120.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 121.35 | 117.97 | 120.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:45:00 | 121.36 | 117.97 | 120.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 120.93 | 118.00 | 120.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:30:00 | 121.11 | 118.00 | 120.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 120.47 | 118.02 | 120.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 119.35 | 118.06 | 120.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 113.38 | 117.76 | 120.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-10-17 09:15:00 | 107.41 | 113.71 | 116.93 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 09:15:00 | 91.66 | 84.62 | 84.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 94.07 | 85.25 | 84.95 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-31 10:15:00 | 149.35 | 2024-06-03 09:15:00 | 153.80 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2024-05-31 14:15:00 | 148.75 | 2024-06-03 09:15:00 | 153.80 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2024-06-04 10:45:00 | 147.60 | 2024-06-04 12:15:00 | 132.84 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-25 12:45:00 | 148.82 | 2024-06-26 11:15:00 | 154.75 | STOP_HIT | 1.00 | -3.98% |
| SELL | retest2 | 2024-06-27 13:30:00 | 152.40 | 2024-07-10 10:15:00 | 146.18 | PARTIAL | 0.50 | 4.08% |
| SELL | retest2 | 2024-06-28 10:30:00 | 153.61 | 2024-07-10 10:15:00 | 146.25 | PARTIAL | 0.50 | 4.79% |
| SELL | retest2 | 2024-07-01 11:15:00 | 153.87 | 2024-07-10 10:15:00 | 146.96 | PARTIAL | 0.50 | 4.49% |
| SELL | retest2 | 2024-07-01 11:45:00 | 153.95 | 2024-07-10 15:15:00 | 145.93 | PARTIAL | 0.50 | 5.21% |
| SELL | retest2 | 2024-06-27 13:30:00 | 152.40 | 2024-07-12 09:15:00 | 155.11 | STOP_HIT | 0.50 | -1.78% |
| SELL | retest2 | 2024-06-28 10:30:00 | 153.61 | 2024-07-12 09:15:00 | 155.11 | STOP_HIT | 0.50 | -0.98% |
| SELL | retest2 | 2024-07-01 11:15:00 | 153.87 | 2024-07-12 09:15:00 | 155.11 | STOP_HIT | 0.50 | -0.81% |
| SELL | retest2 | 2024-07-01 11:45:00 | 153.95 | 2024-07-12 09:15:00 | 155.11 | STOP_HIT | 0.50 | -0.75% |
| SELL | retest2 | 2024-07-02 12:15:00 | 154.70 | 2024-07-15 13:15:00 | 159.74 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2024-07-18 09:15:00 | 145.65 | 2024-07-19 13:15:00 | 138.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-18 09:15:00 | 145.65 | 2024-07-23 12:15:00 | 131.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-03 13:00:00 | 126.35 | 2025-01-07 14:15:00 | 129.61 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-01-03 14:00:00 | 126.71 | 2025-01-07 14:15:00 | 129.61 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-01-03 14:45:00 | 126.00 | 2025-01-07 14:15:00 | 129.61 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-01-10 14:30:00 | 126.25 | 2025-01-13 14:15:00 | 119.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 14:30:00 | 126.25 | 2025-01-27 09:15:00 | 113.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-25 09:30:00 | 104.77 | 2025-03-28 12:15:00 | 99.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 09:30:00 | 104.77 | 2025-04-03 10:15:00 | 103.85 | STOP_HIT | 0.50 | 0.88% |
| SELL | retest2 | 2025-04-03 14:15:00 | 104.70 | 2025-04-03 15:15:00 | 110.50 | STOP_HIT | 1.00 | -5.54% |
| SELL | retest2 | 2025-04-04 10:15:00 | 104.70 | 2025-04-07 09:15:00 | 99.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 11:45:00 | 104.96 | 2025-04-07 09:15:00 | 99.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 10:15:00 | 104.70 | 2025-04-08 14:15:00 | 107.54 | STOP_HIT | 0.50 | -2.71% |
| SELL | retest2 | 2025-04-04 11:45:00 | 104.96 | 2025-04-08 14:15:00 | 107.54 | STOP_HIT | 0.50 | -2.46% |
| SELL | retest2 | 2025-04-09 09:15:00 | 104.92 | 2025-04-16 09:15:00 | 111.31 | STOP_HIT | 1.00 | -6.09% |
| SELL | retest2 | 2025-04-15 15:15:00 | 107.20 | 2025-04-16 09:15:00 | 111.31 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2025-04-25 11:30:00 | 107.50 | 2025-04-28 09:15:00 | 110.21 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-04-29 11:45:00 | 107.24 | 2025-05-05 10:15:00 | 108.59 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-04-30 12:15:00 | 107.62 | 2025-05-07 12:15:00 | 108.53 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-05-06 09:15:00 | 107.40 | 2025-05-08 09:15:00 | 111.90 | STOP_HIT | 1.00 | -4.19% |
| SELL | retest2 | 2025-09-23 09:15:00 | 119.35 | 2025-09-26 09:15:00 | 113.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 09:15:00 | 119.35 | 2025-10-17 09:15:00 | 107.41 | TARGET_HIT | 0.50 | 10.00% |
