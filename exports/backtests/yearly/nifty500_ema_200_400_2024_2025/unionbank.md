# Union Bank of India (UNIONBANK)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 166.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 40 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 44 |
| PARTIAL | 25 |
| TARGET_HIT | 16 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 73 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 56 / 17
- **Target hits / Stop hits / Partials:** 15 / 33 / 25
- **Avg / median % per leg:** 3.85% / 5.00%
- **Sum % (uncompounded):** 280.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 7 | 63.6% | 7 | 4 | 0 | 5.74% | 63.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 7 | 63.6% | 7 | 4 | 0 | 5.74% | 63.2% |
| SELL (all) | 62 | 49 | 79.0% | 8 | 29 | 25 | 3.51% | 217.5% |
| SELL @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| SELL @ 3rd Alert (retest2) | 54 | 41 | 75.9% | 4 | 29 | 21 | 2.92% | 157.5% |
| retest1 (combined) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| retest2 (combined) | 65 | 48 | 73.8% | 11 | 33 | 21 | 3.40% | 220.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 10:15:00 | 138.97 | 146.62 | 146.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 138.26 | 146.38 | 146.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 140.90 | 140.84 | 143.07 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-16 13:45:00 | 139.71 | 140.82 | 143.02 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 11:00:00 | 139.71 | 140.80 | 142.97 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 11:45:00 | 139.63 | 140.79 | 142.95 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 12:15:00 | 139.50 | 140.79 | 142.95 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 132.72 | 140.00 | 142.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 132.72 | 140.00 | 142.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 132.65 | 140.00 | 142.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 132.53 | 140.00 | 142.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-08-05 11:15:00 | 125.74 | 137.15 | 140.06 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 2 — BUY (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 09:15:00 | 128.69 | 120.36 | 120.35 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 09:15:00 | 114.42 | 121.18 | 121.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 14:15:00 | 112.67 | 120.85 | 121.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 113.01 | 112.67 | 115.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 09:45:00 | 113.12 | 112.67 | 115.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 13:15:00 | 114.71 | 112.66 | 115.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 11:45:00 | 114.14 | 112.80 | 115.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-03 14:15:00 | 108.43 | 112.61 | 115.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-04 09:15:00 | 113.43 | 112.58 | 115.22 | SL hit (close>ema200) qty=0.50 sl=112.58 alert=retest2 |

### Cycle 4 — BUY (started 2025-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 14:15:00 | 122.93 | 115.19 | 115.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 09:15:00 | 125.35 | 115.36 | 115.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 115.76 | 120.02 | 117.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 115.76 | 120.02 | 117.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 115.76 | 120.02 | 117.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 15:15:00 | 119.08 | 119.85 | 117.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 12:30:00 | 118.14 | 119.75 | 117.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 09:15:00 | 118.30 | 119.67 | 117.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 14:30:00 | 117.93 | 119.61 | 117.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 15:15:00 | 117.98 | 119.60 | 117.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 09:15:00 | 120.53 | 119.60 | 117.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-21 10:15:00 | 129.72 | 120.75 | 118.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 128.68 | 139.96 | 140.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 13:15:00 | 127.21 | 139.83 | 139.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 14:15:00 | 137.43 | 137.12 | 138.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 14:45:00 | 137.32 | 137.12 | 138.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 137.67 | 137.14 | 138.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 11:30:00 | 137.40 | 137.14 | 138.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 12:00:00 | 137.49 | 137.14 | 138.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 14:15:00 | 136.90 | 137.16 | 138.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 15:00:00 | 136.94 | 137.15 | 138.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 138.08 | 137.17 | 138.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:30:00 | 137.84 | 137.17 | 138.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 137.51 | 137.16 | 138.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:30:00 | 136.20 | 137.14 | 138.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 14:00:00 | 136.16 | 137.13 | 138.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:15:00 | 130.53 | 136.78 | 137.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:15:00 | 130.62 | 136.78 | 137.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 11:15:00 | 130.06 | 136.64 | 137.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 11:15:00 | 130.09 | 136.64 | 137.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 12:15:00 | 129.39 | 136.57 | 137.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 12:15:00 | 129.35 | 136.57 | 137.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 13:15:00 | 132.30 | 132.25 | 134.90 | SL hit (close>ema200) qty=0.50 sl=132.25 alert=retest2 |

### Cycle 6 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 138.46 | 136.02 | 136.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 139.80 | 136.10 | 136.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 136.41 | 136.91 | 136.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 136.41 | 136.91 | 136.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 136.41 | 136.91 | 136.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 136.41 | 136.91 | 136.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 136.27 | 136.90 | 136.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 136.21 | 136.90 | 136.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 136.72 | 136.90 | 136.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 138.88 | 136.90 | 136.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-07 13:15:00 | 152.77 | 142.80 | 140.07 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 13:15:00 | 166.46 | 177.08 | 177.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 10:15:00 | 165.18 | 176.04 | 176.57 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-07-16 13:45:00 | 139.71 | 2024-07-23 12:15:00 | 132.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-07-18 11:00:00 | 139.71 | 2024-07-23 12:15:00 | 132.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-07-18 11:45:00 | 139.63 | 2024-07-23 12:15:00 | 132.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-07-18 12:15:00 | 139.50 | 2024-07-23 12:15:00 | 132.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-07-16 13:45:00 | 139.71 | 2024-08-05 11:15:00 | 125.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2024-07-18 11:00:00 | 139.71 | 2024-08-05 11:15:00 | 125.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2024-07-18 11:45:00 | 139.63 | 2024-08-05 11:15:00 | 125.67 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2024-07-18 12:15:00 | 139.50 | 2024-08-05 11:15:00 | 125.55 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-24 11:45:00 | 126.80 | 2024-10-03 09:15:00 | 120.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-25 10:15:00 | 126.53 | 2024-10-03 09:15:00 | 120.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-26 09:15:00 | 125.30 | 2024-10-03 09:15:00 | 120.31 | PARTIAL | 0.50 | 3.98% |
| SELL | retest2 | 2024-09-26 10:45:00 | 126.64 | 2024-10-03 13:15:00 | 119.03 | PARTIAL | 0.50 | 6.01% |
| SELL | retest2 | 2024-09-24 11:45:00 | 126.80 | 2024-10-07 10:15:00 | 114.12 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-25 10:15:00 | 126.53 | 2024-10-07 10:15:00 | 113.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-26 09:15:00 | 125.30 | 2024-10-07 10:15:00 | 112.77 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-26 10:45:00 | 126.64 | 2024-10-07 10:15:00 | 113.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-08 09:30:00 | 118.41 | 2024-11-18 09:15:00 | 112.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 11:00:00 | 118.64 | 2024-11-18 09:15:00 | 112.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 10:45:00 | 118.72 | 2024-11-18 09:15:00 | 112.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 13:30:00 | 118.36 | 2024-11-18 09:15:00 | 112.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 09:30:00 | 118.41 | 2024-11-19 09:15:00 | 117.26 | STOP_HIT | 0.50 | 0.97% |
| SELL | retest2 | 2024-11-11 11:00:00 | 118.64 | 2024-11-19 09:15:00 | 117.26 | STOP_HIT | 0.50 | 1.16% |
| SELL | retest2 | 2024-11-12 10:45:00 | 118.72 | 2024-11-19 09:15:00 | 117.26 | STOP_HIT | 0.50 | 1.23% |
| SELL | retest2 | 2024-11-12 13:30:00 | 118.36 | 2024-11-19 09:15:00 | 117.26 | STOP_HIT | 0.50 | 0.93% |
| SELL | retest2 | 2024-11-25 13:45:00 | 118.17 | 2024-11-25 14:15:00 | 119.97 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-02-01 11:45:00 | 114.14 | 2025-02-03 14:15:00 | 108.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 11:45:00 | 114.14 | 2025-02-04 09:15:00 | 113.43 | STOP_HIT | 0.50 | 0.62% |
| SELL | retest2 | 2025-02-10 12:00:00 | 114.41 | 2025-02-14 12:15:00 | 108.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 113.71 | 2025-02-14 12:15:00 | 108.80 | PARTIAL | 0.50 | 4.31% |
| SELL | retest2 | 2025-02-12 13:45:00 | 113.96 | 2025-02-14 13:15:00 | 108.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 11:30:00 | 114.53 | 2025-02-14 13:15:00 | 108.34 | PARTIAL | 0.50 | 5.41% |
| SELL | retest2 | 2025-02-13 12:00:00 | 114.04 | 2025-02-17 09:15:00 | 108.02 | PARTIAL | 0.50 | 5.27% |
| SELL | retest2 | 2025-02-10 12:00:00 | 114.41 | 2025-02-19 09:15:00 | 113.18 | STOP_HIT | 0.50 | 1.08% |
| SELL | retest2 | 2025-02-11 09:15:00 | 113.71 | 2025-02-19 09:15:00 | 113.18 | STOP_HIT | 0.50 | 0.47% |
| SELL | retest2 | 2025-02-12 13:45:00 | 113.96 | 2025-02-19 09:15:00 | 113.18 | STOP_HIT | 0.50 | 0.68% |
| SELL | retest2 | 2025-02-13 11:30:00 | 114.53 | 2025-02-19 09:15:00 | 113.18 | STOP_HIT | 0.50 | 1.18% |
| SELL | retest2 | 2025-02-13 12:00:00 | 114.04 | 2025-02-19 09:15:00 | 113.18 | STOP_HIT | 0.50 | 0.75% |
| SELL | retest2 | 2025-02-19 12:30:00 | 114.58 | 2025-02-20 09:15:00 | 115.85 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-02-19 13:00:00 | 114.52 | 2025-02-20 09:15:00 | 115.85 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-02-20 09:15:00 | 114.02 | 2025-02-20 09:15:00 | 115.85 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-02-24 09:30:00 | 114.35 | 2025-02-24 10:15:00 | 115.35 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-02-27 15:15:00 | 114.50 | 2025-03-03 09:15:00 | 108.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 15:15:00 | 114.50 | 2025-03-05 09:15:00 | 115.52 | STOP_HIT | 0.50 | -0.89% |
| SELL | retest2 | 2025-03-10 14:15:00 | 114.50 | 2025-03-11 15:15:00 | 115.25 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-03-11 13:00:00 | 113.93 | 2025-03-11 15:15:00 | 115.25 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-03-11 14:30:00 | 114.20 | 2025-03-18 10:15:00 | 115.27 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-03-12 10:15:00 | 114.14 | 2025-03-18 10:15:00 | 115.27 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-03-12 10:45:00 | 113.95 | 2025-03-18 11:15:00 | 115.90 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-04-07 15:15:00 | 119.08 | 2025-04-21 10:15:00 | 129.72 | TARGET_HIT | 1.00 | 8.94% |
| BUY | retest2 | 2025-04-09 12:30:00 | 118.14 | 2025-04-22 09:15:00 | 129.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-11 09:15:00 | 118.30 | 2025-04-22 09:15:00 | 130.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-11 14:30:00 | 117.93 | 2025-04-29 09:15:00 | 130.99 | TARGET_HIT | 1.00 | 11.07% |
| BUY | retest2 | 2025-04-15 09:15:00 | 120.53 | 2025-05-08 12:15:00 | 116.70 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2025-05-06 15:00:00 | 118.21 | 2025-05-08 12:15:00 | 116.70 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-05-08 10:00:00 | 118.07 | 2025-05-08 12:15:00 | 116.70 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-05-08 10:45:00 | 118.12 | 2025-05-08 12:15:00 | 116.70 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-05-09 11:15:00 | 122.90 | 2025-05-19 10:15:00 | 135.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-09 15:00:00 | 122.96 | 2025-05-19 10:15:00 | 135.26 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-20 11:30:00 | 137.40 | 2025-08-26 09:15:00 | 130.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-20 12:00:00 | 137.49 | 2025-08-26 09:15:00 | 130.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-20 14:15:00 | 136.90 | 2025-08-26 11:15:00 | 130.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-20 15:00:00 | 136.94 | 2025-08-26 11:15:00 | 130.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 12:30:00 | 136.20 | 2025-08-26 12:15:00 | 129.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 14:00:00 | 136.16 | 2025-08-26 12:15:00 | 129.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-20 11:30:00 | 137.40 | 2025-09-10 13:15:00 | 132.30 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2025-08-20 12:00:00 | 137.49 | 2025-09-10 13:15:00 | 132.30 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2025-08-20 14:15:00 | 136.90 | 2025-09-10 13:15:00 | 132.30 | STOP_HIT | 0.50 | 3.36% |
| SELL | retest2 | 2025-08-20 15:00:00 | 136.94 | 2025-09-10 13:15:00 | 132.30 | STOP_HIT | 0.50 | 3.39% |
| SELL | retest2 | 2025-08-22 12:30:00 | 136.20 | 2025-09-10 13:15:00 | 132.30 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2025-08-22 14:00:00 | 136.16 | 2025-09-10 13:15:00 | 132.30 | STOP_HIT | 0.50 | 2.83% |
| SELL | retest2 | 2025-09-17 11:30:00 | 135.87 | 2025-09-19 09:15:00 | 138.65 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-09-18 09:15:00 | 136.18 | 2025-09-19 09:15:00 | 138.65 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-10-20 09:15:00 | 138.88 | 2025-11-07 13:15:00 | 152.77 | TARGET_HIT | 1.00 | 10.00% |
