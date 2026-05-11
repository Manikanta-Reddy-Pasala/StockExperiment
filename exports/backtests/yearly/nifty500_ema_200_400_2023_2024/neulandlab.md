# Neuland Laboratories Ltd. (NEULANDLAB)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 17713.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 44 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 35 |
| PARTIAL | 2 |
| TARGET_HIT | 8 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 37 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 26
- **Target hits / Stop hits / Partials:** 8 / 27 / 2
- **Avg / median % per leg:** 0.16% / -2.88%
- **Sum % (uncompounded):** 6.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 5 | 38.5% | 5 | 8 | 0 | 2.40% | 31.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 5 | 38.5% | 5 | 8 | 0 | 2.40% | 31.2% |
| SELL (all) | 24 | 6 | 25.0% | 3 | 19 | 2 | -1.05% | -25.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 24 | 6 | 25.0% | 3 | 19 | 2 | -1.05% | -25.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 37 | 11 | 29.7% | 8 | 27 | 2 | 0.16% | 6.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 12:15:00 | 6060.80 | 6566.47 | 6566.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 5785.50 | 6481.47 | 6522.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 6446.35 | 6439.08 | 6497.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-06 10:00:00 | 6446.35 | 6439.08 | 6497.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 6501.05 | 6414.54 | 6477.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-11 14:30:00 | 6506.90 | 6414.54 | 6477.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 6500.00 | 6415.39 | 6477.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 6400.00 | 6415.39 | 6477.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 12:15:00 | 6584.55 | 6413.88 | 6473.31 | SL hit (close>static) qty=1.00 sl=6525.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 09:15:00 | 7319.00 | 6515.57 | 6513.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 7829.45 | 6695.45 | 6608.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 11875.90 | 12184.79 | 11077.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-26 10:00:00 | 11875.90 | 12184.79 | 11077.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 12:15:00 | 11244.30 | 12220.22 | 11322.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 13:00:00 | 11244.30 | 12220.22 | 11322.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 13:15:00 | 11284.95 | 12210.92 | 11322.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 14:30:00 | 11355.00 | 12202.20 | 11322.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-10 10:15:00 | 12490.50 | 12138.65 | 11360.91 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 13700.10 | 14287.96 | 14290.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 12:15:00 | 13345.75 | 14210.01 | 14250.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 12:15:00 | 14166.05 | 14130.73 | 14203.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 12:15:00 | 14166.05 | 14130.73 | 14203.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 12:15:00 | 14166.05 | 14130.73 | 14203.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:15:00 | 13944.55 | 14129.59 | 14202.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 13247.32 | 14077.27 | 14172.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 10:15:00 | 12550.09 | 14066.55 | 14166.89 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 10:15:00 | 13274.00 | 12131.71 | 12128.59 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 10:15:00 | 11646.00 | 12166.22 | 12167.61 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 14761.00 | 12174.37 | 12163.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 14848.00 | 12320.77 | 12237.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 13229.00 | 13276.14 | 12854.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 15:00:00 | 13229.00 | 13276.14 | 12854.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 13021.00 | 13274.13 | 12858.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 14:45:00 | 13532.00 | 13162.86 | 12918.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 15:15:00 | 13500.00 | 13162.86 | 12918.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 11:30:00 | 13580.00 | 13169.89 | 12934.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 14:00:00 | 13494.00 | 13296.68 | 13054.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-05 09:15:00 | 14885.20 | 13473.30 | 13181.21 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 13:15:00 | 15009.00 | 15950.67 | 15954.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 14646.00 | 15772.86 | 15859.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 14:15:00 | 14123.00 | 13925.62 | 14582.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 15:00:00 | 14123.00 | 13925.62 | 14582.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 13296.00 | 12586.65 | 13167.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 12:00:00 | 13296.00 | 12586.65 | 13167.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 12:15:00 | 13100.00 | 12591.75 | 13166.74 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 13:15:00 | 15153.00 | 13558.68 | 13555.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 15388.00 | 13608.47 | 13580.43 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-12 09:15:00 | 6400.00 | 2024-06-13 12:15:00 | 6584.55 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2024-06-14 15:00:00 | 6465.00 | 2024-06-14 15:15:00 | 6542.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-06-18 09:15:00 | 6442.60 | 2024-06-20 11:15:00 | 6528.20 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-06-19 10:15:00 | 6464.80 | 2024-06-20 11:15:00 | 6528.20 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-10-07 14:30:00 | 11355.00 | 2024-10-10 10:15:00 | 12490.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-24 14:15:00 | 13944.55 | 2025-01-28 09:15:00 | 13247.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 14:15:00 | 13944.55 | 2025-01-28 10:15:00 | 12550.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-30 11:15:00 | 13862.05 | 2025-02-04 09:15:00 | 14571.40 | STOP_HIT | 1.00 | -5.12% |
| SELL | retest2 | 2025-01-30 12:30:00 | 13959.95 | 2025-02-04 09:15:00 | 14571.40 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2025-01-31 10:00:00 | 13931.35 | 2025-02-04 09:15:00 | 14571.40 | STOP_HIT | 1.00 | -4.59% |
| SELL | retest2 | 2025-02-01 11:00:00 | 13973.65 | 2025-02-04 09:15:00 | 14571.40 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2025-02-01 11:45:00 | 13852.50 | 2025-02-04 09:15:00 | 14571.40 | STOP_HIT | 1.00 | -5.19% |
| SELL | retest2 | 2025-02-03 09:30:00 | 13930.00 | 2025-02-04 09:15:00 | 14571.40 | STOP_HIT | 1.00 | -4.60% |
| SELL | retest2 | 2025-02-03 12:45:00 | 14029.80 | 2025-02-04 09:15:00 | 14571.40 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-03-28 12:15:00 | 12332.00 | 2025-04-01 09:15:00 | 11715.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 12:15:00 | 12332.00 | 2025-04-03 11:15:00 | 11857.50 | STOP_HIT | 0.50 | 3.85% |
| SELL | retest2 | 2025-04-15 13:00:00 | 12423.00 | 2025-04-17 09:15:00 | 12829.00 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2025-04-25 11:15:00 | 12330.00 | 2025-05-08 11:15:00 | 12715.00 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-04-25 15:00:00 | 12292.00 | 2025-05-08 11:15:00 | 12715.00 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2025-04-30 11:45:00 | 12135.00 | 2025-05-08 11:15:00 | 12715.00 | STOP_HIT | 1.00 | -4.78% |
| SELL | retest2 | 2025-05-05 12:00:00 | 12197.00 | 2025-05-08 11:15:00 | 12715.00 | STOP_HIT | 1.00 | -4.25% |
| SELL | retest2 | 2025-05-06 09:45:00 | 12141.00 | 2025-05-08 12:15:00 | 12835.00 | STOP_HIT | 1.00 | -5.72% |
| SELL | retest2 | 2025-05-06 14:45:00 | 12108.00 | 2025-05-08 12:15:00 | 12835.00 | STOP_HIT | 1.00 | -6.00% |
| SELL | retest2 | 2025-05-09 09:15:00 | 11933.00 | 2025-05-16 09:15:00 | 10739.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-15 11:00:00 | 12071.00 | 2025-05-16 09:15:00 | 10863.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-18 14:45:00 | 13532.00 | 2025-09-05 09:15:00 | 14885.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-18 15:15:00 | 13500.00 | 2025-09-05 09:15:00 | 14850.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-20 11:30:00 | 13580.00 | 2025-09-05 09:15:00 | 14938.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-29 14:00:00 | 13494.00 | 2025-09-05 09:15:00 | 14843.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-09 09:15:00 | 16220.00 | 2025-12-12 12:15:00 | 16061.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-12-11 10:00:00 | 16185.00 | 2025-12-17 09:15:00 | 16066.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-12-11 14:15:00 | 16202.00 | 2025-12-17 09:15:00 | 16066.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-12-11 15:15:00 | 16285.00 | 2025-12-17 11:15:00 | 15821.00 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-12-12 09:45:00 | 16322.00 | 2025-12-17 11:15:00 | 15821.00 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2025-12-15 10:45:00 | 16376.00 | 2025-12-17 11:15:00 | 15821.00 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2025-12-15 11:45:00 | 16321.00 | 2025-12-17 11:15:00 | 15821.00 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2025-12-19 09:15:00 | 16547.00 | 2025-12-19 11:15:00 | 15912.00 | STOP_HIT | 1.00 | -3.84% |
