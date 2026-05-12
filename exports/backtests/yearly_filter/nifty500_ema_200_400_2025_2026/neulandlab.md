# Neuland Laboratories Ltd. (NEULANDLAB)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 17713.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 0 |
| TARGET_HIT | 5 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 8
- **Target hits / Stop hits / Partials:** 4 / 8 / 0
- **Avg / median % per leg:** 1.77% / -0.84%
- **Sum % (uncompounded):** 21.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 4 | 33.3% | 4 | 8 | 0 | 1.77% | 21.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 4 | 33.3% | 4 | 8 | 0 | 1.77% | 21.2% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 4 | 33.3% | 4 | 8 | 0 | 1.77% | 21.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 10:15:00 | 13274.00 | 12131.71 | 12128.61 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 10:15:00 | 11646.00 | 12166.22 | 12167.62 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 14761.00 | 12174.37 | 12163.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 14848.00 | 12320.77 | 12237.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 13229.00 | 13276.14 | 12854.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 15:00:00 | 13229.00 | 13276.14 | 12854.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 13021.00 | 13274.13 | 12858.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 14:45:00 | 13532.00 | 13162.86 | 12918.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 15:15:00 | 13500.00 | 13162.86 | 12918.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 11:30:00 | 13580.00 | 13169.89 | 12934.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 14:00:00 | 13494.00 | 13296.68 | 13054.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-05 09:15:00 | 14885.20 | 13473.30 | 13181.22 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 13:15:00 | 15009.00 | 15950.67 | 15954.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 14646.00 | 15772.86 | 15859.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 14:15:00 | 14123.00 | 13925.62 | 14582.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 15:00:00 | 14123.00 | 13925.62 | 14582.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 13296.00 | 12586.65 | 13167.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 12:00:00 | 13296.00 | 12586.65 | 13167.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 12:15:00 | 13100.00 | 12591.75 | 13166.74 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 13:15:00 | 15153.00 | 13558.68 | 13555.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 15388.00 | 13608.47 | 13580.43 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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
