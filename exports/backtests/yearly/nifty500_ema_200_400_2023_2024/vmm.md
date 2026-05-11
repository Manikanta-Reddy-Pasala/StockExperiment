# Vishal Mega Mart Ltd. (VMM)

## Backtest Summary

- **Window:** 2024-12-18 09:15:00 → 2026-05-08 15:15:00 (2396 bars)
- **Last close:** 124.00
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
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 7 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 5 |
| TARGET_HIT | 6 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 3
- **Target hits / Stop hits / Partials:** 6 / 3 / 5
- **Avg / median % per leg:** 5.67% / 5.00%
- **Sum % (uncompounded):** 79.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| SELL (all) | 13 | 10 | 76.9% | 5 | 3 | 5 | 5.34% | 69.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 10 | 76.9% | 5 | 3 | 5 | 5.34% | 69.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 11 | 78.6% | 6 | 3 | 5 | 5.67% | 79.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 11:15:00 | 110.99 | 105.81 | 105.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 12:15:00 | 111.23 | 105.86 | 105.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 11:15:00 | 124.00 | 124.15 | 119.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 11:45:00 | 123.89 | 124.15 | 119.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 119.56 | 124.08 | 119.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 10:15:00 | 120.67 | 124.08 | 119.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-18 09:15:00 | 132.74 | 124.09 | 119.66 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 137.72 | 144.50 | 144.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 137.34 | 144.18 | 144.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 11:15:00 | 136.18 | 135.53 | 138.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 11:30:00 | 136.10 | 135.53 | 138.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 138.43 | 135.34 | 138.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 138.43 | 135.34 | 138.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 137.67 | 135.37 | 138.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 137.67 | 135.37 | 138.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 137.80 | 135.39 | 138.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:30:00 | 137.14 | 135.41 | 138.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:15:00 | 136.85 | 135.41 | 138.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 10:00:00 | 137.07 | 135.48 | 138.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 10:45:00 | 137.10 | 135.57 | 137.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 10:15:00 | 130.28 | 135.49 | 137.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 10:15:00 | 130.01 | 135.49 | 137.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 10:15:00 | 130.22 | 135.49 | 137.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 10:15:00 | 130.24 | 135.49 | 137.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 09:15:00 | 123.43 | 132.33 | 135.04 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 123.48 | 118.31 | 118.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 12:15:00 | 124.02 | 118.41 | 118.36 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-17 10:15:00 | 120.67 | 2025-06-18 09:15:00 | 132.74 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-22 09:30:00 | 137.14 | 2026-01-06 10:15:00 | 130.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-22 10:15:00 | 136.85 | 2026-01-06 10:15:00 | 130.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 10:00:00 | 137.07 | 2026-01-06 10:15:00 | 130.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 10:45:00 | 137.10 | 2026-01-06 10:15:00 | 130.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-22 09:30:00 | 137.14 | 2026-01-20 09:15:00 | 123.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-22 10:15:00 | 136.85 | 2026-01-20 09:15:00 | 123.16 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-23 10:00:00 | 137.07 | 2026-01-20 09:15:00 | 123.36 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-24 10:45:00 | 137.10 | 2026-01-20 09:15:00 | 123.39 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-26 13:15:00 | 125.56 | 2026-02-26 14:15:00 | 128.03 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-02-27 09:15:00 | 119.18 | 2026-03-04 09:15:00 | 113.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 09:15:00 | 119.18 | 2026-03-09 09:15:00 | 107.26 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-23 11:30:00 | 125.70 | 2026-04-28 11:15:00 | 128.09 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-04-23 15:00:00 | 125.86 | 2026-04-28 11:15:00 | 128.09 | STOP_HIT | 1.00 | -1.77% |
