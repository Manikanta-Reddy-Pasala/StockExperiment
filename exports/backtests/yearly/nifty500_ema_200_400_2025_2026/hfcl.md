# HFCL Ltd. (HFCL)

## Backtest Summary

- **Window:** 2025-07-25 09:15:00 → 2026-05-08 15:15:00 (1346 bars)
- **Last close:** 139.75
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
| ALERT2_SKIP | 2 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 9 |
| TARGET_HIT | 3 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 12
- **Target hits / Stop hits / Partials:** 3 / 16 / 9
- **Avg / median % per leg:** 1.58% / 1.71%
- **Sum % (uncompounded):** 44.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 2 | 5 | 0 | -0.61% | -4.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 2 | 28.6% | 2 | 5 | 0 | -0.61% | -4.3% |
| SELL (all) | 21 | 14 | 66.7% | 1 | 11 | 9 | 2.31% | 48.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 21 | 14 | 66.7% | 1 | 11 | 9 | 2.31% | 48.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 28 | 16 | 57.1% | 3 | 16 | 9 | 1.58% | 44.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 71.98 | 74.44 | 74.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 12:15:00 | 71.37 | 74.38 | 74.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 70.52 | 67.78 | 70.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 70.52 | 67.78 | 70.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 70.52 | 67.78 | 70.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 70.52 | 67.78 | 70.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 68.76 | 67.79 | 70.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:15:00 | 68.48 | 67.79 | 70.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 12:30:00 | 68.46 | 67.81 | 70.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 13:15:00 | 68.47 | 67.81 | 70.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 14:00:00 | 68.49 | 67.82 | 70.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 68.94 | 67.82 | 70.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 67.18 | 67.89 | 69.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 12:00:00 | 67.11 | 67.88 | 69.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 12:30:00 | 67.17 | 67.88 | 69.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 13:15:00 | 66.93 | 67.88 | 69.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 69.75 | 67.85 | 69.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 70.05 | 67.85 | 69.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 70.47 | 67.88 | 69.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 68.55 | 67.99 | 69.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 14:15:00 | 65.06 | 67.80 | 69.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 14:15:00 | 65.04 | 67.80 | 69.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 14:15:00 | 65.05 | 67.80 | 69.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 14:15:00 | 65.07 | 67.80 | 69.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 14:15:00 | 65.12 | 67.80 | 69.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 63.82 | 67.73 | 69.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 63.75 | 67.73 | 69.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 63.81 | 67.73 | 69.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 63.58 | 67.73 | 69.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-13 14:15:00 | 61.70 | 67.21 | 69.13 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 13:15:00 | 72.89 | 68.57 | 68.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 14:15:00 | 73.85 | 68.63 | 68.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 11:15:00 | 68.84 | 69.12 | 68.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 11:15:00 | 68.84 | 69.12 | 68.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 68.84 | 69.12 | 68.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:00:00 | 68.84 | 69.12 | 68.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 68.32 | 69.11 | 68.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:00:00 | 68.32 | 69.11 | 68.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 68.48 | 69.11 | 68.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:30:00 | 68.17 | 69.11 | 68.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 68.92 | 69.19 | 68.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:30:00 | 69.11 | 69.19 | 68.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 68.95 | 69.19 | 68.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 13:45:00 | 68.96 | 69.19 | 68.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 69.75 | 69.19 | 68.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 70.06 | 68.78 | 68.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 09:45:00 | 70.38 | 68.79 | 68.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 12:45:00 | 70.78 | 68.83 | 68.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 66.31 | 68.84 | 68.77 | SL hit (close<static) qty=1.00 sl=68.80 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-12-31 11:15:00 | 68.48 | 2026-01-09 14:15:00 | 65.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 12:30:00 | 68.46 | 2026-01-09 14:15:00 | 65.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 13:15:00 | 68.47 | 2026-01-09 14:15:00 | 65.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 14:00:00 | 68.49 | 2026-01-09 14:15:00 | 65.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 11:15:00 | 67.18 | 2026-01-09 14:15:00 | 65.12 | PARTIAL | 0.50 | 3.06% |
| SELL | retest2 | 2026-01-05 12:00:00 | 67.11 | 2026-01-12 09:15:00 | 63.82 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2026-01-05 12:30:00 | 67.17 | 2026-01-12 09:15:00 | 63.75 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2026-01-05 13:15:00 | 66.93 | 2026-01-12 09:15:00 | 63.81 | PARTIAL | 0.50 | 4.66% |
| SELL | retest2 | 2026-01-08 09:15:00 | 68.55 | 2026-01-12 09:15:00 | 63.58 | PARTIAL | 0.50 | 7.25% |
| SELL | retest2 | 2025-12-31 11:15:00 | 68.48 | 2026-01-13 14:15:00 | 61.70 | TARGET_HIT | 0.50 | 9.91% |
| SELL | retest2 | 2025-12-31 12:30:00 | 68.46 | 2026-01-16 10:15:00 | 67.32 | STOP_HIT | 0.50 | 1.67% |
| SELL | retest2 | 2025-12-31 13:15:00 | 68.47 | 2026-01-16 10:15:00 | 67.32 | STOP_HIT | 0.50 | 1.68% |
| SELL | retest2 | 2025-12-31 14:00:00 | 68.49 | 2026-01-16 10:15:00 | 67.32 | STOP_HIT | 0.50 | 1.71% |
| SELL | retest2 | 2026-01-05 11:15:00 | 67.18 | 2026-01-16 10:15:00 | 67.32 | STOP_HIT | 0.50 | -0.21% |
| SELL | retest2 | 2026-01-05 12:00:00 | 67.11 | 2026-01-16 10:15:00 | 67.32 | STOP_HIT | 0.50 | -0.31% |
| SELL | retest2 | 2026-01-05 12:30:00 | 67.17 | 2026-01-16 10:15:00 | 67.32 | STOP_HIT | 0.50 | -0.22% |
| SELL | retest2 | 2026-01-05 13:15:00 | 66.93 | 2026-01-16 10:15:00 | 67.32 | STOP_HIT | 0.50 | -0.58% |
| SELL | retest2 | 2026-01-08 09:15:00 | 68.55 | 2026-01-16 10:15:00 | 67.32 | STOP_HIT | 0.50 | 1.79% |
| SELL | retest2 | 2026-02-04 15:00:00 | 68.90 | 2026-02-05 11:15:00 | 71.35 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2026-02-16 11:00:00 | 69.13 | 2026-02-17 10:15:00 | 71.90 | STOP_HIT | 1.00 | -4.01% |
| SELL | retest2 | 2026-02-16 14:00:00 | 68.95 | 2026-02-17 10:15:00 | 71.90 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest2 | 2026-03-06 09:15:00 | 70.06 | 2026-03-09 09:15:00 | 66.31 | STOP_HIT | 1.00 | -5.35% |
| BUY | retest2 | 2026-03-06 09:45:00 | 70.38 | 2026-03-09 09:15:00 | 66.31 | STOP_HIT | 1.00 | -5.78% |
| BUY | retest2 | 2026-03-06 12:45:00 | 70.78 | 2026-03-09 09:15:00 | 66.31 | STOP_HIT | 1.00 | -6.32% |
| BUY | retest2 | 2026-03-10 14:30:00 | 70.20 | 2026-03-23 09:15:00 | 68.03 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2026-03-25 09:15:00 | 71.50 | 2026-03-30 12:15:00 | 68.84 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2026-04-01 09:15:00 | 71.73 | 2026-04-09 09:15:00 | 78.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 11:30:00 | 70.58 | 2026-04-09 09:15:00 | 77.64 | TARGET_HIT | 1.00 | 10.00% |
