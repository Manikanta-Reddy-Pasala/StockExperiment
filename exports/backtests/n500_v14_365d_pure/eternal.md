# Eternal Ltd. (ETERNAL)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 256.15
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
| ALERT2_SKIP | 0 |
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 13 |
| PARTIAL | 4 |
| TARGET_HIT | 5 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 7
- **Target hits / Stop hits / Partials:** 5 / 8 / 4
- **Avg / median % per leg:** 3.22% / 5.00%
- **Sum % (uncompounded):** 54.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 1 | 3 | 0 | 1.09% | 4.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 1 | 25.0% | 1 | 3 | 0 | 1.09% | 4.4% |
| SELL (all) | 13 | 9 | 69.2% | 4 | 5 | 4 | 3.88% | 50.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 9 | 69.2% | 4 | 5 | 4 | 3.88% | 50.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 17 | 10 | 58.8% | 5 | 8 | 4 | 3.22% | 54.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 14:15:00 | 239.45 | 227.33 | 227.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 243.73 | 228.88 | 228.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 11:15:00 | 230.63 | 230.93 | 229.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 12:00:00 | 230.63 | 230.93 | 229.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 229.22 | 230.91 | 229.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 228.48 | 230.91 | 229.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 229.65 | 230.90 | 229.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:15:00 | 229.25 | 230.90 | 229.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 228.40 | 230.87 | 229.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 228.40 | 230.87 | 229.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 228.00 | 230.84 | 229.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 225.54 | 230.84 | 229.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 227.38 | 230.77 | 229.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 227.38 | 230.77 | 229.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 229.49 | 230.76 | 229.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:45:00 | 229.80 | 230.74 | 229.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 15:15:00 | 230.24 | 230.61 | 229.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 10:00:00 | 231.00 | 231.07 | 229.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 12:15:00 | 226.03 | 230.97 | 229.49 | SL hit (close<static) qty=1.00 sl=226.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-26 12:15:00 | 226.03 | 230.97 | 229.49 | SL hit (close<static) qty=1.00 sl=226.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-26 12:15:00 | 226.03 | 230.97 | 229.49 | SL hit (close<static) qty=1.00 sl=226.55 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:30:00 | 230.30 | 229.90 | 229.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-05 09:15:00 | 253.33 | 232.63 | 230.64 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 322.50 | 333.57 | 323.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 322.50 | 333.57 | 323.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 320.95 | 333.45 | 323.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:30:00 | 320.95 | 333.45 | 323.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 322.35 | 332.07 | 323.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 318.35 | 332.07 | 323.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 12:15:00 | 307.65 | 317.76 | 317.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 306.85 | 317.55 | 317.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 291.05 | 288.77 | 296.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 09:45:00 | 292.00 | 288.77 | 296.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 293.20 | 288.27 | 296.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 14:15:00 | 289.45 | 289.29 | 295.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 14:45:00 | 289.25 | 289.26 | 295.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 274.98 | 288.55 | 295.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 274.79 | 288.55 | 295.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 09:30:00 | 285.70 | 286.96 | 293.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 09:15:00 | 271.41 | 286.27 | 293.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-23 13:15:00 | 260.50 | 285.37 | 292.74 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-23 13:15:00 | 260.32 | 285.37 | 292.74 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-27 09:15:00 | 257.13 | 284.58 | 292.24 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 10:15:00 | 289.10 | 280.15 | 287.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 286.90 | 280.30 | 287.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 14:00:00 | 285.45 | 280.41 | 287.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 14:30:00 | 285.05 | 280.47 | 287.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 284.50 | 280.54 | 287.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 14:15:00 | 288.95 | 281.13 | 287.49 | SL hit (close>static) qty=1.00 sl=288.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-09 14:15:00 | 288.95 | 281.13 | 287.49 | SL hit (close>static) qty=1.00 sl=288.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-09 14:15:00 | 288.95 | 281.13 | 287.49 | SL hit (close>static) qty=1.00 sl=288.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-10 10:15:00 | 300.90 | 281.54 | 287.60 | SL hit (close>static) qty=1.00 sl=297.30 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 14:15:00 | 285.25 | 285.04 | 288.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 288.85 | 285.07 | 288.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 288.85 | 285.07 | 288.75 | SL hit (close>static) qty=1.00 sl=288.65 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 288.85 | 285.07 | 288.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 288.40 | 285.11 | 288.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:45:00 | 288.85 | 285.11 | 288.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 287.20 | 285.13 | 288.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 12:45:00 | 286.50 | 285.14 | 288.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 13:15:00 | 272.18 | 283.79 | 287.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-24 09:15:00 | 257.85 | 281.30 | 286.05 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 14:45:00 | 229.80 | 2025-05-26 12:15:00 | 226.03 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-05-22 15:15:00 | 230.24 | 2025-05-26 12:15:00 | 226.03 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-05-26 10:00:00 | 231.00 | 2025-05-26 12:15:00 | 226.03 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-05-30 09:30:00 | 230.30 | 2025-06-05 09:15:00 | 253.33 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-16 14:15:00 | 289.45 | 2026-01-20 09:15:00 | 274.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 14:45:00 | 289.25 | 2026-01-20 09:15:00 | 274.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 09:30:00 | 285.70 | 2026-01-23 09:15:00 | 271.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 14:15:00 | 289.45 | 2026-01-23 13:15:00 | 260.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-16 14:45:00 | 289.25 | 2026-01-23 13:15:00 | 260.32 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-22 09:30:00 | 285.70 | 2026-01-27 09:15:00 | 257.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-05 10:15:00 | 289.10 | 2026-02-09 14:15:00 | 288.95 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2026-02-05 14:00:00 | 285.45 | 2026-02-09 14:15:00 | 288.95 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-02-05 14:30:00 | 285.05 | 2026-02-09 14:15:00 | 288.95 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-02-06 09:15:00 | 284.50 | 2026-02-10 10:15:00 | 300.90 | STOP_HIT | 1.00 | -5.76% |
| SELL | retest2 | 2026-02-13 14:15:00 | 285.25 | 2026-02-16 09:15:00 | 288.85 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-02-16 12:45:00 | 286.50 | 2026-02-19 13:15:00 | 272.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-16 12:45:00 | 286.50 | 2026-02-24 09:15:00 | 257.85 | TARGET_HIT | 0.50 | 10.00% |
