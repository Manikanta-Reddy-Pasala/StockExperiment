# Housing & Urban Development Corporation Ltd. (HUDCO)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 232.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 32 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 3 |
| TARGET_HIT | 4 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 20
- **Target hits / Stop hits / Partials:** 4 / 20 / 3
- **Avg / median % per leg:** -0.06% / -2.55%
- **Sum % (uncompounded):** -1.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 1 | 6.7% | 1 | 14 | 0 | -1.67% | -25.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 1 | 6.7% | 1 | 14 | 0 | -1.67% | -25.1% |
| SELL (all) | 12 | 6 | 50.0% | 3 | 6 | 3 | 1.95% | 23.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 6 | 50.0% | 3 | 6 | 3 | 1.95% | 23.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 27 | 7 | 25.9% | 4 | 20 | 3 | -0.06% | -1.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 12:15:00 | 217.10 | 227.37 | 227.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 212.40 | 226.94 | 227.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 215.05 | 214.67 | 218.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 10:00:00 | 215.05 | 214.67 | 218.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 218.49 | 214.73 | 218.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 12:45:00 | 218.44 | 214.73 | 218.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 217.70 | 215.13 | 218.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:15:00 | 216.35 | 215.13 | 218.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:45:00 | 215.67 | 215.15 | 218.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:30:00 | 216.16 | 215.17 | 218.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:00:00 | 216.37 | 215.28 | 218.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 217.38 | 215.34 | 218.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 218.20 | 215.34 | 218.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 217.10 | 215.38 | 218.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 12:30:00 | 216.72 | 215.39 | 218.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:30:00 | 216.75 | 215.47 | 218.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 224.13 | 215.59 | 218.28 | SL hit (close>static) qty=1.00 sl=220.84 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 13:15:00 | 236.09 | 220.38 | 220.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 11:15:00 | 237.36 | 227.20 | 225.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 228.70 | 229.35 | 226.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 10:00:00 | 228.70 | 229.35 | 226.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 227.21 | 229.33 | 226.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:00:00 | 227.21 | 229.33 | 226.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 223.43 | 229.21 | 226.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:00:00 | 223.43 | 229.21 | 226.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 226.17 | 229.18 | 226.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:15:00 | 227.15 | 229.18 | 226.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 12:30:00 | 227.39 | 229.14 | 226.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 13:45:00 | 226.80 | 229.30 | 226.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:30:00 | 226.80 | 229.04 | 226.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 226.50 | 229.02 | 226.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:30:00 | 225.30 | 229.02 | 226.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 227.00 | 228.97 | 226.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 231.87 | 228.93 | 226.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 09:30:00 | 228.86 | 231.39 | 228.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 14:15:00 | 226.00 | 233.19 | 230.27 | SL hit (close<static) qty=1.00 sl=226.20 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 11:15:00 | 213.19 | 227.80 | 227.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 211.27 | 225.08 | 226.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 222.25 | 220.48 | 223.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 11:00:00 | 222.25 | 220.48 | 223.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 224.05 | 220.56 | 223.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:45:00 | 225.05 | 220.56 | 223.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 224.10 | 220.59 | 223.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:45:00 | 224.74 | 220.59 | 223.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 223.99 | 220.63 | 223.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:45:00 | 224.34 | 220.63 | 223.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 224.18 | 220.97 | 223.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:00:00 | 224.18 | 220.97 | 223.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 224.81 | 221.01 | 223.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:45:00 | 224.45 | 221.01 | 223.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 224.85 | 221.07 | 223.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 227.30 | 221.07 | 223.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 224.98 | 222.79 | 224.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 224.50 | 222.79 | 224.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 220.90 | 223.04 | 224.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:15:00 | 220.18 | 223.04 | 224.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:45:00 | 219.74 | 223.01 | 224.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 12:30:00 | 218.42 | 222.98 | 224.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 209.17 | 220.18 | 222.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 208.75 | 220.18 | 222.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 207.50 | 219.70 | 222.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-23 13:15:00 | 198.16 | 216.85 | 220.32 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 214.51 | 191.30 | 191.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 216.87 | 191.55 | 191.38 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:15:00 | 219.60 | 2025-05-28 09:15:00 | 241.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-13 10:30:00 | 228.03 | 2025-06-19 10:15:00 | 221.21 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2025-06-13 11:15:00 | 228.00 | 2025-06-19 10:15:00 | 221.21 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2025-06-13 15:00:00 | 228.15 | 2025-06-19 10:15:00 | 221.21 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-06-16 13:30:00 | 229.41 | 2025-06-19 10:15:00 | 221.21 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2025-06-23 10:15:00 | 229.75 | 2025-07-23 09:15:00 | 223.83 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-06-23 13:00:00 | 229.68 | 2025-07-23 09:15:00 | 223.83 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-07-07 09:45:00 | 229.51 | 2025-07-23 09:15:00 | 223.83 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-07-08 15:15:00 | 229.35 | 2025-07-23 09:15:00 | 223.83 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-09-09 10:15:00 | 216.35 | 2025-09-15 09:15:00 | 224.13 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2025-09-09 11:45:00 | 215.67 | 2025-09-15 09:15:00 | 224.13 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2025-09-09 13:30:00 | 216.16 | 2025-09-15 09:15:00 | 224.13 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2025-09-10 13:00:00 | 216.37 | 2025-09-15 09:15:00 | 224.13 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2025-09-11 12:30:00 | 216.72 | 2025-09-15 09:15:00 | 224.13 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2025-09-12 11:30:00 | 216.75 | 2025-09-15 09:15:00 | 224.13 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2025-11-07 11:15:00 | 227.15 | 2025-12-03 14:15:00 | 226.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-11-07 12:30:00 | 227.39 | 2025-12-03 14:15:00 | 226.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-11-11 13:45:00 | 226.80 | 2025-12-04 09:15:00 | 222.06 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-11-14 09:30:00 | 226.80 | 2025-12-04 09:15:00 | 222.06 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-11-17 09:15:00 | 231.87 | 2025-12-04 09:15:00 | 222.06 | STOP_HIT | 1.00 | -4.23% |
| BUY | retest2 | 2025-11-25 09:30:00 | 228.86 | 2025-12-04 09:15:00 | 222.06 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2026-01-08 11:15:00 | 220.18 | 2026-01-20 09:15:00 | 209.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:45:00 | 219.74 | 2026-01-20 09:15:00 | 208.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 12:30:00 | 218.42 | 2026-01-20 13:15:00 | 207.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:15:00 | 220.18 | 2026-01-23 13:15:00 | 198.16 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 11:45:00 | 219.74 | 2026-01-23 13:15:00 | 197.77 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 12:30:00 | 218.42 | 2026-01-23 13:15:00 | 196.58 | TARGET_HIT | 0.50 | 10.00% |
