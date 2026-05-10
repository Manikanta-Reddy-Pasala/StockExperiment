# Engineers India Ltd. (ENGINERSIN)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 256.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 20 |
| PARTIAL | 7 |
| TARGET_HIT | 0 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 13
- **Target hits / Stop hits / Partials:** 0 / 20 / 7
- **Avg / median % per leg:** 0.83% / 0.79%
- **Sum % (uncompounded):** 22.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 27 | 14 | 51.9% | 0 | 20 | 7 | 0.83% | 22.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 14 | 51.9% | 0 | 20 | 7 | 0.83% | 22.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 27 | 14 | 51.9% | 0 | 20 | 7 | 0.83% | 22.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 191.77 | 217.05 | 217.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 190.70 | 216.79 | 216.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 210.84 | 204.98 | 209.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 210.84 | 204.98 | 209.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 210.84 | 204.98 | 209.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 211.10 | 204.98 | 209.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 210.42 | 205.04 | 209.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 211.78 | 205.04 | 209.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 214.65 | 205.21 | 209.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 215.35 | 205.21 | 209.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 213.61 | 205.29 | 209.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:30:00 | 214.65 | 205.29 | 209.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 210.33 | 205.58 | 209.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 207.78 | 205.58 | 209.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:00:00 | 208.54 | 205.65 | 209.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 13:45:00 | 208.30 | 205.73 | 209.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:30:00 | 208.45 | 205.82 | 209.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 208.32 | 205.85 | 209.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:15:00 | 207.98 | 205.85 | 209.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 212.66 | 206.04 | 209.26 | SL hit (close>static) qty=1.00 sl=210.52 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 212.66 | 206.04 | 209.26 | SL hit (close>static) qty=1.00 sl=210.52 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 212.66 | 206.04 | 209.26 | SL hit (close>static) qty=1.00 sl=210.52 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 212.66 | 206.04 | 209.26 | SL hit (close>static) qty=1.00 sl=210.52 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 212.66 | 206.04 | 209.26 | SL hit (close>static) qty=1.00 sl=209.55 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 13:15:00 | 207.08 | 207.02 | 209.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:45:00 | 207.34 | 207.03 | 209.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 11:30:00 | 207.94 | 207.05 | 209.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 209.80 | 207.07 | 209.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 209.80 | 207.07 | 209.30 | SL hit (close>static) qty=1.00 sl=209.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 209.80 | 207.07 | 209.30 | SL hit (close>static) qty=1.00 sl=209.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 209.80 | 207.07 | 209.30 | SL hit (close>static) qty=1.00 sl=209.55 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 211.03 | 207.07 | 209.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 208.80 | 207.09 | 209.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 12:15:00 | 208.68 | 207.09 | 209.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:45:00 | 208.65 | 207.13 | 209.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 14:30:00 | 208.38 | 207.14 | 209.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 198.25 | 206.21 | 208.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 198.22 | 206.21 | 208.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 197.96 | 206.21 | 208.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-09 12:15:00 | 203.38 | 202.64 | 205.88 | SL hit (close>ema200) qty=0.50 sl=202.64 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-09 12:15:00 | 203.38 | 202.64 | 205.88 | SL hit (close>ema200) qty=0.50 sl=202.64 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-09 12:15:00 | 203.38 | 202.64 | 205.88 | SL hit (close>ema200) qty=0.50 sl=202.64 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:15:00 | 207.20 | 198.77 | 201.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 200.80 | 199.29 | 201.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:30:00 | 202.07 | 199.29 | 201.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 201.70 | 199.31 | 201.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 201.70 | 199.31 | 201.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 200.00 | 199.32 | 201.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:15:00 | 199.25 | 199.49 | 201.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 09:15:00 | 196.84 | 199.42 | 201.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 200.51 | 199.21 | 201.41 | SL hit (close>ema200) qty=0.50 sl=199.21 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 10:45:00 | 199.20 | 199.21 | 201.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 196.62 | 199.19 | 201.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 11:00:00 | 199.50 | 198.95 | 201.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 201.16 | 198.98 | 201.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 12:15:00 | 199.47 | 199.01 | 201.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:00:00 | 199.60 | 199.02 | 201.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 198.34 | 199.04 | 201.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 202.50 | 198.79 | 200.56 | SL hit (close>static) qty=1.00 sl=202.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 202.50 | 198.79 | 200.56 | SL hit (close>static) qty=1.00 sl=202.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 202.50 | 198.79 | 200.56 | SL hit (close>static) qty=1.00 sl=202.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 202.50 | 198.79 | 200.56 | SL hit (close>static) qty=1.00 sl=202.20 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:30:00 | 199.00 | 198.97 | 200.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 189.50 | 198.76 | 200.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 189.62 | 198.76 | 200.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 14:15:00 | 189.05 | 198.67 | 200.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 196.78 | 196.64 | 198.76 | SL hit (close>ema200) qty=0.50 sl=196.64 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 196.78 | 196.64 | 198.76 | SL hit (close>ema200) qty=0.50 sl=196.64 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 196.78 | 196.64 | 198.76 | SL hit (close>ema200) qty=0.50 sl=196.64 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 199.19 | 196.66 | 198.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 199.19 | 196.66 | 198.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 199.35 | 196.69 | 198.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 201.55 | 196.69 | 198.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 199.21 | 196.88 | 198.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 199.95 | 196.88 | 198.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 200.86 | 196.92 | 198.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:45:00 | 201.39 | 196.92 | 198.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 199.29 | 197.09 | 198.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 201.47 | 197.09 | 198.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 207.99 | 197.49 | 198.97 | SL hit (close>static) qty=1.00 sl=202.75 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 200.40 | 198.31 | 199.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:45:00 | 201.40 | 198.31 | 199.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 199.29 | 198.33 | 199.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:45:00 | 199.12 | 198.33 | 199.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 200.85 | 198.36 | 199.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 200.85 | 198.36 | 199.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 198.79 | 198.36 | 199.30 | EMA400 retest candle locked (from downside) |
| CROSSOVER_SKIP | 2026-01-06 14:15:00 | 205.60 | 200.07 | 200.06 | min_gap filter: gap=0.002% < 0.030% |
| TREND_RESET | 2026-01-06 14:15:00 | 205.60 | 200.07 | 200.06 | EMA inversion without crossover edge (EMA200=200.07 EMA400=200.06) — end cycle |
| CROSSOVER_SKIP | 2026-01-12 09:15:00 | 191.01 | 200.01 | 200.05 | min_gap filter: gap=0.020% < 0.030% |
| CROSSOVER_SKIP | 2026-02-20 12:15:00 | 214.27 | 192.63 | 192.58 | min_gap filter: gap=0.025% < 0.030% |
| CROSSOVER_SKIP | 2026-03-24 14:15:00 | 187.43 | 196.14 | 196.16 | min_gap filter: gap=0.010% < 0.030% |
| CROSSOVER_SKIP | 2026-04-08 11:15:00 | 207.24 | 196.13 | 196.07 | min_gap filter: gap=0.027% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-11 09:15:00 | 207.78 | 2025-09-15 09:15:00 | 212.66 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-09-11 11:00:00 | 208.54 | 2025-09-15 09:15:00 | 212.66 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-09-11 13:45:00 | 208.30 | 2025-09-15 09:15:00 | 212.66 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-09-12 09:30:00 | 208.45 | 2025-09-15 09:15:00 | 212.66 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-09-12 11:15:00 | 207.98 | 2025-09-15 09:15:00 | 212.66 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-09-18 13:15:00 | 207.08 | 2025-09-22 10:15:00 | 209.80 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-09-19 09:45:00 | 207.34 | 2025-09-22 10:15:00 | 209.80 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-09-19 11:30:00 | 207.94 | 2025-09-22 10:15:00 | 209.80 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-09-22 12:15:00 | 208.68 | 2025-09-26 09:15:00 | 198.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 13:45:00 | 208.65 | 2025-09-26 09:15:00 | 198.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 14:30:00 | 208.38 | 2025-09-26 09:15:00 | 197.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 12:15:00 | 208.68 | 2025-10-09 12:15:00 | 203.38 | STOP_HIT | 0.50 | 2.54% |
| SELL | retest2 | 2025-09-22 13:45:00 | 208.65 | 2025-10-09 12:15:00 | 203.38 | STOP_HIT | 0.50 | 2.53% |
| SELL | retest2 | 2025-09-22 14:30:00 | 208.38 | 2025-10-09 12:15:00 | 203.38 | STOP_HIT | 0.50 | 2.40% |
| SELL | retest2 | 2025-11-17 11:15:00 | 207.20 | 2025-11-21 09:15:00 | 196.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 11:15:00 | 207.20 | 2025-11-24 09:15:00 | 200.51 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2025-11-20 13:15:00 | 199.25 | 2025-12-05 09:15:00 | 202.50 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-11-24 10:45:00 | 199.20 | 2025-12-05 09:15:00 | 202.50 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-11-25 09:15:00 | 196.62 | 2025-12-05 09:15:00 | 202.50 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2025-11-26 11:00:00 | 199.50 | 2025-12-05 09:15:00 | 202.50 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-11-27 12:15:00 | 199.47 | 2025-12-08 13:15:00 | 189.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 13:00:00 | 199.60 | 2025-12-08 13:15:00 | 189.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 09:15:00 | 198.34 | 2025-12-08 14:15:00 | 189.05 | PARTIAL | 0.50 | 4.68% |
| SELL | retest2 | 2025-11-27 12:15:00 | 199.47 | 2025-12-19 13:15:00 | 196.78 | STOP_HIT | 0.50 | 1.35% |
| SELL | retest2 | 2025-11-27 13:00:00 | 199.60 | 2025-12-19 13:15:00 | 196.78 | STOP_HIT | 0.50 | 1.41% |
| SELL | retest2 | 2025-11-28 09:15:00 | 198.34 | 2025-12-19 13:15:00 | 196.78 | STOP_HIT | 0.50 | 0.79% |
| SELL | retest2 | 2025-12-08 09:30:00 | 199.00 | 2025-12-26 09:15:00 | 207.99 | STOP_HIT | 1.00 | -4.52% |
