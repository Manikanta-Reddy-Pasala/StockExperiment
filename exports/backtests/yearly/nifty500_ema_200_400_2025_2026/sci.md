# Shipping Corporation of India Ltd. (SCI)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 339.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 13 |
| PARTIAL | 8 |
| TARGET_HIT | 7 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 5
- **Target hits / Stop hits / Partials:** 7 / 6 / 8
- **Avg / median % per leg:** 4.77% / 5.01%
- **Sum % (uncompounded):** 100.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.67% | -8.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.67% | -8.0% |
| SELL (all) | 18 | 16 | 88.9% | 7 | 3 | 8 | 6.00% | 108.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 16 | 88.9% | 7 | 3 | 8 | 6.00% | 108.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 16 | 76.2% | 7 | 6 | 8 | 4.77% | 100.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 219.95 | 235.06 | 235.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 10:15:00 | 219.30 | 234.91 | 235.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 227.60 | 226.62 | 230.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 11:00:00 | 227.60 | 226.62 | 230.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 234.81 | 226.63 | 230.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:45:00 | 233.58 | 226.63 | 230.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 233.99 | 226.71 | 230.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 229.43 | 227.05 | 230.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 14:15:00 | 231.40 | 227.41 | 230.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 15:15:00 | 231.64 | 227.45 | 230.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:45:00 | 231.62 | 227.53 | 230.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 229.64 | 227.70 | 230.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:30:00 | 230.83 | 227.70 | 230.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 232.00 | 227.74 | 230.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 232.00 | 227.74 | 230.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 231.40 | 227.77 | 230.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 10:45:00 | 231.03 | 228.09 | 230.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 230.63 | 228.09 | 230.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 13:00:00 | 231.00 | 228.15 | 230.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 12:15:00 | 219.83 | 227.96 | 230.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 12:15:00 | 220.06 | 227.96 | 230.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 12:15:00 | 220.04 | 227.96 | 230.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 12:15:00 | 219.48 | 227.96 | 230.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 12:15:00 | 219.10 | 227.96 | 230.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 12:15:00 | 219.45 | 227.96 | 230.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 15:15:00 | 217.96 | 227.67 | 229.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-12 09:15:00 | 208.48 | 226.77 | 229.35 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 11:15:00 | 266.96 | 226.29 | 226.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 12:15:00 | 268.40 | 226.71 | 226.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 10:15:00 | 249.80 | 249.96 | 241.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-04 11:00:00 | 249.80 | 249.96 | 241.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 238.45 | 249.76 | 241.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:30:00 | 240.80 | 249.76 | 241.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 233.00 | 249.59 | 241.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:45:00 | 232.95 | 249.59 | 241.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 240.50 | 249.33 | 241.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:00:00 | 240.50 | 249.33 | 241.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 242.40 | 249.26 | 241.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:30:00 | 241.95 | 249.26 | 241.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 242.80 | 249.20 | 241.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 242.95 | 249.20 | 241.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 243.25 | 249.14 | 241.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:15:00 | 239.15 | 249.14 | 241.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 239.00 | 249.04 | 241.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 239.40 | 249.04 | 241.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 243.05 | 248.98 | 241.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 14:30:00 | 244.10 | 247.20 | 240.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:30:00 | 246.00 | 247.00 | 241.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 10:15:00 | 245.00 | 247.20 | 241.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 15:15:00 | 238.50 | 246.83 | 241.35 | SL hit (close<static) qty=1.00 sl=238.60 alert=retest2 |

### Cycle 3 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 222.91 | 237.86 | 237.88 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 10:15:00 | 247.82 | 237.79 | 237.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 11:15:00 | 249.32 | 237.91 | 237.84 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-12-30 09:15:00 | 229.43 | 2026-01-08 12:15:00 | 219.83 | PARTIAL | 0.50 | 4.18% |
| SELL | retest2 | 2025-12-31 14:15:00 | 231.40 | 2026-01-08 12:15:00 | 220.06 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2025-12-31 15:15:00 | 231.64 | 2026-01-08 12:15:00 | 220.04 | PARTIAL | 0.50 | 5.01% |
| SELL | retest2 | 2026-01-01 09:45:00 | 231.62 | 2026-01-08 12:15:00 | 219.48 | PARTIAL | 0.50 | 5.24% |
| SELL | retest2 | 2026-01-05 10:45:00 | 231.03 | 2026-01-08 12:15:00 | 219.10 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2026-01-05 11:15:00 | 230.63 | 2026-01-08 12:15:00 | 219.45 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2026-01-05 13:00:00 | 231.00 | 2026-01-08 15:15:00 | 217.96 | PARTIAL | 0.50 | 5.65% |
| SELL | retest2 | 2025-12-30 09:15:00 | 229.43 | 2026-01-12 09:15:00 | 208.48 | TARGET_HIT | 0.50 | 9.13% |
| SELL | retest2 | 2025-12-31 14:15:00 | 231.40 | 2026-01-12 09:15:00 | 208.46 | TARGET_HIT | 0.50 | 9.91% |
| SELL | retest2 | 2025-12-31 15:15:00 | 231.64 | 2026-01-13 14:15:00 | 208.26 | TARGET_HIT | 0.50 | 10.09% |
| SELL | retest2 | 2026-01-01 09:45:00 | 231.62 | 2026-01-20 09:15:00 | 207.93 | TARGET_HIT | 0.50 | 10.23% |
| SELL | retest2 | 2026-01-05 10:45:00 | 231.03 | 2026-01-20 09:15:00 | 207.57 | TARGET_HIT | 0.50 | 10.16% |
| SELL | retest2 | 2026-01-05 11:15:00 | 230.63 | 2026-01-20 09:15:00 | 207.90 | TARGET_HIT | 0.50 | 9.86% |
| SELL | retest2 | 2026-01-05 13:00:00 | 231.00 | 2026-01-20 11:15:00 | 206.49 | TARGET_HIT | 0.50 | 10.61% |
| SELL | retest2 | 2026-01-30 10:15:00 | 227.65 | 2026-02-01 09:15:00 | 232.50 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2026-01-30 11:45:00 | 226.89 | 2026-02-01 09:15:00 | 232.50 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2026-02-01 11:30:00 | 226.75 | 2026-02-01 12:15:00 | 215.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 11:30:00 | 226.75 | 2026-02-01 12:15:00 | 220.65 | STOP_HIT | 0.50 | 2.69% |
| BUY | retest2 | 2026-03-10 14:30:00 | 244.10 | 2026-03-13 15:15:00 | 238.50 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2026-03-12 10:30:00 | 246.00 | 2026-03-13 15:15:00 | 238.50 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2026-03-13 10:15:00 | 245.00 | 2026-03-13 15:15:00 | 238.50 | STOP_HIT | 1.00 | -2.65% |
