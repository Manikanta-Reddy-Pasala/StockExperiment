# NCC Ltd. (NCC)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 170.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 34 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 48 |
| PARTIAL | 14 |
| TARGET_HIT | 9 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 62 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 29
- **Target hits / Stop hits / Partials:** 9 / 39 / 14
- **Avg / median % per leg:** 2.06% / 1.69%
- **Sum % (uncompounded):** 127.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 0 | 0.0% | 0 | 16 | 0 | -1.61% | -25.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 0 | 0.0% | 0 | 16 | 0 | -1.61% | -25.7% |
| SELL (all) | 46 | 33 | 71.7% | 9 | 23 | 14 | 3.33% | 153.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 46 | 33 | 71.7% | 9 | 23 | 14 | 3.33% | 153.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 62 | 33 | 53.2% | 9 | 39 | 14 | 2.06% | 127.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 302.35 | 315.58 | 315.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 15:15:00 | 301.90 | 315.45 | 315.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 11:15:00 | 308.95 | 307.31 | 310.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-16 11:45:00 | 309.20 | 307.31 | 310.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 309.00 | 307.36 | 310.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 09:15:00 | 307.45 | 307.36 | 310.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 11:45:00 | 307.60 | 307.37 | 310.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 12:15:00 | 311.00 | 307.40 | 310.57 | SL hit (close>static) qty=1.00 sl=310.95 alert=retest2 |

### Cycle 2 — BUY (started 2024-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 15:15:00 | 312.20 | 303.59 | 303.58 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 10:15:00 | 299.40 | 303.54 | 303.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 09:15:00 | 294.20 | 303.26 | 303.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 09:15:00 | 257.85 | 255.85 | 270.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-01 10:00:00 | 257.85 | 255.85 | 270.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 216.28 | 198.81 | 216.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:00:00 | 216.28 | 198.81 | 216.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 213.50 | 198.96 | 216.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 13:00:00 | 212.02 | 199.23 | 216.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 09:30:00 | 211.64 | 199.64 | 215.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 211.78 | 201.67 | 215.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 12:45:00 | 212.58 | 202.13 | 215.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 213.82 | 202.58 | 215.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:45:00 | 208.16 | 203.29 | 215.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 11:15:00 | 208.65 | 203.34 | 215.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 190.82 | 203.40 | 214.92 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 09:15:00 | 238.13 | 216.85 | 216.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 11:15:00 | 239.94 | 217.28 | 217.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 227.00 | 230.46 | 225.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 227.00 | 230.46 | 225.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 227.00 | 230.46 | 225.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:45:00 | 226.16 | 230.46 | 225.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 224.92 | 230.33 | 225.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 13:00:00 | 224.92 | 230.33 | 225.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 226.00 | 230.29 | 225.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 15:00:00 | 226.99 | 230.25 | 225.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:30:00 | 228.09 | 230.10 | 225.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 13:00:00 | 227.22 | 229.98 | 225.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 13:30:00 | 227.11 | 229.95 | 225.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 226.10 | 229.91 | 225.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 14:30:00 | 226.20 | 229.91 | 225.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 225.75 | 229.87 | 225.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 226.10 | 229.87 | 225.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 227.15 | 229.84 | 225.78 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 224.09 | 229.74 | 225.77 | SL hit (close<static) qty=1.00 sl=224.65 alert=retest2 |

### Cycle 5 — SELL (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 15:15:00 | 222.59 | 225.27 | 225.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 218.82 | 225.21 | 225.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 224.14 | 221.91 | 223.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 15:15:00 | 224.14 | 221.91 | 223.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 224.14 | 221.91 | 223.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 225.43 | 221.91 | 223.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 223.87 | 221.93 | 223.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 219.83 | 222.16 | 223.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 11:45:00 | 220.00 | 221.92 | 223.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 13:45:00 | 220.53 | 221.76 | 223.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 220.49 | 221.77 | 223.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 209.00 | 220.55 | 222.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 209.50 | 220.55 | 222.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 209.47 | 220.55 | 222.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 14:15:00 | 208.84 | 220.04 | 221.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 215.60 | 213.65 | 217.37 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 215.60 | 213.65 | 217.37 | SL hit (close>ema200) qty=0.50 sl=213.65 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 15:15:00 | 162.00 | 151.54 | 151.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 164.20 | 151.67 | 151.61 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-10-17 09:15:00 | 307.45 | 2024-10-17 12:15:00 | 311.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-10-17 11:45:00 | 307.60 | 2024-10-17 12:15:00 | 311.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-10-18 09:15:00 | 303.85 | 2024-10-18 12:15:00 | 311.05 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2024-10-21 09:30:00 | 307.55 | 2024-10-22 12:15:00 | 292.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:30:00 | 307.55 | 2024-10-25 09:15:00 | 276.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-11 09:30:00 | 303.60 | 2024-11-11 11:15:00 | 306.85 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-11-11 15:00:00 | 303.60 | 2024-11-13 09:15:00 | 288.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 15:00:00 | 303.60 | 2024-11-18 09:15:00 | 273.24 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-28 10:30:00 | 303.45 | 2024-11-29 09:15:00 | 310.60 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-12-12 11:30:00 | 303.50 | 2024-12-16 09:15:00 | 303.60 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2024-12-12 14:45:00 | 301.25 | 2024-12-16 09:15:00 | 303.60 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-12-13 09:15:00 | 299.10 | 2024-12-16 13:15:00 | 308.20 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2024-12-16 11:00:00 | 301.10 | 2024-12-16 13:15:00 | 308.20 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-12-16 11:30:00 | 301.00 | 2024-12-16 13:15:00 | 308.20 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-03-26 13:00:00 | 212.02 | 2025-04-07 09:15:00 | 190.82 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-27 09:30:00 | 211.64 | 2025-04-07 09:15:00 | 190.48 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-02 09:15:00 | 211.78 | 2025-04-07 09:15:00 | 190.60 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-02 12:45:00 | 212.58 | 2025-04-07 09:15:00 | 191.32 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-04 09:45:00 | 208.16 | 2025-04-07 09:15:00 | 187.34 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-04 11:15:00 | 208.65 | 2025-04-07 09:15:00 | 187.78 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-06 14:45:00 | 208.40 | 2025-05-08 09:15:00 | 216.40 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2025-05-06 15:15:00 | 207.70 | 2025-05-08 09:15:00 | 216.40 | STOP_HIT | 1.00 | -4.19% |
| SELL | retest2 | 2025-05-08 11:30:00 | 214.47 | 2025-05-09 09:15:00 | 203.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 11:30:00 | 214.47 | 2025-05-12 09:15:00 | 216.26 | STOP_HIT | 0.50 | -0.83% |
| BUY | retest2 | 2025-06-13 15:00:00 | 226.99 | 2025-06-18 11:15:00 | 224.09 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-06-16 11:30:00 | 228.09 | 2025-06-18 11:15:00 | 224.09 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-06-17 13:00:00 | 227.22 | 2025-06-18 11:15:00 | 224.09 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-06-17 13:30:00 | 227.11 | 2025-06-18 11:15:00 | 224.09 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-06-24 13:00:00 | 227.66 | 2025-07-04 11:15:00 | 224.45 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-06-25 09:30:00 | 228.68 | 2025-07-07 11:15:00 | 222.61 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-06-25 13:30:00 | 228.10 | 2025-07-07 11:15:00 | 222.61 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-06-25 14:00:00 | 227.87 | 2025-07-07 11:15:00 | 222.61 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-07-04 09:15:00 | 226.99 | 2025-07-07 11:15:00 | 222.61 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-07-09 11:30:00 | 227.19 | 2025-07-09 15:15:00 | 224.98 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-07-09 12:45:00 | 226.97 | 2025-07-09 15:15:00 | 224.98 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-07-15 10:45:00 | 227.46 | 2025-07-23 09:15:00 | 224.32 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-07-18 14:30:00 | 228.04 | 2025-07-23 09:15:00 | 224.32 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-07-21 09:45:00 | 227.70 | 2025-07-23 09:15:00 | 224.32 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-07-21 14:15:00 | 227.64 | 2025-07-23 09:15:00 | 224.32 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-07-21 14:45:00 | 227.60 | 2025-07-23 09:15:00 | 224.32 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-08-14 09:15:00 | 219.83 | 2025-08-28 09:15:00 | 209.00 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2025-08-18 11:45:00 | 220.00 | 2025-08-28 09:15:00 | 209.50 | PARTIAL | 0.50 | 4.77% |
| SELL | retest2 | 2025-08-19 13:45:00 | 220.53 | 2025-08-28 09:15:00 | 209.47 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2025-08-20 09:15:00 | 220.49 | 2025-08-28 14:15:00 | 208.84 | PARTIAL | 0.50 | 5.28% |
| SELL | retest2 | 2025-08-14 09:15:00 | 219.83 | 2025-09-16 09:15:00 | 215.60 | STOP_HIT | 0.50 | 1.92% |
| SELL | retest2 | 2025-08-18 11:45:00 | 220.00 | 2025-09-16 09:15:00 | 215.60 | STOP_HIT | 0.50 | 2.00% |
| SELL | retest2 | 2025-08-19 13:45:00 | 220.53 | 2025-09-16 09:15:00 | 215.60 | STOP_HIT | 0.50 | 2.24% |
| SELL | retest2 | 2025-08-20 09:15:00 | 220.49 | 2025-09-16 09:15:00 | 215.60 | STOP_HIT | 0.50 | 2.22% |
| SELL | retest2 | 2025-09-16 13:15:00 | 215.34 | 2025-09-26 09:15:00 | 204.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 14:00:00 | 214.98 | 2025-09-26 09:15:00 | 204.72 | PARTIAL | 0.50 | 4.77% |
| SELL | retest2 | 2025-09-17 15:15:00 | 215.50 | 2025-09-26 09:15:00 | 204.59 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2025-09-18 11:30:00 | 215.36 | 2025-09-26 09:15:00 | 205.53 | PARTIAL | 0.50 | 4.56% |
| SELL | retest2 | 2025-09-19 14:15:00 | 216.35 | 2025-09-26 09:15:00 | 204.28 | PARTIAL | 0.50 | 5.58% |
| SELL | retest2 | 2025-09-22 09:15:00 | 215.03 | 2025-09-26 10:15:00 | 204.23 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2025-09-16 13:15:00 | 215.34 | 2025-10-07 15:15:00 | 211.69 | STOP_HIT | 0.50 | 1.69% |
| SELL | retest2 | 2025-09-17 14:00:00 | 214.98 | 2025-10-07 15:15:00 | 211.69 | STOP_HIT | 0.50 | 1.53% |
| SELL | retest2 | 2025-09-17 15:15:00 | 215.50 | 2025-10-07 15:15:00 | 211.69 | STOP_HIT | 0.50 | 1.77% |
| SELL | retest2 | 2025-09-18 11:30:00 | 215.36 | 2025-10-07 15:15:00 | 211.69 | STOP_HIT | 0.50 | 1.70% |
| SELL | retest2 | 2025-09-19 14:15:00 | 216.35 | 2025-10-07 15:15:00 | 211.69 | STOP_HIT | 0.50 | 2.15% |
| SELL | retest2 | 2025-09-22 09:15:00 | 215.03 | 2025-10-07 15:15:00 | 211.69 | STOP_HIT | 0.50 | 1.55% |
| SELL | retest2 | 2025-11-03 10:15:00 | 216.16 | 2025-11-04 14:15:00 | 205.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 10:15:00 | 216.16 | 2025-11-06 14:15:00 | 194.54 | TARGET_HIT | 0.50 | 10.00% |
