# Aditya Birla Capital Ltd. (ABCAPITAL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 362.25
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
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 7 |
| PARTIAL | 4 |
| TARGET_HIT | 6 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 5
- **Target hits / Stop hits / Partials:** 6 / 5 / 4
- **Avg / median % per leg:** 4.66% / 5.00%
- **Sum % (uncompounded):** 69.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 2 | 5 | 0 | 1.42% | 9.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 2 | 28.6% | 2 | 5 | 0 | 1.42% | 9.9% |
| SELL (all) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| SELL @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| retest2 (combined) | 7 | 2 | 28.6% | 2 | 5 | 0 | 1.42% | 9.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 09:15:00 | 203.59 | 222.17 | 222.19 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 12:15:00 | 225.15 | 220.53 | 220.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 09:15:00 | 226.39 | 220.73 | 220.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 14:15:00 | 227.79 | 228.06 | 224.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-04 15:00:00 | 227.79 | 228.06 | 224.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 226.94 | 228.05 | 225.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:00:00 | 226.94 | 228.05 | 225.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 12:15:00 | 225.16 | 228.02 | 225.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 13:00:00 | 225.16 | 228.02 | 225.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 13:15:00 | 223.83 | 227.98 | 225.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 13:30:00 | 223.86 | 227.98 | 225.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 14:15:00 | 224.57 | 227.94 | 225.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 09:15:00 | 225.95 | 227.92 | 225.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 09:45:00 | 226.67 | 227.89 | 225.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 10:15:00 | 226.01 | 227.89 | 225.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 10:45:00 | 225.81 | 227.87 | 225.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 224.10 | 227.83 | 225.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:00:00 | 224.10 | 227.83 | 225.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 223.82 | 227.79 | 225.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:45:00 | 223.17 | 227.79 | 225.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 225.00 | 227.65 | 224.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 225.00 | 227.65 | 224.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 225.50 | 227.62 | 224.99 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-11 11:15:00 | 221.33 | 227.21 | 224.97 | SL hit (close<static) qty=1.00 sl=222.20 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 14:15:00 | 210.63 | 223.48 | 223.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 207.28 | 222.50 | 223.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 199.72 | 199.23 | 206.73 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 11:45:00 | 198.25 | 199.22 | 206.65 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 09:30:00 | 198.12 | 199.19 | 206.45 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 10:00:00 | 198.05 | 199.19 | 206.20 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 10:00:00 | 197.66 | 199.17 | 205.95 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 188.34 | 197.83 | 203.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 188.21 | 197.83 | 203.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 188.15 | 197.83 | 203.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 187.78 | 197.83 | 203.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-12-31 09:15:00 | 178.43 | 193.12 | 199.74 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 4 — BUY (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 13:15:00 | 192.50 | 173.30 | 173.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 193.43 | 173.50 | 173.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 260.60 | 263.69 | 247.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 10:15:00 | 248.20 | 261.92 | 248.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 248.20 | 261.92 | 248.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:30:00 | 248.85 | 261.92 | 248.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 248.60 | 261.79 | 248.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 09:30:00 | 251.85 | 261.16 | 248.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 10:45:00 | 251.85 | 261.08 | 248.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-04 14:15:00 | 277.04 | 260.54 | 249.41 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 316.00 | 342.52 | 342.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 298.00 | 330.50 | 335.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 328.65 | 317.73 | 326.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 328.65 | 317.73 | 326.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 328.65 | 317.73 | 326.79 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 342.25 | 332.43 | 332.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 345.40 | 332.66 | 332.50 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-10-08 09:15:00 | 225.95 | 2024-10-11 11:15:00 | 221.33 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-10-08 09:45:00 | 226.67 | 2024-10-11 11:15:00 | 221.33 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2024-10-08 10:15:00 | 226.01 | 2024-10-11 11:15:00 | 221.33 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-10-08 10:45:00 | 225.81 | 2024-10-11 11:15:00 | 221.33 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-10-15 09:30:00 | 227.69 | 2024-10-15 10:15:00 | 223.98 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest1 | 2024-12-04 11:45:00 | 198.25 | 2024-12-19 09:15:00 | 188.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-12-05 09:30:00 | 198.12 | 2024-12-19 09:15:00 | 188.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-12-06 10:00:00 | 198.05 | 2024-12-19 09:15:00 | 188.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-12-09 10:00:00 | 197.66 | 2024-12-19 09:15:00 | 187.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-12-04 11:45:00 | 198.25 | 2024-12-31 09:15:00 | 178.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2024-12-05 09:30:00 | 198.12 | 2024-12-31 09:15:00 | 178.31 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2024-12-06 10:00:00 | 198.05 | 2024-12-31 09:15:00 | 178.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2024-12-09 10:00:00 | 197.66 | 2024-12-31 09:15:00 | 177.89 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-31 09:30:00 | 251.85 | 2025-08-04 14:15:00 | 277.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 10:45:00 | 251.85 | 2025-08-04 14:15:00 | 277.04 | TARGET_HIT | 1.00 | 10.00% |
