# Aditya Birla Capital Ltd. (ABCAPITAL)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 362.25
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT2_SKIP | 4 |
| ALERT3 | 28 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 21 |
| PARTIAL | 9 |
| TARGET_HIT | 7 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 17
- **Target hits / Stop hits / Partials:** 7 / 20 / 9
- **Avg / median % per leg:** 2.64% / 2.85%
- **Sum % (uncompounded):** 94.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 5 | 38.5% | 3 | 8 | 2 | 1.71% | 22.3% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 2 | 2 | 1.67% | 6.7% |
| BUY @ 3rd Alert (retest2) | 9 | 3 | 33.3% | 3 | 6 | 0 | 1.73% | 15.6% |
| SELL (all) | 23 | 14 | 60.9% | 4 | 12 | 7 | 3.16% | 72.6% |
| SELL @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| SELL @ 3rd Alert (retest2) | 15 | 6 | 40.0% | 0 | 12 | 3 | 0.84% | 12.6% |
| retest1 (combined) | 12 | 10 | 83.3% | 4 | 2 | 6 | 5.56% | 66.7% |
| retest2 (combined) | 24 | 9 | 37.5% | 3 | 18 | 3 | 1.17% | 28.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 10:15:00 | 178.75 | 182.10 | 182.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 11:15:00 | 178.10 | 182.06 | 182.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 13:15:00 | 181.05 | 180.74 | 181.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 13:15:00 | 181.05 | 180.74 | 181.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 13:15:00 | 181.05 | 180.74 | 181.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 14:15:00 | 181.55 | 180.74 | 181.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 14:15:00 | 181.75 | 180.75 | 181.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 09:15:00 | 177.55 | 180.76 | 181.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-12 12:00:00 | 180.75 | 180.17 | 180.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-12 14:00:00 | 180.50 | 180.18 | 180.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-12 15:15:00 | 180.65 | 180.19 | 180.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 15:15:00 | 180.65 | 180.19 | 180.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-13 09:15:00 | 180.50 | 180.19 | 180.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-13 11:15:00 | 181.65 | 180.22 | 180.98 | SL hit (close>static) qty=1.00 sl=181.15 alert=retest2 |

### Cycle 2 — BUY (started 2024-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 11:15:00 | 175.40 | 172.32 | 172.31 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 13:15:00 | 165.75 | 172.30 | 172.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 14:15:00 | 163.35 | 172.21 | 172.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 13:15:00 | 171.50 | 170.91 | 171.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 13:15:00 | 171.50 | 170.91 | 171.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 13:15:00 | 171.50 | 170.91 | 171.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-31 14:00:00 | 171.50 | 170.91 | 171.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 181.70 | 170.71 | 171.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 10:00:00 | 181.70 | 170.71 | 171.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 10:15:00 | 183.70 | 170.84 | 171.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 11:00:00 | 183.70 | 170.84 | 171.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 11:15:00 | 178.25 | 172.05 | 172.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 14:15:00 | 180.00 | 172.25 | 172.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-29 09:15:00 | 180.50 | 181.00 | 177.72 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-29 14:30:00 | 182.75 | 181.04 | 177.82 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-29 15:00:00 | 184.05 | 181.04 | 177.82 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-04 09:15:00 | 191.89 | 181.68 | 178.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-04 09:15:00 | 193.25 | 181.68 | 178.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 180.35 | 182.90 | 179.20 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-03-06 09:15:00 | 180.35 | 182.90 | 179.20 | SL hit (close<ema200) qty=0.50 sl=182.90 alert=retest1 |

### Cycle 5 — SELL (started 2024-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 09:15:00 | 203.59 | 222.17 | 222.19 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-09-17 12:15:00)

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

### Cycle 7 — SELL (started 2024-10-23 14:15:00)

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

### Cycle 8 — BUY (started 2025-04-03 13:15:00)

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

### Cycle 9 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 316.00 | 342.52 | 342.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 298.00 | 330.50 | 335.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 328.65 | 317.73 | 326.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 328.65 | 317.73 | 326.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 328.65 | 317.73 | 326.79 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 342.25 | 332.43 | 332.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 345.40 | 332.66 | 332.50 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-10-09 09:15:00 | 177.55 | 2023-10-13 11:15:00 | 181.65 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2023-10-12 12:00:00 | 180.75 | 2023-10-16 12:15:00 | 181.25 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2023-10-12 14:00:00 | 180.50 | 2023-10-17 09:15:00 | 182.25 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2023-10-12 15:15:00 | 180.65 | 2023-10-17 09:15:00 | 182.25 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2023-10-13 09:15:00 | 180.50 | 2023-10-17 09:15:00 | 182.25 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2023-10-13 15:15:00 | 179.60 | 2023-10-17 09:15:00 | 182.25 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2023-10-16 15:00:00 | 180.50 | 2023-10-17 09:15:00 | 182.25 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2023-10-18 14:30:00 | 180.25 | 2023-10-19 10:15:00 | 182.45 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2023-10-19 09:15:00 | 179.90 | 2023-10-19 10:15:00 | 182.45 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2023-10-23 10:00:00 | 179.05 | 2023-10-26 09:15:00 | 170.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-23 10:00:00 | 179.05 | 2023-11-08 09:15:00 | 176.30 | STOP_HIT | 0.50 | 1.54% |
| SELL | retest2 | 2023-11-17 09:15:00 | 171.95 | 2023-12-07 09:15:00 | 163.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-11-17 09:15:00 | 171.95 | 2024-01-01 09:15:00 | 167.05 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest2 | 2024-01-11 14:30:00 | 180.20 | 2024-01-18 09:15:00 | 171.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-11 14:30:00 | 180.20 | 2024-01-18 09:15:00 | 173.50 | STOP_HIT | 0.50 | 3.72% |
| BUY | retest1 | 2024-02-29 14:30:00 | 182.75 | 2024-03-04 09:15:00 | 191.89 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-02-29 15:00:00 | 184.05 | 2024-03-04 09:15:00 | 193.25 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-02-29 14:30:00 | 182.75 | 2024-03-06 09:15:00 | 180.35 | STOP_HIT | 0.50 | -1.31% |
| BUY | retest1 | 2024-02-29 15:00:00 | 184.05 | 2024-03-06 09:15:00 | 180.35 | STOP_HIT | 0.50 | -2.01% |
| BUY | retest2 | 2024-03-12 09:15:00 | 184.45 | 2024-03-13 09:15:00 | 176.45 | STOP_HIT | 1.00 | -4.34% |
| BUY | retest2 | 2024-04-01 13:15:00 | 180.60 | 2024-04-02 10:15:00 | 198.66 | TARGET_HIT | 1.00 | 10.00% |
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
