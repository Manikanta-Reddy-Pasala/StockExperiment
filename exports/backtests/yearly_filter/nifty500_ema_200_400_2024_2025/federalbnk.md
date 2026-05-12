# Federal Bank Ltd. (FEDERALBNK)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 297.40
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
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 28 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 0 |
| TARGET_HIT | 3 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 4
- **Target hits / Stop hits / Partials:** 3 / 6 / 0
- **Avg / median % per leg:** 3.03% / 0.81%
- **Sum % (uncompounded):** 27.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 5 | 100.0% | 3 | 2 | 0 | 6.33% | 31.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 5 | 100.0% | 3 | 2 | 0 | 6.33% | 31.7% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.10% | -4.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.10% | -4.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 5 | 55.6% | 3 | 6 | 0 | 3.03% | 27.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 09:15:00 | 187.50 | 200.96 | 201.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 178.96 | 195.76 | 197.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 183.20 | 181.37 | 185.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-19 10:00:00 | 183.20 | 181.37 | 185.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 13:15:00 | 186.07 | 181.51 | 185.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 14:00:00 | 186.07 | 181.51 | 185.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 14:15:00 | 186.10 | 181.56 | 185.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 15:00:00 | 186.10 | 181.56 | 185.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 15:15:00 | 186.15 | 181.61 | 185.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 09:15:00 | 187.20 | 181.61 | 185.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 186.62 | 181.70 | 185.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:30:00 | 186.43 | 181.70 | 185.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 11:15:00 | 186.81 | 181.75 | 185.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 11:45:00 | 187.01 | 181.75 | 185.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 10:15:00 | 190.60 | 187.64 | 187.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 11:00:00 | 190.60 | 187.64 | 187.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 11:15:00 | 188.55 | 188.04 | 188.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 11:45:00 | 189.21 | 188.04 | 188.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 12:15:00 | 190.62 | 188.06 | 188.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 13:00:00 | 190.62 | 188.06 | 188.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-04-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 13:15:00 | 190.19 | 188.23 | 188.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 14:15:00 | 190.67 | 188.25 | 188.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 09:15:00 | 190.83 | 193.93 | 191.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 190.83 | 193.93 | 191.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 190.83 | 193.93 | 191.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 10:00:00 | 196.10 | 192.43 | 191.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 11:00:00 | 196.65 | 192.47 | 191.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-01 12:15:00 | 215.71 | 206.24 | 202.11 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 10:15:00 | 197.67 | 204.89 | 204.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 12:15:00 | 197.09 | 204.73 | 204.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 12:15:00 | 196.64 | 196.44 | 199.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 13:00:00 | 196.64 | 196.44 | 199.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 198.87 | 196.42 | 198.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 198.87 | 196.42 | 198.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 199.07 | 196.45 | 198.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 199.07 | 196.45 | 198.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 199.93 | 196.49 | 198.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 14:30:00 | 198.21 | 196.61 | 198.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:15:00 | 198.01 | 196.73 | 198.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:30:00 | 198.11 | 196.77 | 198.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 11:00:00 | 198.01 | 196.78 | 198.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 198.36 | 195.01 | 197.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:30:00 | 198.51 | 195.01 | 197.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 199.29 | 195.06 | 197.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 199.29 | 195.06 | 197.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 200.26 | 195.32 | 197.35 | SL hit (close>static) qty=1.00 sl=200.20 alert=retest2 |

### Cycle 4 — BUY (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 13:15:00 | 213.99 | 199.03 | 199.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 14:15:00 | 215.64 | 199.20 | 199.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 258.20 | 259.38 | 247.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 11:00:00 | 258.20 | 259.38 | 247.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 248.40 | 258.08 | 249.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:00:00 | 248.40 | 258.08 | 249.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 247.60 | 257.97 | 249.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:45:00 | 247.60 | 257.97 | 249.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 247.25 | 257.87 | 249.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:00:00 | 247.25 | 257.87 | 249.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 249.25 | 257.78 | 249.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:30:00 | 248.20 | 257.78 | 249.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 244.65 | 257.57 | 249.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:45:00 | 244.85 | 257.57 | 249.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 271.70 | 287.13 | 277.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:00:00 | 271.70 | 287.13 | 277.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 272.50 | 286.98 | 277.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:45:00 | 272.00 | 286.98 | 277.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 276.05 | 285.93 | 276.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:00:00 | 276.05 | 285.93 | 276.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 277.40 | 285.85 | 276.93 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 09:15:00 | 268.30 | 272.01 | 272.03 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 15:15:00 | 275.65 | 272.06 | 272.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 285.15 | 272.19 | 272.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 15:15:00 | 283.80 | 285.00 | 280.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 09:15:00 | 282.00 | 285.00 | 280.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 282.90 | 284.92 | 280.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:45:00 | 280.95 | 284.92 | 280.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-05 13:15:00 | 163.10 | 2024-06-20 14:15:00 | 179.41 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-13 10:00:00 | 196.10 | 2025-07-01 12:15:00 | 215.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-13 11:00:00 | 196.65 | 2025-07-01 12:15:00 | 216.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-04 11:45:00 | 196.09 | 2025-08-11 10:15:00 | 197.67 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2025-08-04 13:00:00 | 195.99 | 2025-08-11 10:15:00 | 197.67 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2025-09-18 14:30:00 | 198.21 | 2025-10-08 09:15:00 | 200.26 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-09-19 14:15:00 | 198.01 | 2025-10-08 09:15:00 | 200.26 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-09-22 09:30:00 | 198.11 | 2025-10-08 09:15:00 | 200.26 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-09-22 11:00:00 | 198.01 | 2025-10-08 09:15:00 | 200.26 | STOP_HIT | 1.00 | -1.14% |
