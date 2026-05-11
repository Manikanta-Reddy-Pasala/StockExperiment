# Jindal Saw Ltd. (JINDALSAW)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 243.50
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
| ALERT2_SKIP | 3 |
| ALERT3 | 17 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 16 |
| PARTIAL | 3 |
| TARGET_HIT | 11 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 6
- **Target hits / Stop hits / Partials:** 11 / 6 / 3
- **Avg / median % per leg:** 5.63% / 10.00%
- **Sum % (uncompounded):** 112.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 8 | 61.5% | 8 | 5 | 0 | 5.57% | 72.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 8 | 61.5% | 8 | 5 | 0 | 5.57% | 72.4% |
| SELL (all) | 7 | 6 | 85.7% | 3 | 1 | 3 | 5.73% | 40.1% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.87% | -4.9% |
| SELL @ 3rd Alert (retest2) | 6 | 6 | 100.0% | 3 | 0 | 3 | 7.50% | 45.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.87% | -4.9% |
| retest2 (combined) | 19 | 14 | 73.7% | 11 | 5 | 3 | 6.18% | 117.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 14:15:00 | 217.50 | 238.07 | 238.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-18 09:15:00 | 207.80 | 237.56 | 237.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 229.75 | 227.09 | 231.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 09:15:00 | 229.75 | 227.09 | 231.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 229.75 | 227.09 | 231.78 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-04-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 14:15:00 | 246.23 | 235.24 | 235.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-12 09:15:00 | 253.93 | 235.54 | 235.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 235.98 | 238.45 | 236.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 09:15:00 | 235.98 | 238.45 | 236.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 235.98 | 238.45 | 236.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 12:30:00 | 239.05 | 238.41 | 236.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-24 09:15:00 | 262.96 | 239.78 | 237.82 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 11:15:00 | 319.30 | 334.58 | 334.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 13:15:00 | 319.00 | 334.29 | 334.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 316.65 | 314.95 | 322.04 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 10:45:00 | 310.85 | 315.04 | 321.81 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 319.45 | 315.15 | 321.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:30:00 | 319.55 | 315.15 | 321.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 321.15 | 315.36 | 321.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:15:00 | 326.55 | 315.36 | 321.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 326.00 | 315.46 | 321.60 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-06 09:15:00 | 326.00 | 315.46 | 321.60 | SL hit (close>ema400) qty=1.00 sl=321.60 alert=retest1 |

### Cycle 4 — BUY (started 2026-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 13:15:00 | 189.53 | 174.46 | 174.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 191.20 | 177.11 | 175.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 180.15 | 182.13 | 178.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 12:15:00 | 180.56 | 182.08 | 179.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 180.56 | 182.08 | 179.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 13:15:00 | 180.74 | 182.08 | 179.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 14:15:00 | 178.26 | 182.02 | 179.02 | SL hit (close<static) qty=1.00 sl=179.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 13:15:00 | 165.30 | 177.69 | 177.69 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 11:15:00 | 188.06 | 177.77 | 177.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 13:15:00 | 190.20 | 178.01 | 177.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 183.78 | 184.87 | 181.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 10:00:00 | 183.78 | 184.87 | 181.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 182.50 | 184.85 | 181.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 184.95 | 184.57 | 181.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 180.17 | 184.52 | 181.71 | SL hit (close<static) qty=1.00 sl=181.62 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-19 12:30:00 | 239.05 | 2024-04-24 09:15:00 | 262.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 12:30:00 | 244.08 | 2024-06-07 10:15:00 | 268.49 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2024-12-04 10:45:00 | 310.85 | 2024-12-06 09:15:00 | 326.00 | STOP_HIT | 1.00 | -4.87% |
| SELL | retest2 | 2024-12-16 10:15:00 | 318.50 | 2024-12-18 15:15:00 | 302.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 10:45:00 | 318.15 | 2024-12-18 15:15:00 | 302.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 09:30:00 | 316.75 | 2024-12-19 09:15:00 | 300.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 10:15:00 | 318.50 | 2025-01-01 09:15:00 | 286.65 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-16 10:45:00 | 318.15 | 2025-01-01 09:15:00 | 286.33 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-18 09:30:00 | 316.75 | 2025-01-01 09:15:00 | 285.07 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-19 13:15:00 | 180.74 | 2026-02-19 14:15:00 | 178.26 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-02-20 10:45:00 | 180.86 | 2026-02-23 10:15:00 | 178.95 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2026-02-20 11:30:00 | 181.13 | 2026-02-23 10:15:00 | 178.95 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-02-20 12:15:00 | 181.45 | 2026-02-23 10:15:00 | 178.95 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-03-24 09:15:00 | 184.95 | 2026-03-24 09:15:00 | 180.17 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2026-03-25 09:15:00 | 185.96 | 2026-04-09 09:15:00 | 204.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-27 12:15:00 | 184.65 | 2026-04-09 09:15:00 | 203.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-27 13:00:00 | 185.43 | 2026-04-09 09:15:00 | 203.97 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 15:00:00 | 195.50 | 2026-04-15 10:15:00 | 215.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 10:00:00 | 196.55 | 2026-04-15 10:15:00 | 216.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:15:00 | 195.82 | 2026-04-15 10:15:00 | 215.40 | TARGET_HIT | 1.00 | 10.00% |
