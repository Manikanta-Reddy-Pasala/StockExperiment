# Engineers India Ltd. (ENGINERSIN)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 256.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 26 |
| PARTIAL | 7 |
| TARGET_HIT | 1 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 22
- **Target hits / Stop hits / Partials:** 0 / 29 / 7
- **Avg / median % per leg:** -0.19% / -1.50%
- **Sum % (uncompounded):** -6.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -4.03% | -20.2% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.05% | -12.2% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.00% | -8.0% |
| SELL (all) | 31 | 14 | 45.2% | 0 | 24 | 7 | 0.43% | 13.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 31 | 14 | 45.2% | 0 | 24 | 7 | 0.43% | 13.3% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.05% | -12.2% |
| retest2 (combined) | 33 | 14 | 42.4% | 0 | 26 | 7 | 0.16% | 5.3% |

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

### Cycle 2 — BUY (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 14:15:00 | 205.60 | 200.07 | 200.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 09:15:00 | 206.60 | 200.18 | 200.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 11:15:00 | 200.41 | 200.52 | 200.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 11:15:00 | 200.41 | 200.52 | 200.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 200.41 | 200.52 | 200.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 200.41 | 200.52 | 200.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 200.27 | 200.52 | 200.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:45:00 | 200.03 | 200.52 | 200.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 199.91 | 200.51 | 200.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:30:00 | 199.90 | 200.51 | 200.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 197.30 | 200.48 | 200.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 197.30 | 200.48 | 200.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 191.01 | 200.01 | 200.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 15:15:00 | 187.00 | 198.48 | 199.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 10:15:00 | 182.60 | 181.71 | 188.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-10 10:30:00 | 182.50 | 181.71 | 188.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 204.70 | 182.05 | 187.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:15:00 | 205.74 | 182.05 | 187.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 12:15:00 | 214.27 | 192.63 | 192.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 219.65 | 193.55 | 193.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 15:15:00 | 201.10 | 201.94 | 197.90 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 09:15:00 | 205.30 | 201.94 | 197.90 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 12:15:00 | 203.50 | 201.97 | 197.97 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 14:30:00 | 203.49 | 202.00 | 198.05 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 195.82 | 202.26 | 198.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 195.82 | 202.26 | 198.35 | SL hit (close<ema400) qty=1.00 sl=198.35 alert=retest1 |

### Cycle 5 — SELL (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 14:15:00 | 187.43 | 196.14 | 196.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 186.21 | 195.43 | 195.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 11:15:00 | 199.80 | 194.90 | 195.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 11:15:00 | 199.80 | 194.90 | 195.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 199.80 | 194.90 | 195.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:00:00 | 199.80 | 194.90 | 195.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 199.83 | 194.95 | 195.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 197.83 | 194.99 | 195.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 194.09 | 195.11 | 195.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 14:15:00 | 198.83 | 195.14 | 195.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 15:15:00 | 199.00 | 195.19 | 195.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 199.71 | 195.30 | 195.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:45:00 | 200.40 | 195.30 | 195.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-06 13:15:00 | 202.00 | 195.41 | 195.73 | SL hit (close>static) qty=1.00 sl=201.88 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 207.24 | 196.13 | 196.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 10:15:00 | 210.22 | 196.78 | 196.41 | Break + close above crossover candle high |


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
| BUY | retest1 | 2026-03-05 09:15:00 | 205.30 | 2026-03-09 09:15:00 | 195.82 | STOP_HIT | 1.00 | -4.62% |
| BUY | retest1 | 2026-03-05 12:15:00 | 203.50 | 2026-03-09 09:15:00 | 195.82 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest1 | 2026-03-05 14:30:00 | 203.49 | 2026-03-09 09:15:00 | 195.82 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2026-03-10 15:15:00 | 200.59 | 2026-03-13 10:15:00 | 192.25 | STOP_HIT | 1.00 | -4.16% |
| BUY | retest2 | 2026-03-12 11:15:00 | 199.93 | 2026-03-13 10:15:00 | 192.25 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2026-04-01 13:30:00 | 197.83 | 2026-04-06 13:15:00 | 202.00 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2026-04-02 09:15:00 | 194.09 | 2026-04-06 13:15:00 | 202.00 | STOP_HIT | 1.00 | -4.08% |
| SELL | retest2 | 2026-04-02 14:15:00 | 198.83 | 2026-04-06 13:15:00 | 202.00 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-04-02 15:15:00 | 199.00 | 2026-04-06 13:15:00 | 202.00 | STOP_HIT | 1.00 | -1.51% |
