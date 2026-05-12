# The New India Assurance Company Ltd. (NIACL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 163.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 42 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 46 |
| PARTIAL | 19 |
| TARGET_HIT | 14 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 65 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 22
- **Target hits / Stop hits / Partials:** 14 / 32 / 19
- **Avg / median % per leg:** 2.84% / 4.43%
- **Sum % (uncompounded):** 184.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 4 | 36.4% | 4 | 7 | 0 | 1.99% | 21.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 4 | 36.4% | 4 | 7 | 0 | 1.99% | 21.9% |
| SELL (all) | 54 | 39 | 72.2% | 10 | 25 | 19 | 3.01% | 162.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 54 | 39 | 72.2% | 10 | 25 | 19 | 3.01% | 162.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 65 | 43 | 66.2% | 14 | 32 | 19 | 2.84% | 184.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 09:15:00 | 247.62 | 233.65 | 233.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 15:15:00 | 254.00 | 234.52 | 234.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 13:15:00 | 235.20 | 237.38 | 235.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 13:15:00 | 235.20 | 237.38 | 235.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 235.20 | 237.38 | 235.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 235.20 | 237.38 | 235.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 232.50 | 237.34 | 235.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 232.50 | 237.34 | 235.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 234.40 | 237.23 | 235.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 12:30:00 | 235.81 | 237.19 | 235.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 13:45:00 | 235.95 | 237.20 | 235.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-04 15:15:00 | 259.39 | 239.34 | 237.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 11:15:00 | 239.30 | 256.37 | 256.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 236.50 | 255.08 | 255.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 196.22 | 192.56 | 207.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-28 10:00:00 | 196.22 | 192.56 | 207.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 205.50 | 193.41 | 206.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:00:00 | 205.50 | 193.41 | 206.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 204.25 | 193.52 | 206.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 09:30:00 | 202.51 | 194.02 | 206.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 12:15:00 | 202.30 | 194.19 | 205.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 13:15:00 | 202.75 | 194.27 | 205.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 10:15:00 | 206.63 | 194.82 | 205.97 | SL hit (close>static) qty=1.00 sl=206.51 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 15:15:00 | 174.50 | 166.88 | 166.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 176.35 | 166.98 | 166.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 14:15:00 | 182.82 | 183.34 | 177.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 15:00:00 | 182.82 | 183.34 | 177.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 177.73 | 183.14 | 178.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:30:00 | 177.74 | 183.14 | 178.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 177.79 | 183.08 | 178.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 176.83 | 183.08 | 178.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 177.90 | 183.03 | 178.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 177.90 | 183.03 | 178.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 176.40 | 182.97 | 178.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 09:15:00 | 178.00 | 182.97 | 178.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 13:15:00 | 175.61 | 184.43 | 182.36 | SL hit (close<static) qty=1.00 sl=176.01 alert=retest2 |

### Cycle 4 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 182.04 | 189.83 | 189.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 178.82 | 188.48 | 189.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 09:15:00 | 152.40 | 151.79 | 159.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-01 10:00:00 | 152.40 | 151.79 | 159.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 157.75 | 150.74 | 157.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:00:00 | 157.75 | 150.74 | 157.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 159.32 | 150.82 | 157.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:00:00 | 159.32 | 150.82 | 157.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 160.13 | 150.91 | 157.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 14:00:00 | 159.14 | 151.00 | 157.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 14:45:00 | 158.80 | 151.07 | 157.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 156.65 | 151.15 | 157.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 09:15:00 | 151.18 | 151.71 | 157.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 10:15:00 | 150.86 | 151.71 | 157.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 11:15:00 | 151.88 | 151.71 | 157.09 | SL hit (close>ema200) qty=0.50 sl=151.71 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 15:15:00 | 163.20 | 146.10 | 146.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 166.89 | 146.31 | 146.18 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-16 13:15:00 | 231.95 | 2024-05-22 09:15:00 | 248.25 | STOP_HIT | 1.00 | -7.03% |
| SELL | retest2 | 2024-05-17 14:45:00 | 231.90 | 2024-05-22 09:15:00 | 248.25 | STOP_HIT | 1.00 | -7.05% |
| SELL | retest2 | 2024-05-21 13:15:00 | 231.75 | 2024-05-22 09:15:00 | 248.25 | STOP_HIT | 1.00 | -7.12% |
| SELL | retest2 | 2024-05-21 15:15:00 | 231.75 | 2024-05-22 09:15:00 | 248.25 | STOP_HIT | 1.00 | -7.12% |
| SELL | retest2 | 2024-05-28 11:45:00 | 228.70 | 2024-05-29 15:15:00 | 235.95 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2024-05-28 13:30:00 | 228.70 | 2024-05-29 15:15:00 | 235.95 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2024-05-30 10:00:00 | 228.45 | 2024-06-03 09:15:00 | 236.45 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2024-05-30 11:30:00 | 228.70 | 2024-06-03 09:15:00 | 236.45 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2024-06-04 09:15:00 | 223.00 | 2024-06-04 11:15:00 | 200.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-28 12:30:00 | 235.81 | 2024-07-04 15:15:00 | 259.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-28 13:45:00 | 235.95 | 2024-07-04 15:15:00 | 259.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-14 09:45:00 | 237.60 | 2024-08-14 11:15:00 | 232.20 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2024-08-14 12:45:00 | 237.00 | 2024-08-20 15:15:00 | 260.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-20 15:00:00 | 258.05 | 2024-09-06 14:15:00 | 249.75 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2024-09-06 10:45:00 | 256.75 | 2024-09-06 14:15:00 | 249.75 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2024-09-06 11:30:00 | 256.70 | 2024-09-06 14:15:00 | 249.75 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2024-09-06 13:30:00 | 256.35 | 2024-09-06 14:15:00 | 249.75 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2024-09-09 15:00:00 | 258.65 | 2024-09-10 09:15:00 | 250.20 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2024-12-05 09:30:00 | 202.51 | 2024-12-06 10:15:00 | 206.63 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-12-05 12:15:00 | 202.30 | 2024-12-06 10:15:00 | 206.63 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-12-05 13:15:00 | 202.75 | 2024-12-06 10:15:00 | 206.63 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-12-13 10:00:00 | 202.36 | 2024-12-16 09:15:00 | 207.71 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2024-12-26 11:15:00 | 204.50 | 2025-01-06 13:15:00 | 195.44 | PARTIAL | 0.50 | 4.43% |
| SELL | retest2 | 2024-12-26 12:30:00 | 205.19 | 2025-01-06 13:15:00 | 195.79 | PARTIAL | 0.50 | 4.58% |
| SELL | retest2 | 2024-12-27 09:15:00 | 205.73 | 2025-01-06 13:15:00 | 195.88 | PARTIAL | 0.50 | 4.79% |
| SELL | retest2 | 2024-12-30 09:15:00 | 203.75 | 2025-01-06 13:15:00 | 196.09 | PARTIAL | 0.50 | 3.76% |
| SELL | retest2 | 2025-01-01 15:15:00 | 206.10 | 2025-01-06 13:15:00 | 196.00 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2025-01-03 10:15:00 | 206.19 | 2025-01-06 14:15:00 | 194.27 | PARTIAL | 0.50 | 5.78% |
| SELL | retest2 | 2025-01-03 12:15:00 | 206.41 | 2025-01-06 14:15:00 | 194.93 | PARTIAL | 0.50 | 5.56% |
| SELL | retest2 | 2025-01-03 12:45:00 | 206.32 | 2025-01-06 15:15:00 | 193.56 | PARTIAL | 0.50 | 6.18% |
| SELL | retest2 | 2024-12-26 11:15:00 | 204.50 | 2025-01-13 09:15:00 | 184.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-26 12:30:00 | 205.19 | 2025-01-13 09:15:00 | 184.67 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-27 09:15:00 | 205.73 | 2025-01-13 09:15:00 | 185.16 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-30 09:15:00 | 203.75 | 2025-01-13 09:15:00 | 183.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-01 15:15:00 | 206.10 | 2025-01-13 09:15:00 | 185.49 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-03 10:15:00 | 206.19 | 2025-01-13 09:15:00 | 185.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-03 12:15:00 | 206.41 | 2025-01-13 09:15:00 | 185.77 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-03 12:45:00 | 206.32 | 2025-01-13 09:15:00 | 185.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-25 11:00:00 | 161.25 | 2025-03-28 15:15:00 | 153.61 | PARTIAL | 0.50 | 4.74% |
| SELL | retest2 | 2025-03-25 12:15:00 | 161.00 | 2025-04-01 09:15:00 | 153.19 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2025-03-25 13:00:00 | 161.69 | 2025-04-01 09:15:00 | 152.95 | PARTIAL | 0.50 | 5.41% |
| SELL | retest2 | 2025-03-25 15:00:00 | 161.30 | 2025-04-01 09:15:00 | 153.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 11:00:00 | 161.25 | 2025-04-01 11:15:00 | 157.41 | STOP_HIT | 0.50 | 2.38% |
| SELL | retest2 | 2025-03-25 12:15:00 | 161.00 | 2025-04-01 11:15:00 | 157.41 | STOP_HIT | 0.50 | 2.23% |
| SELL | retest2 | 2025-03-25 13:00:00 | 161.69 | 2025-04-01 11:15:00 | 157.41 | STOP_HIT | 0.50 | 2.65% |
| SELL | retest2 | 2025-03-25 15:00:00 | 161.30 | 2025-04-01 11:15:00 | 157.41 | STOP_HIT | 0.50 | 2.41% |
| SELL | retest2 | 2025-03-26 14:30:00 | 159.11 | 2025-04-07 09:15:00 | 151.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-27 12:00:00 | 159.57 | 2025-04-07 09:15:00 | 151.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 11:30:00 | 159.47 | 2025-04-07 09:15:00 | 151.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 09:30:00 | 159.50 | 2025-04-07 09:15:00 | 151.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-26 14:30:00 | 159.11 | 2025-04-08 14:15:00 | 156.93 | STOP_HIT | 0.50 | 1.37% |
| SELL | retest2 | 2025-03-27 12:00:00 | 159.57 | 2025-04-08 14:15:00 | 156.93 | STOP_HIT | 0.50 | 1.65% |
| SELL | retest2 | 2025-03-28 11:30:00 | 159.47 | 2025-04-08 14:15:00 | 156.93 | STOP_HIT | 0.50 | 1.59% |
| SELL | retest2 | 2025-04-04 09:30:00 | 159.50 | 2025-04-08 14:15:00 | 156.93 | STOP_HIT | 0.50 | 1.61% |
| SELL | retest2 | 2025-05-07 10:00:00 | 164.19 | 2025-05-08 09:15:00 | 168.65 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2025-05-08 14:45:00 | 163.80 | 2025-05-12 09:15:00 | 167.37 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-05-08 15:15:00 | 164.01 | 2025-05-12 09:15:00 | 167.37 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-06-20 09:15:00 | 178.00 | 2025-07-28 13:15:00 | 175.61 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-07-30 09:15:00 | 185.84 | 2025-07-30 14:15:00 | 204.42 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-10 14:00:00 | 159.14 | 2026-02-13 09:15:00 | 151.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 14:45:00 | 158.80 | 2026-02-13 10:15:00 | 150.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 14:00:00 | 159.14 | 2026-02-13 11:15:00 | 151.88 | STOP_HIT | 0.50 | 4.56% |
| SELL | retest2 | 2026-02-10 14:45:00 | 158.80 | 2026-02-13 11:15:00 | 151.88 | STOP_HIT | 0.50 | 4.36% |
| SELL | retest2 | 2026-02-11 09:15:00 | 156.65 | 2026-02-24 12:15:00 | 148.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 09:15:00 | 156.65 | 2026-03-02 09:15:00 | 140.99 | TARGET_HIT | 0.50 | 10.00% |
