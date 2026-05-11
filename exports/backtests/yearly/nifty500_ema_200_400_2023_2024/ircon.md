# IRCON International Ltd. (IRCON)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 158.99
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
| ALERT2_SKIP | 1 |
| ALERT3 | 24 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 36 |
| PARTIAL | 22 |
| TARGET_HIT | 15 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 44 / 14
- **Target hits / Stop hits / Partials:** 15 / 21 / 22
- **Avg / median % per leg:** 4.28% / 5.00%
- **Sum % (uncompounded):** 248.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.37% | -6.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.37% | -6.8% |
| SELL (all) | 53 | 44 | 83.0% | 15 | 16 | 22 | 4.82% | 255.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 53 | 44 | 83.0% | 15 | 16 | 22 | 4.82% | 255.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 58 | 44 | 75.9% | 15 | 21 | 22 | 4.28% | 248.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 263.15 | 274.54 | 274.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 09:15:00 | 261.85 | 273.85 | 274.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 220.34 | 218.85 | 232.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:00:00 | 220.34 | 218.85 | 232.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 218.89 | 206.31 | 217.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:15:00 | 221.70 | 206.31 | 217.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 219.19 | 206.44 | 217.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 14:30:00 | 217.23 | 217.73 | 220.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 15:15:00 | 217.00 | 217.73 | 220.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 10:00:00 | 216.91 | 217.71 | 220.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 11:15:00 | 217.00 | 217.71 | 220.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 213.77 | 217.63 | 220.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 15:15:00 | 211.25 | 217.46 | 219.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 13:45:00 | 211.72 | 217.18 | 219.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:30:00 | 210.80 | 216.96 | 219.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 10:00:00 | 211.15 | 215.52 | 218.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 218.16 | 215.54 | 218.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:30:00 | 218.30 | 215.54 | 218.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 215.70 | 215.54 | 218.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 12:30:00 | 214.56 | 215.54 | 218.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 09:15:00 | 219.18 | 215.58 | 218.53 | SL hit (close>static) qty=1.00 sl=219.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 189.66 | 166.60 | 166.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 190.92 | 171.82 | 169.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 194.39 | 194.58 | 184.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 11:00:00 | 194.39 | 194.58 | 184.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 191.15 | 196.76 | 190.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 190.70 | 196.76 | 190.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 191.21 | 196.71 | 190.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 192.85 | 196.36 | 190.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 10:45:00 | 192.17 | 196.28 | 190.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 11:30:00 | 192.14 | 196.23 | 190.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 12:30:00 | 192.34 | 196.19 | 190.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 190.39 | 196.02 | 190.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 190.54 | 196.02 | 190.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 191.15 | 195.97 | 190.75 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-17 11:15:00 | 189.67 | 195.58 | 190.76 | SL hit (close<static) qty=1.00 sl=189.80 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 10:15:00 | 175.00 | 187.99 | 188.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 173.62 | 186.57 | 187.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 12:15:00 | 177.67 | 173.07 | 178.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 12:15:00 | 177.67 | 173.07 | 178.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 177.67 | 173.07 | 178.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:00:00 | 177.67 | 173.07 | 178.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 185.63 | 172.48 | 176.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:45:00 | 185.00 | 172.48 | 176.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 183.03 | 172.59 | 176.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 11:15:00 | 182.48 | 172.59 | 176.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 182.40 | 173.07 | 176.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 10:30:00 | 182.10 | 173.26 | 176.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 14:15:00 | 187.34 | 174.41 | 177.22 | SL hit (close>static) qty=1.00 sl=186.90 alert=retest2 |

### Cycle 4 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 178.04 | 166.29 | 166.27 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 161.10 | 166.45 | 166.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 158.47 | 166.21 | 166.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 14:15:00 | 164.17 | 162.86 | 164.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 15:00:00 | 164.17 | 162.86 | 164.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 164.60 | 162.88 | 164.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:15:00 | 164.64 | 162.88 | 164.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 162.30 | 162.88 | 164.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 161.39 | 162.88 | 164.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 15:15:00 | 161.90 | 162.79 | 164.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:45:00 | 161.86 | 162.91 | 164.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 14:15:00 | 153.32 | 162.71 | 164.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 14:15:00 | 153.81 | 162.71 | 164.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 14:15:00 | 153.77 | 162.71 | 164.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 160.75 | 159.88 | 162.36 | SL hit (close>ema200) qty=0.50 sl=159.88 alert=retest2 |

### Cycle 6 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 160.55 | 144.72 | 144.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 12:15:00 | 163.05 | 146.62 | 145.67 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-12-19 14:30:00 | 217.23 | 2025-01-01 09:15:00 | 219.18 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-12-19 15:15:00 | 217.00 | 2025-01-06 13:15:00 | 206.37 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2024-12-20 10:00:00 | 216.91 | 2025-01-06 14:15:00 | 206.15 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2024-12-20 11:15:00 | 217.00 | 2025-01-06 14:15:00 | 206.06 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2024-12-23 15:15:00 | 211.25 | 2025-01-06 14:15:00 | 206.15 | PARTIAL | 0.50 | 2.41% |
| SELL | retest2 | 2024-12-24 13:45:00 | 211.72 | 2025-01-09 15:15:00 | 200.69 | PARTIAL | 0.50 | 5.21% |
| SELL | retest2 | 2024-12-26 09:30:00 | 210.80 | 2025-01-09 15:15:00 | 201.13 | PARTIAL | 0.50 | 4.59% |
| SELL | retest2 | 2024-12-31 10:00:00 | 211.15 | 2025-01-09 15:15:00 | 200.26 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2024-12-31 12:30:00 | 214.56 | 2025-01-09 15:15:00 | 200.59 | PARTIAL | 0.50 | 6.51% |
| SELL | retest2 | 2025-01-06 09:15:00 | 212.63 | 2025-01-09 15:15:00 | 202.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-19 15:15:00 | 217.00 | 2025-01-10 10:15:00 | 195.51 | TARGET_HIT | 0.50 | 9.90% |
| SELL | retest2 | 2024-12-20 10:00:00 | 216.91 | 2025-01-10 10:15:00 | 195.30 | TARGET_HIT | 0.50 | 9.96% |
| SELL | retest2 | 2024-12-20 11:15:00 | 217.00 | 2025-01-10 10:15:00 | 195.22 | TARGET_HIT | 0.50 | 10.04% |
| SELL | retest2 | 2024-12-23 15:15:00 | 211.25 | 2025-01-10 10:15:00 | 195.30 | TARGET_HIT | 0.50 | 7.55% |
| SELL | retest2 | 2024-12-24 13:45:00 | 211.72 | 2025-01-10 15:15:00 | 191.37 | TARGET_HIT | 0.50 | 9.61% |
| SELL | retest2 | 2024-12-26 09:30:00 | 210.80 | 2025-01-13 09:15:00 | 190.12 | TARGET_HIT | 0.50 | 9.81% |
| SELL | retest2 | 2024-12-31 10:00:00 | 211.15 | 2025-01-13 09:15:00 | 190.55 | TARGET_HIT | 0.50 | 9.76% |
| SELL | retest2 | 2024-12-31 12:30:00 | 214.56 | 2025-01-13 09:15:00 | 189.72 | TARGET_HIT | 0.50 | 11.58% |
| SELL | retest2 | 2025-01-06 09:15:00 | 212.63 | 2025-01-13 09:15:00 | 190.03 | TARGET_HIT | 0.50 | 10.63% |
| SELL | retest2 | 2025-01-21 10:15:00 | 213.30 | 2025-01-27 09:15:00 | 202.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 12:00:00 | 214.31 | 2025-01-27 09:15:00 | 203.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-22 09:15:00 | 207.89 | 2025-01-27 09:15:00 | 200.40 | PARTIAL | 0.50 | 3.60% |
| SELL | retest2 | 2025-01-23 11:30:00 | 210.95 | 2025-01-27 10:15:00 | 197.50 | PARTIAL | 0.50 | 6.38% |
| SELL | retest2 | 2025-01-21 10:15:00 | 213.30 | 2025-01-28 09:15:00 | 191.97 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-21 12:00:00 | 214.31 | 2025-01-28 09:15:00 | 192.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-22 09:15:00 | 207.89 | 2025-01-28 09:15:00 | 189.85 | TARGET_HIT | 0.50 | 8.68% |
| SELL | retest2 | 2025-01-23 11:30:00 | 210.95 | 2025-01-28 10:15:00 | 187.10 | TARGET_HIT | 0.50 | 11.31% |
| SELL | retest2 | 2025-02-01 12:15:00 | 204.56 | 2025-02-03 09:15:00 | 194.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 12:15:00 | 204.56 | 2025-02-11 09:15:00 | 184.10 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-15 09:15:00 | 192.85 | 2025-07-17 11:15:00 | 189.67 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-07-15 10:45:00 | 192.17 | 2025-07-17 11:15:00 | 189.67 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-07-15 11:30:00 | 192.14 | 2025-07-17 11:15:00 | 189.67 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-07-15 12:30:00 | 192.34 | 2025-07-17 11:15:00 | 189.67 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-07-21 09:15:00 | 192.42 | 2025-07-21 13:15:00 | 190.09 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-09-15 11:15:00 | 182.48 | 2025-09-17 14:15:00 | 187.34 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-09-16 09:15:00 | 182.40 | 2025-09-17 14:15:00 | 187.34 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-09-16 10:30:00 | 182.10 | 2025-09-17 14:15:00 | 187.34 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-09-22 13:45:00 | 182.20 | 2025-09-25 14:15:00 | 173.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 13:45:00 | 182.20 | 2025-10-07 12:15:00 | 176.31 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2025-09-24 11:15:00 | 177.00 | 2025-10-07 14:15:00 | 182.70 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2025-09-25 09:30:00 | 176.23 | 2025-10-07 14:15:00 | 182.70 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2025-10-09 09:30:00 | 176.78 | 2025-10-17 13:15:00 | 168.18 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2025-10-09 10:00:00 | 177.03 | 2025-10-20 10:15:00 | 167.94 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2025-10-10 15:15:00 | 176.48 | 2025-10-20 10:15:00 | 167.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-09 09:30:00 | 176.78 | 2025-11-17 09:15:00 | 169.90 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest2 | 2025-10-09 10:00:00 | 177.03 | 2025-11-17 09:15:00 | 169.90 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2025-10-10 15:15:00 | 176.48 | 2025-11-17 09:15:00 | 169.90 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2025-12-29 09:30:00 | 175.86 | 2025-12-31 09:15:00 | 178.25 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-01-01 09:45:00 | 176.56 | 2026-01-01 10:15:00 | 177.36 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-01-05 09:15:00 | 176.45 | 2026-01-05 09:15:00 | 178.37 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-01-29 10:15:00 | 161.39 | 2026-02-01 14:15:00 | 153.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-29 15:15:00 | 161.90 | 2026-02-01 14:15:00 | 153.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 11:45:00 | 161.86 | 2026-02-01 14:15:00 | 153.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-29 10:15:00 | 161.39 | 2026-02-10 09:15:00 | 160.75 | STOP_HIT | 0.50 | 0.40% |
| SELL | retest2 | 2026-01-29 15:15:00 | 161.90 | 2026-02-10 09:15:00 | 160.75 | STOP_HIT | 0.50 | 0.71% |
| SELL | retest2 | 2026-02-01 11:45:00 | 161.86 | 2026-02-10 09:15:00 | 160.75 | STOP_HIT | 0.50 | 0.69% |
| SELL | retest2 | 2026-02-10 09:30:00 | 161.60 | 2026-02-13 09:15:00 | 153.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 09:30:00 | 161.60 | 2026-02-26 12:15:00 | 145.44 | TARGET_HIT | 0.50 | 10.00% |
