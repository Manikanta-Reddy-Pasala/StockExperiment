# Castrol India Ltd. (CASTROLIND)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 185.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 3 |
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 24 |
| PARTIAL | 7 |
| TARGET_HIT | 7 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 35 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 20
- **Target hits / Stop hits / Partials:** 7 / 21 / 7
- **Avg / median % per leg:** 1.55% / -0.94%
- **Sum % (uncompounded):** 54.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 1 | 5.9% | 0 | 17 | 0 | -2.62% | -44.6% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.33% | -17.3% |
| BUY @ 3rd Alert (retest2) | 13 | 1 | 7.7% | 0 | 13 | 0 | -2.10% | -27.3% |
| SELL (all) | 18 | 14 | 77.8% | 7 | 4 | 7 | 5.48% | 98.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 14 | 77.8% | 7 | 4 | 7 | 5.48% | 98.7% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.33% | -17.3% |
| retest2 (combined) | 31 | 15 | 48.4% | 7 | 17 | 7 | 2.30% | 71.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 14:15:00 | 193.15 | 198.11 | 198.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 191.00 | 197.99 | 198.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 196.90 | 195.45 | 196.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 196.90 | 195.45 | 196.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 196.90 | 195.45 | 196.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 12:45:00 | 193.65 | 195.43 | 196.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 10:15:00 | 183.97 | 195.16 | 196.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 11:15:00 | 174.28 | 194.94 | 196.35 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 15:15:00 | 202.90 | 196.99 | 196.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 09:15:00 | 205.51 | 197.07 | 197.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 12:15:00 | 201.15 | 201.43 | 199.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-28 12:45:00 | 200.98 | 201.43 | 199.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 251.70 | 259.70 | 251.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 251.70 | 259.70 | 251.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 250.20 | 259.54 | 251.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 250.20 | 259.54 | 251.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 245.90 | 259.40 | 251.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:00:00 | 245.90 | 259.40 | 251.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 251.50 | 258.56 | 251.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 11:15:00 | 251.80 | 258.56 | 251.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 11:45:00 | 252.05 | 258.50 | 251.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 13:15:00 | 249.35 | 258.35 | 251.34 | SL hit (close<static) qty=1.00 sl=249.45 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 09:15:00 | 230.82 | 247.56 | 247.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 09:15:00 | 226.39 | 241.96 | 244.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 10:15:00 | 208.60 | 205.81 | 217.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-02 11:00:00 | 208.60 | 205.81 | 217.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 215.88 | 206.29 | 217.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:00:00 | 215.88 | 206.29 | 217.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 216.22 | 206.85 | 216.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:30:00 | 218.36 | 206.85 | 216.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 216.30 | 206.94 | 216.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 12:00:00 | 215.33 | 210.20 | 217.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 09:15:00 | 204.56 | 210.13 | 216.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-30 09:15:00 | 193.80 | 205.85 | 212.37 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 217.14 | 199.81 | 199.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 09:15:00 | 219.31 | 201.00 | 200.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 15:15:00 | 218.50 | 218.52 | 211.36 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:15:00 | 222.80 | 218.52 | 211.36 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 11:45:00 | 219.44 | 218.55 | 211.48 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 12:15:00 | 219.85 | 218.55 | 211.48 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 12:45:00 | 219.67 | 218.55 | 211.52 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 212.25 | 218.40 | 212.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 212.69 | 218.40 | 212.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 210.89 | 218.32 | 212.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-25 11:15:00 | 210.89 | 218.32 | 212.07 | SL hit (close<ema400) qty=1.00 sl=212.07 alert=retest1 |

### Cycle 5 — SELL (started 2025-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-15 11:15:00 | 201.85 | 208.38 | 208.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 10:15:00 | 201.27 | 207.07 | 207.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 15:15:00 | 204.70 | 204.43 | 205.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-13 09:15:00 | 206.00 | 204.43 | 205.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 205.58 | 204.44 | 205.98 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 218.70 | 207.03 | 206.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 222.25 | 208.66 | 207.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 212.50 | 213.14 | 210.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 212.50 | 213.14 | 210.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 212.50 | 213.14 | 210.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 213.45 | 213.14 | 210.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 14:30:00 | 213.52 | 212.90 | 210.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 09:45:00 | 213.43 | 212.89 | 210.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 15:15:00 | 208.00 | 212.52 | 210.59 | SL hit (close<static) qty=1.00 sl=208.08 alert=retest2 |

### Cycle 7 — SELL (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 11:15:00 | 206.55 | 215.56 | 215.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 205.51 | 213.36 | 214.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 09:15:00 | 206.12 | 204.13 | 208.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 09:15:00 | 206.12 | 204.13 | 208.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 206.12 | 204.13 | 208.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 203.70 | 204.21 | 208.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 11:00:00 | 203.70 | 204.20 | 208.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 12:15:00 | 203.75 | 204.06 | 207.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 12:45:00 | 203.57 | 204.06 | 207.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 204.55 | 201.57 | 204.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:30:00 | 203.79 | 201.59 | 204.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 193.51 | 199.98 | 202.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 193.51 | 199.98 | 202.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 193.56 | 199.98 | 202.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 193.39 | 199.98 | 202.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 193.60 | 199.98 | 202.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-15 09:15:00 | 183.33 | 190.91 | 194.69 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-03 12:45:00 | 193.65 | 2024-06-04 10:15:00 | 183.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 12:45:00 | 193.65 | 2024-06-04 11:15:00 | 174.28 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-07 15:15:00 | 194.35 | 2024-06-10 10:15:00 | 201.08 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2024-09-20 11:15:00 | 251.80 | 2024-09-20 13:15:00 | 249.35 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-09-20 11:45:00 | 252.05 | 2024-09-20 13:15:00 | 249.35 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-09-23 10:30:00 | 252.55 | 2024-09-24 14:15:00 | 249.90 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-09-23 13:15:00 | 251.90 | 2024-09-24 14:15:00 | 249.90 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-09-24 09:15:00 | 253.25 | 2024-09-25 09:15:00 | 248.20 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-09-24 13:00:00 | 252.95 | 2024-09-25 09:15:00 | 248.20 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-12-11 12:00:00 | 215.33 | 2024-12-18 09:15:00 | 204.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-11 12:00:00 | 215.33 | 2024-12-30 09:15:00 | 193.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-21 10:00:00 | 215.12 | 2025-02-25 09:15:00 | 217.14 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-02-21 10:30:00 | 215.31 | 2025-02-25 09:15:00 | 217.14 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-02-21 15:15:00 | 214.88 | 2025-02-25 09:15:00 | 217.14 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest1 | 2025-03-20 09:15:00 | 222.80 | 2025-03-25 11:15:00 | 210.89 | STOP_HIT | 1.00 | -5.35% |
| BUY | retest1 | 2025-03-20 11:45:00 | 219.44 | 2025-03-25 11:15:00 | 210.89 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest1 | 2025-03-20 12:15:00 | 219.85 | 2025-03-25 11:15:00 | 210.89 | STOP_HIT | 1.00 | -4.08% |
| BUY | retest1 | 2025-03-20 12:45:00 | 219.67 | 2025-03-25 11:15:00 | 210.89 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2025-06-13 10:15:00 | 213.45 | 2025-06-18 15:15:00 | 208.00 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-06-16 14:30:00 | 213.52 | 2025-06-18 15:15:00 | 208.00 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-06-17 09:45:00 | 213.43 | 2025-06-18 15:15:00 | 208.00 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-06-26 10:00:00 | 213.91 | 2025-08-06 11:15:00 | 215.50 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2025-08-01 09:15:00 | 220.56 | 2025-08-06 11:15:00 | 215.50 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-08-01 09:45:00 | 221.24 | 2025-08-06 11:15:00 | 215.50 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-08-05 09:15:00 | 223.15 | 2025-08-14 09:15:00 | 205.95 | STOP_HIT | 1.00 | -7.71% |
| SELL | retest2 | 2025-09-18 09:15:00 | 203.70 | 2025-11-03 09:15:00 | 193.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 11:00:00 | 203.70 | 2025-11-03 09:15:00 | 193.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 12:15:00 | 203.75 | 2025-11-03 09:15:00 | 193.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 12:45:00 | 203.57 | 2025-11-03 09:15:00 | 193.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 09:30:00 | 203.79 | 2025-11-03 09:15:00 | 193.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 09:15:00 | 203.70 | 2025-12-15 09:15:00 | 183.33 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-18 11:00:00 | 203.70 | 2025-12-15 09:15:00 | 183.33 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-22 12:15:00 | 203.75 | 2025-12-15 09:15:00 | 183.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-22 12:45:00 | 203.57 | 2025-12-15 09:15:00 | 183.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-16 09:30:00 | 203.79 | 2025-12-15 09:15:00 | 183.41 | TARGET_HIT | 0.50 | 10.00% |
