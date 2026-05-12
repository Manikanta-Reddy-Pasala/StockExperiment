# Godawari Power & Ispat Ltd. (GPIL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 295.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 70 |
| ALERT1 | 52 |
| ALERT2 | 51 |
| ALERT2_SKIP | 24 |
| ALERT3 | 150 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 67 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 42
- **Target hits / Stop hits / Partials:** 4 / 63 / 5
- **Avg / median % per leg:** 0.26% / -0.71%
- **Sum % (uncompounded):** 18.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 9 | 36.0% | 4 | 21 | 0 | 0.96% | 24.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 25 | 9 | 36.0% | 4 | 21 | 0 | 0.96% | 24.0% |
| SELL (all) | 47 | 21 | 44.7% | 0 | 42 | 5 | -0.11% | -5.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 47 | 21 | 44.7% | 0 | 42 | 5 | -0.11% | -5.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 72 | 30 | 41.7% | 4 | 63 | 5 | 0.26% | 18.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 185.44 | 180.09 | 179.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 190.10 | 184.67 | 182.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 200.78 | 202.04 | 197.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 10:00:00 | 200.78 | 202.04 | 197.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 202.08 | 201.84 | 200.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 14:45:00 | 200.90 | 201.84 | 200.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 201.76 | 201.82 | 200.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:15:00 | 203.25 | 201.82 | 200.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 204.22 | 202.30 | 200.76 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 198.53 | 201.16 | 201.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 15:15:00 | 197.90 | 200.05 | 200.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 11:15:00 | 198.00 | 196.03 | 197.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 11:15:00 | 198.00 | 196.03 | 197.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 198.00 | 196.03 | 197.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:45:00 | 198.48 | 196.03 | 197.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 197.88 | 196.40 | 197.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:30:00 | 198.02 | 196.40 | 197.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 198.51 | 196.82 | 197.56 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 198.74 | 197.95 | 197.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 11:15:00 | 204.59 | 199.28 | 198.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 13:15:00 | 201.94 | 202.32 | 201.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 14:00:00 | 201.94 | 202.32 | 201.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 201.13 | 202.09 | 201.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:30:00 | 201.31 | 202.09 | 201.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 200.90 | 201.85 | 201.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 200.27 | 201.85 | 201.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 200.18 | 201.51 | 200.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 200.18 | 201.51 | 200.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 201.10 | 201.43 | 201.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:00:00 | 201.10 | 201.43 | 201.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 200.12 | 201.17 | 200.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:45:00 | 200.17 | 201.17 | 200.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 199.20 | 200.78 | 200.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:30:00 | 199.88 | 200.78 | 200.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 196.75 | 199.97 | 200.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 11:15:00 | 196.62 | 198.06 | 199.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 10:15:00 | 195.17 | 193.46 | 194.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 10:15:00 | 195.17 | 193.46 | 194.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 195.17 | 193.46 | 194.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:30:00 | 195.57 | 193.46 | 194.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 193.81 | 193.53 | 194.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 12:45:00 | 193.23 | 193.61 | 194.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 14:30:00 | 193.37 | 193.72 | 194.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 15:15:00 | 193.45 | 193.72 | 194.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 10:45:00 | 192.68 | 193.58 | 194.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 190.67 | 192.06 | 193.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 14:45:00 | 190.10 | 191.18 | 192.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:30:00 | 189.87 | 190.94 | 191.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 14:15:00 | 183.57 | 188.29 | 189.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 14:15:00 | 183.70 | 188.29 | 189.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 14:15:00 | 183.78 | 188.29 | 189.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 14:15:00 | 183.05 | 188.29 | 189.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-09 11:15:00 | 188.88 | 188.07 | 189.11 | SL hit (close>ema200) qty=0.50 sl=188.07 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 10:15:00 | 195.69 | 189.00 | 188.45 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 186.29 | 188.62 | 188.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 185.72 | 187.59 | 188.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 184.53 | 184.33 | 185.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 184.53 | 184.33 | 185.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 185.35 | 184.53 | 185.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 185.35 | 184.53 | 185.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 185.58 | 184.74 | 185.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 185.30 | 184.74 | 185.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 186.21 | 185.04 | 185.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 186.89 | 185.04 | 185.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 185.24 | 185.08 | 185.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 184.88 | 185.08 | 185.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 184.18 | 184.87 | 185.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 09:15:00 | 175.64 | 178.55 | 180.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 180.11 | 178.86 | 180.34 | SL hit (close>ema200) qty=0.50 sl=178.86 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 181.10 | 179.29 | 179.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 15:15:00 | 182.01 | 180.29 | 179.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 188.52 | 188.73 | 186.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 10:00:00 | 188.52 | 188.73 | 186.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 187.43 | 188.46 | 187.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:00:00 | 187.43 | 188.46 | 187.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 189.08 | 188.59 | 187.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 15:15:00 | 189.90 | 188.59 | 187.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 185.72 | 188.22 | 187.55 | SL hit (close<static) qty=1.00 sl=186.94 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 186.00 | 187.02 | 187.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 14:15:00 | 183.99 | 186.16 | 186.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 184.93 | 184.51 | 185.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 14:15:00 | 184.93 | 184.51 | 185.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 184.93 | 184.51 | 185.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:00:00 | 184.93 | 184.51 | 185.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 184.61 | 184.57 | 185.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 10:45:00 | 183.86 | 184.40 | 185.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:45:00 | 183.64 | 184.40 | 184.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 09:30:00 | 183.85 | 184.20 | 184.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 10:30:00 | 183.66 | 184.02 | 184.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 183.75 | 183.04 | 183.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 184.55 | 183.04 | 183.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 183.77 | 183.19 | 183.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 183.77 | 183.19 | 183.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 183.00 | 183.15 | 183.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 12:45:00 | 182.39 | 183.02 | 183.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 14:30:00 | 182.54 | 183.04 | 183.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:15:00 | 182.70 | 183.04 | 183.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:45:00 | 182.51 | 182.81 | 183.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 181.94 | 180.90 | 181.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:00:00 | 181.94 | 180.90 | 181.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 182.00 | 181.12 | 181.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 15:15:00 | 181.29 | 181.12 | 181.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 12:15:00 | 182.50 | 181.93 | 181.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 12:15:00 | 182.50 | 181.93 | 181.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 13:15:00 | 183.39 | 182.22 | 182.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 10:15:00 | 185.74 | 185.82 | 184.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 11:00:00 | 185.74 | 185.82 | 184.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 184.25 | 185.50 | 184.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:30:00 | 184.39 | 185.50 | 184.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 183.41 | 185.08 | 184.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:00:00 | 183.41 | 185.08 | 184.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 182.99 | 184.67 | 184.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:45:00 | 182.91 | 184.67 | 184.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 15:15:00 | 182.35 | 183.98 | 184.08 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 185.23 | 184.23 | 184.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 186.37 | 184.66 | 184.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 186.90 | 186.92 | 185.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 09:45:00 | 187.03 | 186.92 | 185.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 186.00 | 186.74 | 185.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:30:00 | 186.09 | 186.74 | 185.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 186.68 | 186.73 | 186.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:30:00 | 186.35 | 186.73 | 186.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 186.67 | 186.72 | 186.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:45:00 | 186.30 | 186.72 | 186.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 185.61 | 186.50 | 186.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 15:00:00 | 185.61 | 186.50 | 186.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 186.50 | 186.50 | 186.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 196.73 | 186.50 | 186.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 190.63 | 192.47 | 192.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 190.63 | 192.47 | 192.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 189.15 | 191.18 | 191.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 189.76 | 189.06 | 189.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 11:15:00 | 189.76 | 189.06 | 189.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 189.76 | 189.06 | 189.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:00:00 | 189.76 | 189.06 | 189.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 189.66 | 189.18 | 189.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 189.66 | 189.18 | 189.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 194.10 | 190.17 | 190.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 194.10 | 190.17 | 190.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 194.12 | 190.96 | 190.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 10:15:00 | 195.46 | 192.81 | 191.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 194.18 | 195.24 | 194.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 14:15:00 | 194.18 | 195.24 | 194.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 194.18 | 195.24 | 194.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 194.18 | 195.24 | 194.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 194.00 | 194.99 | 194.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 193.82 | 194.99 | 194.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 192.32 | 194.46 | 194.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 192.32 | 194.46 | 194.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 191.70 | 193.91 | 193.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 191.70 | 193.91 | 193.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 190.05 | 193.13 | 193.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 189.34 | 191.65 | 192.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 190.99 | 190.98 | 192.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 190.99 | 190.98 | 192.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 190.99 | 190.98 | 192.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 193.84 | 190.98 | 192.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 191.03 | 190.99 | 192.02 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 195.76 | 192.84 | 192.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 14:15:00 | 196.18 | 193.50 | 193.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 190.43 | 196.84 | 195.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 190.43 | 196.84 | 195.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 190.43 | 196.84 | 195.75 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 189.68 | 194.09 | 194.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 186.22 | 190.86 | 192.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 13:15:00 | 189.75 | 189.26 | 191.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:00:00 | 189.75 | 189.26 | 191.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 190.86 | 189.58 | 191.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:30:00 | 191.45 | 189.58 | 191.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 190.90 | 189.84 | 191.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 192.58 | 189.84 | 191.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 190.36 | 189.94 | 191.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:00:00 | 189.50 | 189.91 | 190.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:45:00 | 189.17 | 189.62 | 190.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 12:15:00 | 196.20 | 190.74 | 190.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 196.20 | 190.74 | 190.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 11:15:00 | 197.58 | 195.03 | 193.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 15:15:00 | 202.25 | 202.70 | 200.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 199.95 | 202.15 | 200.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 199.95 | 202.15 | 200.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 199.95 | 202.15 | 200.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 199.50 | 201.62 | 200.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:30:00 | 198.81 | 201.62 | 200.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 202.84 | 201.65 | 200.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 14:30:00 | 203.44 | 201.65 | 200.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 239.50 | 240.62 | 237.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 235.01 | 240.62 | 237.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 239.20 | 240.34 | 237.91 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 232.08 | 236.81 | 236.99 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 247.80 | 236.66 | 236.22 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 243.37 | 243.58 | 243.58 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 244.31 | 243.73 | 243.65 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 14:15:00 | 242.12 | 243.40 | 243.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 10:15:00 | 241.58 | 242.99 | 243.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 11:15:00 | 243.16 | 243.03 | 243.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 11:15:00 | 243.16 | 243.03 | 243.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 243.16 | 243.03 | 243.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:00:00 | 243.16 | 243.03 | 243.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 242.39 | 242.90 | 243.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:30:00 | 243.53 | 242.90 | 243.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 239.99 | 240.13 | 241.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:45:00 | 239.10 | 239.99 | 241.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 15:15:00 | 238.77 | 239.97 | 241.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 242.29 | 240.24 | 240.99 | SL hit (close>static) qty=1.00 sl=241.40 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 243.03 | 241.58 | 241.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 13:15:00 | 243.76 | 242.02 | 241.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 240.69 | 242.08 | 241.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 240.69 | 242.08 | 241.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 240.69 | 242.08 | 241.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 241.12 | 242.08 | 241.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 241.53 | 241.97 | 241.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 242.85 | 242.01 | 241.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 13:15:00 | 241.41 | 241.71 | 241.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 241.41 | 241.71 | 241.72 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 244.21 | 242.07 | 241.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 249.27 | 244.77 | 243.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 12:15:00 | 264.90 | 265.32 | 258.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 12:45:00 | 263.93 | 265.32 | 258.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 263.90 | 264.96 | 259.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:00:00 | 263.90 | 264.96 | 259.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 260.05 | 263.92 | 262.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:15:00 | 258.49 | 263.92 | 262.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 259.30 | 262.99 | 261.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:30:00 | 258.14 | 262.99 | 261.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 13:15:00 | 260.56 | 261.21 | 261.26 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 270.75 | 262.47 | 261.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 10:15:00 | 274.32 | 269.94 | 267.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 13:15:00 | 268.34 | 270.48 | 268.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 13:15:00 | 268.34 | 270.48 | 268.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 268.34 | 270.48 | 268.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:45:00 | 267.65 | 270.48 | 268.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 265.45 | 269.48 | 267.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 265.45 | 269.48 | 267.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 266.99 | 268.98 | 267.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 261.60 | 268.98 | 267.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 10:15:00 | 263.51 | 267.20 | 267.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 11:15:00 | 262.04 | 266.17 | 266.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 258.00 | 257.77 | 260.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 258.00 | 257.77 | 260.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 247.70 | 246.14 | 248.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 11:15:00 | 245.26 | 246.92 | 247.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 09:30:00 | 245.27 | 245.78 | 246.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 10:30:00 | 244.89 | 245.59 | 246.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 15:15:00 | 243.80 | 241.53 | 241.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 15:15:00 | 243.80 | 241.53 | 241.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 09:15:00 | 246.32 | 242.49 | 241.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 10:15:00 | 246.05 | 246.23 | 244.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 11:00:00 | 246.05 | 246.23 | 244.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 244.52 | 245.71 | 244.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:45:00 | 243.07 | 245.71 | 244.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 243.10 | 245.19 | 244.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 243.10 | 245.19 | 244.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 242.90 | 244.73 | 244.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:30:00 | 242.75 | 244.73 | 244.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 277.31 | 280.10 | 277.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:45:00 | 276.09 | 280.10 | 277.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 277.70 | 279.62 | 277.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:00:00 | 277.70 | 279.62 | 277.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 279.75 | 279.65 | 277.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 13:45:00 | 281.20 | 280.18 | 278.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 13:30:00 | 281.20 | 280.51 | 279.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 14:00:00 | 281.05 | 280.51 | 279.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 10:15:00 | 282.60 | 280.76 | 279.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 281.50 | 280.91 | 280.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 272.60 | 278.53 | 279.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 272.60 | 278.53 | 279.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 271.10 | 276.03 | 277.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 271.45 | 271.08 | 273.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 11:15:00 | 271.45 | 271.08 | 273.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 271.45 | 271.08 | 273.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:30:00 | 272.95 | 271.08 | 273.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 275.80 | 272.03 | 274.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 275.80 | 272.03 | 274.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 276.70 | 272.96 | 274.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 276.75 | 272.96 | 274.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 274.00 | 273.49 | 274.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 275.05 | 273.49 | 274.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 274.60 | 273.72 | 274.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 277.20 | 273.72 | 274.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 272.50 | 273.47 | 274.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 11:30:00 | 271.75 | 273.12 | 273.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:30:00 | 271.50 | 272.75 | 273.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:15:00 | 271.30 | 272.49 | 273.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 12:45:00 | 271.85 | 272.17 | 272.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 274.10 | 272.56 | 272.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:00:00 | 274.10 | 272.56 | 272.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 275.25 | 273.10 | 273.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:30:00 | 277.30 | 273.10 | 273.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 274.00 | 273.28 | 273.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 274.00 | 273.28 | 273.23 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 09:15:00 | 271.75 | 272.97 | 273.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 12:15:00 | 270.65 | 272.07 | 272.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 11:15:00 | 270.00 | 268.84 | 270.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:00:00 | 270.00 | 268.84 | 270.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 267.45 | 268.56 | 270.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 14:00:00 | 265.35 | 267.92 | 269.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 14:45:00 | 263.00 | 267.03 | 269.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 09:15:00 | 274.50 | 267.97 | 269.20 | SL hit (close>static) qty=1.00 sl=270.50 alert=retest2 |

### Cycle 33 — BUY (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 10:15:00 | 278.40 | 270.06 | 270.04 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 269.75 | 272.61 | 272.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 266.15 | 270.81 | 272.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 240.20 | 238.13 | 240.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 240.20 | 238.13 | 240.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 240.20 | 238.13 | 240.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:30:00 | 234.55 | 237.29 | 239.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:30:00 | 236.24 | 235.71 | 236.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 12:15:00 | 238.09 | 236.86 | 236.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 12:15:00 | 238.09 | 236.86 | 236.76 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 13:15:00 | 236.25 | 236.90 | 236.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 233.08 | 235.94 | 236.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 11:15:00 | 235.57 | 235.30 | 236.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 11:45:00 | 235.29 | 235.30 | 236.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 237.93 | 235.82 | 236.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:00:00 | 237.93 | 235.82 | 236.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 238.40 | 236.34 | 236.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:00:00 | 238.40 | 236.34 | 236.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 14:15:00 | 238.40 | 236.75 | 236.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 09:15:00 | 239.76 | 237.55 | 237.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 11:15:00 | 236.60 | 237.44 | 237.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 11:15:00 | 236.60 | 237.44 | 237.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 236.60 | 237.44 | 237.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:45:00 | 236.94 | 237.44 | 237.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 236.25 | 237.20 | 236.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:45:00 | 235.85 | 237.20 | 236.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 14:15:00 | 235.92 | 236.75 | 236.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 10:15:00 | 232.15 | 235.48 | 236.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 228.56 | 227.46 | 229.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 228.56 | 227.46 | 229.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 230.50 | 228.07 | 229.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 230.50 | 228.07 | 229.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 230.28 | 228.51 | 229.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:15:00 | 229.67 | 228.51 | 229.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 10:15:00 | 233.84 | 230.55 | 230.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 233.84 | 230.55 | 230.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 13:15:00 | 234.05 | 231.67 | 231.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 11:15:00 | 233.63 | 233.68 | 232.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-11 11:45:00 | 233.22 | 233.68 | 232.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 232.34 | 233.41 | 232.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 232.10 | 233.41 | 232.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 232.23 | 233.18 | 232.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 234.62 | 233.24 | 232.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 238.21 | 241.81 | 241.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 238.21 | 241.81 | 241.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 237.68 | 240.41 | 241.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 238.80 | 236.81 | 237.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 238.80 | 236.81 | 237.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 238.80 | 236.81 | 237.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:30:00 | 238.95 | 236.81 | 237.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 239.06 | 237.26 | 237.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:00:00 | 238.36 | 237.48 | 237.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 13:15:00 | 239.27 | 238.12 | 238.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 239.27 | 238.12 | 238.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 241.51 | 238.80 | 238.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 252.96 | 254.32 | 249.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:00:00 | 252.96 | 254.32 | 249.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 251.27 | 253.23 | 250.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:45:00 | 251.06 | 253.23 | 250.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 251.15 | 252.82 | 250.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 251.15 | 252.82 | 250.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 253.51 | 252.95 | 250.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:45:00 | 251.68 | 252.95 | 250.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 252.74 | 253.11 | 251.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 252.25 | 253.11 | 251.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 261.00 | 257.67 | 255.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:45:00 | 265.21 | 258.71 | 256.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 13:15:00 | 261.80 | 259.72 | 256.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 13:15:00 | 271.05 | 273.16 | 273.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 271.05 | 273.16 | 273.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 268.75 | 272.27 | 272.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 256.85 | 256.78 | 261.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 10:15:00 | 261.00 | 256.78 | 261.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 259.75 | 257.37 | 261.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:30:00 | 259.05 | 257.37 | 261.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 261.75 | 258.72 | 260.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 261.75 | 258.72 | 260.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 265.55 | 260.09 | 260.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 265.55 | 260.09 | 260.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 266.30 | 261.71 | 261.27 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2026-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 14:15:00 | 258.10 | 261.49 | 261.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 15:15:00 | 257.55 | 260.70 | 261.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 11:15:00 | 243.05 | 242.81 | 247.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:00:00 | 243.05 | 242.81 | 247.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 250.25 | 244.26 | 246.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 251.00 | 244.26 | 246.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 248.10 | 245.03 | 246.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 247.50 | 245.03 | 246.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:15:00 | 248.00 | 246.14 | 246.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 250.15 | 247.35 | 247.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 250.15 | 247.35 | 247.26 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 245.65 | 247.24 | 247.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 243.30 | 246.45 | 247.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 11:15:00 | 245.45 | 243.50 | 245.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 11:15:00 | 245.45 | 243.50 | 245.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 245.45 | 243.50 | 245.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:00:00 | 245.45 | 243.50 | 245.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 12:15:00 | 245.10 | 243.82 | 245.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:30:00 | 245.15 | 243.82 | 245.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 245.30 | 244.12 | 245.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 14:15:00 | 246.45 | 244.12 | 245.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 247.25 | 244.75 | 245.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 14:30:00 | 247.35 | 244.75 | 245.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 249.40 | 245.68 | 245.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 09:15:00 | 253.70 | 247.28 | 246.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 15:15:00 | 256.00 | 257.29 | 254.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 09:15:00 | 246.20 | 257.29 | 254.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 248.25 | 255.49 | 253.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:30:00 | 249.70 | 255.49 | 253.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 249.00 | 254.19 | 253.46 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 247.95 | 252.94 | 252.96 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 258.40 | 252.67 | 252.44 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 246.98 | 251.99 | 252.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 243.87 | 250.37 | 251.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 248.68 | 245.32 | 247.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 248.68 | 245.32 | 247.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 248.68 | 245.32 | 247.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 248.68 | 245.32 | 247.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 251.00 | 246.46 | 247.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 251.00 | 246.46 | 247.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 252.52 | 248.24 | 248.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 11:00:00 | 252.52 | 248.24 | 248.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 252.87 | 249.16 | 248.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 254.33 | 250.81 | 249.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 255.94 | 257.20 | 254.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 255.94 | 257.20 | 254.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 255.94 | 257.20 | 254.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 255.00 | 257.20 | 254.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 256.76 | 257.11 | 255.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 15:15:00 | 258.00 | 257.11 | 255.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 252.57 | 256.34 | 255.53 | SL hit (close<static) qty=1.00 sl=255.25 alert=retest2 |

### Cycle 52 — SELL (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 13:15:00 | 252.77 | 254.82 | 255.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 14:15:00 | 251.76 | 254.21 | 254.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 252.89 | 252.12 | 253.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 11:00:00 | 252.89 | 252.12 | 253.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 252.72 | 252.24 | 253.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:15:00 | 253.57 | 252.24 | 253.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 255.94 | 252.98 | 253.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:00:00 | 255.94 | 252.98 | 253.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 256.37 | 253.66 | 253.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:45:00 | 256.12 | 253.66 | 253.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 257.48 | 254.42 | 254.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 261.44 | 256.24 | 255.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 258.22 | 259.51 | 257.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 258.22 | 259.51 | 257.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 258.22 | 259.51 | 257.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 257.95 | 259.51 | 257.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 257.20 | 258.81 | 257.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:45:00 | 256.65 | 258.81 | 257.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 257.54 | 258.56 | 257.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:30:00 | 260.45 | 259.61 | 258.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 10:30:00 | 260.38 | 261.96 | 260.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 14:00:00 | 258.84 | 260.73 | 260.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 14:15:00 | 258.20 | 260.22 | 260.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 258.20 | 260.22 | 260.36 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 09:15:00 | 263.09 | 260.57 | 260.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 10:15:00 | 266.79 | 261.81 | 261.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 13:15:00 | 271.00 | 272.97 | 270.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 14:00:00 | 271.00 | 272.97 | 270.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 269.81 | 272.34 | 270.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:30:00 | 268.73 | 272.34 | 270.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 269.90 | 271.85 | 270.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 268.91 | 271.85 | 270.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 266.78 | 270.84 | 269.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 266.78 | 270.84 | 269.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 265.47 | 269.77 | 269.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 265.47 | 269.77 | 269.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 263.61 | 268.53 | 268.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 262.09 | 265.90 | 267.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 12:15:00 | 264.38 | 263.85 | 265.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 12:45:00 | 264.00 | 263.85 | 265.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 265.00 | 263.79 | 265.04 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 269.06 | 266.22 | 265.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 270.65 | 268.05 | 267.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 267.28 | 268.43 | 267.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 12:15:00 | 267.28 | 268.43 | 267.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 267.28 | 268.43 | 267.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:45:00 | 267.50 | 268.43 | 267.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 268.99 | 268.54 | 267.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 14:45:00 | 271.03 | 268.79 | 267.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 14:15:00 | 266.32 | 269.30 | 269.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 266.32 | 269.30 | 269.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 15:15:00 | 265.50 | 268.54 | 269.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 246.40 | 245.78 | 250.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 246.40 | 245.78 | 250.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 253.00 | 247.22 | 250.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 239.90 | 248.85 | 250.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 10:15:00 | 245.95 | 248.34 | 249.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 12:00:00 | 245.85 | 247.24 | 249.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 252.75 | 249.36 | 249.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 252.75 | 249.36 | 249.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 253.90 | 251.37 | 250.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 13:15:00 | 254.45 | 254.55 | 252.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 14:00:00 | 254.45 | 254.55 | 252.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 247.80 | 253.00 | 252.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:15:00 | 250.30 | 253.00 | 252.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 255.60 | 253.52 | 252.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:30:00 | 257.75 | 255.54 | 253.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 250.70 | 255.32 | 255.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 250.70 | 255.32 | 255.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 250.10 | 252.45 | 253.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 11:15:00 | 252.60 | 252.48 | 253.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 12:00:00 | 252.60 | 252.48 | 253.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 252.35 | 250.72 | 252.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 252.35 | 250.72 | 252.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 257.00 | 251.97 | 252.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 257.55 | 251.97 | 252.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 256.85 | 253.49 | 253.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 13:15:00 | 258.35 | 254.46 | 253.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 15:15:00 | 259.55 | 260.02 | 257.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 09:15:00 | 257.15 | 260.02 | 257.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 255.50 | 259.11 | 257.66 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 253.15 | 256.51 | 256.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 251.45 | 255.50 | 256.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 261.50 | 255.39 | 255.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 261.50 | 255.39 | 255.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 261.50 | 255.39 | 255.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 261.50 | 255.39 | 255.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 266.10 | 257.54 | 256.81 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 11:15:00 | 252.10 | 256.98 | 257.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 250.55 | 255.69 | 256.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 255.65 | 254.34 | 255.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 255.65 | 254.34 | 255.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 255.65 | 254.34 | 255.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:45:00 | 256.15 | 254.34 | 255.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 260.95 | 255.66 | 256.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 260.95 | 255.66 | 256.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 263.15 | 257.16 | 256.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 14:15:00 | 263.85 | 258.50 | 257.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 11:15:00 | 273.55 | 273.98 | 269.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 11:30:00 | 274.40 | 273.98 | 269.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 272.45 | 273.39 | 270.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 278.55 | 271.08 | 270.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:45:00 | 274.90 | 274.92 | 274.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 12:15:00 | 274.95 | 275.41 | 275.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:00:00 | 276.20 | 275.57 | 275.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 274.90 | 275.43 | 275.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 13:30:00 | 274.25 | 275.43 | 275.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 275.70 | 275.49 | 275.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:45:00 | 275.50 | 275.49 | 275.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 15:15:00 | 275.10 | 275.41 | 275.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:15:00 | 278.65 | 275.41 | 275.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 281.35 | 276.60 | 275.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:15:00 | 284.55 | 276.60 | 275.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:00:00 | 282.65 | 278.94 | 277.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 10:15:00 | 302.39 | 293.03 | 289.95 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 12:15:00 | 297.55 | 299.94 | 300.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 13:15:00 | 295.50 | 299.06 | 299.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 299.60 | 298.21 | 299.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 299.60 | 298.21 | 299.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 299.60 | 298.21 | 299.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 299.60 | 298.21 | 299.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 299.20 | 298.41 | 299.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:45:00 | 300.15 | 298.41 | 299.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 297.85 | 298.30 | 298.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:30:00 | 296.55 | 297.96 | 298.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 296.75 | 294.73 | 295.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:15:00 | 296.55 | 295.76 | 295.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 13:15:00 | 298.85 | 296.37 | 296.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 298.85 | 296.37 | 296.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 310.70 | 299.74 | 297.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 304.10 | 305.30 | 303.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 14:00:00 | 304.10 | 305.30 | 303.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 302.90 | 304.82 | 303.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:15:00 | 304.25 | 304.82 | 303.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 304.25 | 304.71 | 303.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 299.40 | 304.71 | 303.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 297.15 | 303.20 | 302.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 297.15 | 303.20 | 302.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 293.20 | 301.20 | 301.83 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-05-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 13:15:00 | 304.45 | 300.71 | 300.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 305.80 | 302.42 | 301.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 302.00 | 302.33 | 301.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 302.00 | 302.33 | 301.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 302.00 | 302.33 | 301.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:30:00 | 302.80 | 302.33 | 301.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 302.95 | 302.46 | 301.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:30:00 | 301.40 | 302.46 | 301.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 302.75 | 302.51 | 301.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 12:15:00 | 303.95 | 302.51 | 301.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 12:45:00 | 303.15 | 303.41 | 302.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:30:00 | 304.60 | 303.79 | 303.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 09:15:00 | 297.80 | 302.85 | 303.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 297.80 | 302.85 | 303.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 13:15:00 | 295.70 | 299.49 | 301.33 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-02 12:45:00 | 193.23 | 2025-06-06 14:15:00 | 183.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-02 14:30:00 | 193.37 | 2025-06-06 14:15:00 | 183.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-02 15:15:00 | 193.45 | 2025-06-06 14:15:00 | 183.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-03 10:45:00 | 192.68 | 2025-06-06 14:15:00 | 183.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-02 12:45:00 | 193.23 | 2025-06-09 11:15:00 | 188.88 | STOP_HIT | 0.50 | 2.25% |
| SELL | retest2 | 2025-06-02 14:30:00 | 193.37 | 2025-06-09 11:15:00 | 188.88 | STOP_HIT | 0.50 | 2.32% |
| SELL | retest2 | 2025-06-02 15:15:00 | 193.45 | 2025-06-09 11:15:00 | 188.88 | STOP_HIT | 0.50 | 2.36% |
| SELL | retest2 | 2025-06-03 10:45:00 | 192.68 | 2025-06-09 11:15:00 | 188.88 | STOP_HIT | 0.50 | 1.97% |
| SELL | retest2 | 2025-06-04 14:45:00 | 190.10 | 2025-06-11 10:15:00 | 195.69 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2025-06-05 13:30:00 | 189.87 | 2025-06-11 10:15:00 | 195.69 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-06-17 11:15:00 | 184.88 | 2025-06-20 09:15:00 | 175.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:15:00 | 184.88 | 2025-06-20 10:15:00 | 180.11 | STOP_HIT | 0.50 | 2.58% |
| SELL | retest2 | 2025-06-17 11:45:00 | 184.18 | 2025-06-24 12:15:00 | 181.10 | STOP_HIT | 1.00 | 1.67% |
| BUY | retest2 | 2025-06-30 15:15:00 | 189.90 | 2025-07-01 09:15:00 | 185.72 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-07-03 10:45:00 | 183.86 | 2025-07-10 12:15:00 | 182.50 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2025-07-03 13:45:00 | 183.64 | 2025-07-10 12:15:00 | 182.50 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2025-07-04 09:30:00 | 183.85 | 2025-07-10 12:15:00 | 182.50 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2025-07-04 10:30:00 | 183.66 | 2025-07-10 12:15:00 | 182.50 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2025-07-07 12:45:00 | 182.39 | 2025-07-10 12:15:00 | 182.50 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-07-07 14:30:00 | 182.54 | 2025-07-10 12:15:00 | 182.50 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2025-07-07 15:15:00 | 182.70 | 2025-07-10 12:15:00 | 182.50 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-07-08 09:45:00 | 182.51 | 2025-07-10 12:15:00 | 182.50 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-07-09 15:15:00 | 181.29 | 2025-07-10 12:15:00 | 182.50 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-07-17 09:15:00 | 196.73 | 2025-07-25 10:15:00 | 190.63 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-08-08 14:00:00 | 189.50 | 2025-08-11 12:15:00 | 196.20 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2025-08-08 14:45:00 | 189.17 | 2025-08-11 12:15:00 | 196.20 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2025-09-09 13:45:00 | 239.10 | 2025-09-10 09:15:00 | 242.29 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-09-09 15:15:00 | 238.77 | 2025-09-10 09:15:00 | 242.29 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-09-11 11:30:00 | 242.85 | 2025-09-11 13:15:00 | 241.41 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-10-06 11:15:00 | 245.26 | 2025-10-14 15:15:00 | 243.80 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2025-10-07 09:30:00 | 245.27 | 2025-10-14 15:15:00 | 243.80 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2025-10-07 10:30:00 | 244.89 | 2025-10-14 15:15:00 | 243.80 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2025-10-31 13:45:00 | 281.20 | 2025-11-06 09:15:00 | 272.60 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2025-11-03 13:30:00 | 281.20 | 2025-11-06 09:15:00 | 272.60 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2025-11-03 14:00:00 | 281.05 | 2025-11-06 09:15:00 | 272.60 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2025-11-04 10:15:00 | 282.60 | 2025-11-06 09:15:00 | 272.60 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2025-11-10 11:30:00 | 271.75 | 2025-11-11 15:15:00 | 274.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-11-10 12:30:00 | 271.50 | 2025-11-11 15:15:00 | 274.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-11-10 15:15:00 | 271.30 | 2025-11-11 15:15:00 | 274.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-11-11 12:45:00 | 271.85 | 2025-11-11 15:15:00 | 274.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-11-13 14:00:00 | 265.35 | 2025-11-14 09:15:00 | 274.50 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2025-11-13 14:45:00 | 263.00 | 2025-11-14 09:15:00 | 274.50 | STOP_HIT | 1.00 | -4.37% |
| SELL | retest2 | 2025-11-27 09:30:00 | 234.55 | 2025-12-01 12:15:00 | 238.09 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-12-01 09:30:00 | 236.24 | 2025-12-01 12:15:00 | 238.09 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-12-09 14:15:00 | 229.67 | 2025-12-10 10:15:00 | 233.84 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-12-12 09:15:00 | 234.62 | 2025-12-17 11:15:00 | 238.21 | STOP_HIT | 1.00 | 1.53% |
| SELL | retest2 | 2025-12-22 12:00:00 | 238.36 | 2025-12-22 13:15:00 | 239.27 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-12-30 10:45:00 | 265.21 | 2026-01-08 13:15:00 | 271.05 | STOP_HIT | 1.00 | 2.20% |
| BUY | retest2 | 2025-12-30 13:15:00 | 261.80 | 2026-01-08 13:15:00 | 271.05 | STOP_HIT | 1.00 | 3.53% |
| SELL | retest2 | 2026-01-22 11:15:00 | 247.50 | 2026-01-22 14:15:00 | 250.15 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-01-22 13:15:00 | 248.00 | 2026-01-22 14:15:00 | 250.15 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-02-05 15:15:00 | 258.00 | 2026-02-06 09:15:00 | 252.57 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2026-02-12 09:30:00 | 260.45 | 2026-02-13 14:15:00 | 258.20 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-02-13 10:30:00 | 260.38 | 2026-02-13 14:15:00 | 258.20 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2026-02-13 14:00:00 | 258.84 | 2026-02-13 14:15:00 | 258.20 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2026-02-25 14:45:00 | 271.03 | 2026-02-27 14:15:00 | 266.32 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-03-09 09:15:00 | 239.90 | 2026-03-10 10:15:00 | 252.75 | STOP_HIT | 1.00 | -5.36% |
| SELL | retest2 | 2026-03-09 10:15:00 | 245.95 | 2026-03-10 10:15:00 | 252.75 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2026-03-09 12:00:00 | 245.85 | 2026-03-10 10:15:00 | 252.75 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2026-03-12 11:30:00 | 257.75 | 2026-03-13 12:15:00 | 250.70 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2026-04-01 09:15:00 | 278.55 | 2026-04-15 10:15:00 | 302.39 | TARGET_HIT | 1.00 | 8.56% |
| BUY | retest2 | 2026-04-02 12:45:00 | 274.90 | 2026-04-15 10:15:00 | 302.44 | TARGET_HIT | 1.00 | 10.02% |
| BUY | retest2 | 2026-04-06 12:15:00 | 274.95 | 2026-04-15 15:15:00 | 303.82 | TARGET_HIT | 1.00 | 10.50% |
| BUY | retest2 | 2026-04-06 13:00:00 | 276.20 | 2026-04-16 09:15:00 | 306.41 | TARGET_HIT | 1.00 | 10.94% |
| BUY | retest2 | 2026-04-07 10:15:00 | 284.55 | 2026-04-21 12:15:00 | 297.55 | STOP_HIT | 1.00 | 4.57% |
| BUY | retest2 | 2026-04-07 13:00:00 | 282.65 | 2026-04-21 12:15:00 | 297.55 | STOP_HIT | 1.00 | 5.27% |
| SELL | retest2 | 2026-04-23 10:30:00 | 296.55 | 2026-04-27 13:15:00 | 298.85 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-04-27 11:15:00 | 296.75 | 2026-04-27 13:15:00 | 298.85 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-04-27 13:15:00 | 296.55 | 2026-04-27 13:15:00 | 298.85 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-05-05 12:15:00 | 303.95 | 2026-05-08 09:15:00 | 297.80 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2026-05-06 12:45:00 | 303.15 | 2026-05-08 09:15:00 | 297.80 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-05-06 13:30:00 | 304.60 | 2026-05-08 09:15:00 | 297.80 | STOP_HIT | 1.00 | -2.23% |
