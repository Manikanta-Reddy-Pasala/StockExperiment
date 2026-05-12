# Godawari Power & Ispat Ltd. (GPIL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 295.00
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
| ALERT2 | 9 |
| ALERT2_SKIP | 2 |
| ALERT3 | 34 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 44 |
| PARTIAL | 6 |
| TARGET_HIT | 18 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 29
- **Target hits / Stop hits / Partials:** 18 / 31 / 6
- **Avg / median % per leg:** 1.78% / -1.51%
- **Sum % (uncompounded):** 97.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 14 | 41.2% | 14 | 20 | 0 | 2.66% | 90.3% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.14% | -4.1% |
| BUY @ 3rd Alert (retest2) | 33 | 14 | 42.4% | 14 | 19 | 0 | 2.86% | 94.5% |
| SELL (all) | 21 | 12 | 57.1% | 4 | 11 | 6 | 0.36% | 7.5% |
| SELL @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| SELL @ 3rd Alert (retest2) | 13 | 4 | 30.8% | 0 | 11 | 2 | -4.04% | -52.5% |
| retest1 (combined) | 9 | 8 | 88.9% | 4 | 1 | 4 | 6.20% | 55.8% |
| retest2 (combined) | 46 | 18 | 39.1% | 14 | 30 | 2 | 0.91% | 42.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 184.70 | 205.01 | 205.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 12:15:00 | 184.20 | 204.81 | 204.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 10:15:00 | 194.29 | 192.18 | 196.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-23 11:00:00 | 194.29 | 192.18 | 196.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 12:15:00 | 197.40 | 192.26 | 196.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 13:00:00 | 197.40 | 192.26 | 196.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 13:15:00 | 198.48 | 192.33 | 196.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 14:00:00 | 198.48 | 192.33 | 196.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 10:15:00 | 212.20 | 200.28 | 200.23 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 13:15:00 | 188.65 | 200.23 | 200.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 10:15:00 | 186.95 | 197.36 | 198.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 15:15:00 | 190.10 | 188.67 | 193.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 15:15:00 | 190.10 | 188.67 | 193.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 190.10 | 188.67 | 193.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 14:45:00 | 181.29 | 191.04 | 193.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 10:30:00 | 181.49 | 190.77 | 193.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 11:15:00 | 181.47 | 190.77 | 193.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 12:30:00 | 181.22 | 190.60 | 193.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 191.22 | 189.69 | 192.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:45:00 | 191.62 | 189.69 | 192.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 192.54 | 189.58 | 191.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 14:45:00 | 192.40 | 189.58 | 191.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 192.00 | 189.61 | 191.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:15:00 | 198.42 | 189.61 | 191.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-03 10:15:00 | 201.81 | 189.81 | 192.02 | SL hit (close>static) qty=1.00 sl=200.10 alert=retest2 |

### Cycle 4 — BUY (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 09:15:00 | 220.43 | 193.91 | 193.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 09:15:00 | 229.83 | 195.87 | 194.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-24 13:15:00 | 211.05 | 211.60 | 204.74 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 09:15:00 | 213.45 | 211.62 | 204.81 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 207.50 | 211.13 | 205.05 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-30 10:15:00 | 204.62 | 211.06 | 205.05 | SL hit (close<ema400) qty=1.00 sl=205.05 alert=retest1 |

### Cycle 5 — SELL (started 2025-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 13:15:00 | 186.75 | 202.13 | 202.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 14:15:00 | 186.05 | 201.97 | 202.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 15:15:00 | 187.65 | 187.48 | 192.82 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 09:15:00 | 183.80 | 187.48 | 192.82 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 12:00:00 | 186.63 | 187.41 | 192.71 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 14:00:00 | 186.75 | 187.41 | 192.65 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 15:15:00 | 187.00 | 187.41 | 192.63 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 12:15:00 | 177.30 | 186.88 | 192.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 12:15:00 | 177.41 | 186.88 | 192.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 12:15:00 | 177.65 | 186.88 | 192.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 14:15:00 | 174.61 | 186.68 | 191.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-02-13 09:15:00 | 165.42 | 185.66 | 191.15 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 6 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 192.58 | 183.14 | 183.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 194.50 | 183.35 | 183.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 12:15:00 | 188.83 | 189.18 | 186.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-29 13:00:00 | 188.83 | 189.18 | 186.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 185.15 | 189.11 | 186.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 185.40 | 189.11 | 186.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 186.75 | 189.09 | 186.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 13:00:00 | 187.69 | 185.92 | 185.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 14:15:00 | 187.66 | 185.93 | 185.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-15 09:15:00 | 206.46 | 188.05 | 186.57 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 14:15:00 | 178.65 | 189.19 | 189.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 176.20 | 189.06 | 189.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 187.64 | 186.93 | 187.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 14:15:00 | 187.64 | 186.93 | 187.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 187.64 | 186.93 | 187.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 15:00:00 | 187.64 | 186.93 | 187.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 190.00 | 186.97 | 187.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 188.99 | 186.97 | 187.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 189.35 | 186.99 | 187.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 14:00:00 | 187.43 | 187.19 | 188.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 187.96 | 187.24 | 188.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 187.60 | 185.46 | 186.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 195.09 | 185.62 | 186.74 | SL hit (close>static) qty=1.00 sl=191.94 alert=retest2 |

### Cycle 8 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 192.23 | 187.68 | 187.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 15:15:00 | 194.40 | 188.02 | 187.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 15:15:00 | 188.10 | 188.14 | 187.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:15:00 | 189.95 | 188.14 | 187.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 188.71 | 188.16 | 187.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 11:30:00 | 189.52 | 188.20 | 187.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 12:00:00 | 189.76 | 188.20 | 187.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 13:00:00 | 189.66 | 188.21 | 187.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 193.38 | 189.38 | 188.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 190.99 | 189.40 | 188.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 12:30:00 | 194.58 | 189.50 | 188.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 09:15:00 | 186.22 | 190.30 | 189.19 | SL hit (close<static) qty=1.00 sl=187.92 alert=retest2 |

### Cycle 9 — SELL (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 15:15:00 | 230.20 | 246.41 | 246.49 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 14:15:00 | 256.34 | 245.73 | 245.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 09:15:00 | 261.00 | 245.99 | 245.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 10:15:00 | 256.35 | 257.03 | 252.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-12 11:00:00 | 256.35 | 257.03 | 252.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 254.60 | 257.67 | 253.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 254.60 | 257.67 | 253.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 251.20 | 257.60 | 253.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:00:00 | 251.20 | 257.60 | 253.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 250.45 | 257.53 | 253.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 12:00:00 | 250.45 | 257.53 | 253.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 248.25 | 254.03 | 252.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:30:00 | 249.70 | 254.03 | 252.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 249.00 | 253.98 | 252.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 253.15 | 253.70 | 252.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 13:15:00 | 246.98 | 253.70 | 252.09 | SL hit (close<static) qty=1.00 sl=247.05 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-07-25 10:15:00 | 213.03 | 2024-08-08 09:15:00 | 232.43 | TARGET_HIT | 1.00 | 9.11% |
| BUY | retest2 | 2024-07-25 13:15:00 | 211.30 | 2024-08-08 09:15:00 | 232.34 | TARGET_HIT | 1.00 | 9.96% |
| BUY | retest2 | 2024-07-25 13:45:00 | 211.22 | 2024-08-08 09:15:00 | 233.00 | TARGET_HIT | 1.00 | 10.31% |
| BUY | retest2 | 2024-07-25 15:15:00 | 211.82 | 2024-08-13 13:15:00 | 204.16 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2024-11-21 14:45:00 | 181.29 | 2024-12-03 10:15:00 | 201.81 | STOP_HIT | 1.00 | -11.32% |
| SELL | retest2 | 2024-11-22 10:30:00 | 181.49 | 2024-12-03 10:15:00 | 201.81 | STOP_HIT | 1.00 | -11.20% |
| SELL | retest2 | 2024-11-22 11:15:00 | 181.47 | 2024-12-03 10:15:00 | 201.81 | STOP_HIT | 1.00 | -11.21% |
| SELL | retest2 | 2024-11-22 12:30:00 | 181.22 | 2024-12-03 10:15:00 | 201.81 | STOP_HIT | 1.00 | -11.36% |
| BUY | retest1 | 2024-12-26 09:15:00 | 213.45 | 2024-12-30 10:15:00 | 204.62 | STOP_HIT | 1.00 | -4.14% |
| BUY | retest2 | 2025-01-01 13:15:00 | 209.56 | 2025-01-06 09:15:00 | 204.00 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-01-02 12:30:00 | 209.67 | 2025-01-06 09:15:00 | 204.00 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-01-03 09:15:00 | 209.72 | 2025-01-06 09:15:00 | 204.00 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-01-03 11:15:00 | 209.05 | 2025-01-06 09:15:00 | 204.00 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-01-07 11:45:00 | 206.00 | 2025-01-08 11:15:00 | 202.31 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-01-07 12:30:00 | 205.41 | 2025-01-08 11:15:00 | 202.31 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-01-07 13:30:00 | 206.42 | 2025-01-08 11:15:00 | 202.31 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest1 | 2025-02-07 09:15:00 | 183.80 | 2025-02-11 12:15:00 | 177.30 | PARTIAL | 0.50 | 3.54% |
| SELL | retest1 | 2025-02-07 12:00:00 | 186.63 | 2025-02-11 12:15:00 | 177.41 | PARTIAL | 0.50 | 4.94% |
| SELL | retest1 | 2025-02-07 14:00:00 | 186.75 | 2025-02-11 12:15:00 | 177.65 | PARTIAL | 0.50 | 4.87% |
| SELL | retest1 | 2025-02-07 15:15:00 | 187.00 | 2025-02-11 14:15:00 | 174.61 | PARTIAL | 0.50 | 6.63% |
| SELL | retest1 | 2025-02-07 09:15:00 | 183.80 | 2025-02-13 09:15:00 | 165.42 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2025-02-07 12:00:00 | 186.63 | 2025-02-13 09:15:00 | 167.97 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2025-02-07 14:00:00 | 186.75 | 2025-02-13 09:15:00 | 168.08 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2025-02-07 15:15:00 | 187.00 | 2025-02-13 09:15:00 | 168.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-07 15:15:00 | 178.20 | 2025-03-10 09:15:00 | 185.74 | STOP_HIT | 1.00 | -4.23% |
| SELL | retest2 | 2025-03-11 09:15:00 | 176.50 | 2025-03-12 11:15:00 | 168.73 | PARTIAL | 0.50 | 4.40% |
| SELL | retest2 | 2025-03-11 14:00:00 | 177.61 | 2025-03-13 09:15:00 | 167.67 | PARTIAL | 0.50 | 5.59% |
| SELL | retest2 | 2025-03-11 09:15:00 | 176.50 | 2025-03-17 09:15:00 | 176.33 | STOP_HIT | 0.50 | 0.10% |
| SELL | retest2 | 2025-03-11 14:00:00 | 177.61 | 2025-03-17 09:15:00 | 176.33 | STOP_HIT | 0.50 | 0.72% |
| SELL | retest2 | 2025-03-26 14:45:00 | 178.00 | 2025-03-27 14:15:00 | 181.76 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-05-12 13:00:00 | 187.69 | 2025-05-15 09:15:00 | 206.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-12 14:15:00 | 187.66 | 2025-05-15 09:15:00 | 206.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-06 14:45:00 | 188.00 | 2025-06-13 12:15:00 | 184.70 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-06-09 10:30:00 | 187.76 | 2025-06-13 12:15:00 | 184.70 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-06-11 09:15:00 | 187.84 | 2025-06-16 09:15:00 | 182.75 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-06-12 14:15:00 | 186.82 | 2025-06-16 09:15:00 | 182.75 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-06-17 09:30:00 | 186.59 | 2025-06-17 11:15:00 | 184.04 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-06-30 14:00:00 | 187.43 | 2025-07-17 09:15:00 | 195.09 | STOP_HIT | 1.00 | -4.09% |
| SELL | retest2 | 2025-07-01 09:15:00 | 187.96 | 2025-07-17 09:15:00 | 195.09 | STOP_HIT | 1.00 | -3.79% |
| SELL | retest2 | 2025-07-16 09:15:00 | 187.60 | 2025-07-17 09:15:00 | 195.09 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2025-07-29 11:30:00 | 189.52 | 2025-08-07 09:15:00 | 186.22 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-07-29 12:00:00 | 189.76 | 2025-08-07 09:15:00 | 186.22 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-07-29 13:00:00 | 189.66 | 2025-08-07 09:15:00 | 186.22 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-08-04 09:15:00 | 193.38 | 2025-08-07 09:15:00 | 186.22 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2025-08-04 12:30:00 | 194.58 | 2025-08-07 09:15:00 | 186.22 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2025-08-11 13:00:00 | 196.20 | 2025-08-19 13:15:00 | 215.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-11 14:15:00 | 195.06 | 2025-08-19 13:15:00 | 214.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-01 09:15:00 | 253.15 | 2026-02-01 13:15:00 | 246.98 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2026-02-02 15:15:00 | 251.00 | 2026-02-18 09:15:00 | 276.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-03 10:45:00 | 250.87 | 2026-02-18 09:15:00 | 275.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-09 10:45:00 | 251.74 | 2026-02-18 10:15:00 | 276.91 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-12 13:15:00 | 266.65 | 2026-04-08 09:15:00 | 293.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-20 10:45:00 | 267.55 | 2026-04-08 09:15:00 | 294.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-20 12:00:00 | 266.50 | 2026-04-08 09:15:00 | 293.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-25 09:15:00 | 271.60 | 2026-04-15 09:15:00 | 298.76 | TARGET_HIT | 1.00 | 10.00% |
