# Emmvee Photovoltaic Power Ltd. (EMMVEE)

## Backtest Summary

- **Window:** 2025-11-18 09:15:00 → 2026-05-11 15:15:00 (819 bars)
- **Last close:** 275.55
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 28 |
| ALERT1 | 22 |
| ALERT2 | 22 |
| ALERT2_SKIP | 12 |
| ALERT3 | 53 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 26 |
| PARTIAL | 4 |
| TARGET_HIT | 6 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 16
- **Target hits / Stop hits / Partials:** 6 / 20 / 4
- **Avg / median % per leg:** 1.35% / -0.52%
- **Sum % (uncompounded):** 40.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 6 | 42.9% | 4 | 10 | 0 | 1.17% | 16.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 6 | 42.9% | 4 | 10 | 0 | 1.17% | 16.4% |
| SELL (all) | 16 | 8 | 50.0% | 2 | 10 | 4 | 1.51% | 24.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 8 | 50.0% | 2 | 10 | 4 | 1.51% | 24.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 30 | 14 | 46.7% | 6 | 20 | 4 | 1.35% | 40.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 09:15:00 | 219.48 | 228.42 | 228.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 15:15:00 | 215.00 | 220.15 | 223.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 13:15:00 | 212.54 | 211.78 | 213.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-27 14:00:00 | 212.54 | 211.78 | 213.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 214.08 | 212.56 | 213.80 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 219.35 | 215.30 | 214.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 234.83 | 221.45 | 218.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 234.13 | 234.64 | 228.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 09:45:00 | 235.75 | 234.64 | 228.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 233.88 | 238.11 | 233.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 234.68 | 238.11 | 233.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 234.04 | 237.30 | 233.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 234.47 | 237.30 | 233.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 232.50 | 236.34 | 233.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:45:00 | 232.00 | 236.34 | 233.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 227.33 | 234.54 | 232.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:45:00 | 226.96 | 234.54 | 232.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 14:15:00 | 220.02 | 230.34 | 231.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 15:15:00 | 219.00 | 228.07 | 230.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 205.12 | 203.82 | 209.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:30:00 | 204.47 | 203.82 | 209.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 200.25 | 197.28 | 200.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 200.25 | 197.28 | 200.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 200.00 | 197.83 | 200.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 198.35 | 197.83 | 200.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 196.70 | 197.60 | 199.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:15:00 | 194.80 | 197.31 | 199.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:00:00 | 195.26 | 196.90 | 199.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 09:15:00 | 185.06 | 193.96 | 196.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 09:15:00 | 185.50 | 193.96 | 196.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 190.20 | 189.62 | 192.63 | SL hit (close>ema200) qty=0.50 sl=189.62 alert=retest2 |

### Cycle 4 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 185.75 | 181.21 | 180.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 190.64 | 183.60 | 182.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 14:15:00 | 188.02 | 189.11 | 185.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 15:00:00 | 188.02 | 189.11 | 185.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 185.64 | 188.29 | 186.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 13:00:00 | 188.52 | 186.48 | 185.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 184.00 | 185.85 | 186.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 12:15:00 | 184.00 | 185.85 | 186.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 178.56 | 183.71 | 184.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 182.82 | 180.97 | 182.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 14:15:00 | 182.82 | 180.97 | 182.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 182.82 | 180.97 | 182.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 182.82 | 180.97 | 182.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 182.90 | 181.36 | 182.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 178.01 | 181.36 | 182.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 13:15:00 | 185.00 | 182.03 | 182.47 | SL hit (close>static) qty=1.00 sl=184.95 alert=retest2 |

### Cycle 6 — BUY (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 15:15:00 | 186.58 | 183.45 | 183.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 188.21 | 184.40 | 183.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 11:15:00 | 189.60 | 189.77 | 187.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 11:45:00 | 189.90 | 189.77 | 187.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 182.73 | 187.87 | 187.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:30:00 | 182.06 | 187.87 | 187.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 10:15:00 | 184.30 | 187.16 | 187.23 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 203.40 | 188.73 | 187.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 09:15:00 | 216.45 | 200.97 | 195.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 208.70 | 211.52 | 205.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 11:00:00 | 208.70 | 211.52 | 205.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 202.51 | 208.08 | 205.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 202.51 | 208.08 | 205.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 201.79 | 206.82 | 205.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:30:00 | 204.66 | 206.71 | 205.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-16 09:15:00 | 225.13 | 217.20 | 214.46 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 211.92 | 214.44 | 214.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 207.20 | 210.72 | 212.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 206.15 | 199.79 | 203.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 206.15 | 199.79 | 203.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 206.15 | 199.79 | 203.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 206.15 | 199.79 | 203.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 206.58 | 201.15 | 203.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:45:00 | 209.95 | 201.15 | 203.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 209.46 | 204.93 | 204.86 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 200.92 | 204.40 | 204.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 199.80 | 203.48 | 204.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 196.96 | 196.03 | 198.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 09:15:00 | 195.60 | 196.03 | 198.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 192.41 | 190.77 | 192.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:30:00 | 194.40 | 190.77 | 192.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 190.31 | 190.68 | 192.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 13:15:00 | 190.01 | 190.68 | 192.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 09:15:00 | 193.12 | 191.51 | 192.01 | SL hit (close>static) qty=1.00 sl=192.41 alert=retest2 |

### Cycle 12 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 194.59 | 192.02 | 191.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 209.82 | 195.58 | 193.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 204.38 | 206.59 | 203.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 204.38 | 206.59 | 203.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 204.38 | 206.59 | 203.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 204.40 | 206.59 | 203.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 205.17 | 206.31 | 203.88 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 199.29 | 203.28 | 203.30 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 204.24 | 202.85 | 202.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 205.14 | 203.45 | 203.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 14:15:00 | 205.11 | 205.68 | 204.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 15:00:00 | 205.11 | 205.68 | 204.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 202.22 | 204.99 | 204.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 206.41 | 204.99 | 204.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:15:00 | 206.70 | 205.13 | 204.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:30:00 | 207.32 | 206.06 | 205.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 15:00:00 | 206.47 | 206.83 | 205.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 205.60 | 206.58 | 205.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 197.29 | 206.58 | 205.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 197.57 | 204.78 | 204.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 197.57 | 204.78 | 204.95 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 15:15:00 | 208.55 | 204.38 | 204.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 10:15:00 | 212.67 | 206.54 | 205.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 15:15:00 | 209.00 | 209.23 | 207.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-16 09:15:00 | 207.11 | 209.23 | 207.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 207.65 | 208.92 | 207.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 10:00:00 | 210.27 | 208.05 | 207.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 10:45:00 | 209.90 | 208.28 | 207.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 11:30:00 | 212.80 | 209.06 | 208.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 13:15:00 | 211.69 | 214.88 | 215.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 13:15:00 | 211.69 | 214.88 | 215.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 14:15:00 | 210.88 | 214.08 | 214.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 214.09 | 213.91 | 214.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 214.09 | 213.91 | 214.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 214.09 | 213.91 | 214.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:15:00 | 212.40 | 213.91 | 214.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:15:00 | 207.30 | 211.87 | 212.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:15:00 | 201.78 | 207.37 | 209.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:15:00 | 196.94 | 204.16 | 207.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-26 14:15:00 | 191.16 | 199.32 | 204.28 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 18 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 193.09 | 190.25 | 190.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 15:15:00 | 194.24 | 191.05 | 190.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 09:15:00 | 190.71 | 190.98 | 190.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 09:15:00 | 190.71 | 190.98 | 190.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 190.71 | 190.98 | 190.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 191.14 | 190.98 | 190.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 191.63 | 191.11 | 190.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 191.28 | 191.11 | 190.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 189.17 | 191.68 | 191.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 10:30:00 | 189.96 | 191.33 | 191.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-11 10:15:00 | 208.96 | 204.08 | 199.35 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2026-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 13:15:00 | 200.92 | 202.91 | 203.06 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 11:15:00 | 206.75 | 203.10 | 202.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-16 12:15:00 | 212.59 | 205.00 | 203.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 232.51 | 233.20 | 226.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 229.24 | 231.96 | 227.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 229.24 | 231.96 | 227.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 232.50 | 231.96 | 227.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 229.44 | 231.45 | 227.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 11:15:00 | 232.00 | 231.45 | 227.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 11:15:00 | 227.20 | 230.60 | 227.51 | SL hit (close<static) qty=1.00 sl=227.31 alert=retest2 |

### Cycle 21 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 11:15:00 | 223.00 | 226.10 | 226.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 220.93 | 225.06 | 225.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 221.65 | 220.23 | 222.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 221.65 | 220.23 | 222.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 221.65 | 220.23 | 222.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 222.20 | 220.23 | 222.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 226.29 | 221.44 | 222.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 226.29 | 221.44 | 222.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 226.69 | 222.49 | 223.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:15:00 | 228.00 | 222.49 | 223.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 228.00 | 223.59 | 223.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 232.58 | 223.59 | 223.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 231.08 | 225.09 | 224.30 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 224.37 | 225.58 | 225.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 221.59 | 224.79 | 225.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 10:15:00 | 220.62 | 218.57 | 220.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 10:15:00 | 220.62 | 218.57 | 220.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 220.62 | 218.57 | 220.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 220.62 | 218.57 | 220.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 220.82 | 219.02 | 220.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:15:00 | 221.99 | 219.02 | 220.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 221.90 | 219.59 | 220.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:30:00 | 222.15 | 219.59 | 220.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 220.95 | 219.87 | 220.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:30:00 | 221.23 | 219.87 | 220.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 220.11 | 219.91 | 220.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 15:15:00 | 218.10 | 219.91 | 220.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:30:00 | 218.47 | 216.23 | 217.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 223.51 | 218.70 | 218.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 223.51 | 218.70 | 218.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 13:15:00 | 226.60 | 221.98 | 220.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 10:15:00 | 222.60 | 225.56 | 222.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 10:15:00 | 222.60 | 225.56 | 222.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 222.60 | 225.56 | 222.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 222.60 | 225.56 | 222.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 226.18 | 225.69 | 223.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 235.81 | 224.56 | 223.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 10:00:00 | 231.38 | 225.92 | 224.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 12:15:00 | 254.52 | 244.69 | 238.73 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 261.21 | 268.28 | 268.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 258.10 | 266.25 | 267.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 265.05 | 263.44 | 265.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 265.05 | 263.44 | 265.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 265.05 | 263.44 | 265.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 266.52 | 263.44 | 265.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 265.39 | 263.83 | 265.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 265.39 | 263.83 | 265.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 264.14 | 263.90 | 265.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 15:00:00 | 263.25 | 265.03 | 265.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 268.69 | 265.67 | 265.94 | SL hit (close>static) qty=1.00 sl=266.11 alert=retest2 |

### Cycle 26 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 271.30 | 266.80 | 266.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 15:15:00 | 274.00 | 269.22 | 267.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 269.19 | 283.72 | 278.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 269.19 | 283.72 | 278.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 269.19 | 283.72 | 278.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 269.19 | 283.72 | 278.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 270.50 | 281.08 | 278.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 11:30:00 | 273.05 | 279.60 | 277.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 13:15:00 | 266.26 | 275.86 | 276.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 13:15:00 | 266.26 | 275.86 | 276.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 14:15:00 | 261.76 | 273.04 | 274.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 14:15:00 | 266.90 | 266.18 | 269.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 14:45:00 | 267.60 | 266.18 | 269.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 266.10 | 266.37 | 269.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:30:00 | 267.20 | 266.37 | 269.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 262.10 | 260.51 | 262.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:45:00 | 263.80 | 260.51 | 262.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 264.35 | 261.28 | 262.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:00:00 | 264.35 | 261.28 | 262.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 266.05 | 262.23 | 263.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 12:00:00 | 266.05 | 262.23 | 263.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2026-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 15:15:00 | 264.60 | 263.75 | 263.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 266.65 | 264.33 | 263.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 10:15:00 | 263.80 | 264.23 | 263.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 10:15:00 | 263.80 | 264.23 | 263.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 263.80 | 264.23 | 263.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:15:00 | 261.70 | 264.23 | 263.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 262.80 | 263.94 | 263.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:45:00 | 261.70 | 263.94 | 263.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 263.55 | 263.86 | 263.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:30:00 | 262.70 | 263.86 | 263.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-12-12 11:15:00 | 194.80 | 2025-12-15 09:15:00 | 185.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 12:00:00 | 195.26 | 2025-12-15 09:15:00 | 185.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 11:15:00 | 194.80 | 2025-12-16 09:15:00 | 190.20 | STOP_HIT | 0.50 | 2.36% |
| SELL | retest2 | 2025-12-12 12:00:00 | 195.26 | 2025-12-16 09:15:00 | 190.20 | STOP_HIT | 0.50 | 2.59% |
| BUY | retest2 | 2025-12-24 13:00:00 | 188.52 | 2025-12-26 12:15:00 | 184.00 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-12-30 09:15:00 | 178.01 | 2025-12-30 13:15:00 | 185.00 | STOP_HIT | 1.00 | -3.93% |
| BUY | retest2 | 2026-01-09 09:30:00 | 204.66 | 2026-01-16 09:15:00 | 225.13 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-30 13:15:00 | 190.01 | 2026-02-01 09:15:00 | 193.12 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-02-01 12:15:00 | 186.60 | 2026-02-01 15:15:00 | 194.00 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2026-02-02 09:15:00 | 189.03 | 2026-02-02 14:15:00 | 193.67 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2026-02-02 10:45:00 | 190.02 | 2026-02-02 14:15:00 | 193.67 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2026-02-11 09:15:00 | 206.41 | 2026-02-12 09:15:00 | 197.57 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest2 | 2026-02-11 10:15:00 | 206.70 | 2026-02-12 09:15:00 | 197.57 | STOP_HIT | 1.00 | -4.42% |
| BUY | retest2 | 2026-02-11 11:30:00 | 207.32 | 2026-02-12 09:15:00 | 197.57 | STOP_HIT | 1.00 | -4.70% |
| BUY | retest2 | 2026-02-11 15:00:00 | 206.47 | 2026-02-12 09:15:00 | 197.57 | STOP_HIT | 1.00 | -4.31% |
| BUY | retest2 | 2026-02-17 10:00:00 | 210.27 | 2026-02-20 13:15:00 | 211.69 | STOP_HIT | 1.00 | 0.68% |
| BUY | retest2 | 2026-02-17 10:45:00 | 209.90 | 2026-02-20 13:15:00 | 211.69 | STOP_HIT | 1.00 | 0.85% |
| BUY | retest2 | 2026-02-17 11:30:00 | 212.80 | 2026-02-20 13:15:00 | 211.69 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2026-02-23 10:15:00 | 212.40 | 2026-02-26 09:15:00 | 201.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 09:15:00 | 207.30 | 2026-02-26 11:15:00 | 196.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 10:15:00 | 212.40 | 2026-02-26 14:15:00 | 191.16 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 09:15:00 | 207.30 | 2026-03-02 09:15:00 | 186.57 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-09 10:30:00 | 189.96 | 2026-03-11 10:15:00 | 208.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-20 11:15:00 | 232.00 | 2026-03-20 11:15:00 | 227.20 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-04-01 15:15:00 | 218.10 | 2026-04-06 14:15:00 | 223.51 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2026-04-06 09:30:00 | 218.47 | 2026-04-06 14:15:00 | 223.51 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2026-04-10 09:15:00 | 235.81 | 2026-04-15 12:15:00 | 254.52 | TARGET_HIT | 1.00 | 7.93% |
| BUY | retest2 | 2026-04-10 10:00:00 | 231.38 | 2026-04-15 14:15:00 | 259.39 | TARGET_HIT | 1.00 | 12.11% |
| SELL | retest2 | 2026-04-27 15:00:00 | 263.25 | 2026-04-28 09:15:00 | 268.69 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2026-04-30 11:30:00 | 273.05 | 2026-04-30 13:15:00 | 266.26 | STOP_HIT | 1.00 | -2.49% |
