# WIPRO (WIPRO)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 197.91
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 5 |
| ALERT3 | 11 |
| PENDING | 32 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 10 |
| ENTRY2 | 17 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 23
- **Target hits / Stop hits / Partials:** 0 / 26 / 1
- **Avg / median % per leg:** -2.32% / -3.72%
- **Sum % (uncompounded):** -62.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 2 | 15.4% | 0 | 12 | 1 | -1.54% | -20.1% |
| BUY @ 2nd Alert (retest1) | 7 | 0 | 0.0% | 0 | 7 | 0 | -5.24% | -36.6% |
| BUY @ 3rd Alert (retest2) | 6 | 2 | 33.3% | 0 | 5 | 1 | 2.76% | 16.6% |
| SELL (all) | 14 | 2 | 14.3% | 0 | 14 | 0 | -3.05% | -42.7% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.84% | -11.5% |
| SELL @ 3rd Alert (retest2) | 11 | 2 | 18.2% | 0 | 11 | 0 | -2.83% | -31.2% |
| retest1 (combined) | 10 | 0 | 0.0% | 0 | 10 | 0 | -4.82% | -48.2% |
| retest2 (combined) | 17 | 4 | 23.5% | 0 | 16 | 1 | -0.86% | -14.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 11:15:00 | 196.57 | 206.38 | 206.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 12:15:00 | 196.05 | 206.28 | 206.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 09:15:00 | 196.38 | 196.31 | 199.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 09:15:00 | 199.50 | 196.57 | 199.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 199.50 | 196.57 | 199.80 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2023-11-28 13:15:00 | 197.00 | 197.58 | 199.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-11-28 14:15:00 | 198.45 | 197.59 | 199.78 | ENTRY2 sustain failed after 60m |

### Cycle 2 — BUY (started 2023-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 10:15:00 | 209.98 | 201.36 | 201.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 13:15:00 | 210.65 | 201.62 | 201.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 252.23 | 253.87 | 242.95 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-03-06 15:15:00 | 256.58 | 253.86 | 243.27 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 09:15:00 | 256.60 | 253.88 | 243.33 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Cross detected — sustain check pending | 2024-03-11 11:15:00 | 257.35 | 254.23 | 243.98 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 12:15:00 | 258.42 | 254.27 | 244.05 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-14 14:15:00 | 259.12 | 254.45 | 245.26 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-14 15:15:00 | 258.98 | 254.50 | 245.33 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-15 12:15:00 | 256.65 | 254.53 | 245.52 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-15 13:15:00 | 257.25 | 254.55 | 245.58 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 245.27 | 254.27 | 246.17 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-03-20 09:15:00 | 245.27 | 254.27 | 246.17 | SL hit (close<ema400) qty=1.00 sl=246.17 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-03-20 09:15:00 | 245.27 | 254.27 | 246.17 | SL hit (close<ema400) qty=1.00 sl=246.17 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-03-20 09:15:00 | 245.27 | 254.27 | 246.17 | SL hit (close<ema400) qty=1.00 sl=246.17 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-03-20 09:15:00 | 245.27 | 254.27 | 246.17 | SL hit (close<ema400) qty=1.00 sl=246.17 alert=retest1 |
| Cross detected — sustain check pending | 2024-03-21 09:15:00 | 251.80 | 253.84 | 246.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-21 10:15:00 | 251.80 | 253.82 | 246.26 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-03-22 09:15:00 | 243.25 | 253.54 | 246.34 | SL hit (close<static) qty=1.00 sl=244.82 alert=retest2 |

### Cycle 3 — SELL (started 2024-04-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 12:15:00 | 225.00 | 242.68 | 242.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 13:15:00 | 223.93 | 242.50 | 242.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 09:15:00 | 233.12 | 232.36 | 235.61 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-05-27 09:15:00 | 227.85 | 232.40 | 235.41 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 10:15:00 | 227.18 | 232.35 | 235.37 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 242.05 | 228.63 | 232.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-07 09:15:00 | 242.05 | 228.63 | 232.41 | SL hit (close>ema400) qty=1.00 sl=232.41 alert=retest1 |

### Cycle 4 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 248.38 | 235.04 | 235.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 11:15:00 | 253.18 | 238.18 | 236.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-22 09:15:00 | 255.40 | 261.20 | 251.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 12:15:00 | 251.60 | 260.98 | 251.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 251.60 | 260.98 | 251.55 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-07-26 09:15:00 | 257.92 | 258.91 | 251.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 10:15:00 | 259.33 | 258.92 | 251.59 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 247.20 | 259.18 | 253.13 | SL hit (close<static) qty=1.00 sl=250.77 alert=retest2 |
| Cross detected — sustain check pending | 2024-08-16 11:15:00 | 255.57 | 253.50 | 251.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 12:15:00 | 256.55 | 253.53 | 251.40 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 15:15:00 | 295.03 | 279.41 | 273.70 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-31 09:15:00 | 295.65 | 299.81 | 290.40 | SL hit (close<ema200) qty=0.50 sl=299.81 alert=retest2 |

### Cycle 5 — SELL (started 2025-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-06 15:15:00 | 285.20 | 299.05 | 299.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 11:15:00 | 283.25 | 298.65 | 298.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 250.78 | 249.04 | 261.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 11:15:00 | 254.80 | 249.45 | 255.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 254.80 | 249.45 | 255.27 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 14:15:00 | 265.01 | 258.50 | 258.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 14:15:00 | 266.11 | 258.92 | 258.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 260.55 | 262.53 | 260.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 260.55 | 262.53 | 260.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 260.55 | 262.53 | 260.82 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-07-16 13:15:00 | 262.45 | 261.40 | 260.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:15:00 | 262.75 | 261.42 | 260.43 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-18 09:15:00 | 267.85 | 261.47 | 260.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 267.75 | 261.53 | 260.54 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-21 10:15:00 | 262.35 | 261.78 | 260.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-21 11:15:00 | 261.80 | 261.78 | 260.71 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 257.80 | 261.48 | 260.68 | SL hit (close<static) qty=1.00 sl=258.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 257.80 | 261.48 | 260.68 | SL hit (close<static) qty=1.00 sl=258.50 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 15:15:00 | 251.35 | 259.90 | 259.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 09:15:00 | 250.20 | 259.81 | 259.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 251.85 | 250.05 | 253.77 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-08-22 09:15:00 | 248.27 | 250.12 | 253.59 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 10:15:00 | 248.40 | 250.10 | 253.56 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-22 14:15:00 | 248.77 | 250.07 | 253.48 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 15:15:00 | 248.74 | 250.06 | 253.45 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 254.75 | 250.11 | 253.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 254.75 | 250.11 | 253.46 | SL hit (close>ema400) qty=1.00 sl=253.46 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 254.75 | 250.11 | 253.46 | SL hit (close>ema400) qty=1.00 sl=253.46 alert=retest1 |
| Cross detected — sustain check pending | 2025-08-28 09:15:00 | 250.88 | 250.51 | 253.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 10:15:00 | 250.19 | 250.51 | 253.43 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-28 14:15:00 | 250.73 | 250.52 | 253.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 15:15:00 | 249.98 | 250.52 | 253.36 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-01 11:15:00 | 251.48 | 250.51 | 253.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 12:15:00 | 250.65 | 250.51 | 253.21 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-02 13:15:00 | 250.70 | 250.56 | 253.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 14:15:00 | 251.00 | 250.57 | 253.12 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 255.20 | 249.40 | 252.05 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 257.00 | 249.54 | 252.09 | SL hit (close>static) qty=1.00 sl=256.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 257.00 | 249.54 | 252.09 | SL hit (close>static) qty=1.00 sl=256.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 257.00 | 249.54 | 252.09 | SL hit (close>static) qty=1.00 sl=256.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 257.00 | 249.54 | 252.09 | SL hit (close>static) qty=1.00 sl=256.35 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-23 09:15:00 | 249.12 | 251.29 | 252.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-23 10:15:00 | 249.74 | 251.27 | 252.42 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-24 09:15:00 | 246.29 | 251.17 | 252.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:15:00 | 245.30 | 251.11 | 252.30 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-10 13:15:00 | 248.66 | 246.44 | 249.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 14:15:00 | 248.80 | 246.46 | 249.02 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-14 11:15:00 | 248.36 | 246.43 | 248.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 12:15:00 | 247.80 | 246.44 | 248.86 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-15 10:15:00 | 248.88 | 246.57 | 248.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-15 11:15:00 | 249.60 | 246.60 | 248.87 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-17 09:15:00 | 243.04 | 247.07 | 248.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:15:00 | 242.33 | 247.03 | 248.95 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 244.88 | 243.03 | 245.71 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-11-14 09:15:00 | 242.72 | 243.30 | 245.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:15:00 | 243.86 | 243.30 | 245.67 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-17 09:15:00 | 243.67 | 243.33 | 245.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 10:15:00 | 243.64 | 243.33 | 245.60 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-17 15:15:00 | 243.70 | 243.35 | 245.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 241.91 | 243.34 | 245.54 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-11-19 11:15:00 | 245.93 | 243.28 | 245.41 | SL hit (close>static) qty=1.00 sl=245.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 11:15:00 | 245.93 | 243.28 | 245.41 | SL hit (close>static) qty=1.00 sl=245.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 11:15:00 | 245.93 | 243.28 | 245.41 | SL hit (close>static) qty=1.00 sl=245.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 256.52 | 245.93 | 246.29 | SL hit (close>static) qty=1.00 sl=256.14 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 256.52 | 245.93 | 246.29 | SL hit (close>static) qty=1.00 sl=256.14 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 256.52 | 245.93 | 246.29 | SL hit (close>static) qty=1.00 sl=256.14 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 256.52 | 245.93 | 246.29 | SL hit (close>static) qty=1.00 sl=256.14 alert=retest2 |

### Cycle 8 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 257.33 | 246.72 | 246.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 258.71 | 247.24 | 246.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 10:15:00 | 261.40 | 261.79 | 257.19 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-12 12:15:00 | 263.70 | 261.82 | 257.24 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 13:15:00 | 263.65 | 261.83 | 257.28 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-13 13:15:00 | 262.90 | 261.91 | 257.47 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-13 14:15:00 | 264.25 | 261.94 | 257.51 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 09:15:00 | 268.95 | 261.96 | 257.72 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:15:00 | 267.75 | 262.02 | 257.77 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 250.00 | 262.18 | 257.97 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 250.00 | 262.18 | 257.97 | SL hit (close<ema400) qty=1.00 sl=257.97 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 250.00 | 262.18 | 257.97 | SL hit (close<ema400) qty=1.00 sl=257.97 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 250.00 | 262.18 | 257.97 | SL hit (close<ema400) qty=1.00 sl=257.97 alert=retest1 |

### Cycle 9 — SELL (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 15:15:00 | 235.55 | 254.62 | 254.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 09:15:00 | 233.70 | 250.08 | 252.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 09:15:00 | 202.49 | 199.79 | 212.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 209.45 | 201.40 | 211.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 209.45 | 201.40 | 211.07 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-16 11:15:00 | 208.93 | 201.55 | 211.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-16 12:15:00 | 209.53 | 201.63 | 211.05 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-17 09:15:00 | 204.31 | 201.91 | 211.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 10:15:00 | 204.17 | 201.93 | 210.97 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-03-07 09:15:00 | 256.60 | 2024-03-20 09:15:00 | 245.27 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest1 | 2024-03-11 12:15:00 | 258.42 | 2024-03-20 09:15:00 | 245.27 | STOP_HIT | 1.00 | -5.09% |
| BUY | retest1 | 2024-03-14 15:15:00 | 258.98 | 2024-03-20 09:15:00 | 245.27 | STOP_HIT | 1.00 | -5.29% |
| BUY | retest1 | 2024-03-15 13:15:00 | 257.25 | 2024-03-20 09:15:00 | 245.27 | STOP_HIT | 1.00 | -4.66% |
| BUY | retest2 | 2024-03-21 10:15:00 | 251.80 | 2024-03-22 09:15:00 | 243.25 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest1 | 2024-05-27 10:15:00 | 227.18 | 2024-06-07 09:15:00 | 242.05 | STOP_HIT | 1.00 | -6.55% |
| BUY | retest2 | 2024-07-26 10:15:00 | 259.33 | 2024-08-05 09:15:00 | 247.20 | STOP_HIT | 1.00 | -4.68% |
| BUY | retest2 | 2024-08-16 12:15:00 | 256.55 | 2024-11-26 15:15:00 | 295.03 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-08-16 12:15:00 | 256.55 | 2024-12-31 09:15:00 | 295.65 | STOP_HIT | 0.50 | 15.24% |
| BUY | retest2 | 2025-07-16 14:15:00 | 262.75 | 2025-07-25 09:15:00 | 257.80 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-07-18 10:15:00 | 267.75 | 2025-07-25 09:15:00 | 257.80 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest1 | 2025-08-22 10:15:00 | 248.40 | 2025-08-25 09:15:00 | 254.75 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest1 | 2025-08-22 15:15:00 | 248.74 | 2025-08-25 09:15:00 | 254.75 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-08-28 10:15:00 | 250.19 | 2025-09-10 11:15:00 | 257.00 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2025-08-28 15:15:00 | 249.98 | 2025-09-10 11:15:00 | 257.00 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-09-01 12:15:00 | 250.65 | 2025-09-10 11:15:00 | 257.00 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-09-02 14:15:00 | 251.00 | 2025-09-10 11:15:00 | 257.00 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-09-24 10:15:00 | 245.30 | 2025-11-19 11:15:00 | 245.93 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-10-10 14:15:00 | 248.80 | 2025-11-19 11:15:00 | 245.93 | STOP_HIT | 1.00 | 1.15% |
| SELL | retest2 | 2025-10-14 12:15:00 | 247.80 | 2025-11-19 11:15:00 | 245.93 | STOP_HIT | 1.00 | 0.75% |
| SELL | retest2 | 2025-10-17 10:15:00 | 242.33 | 2025-12-03 10:15:00 | 256.52 | STOP_HIT | 1.00 | -5.86% |
| SELL | retest2 | 2025-11-14 10:15:00 | 243.86 | 2025-12-03 10:15:00 | 256.52 | STOP_HIT | 1.00 | -5.19% |
| SELL | retest2 | 2025-11-17 10:15:00 | 243.64 | 2025-12-03 10:15:00 | 256.52 | STOP_HIT | 1.00 | -5.29% |
| SELL | retest2 | 2025-11-18 09:15:00 | 241.91 | 2025-12-03 10:15:00 | 256.52 | STOP_HIT | 1.00 | -6.04% |
| BUY | retest1 | 2026-01-12 13:15:00 | 263.65 | 2026-01-19 09:15:00 | 250.00 | STOP_HIT | 1.00 | -5.18% |
| BUY | retest1 | 2026-01-13 14:15:00 | 264.25 | 2026-01-19 09:15:00 | 250.00 | STOP_HIT | 1.00 | -5.39% |
| BUY | retest1 | 2026-01-16 10:15:00 | 267.75 | 2026-01-19 09:15:00 | 250.00 | STOP_HIT | 1.00 | -6.63% |
