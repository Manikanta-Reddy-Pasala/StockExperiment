# WIPRO (WIPRO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 199.12
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 7 |
| PENDING | 29 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 9 |
| ENTRY2 | 17 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 5
- **Winners / losers:** 1 / 21
- **Target hits / Stop hits / Partials:** 0 / 21 / 1
- **Avg / median % per leg:** -1.76% / -2.27%
- **Sum % (uncompounded):** -38.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 1 | 6.2% | 0 | 15 | 1 | -1.59% | -25.4% |
| BUY @ 2nd Alert (retest1) | 7 | 0 | 0.0% | 0 | 7 | 0 | -3.75% | -26.2% |
| BUY @ 3rd Alert (retest2) | 9 | 1 | 11.1% | 0 | 8 | 1 | 0.10% | 0.9% |
| SELL (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.23% | -13.4% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.97% | -3.9% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.35% | -9.4% |
| retest1 (combined) | 9 | 0 | 0.0% | 0 | 9 | 0 | -3.35% | -30.2% |
| retest2 (combined) | 13 | 1 | 7.7% | 0 | 12 | 1 | -0.66% | -8.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 10:15:00 | 209.98 | 201.36 | 201.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 13:15:00 | 210.65 | 201.62 | 201.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 252.23 | 253.87 | 242.95 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-03-06 15:15:00 | 256.58 | 253.86 | 243.27 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 09:15:00 | 256.60 | 253.88 | 243.33 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Cross detected — sustain check pending | 2024-03-11 11:15:00 | 257.35 | 254.23 | 243.98 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 12:15:00 | 258.42 | 254.27 | 244.05 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-14 14:15:00 | 259.12 | 254.45 | 245.26 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-14 15:15:00 | 258.98 | 254.50 | 245.33 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-15 12:15:00 | 256.65 | 254.53 | 245.52 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-15 13:15:00 | 257.25 | 254.55 | 245.58 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 245.27 | 254.27 | 246.17 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-03-20 09:15:00 | 246.17 | 254.27 | 246.17 | SL hit qty=1.00 sl=246.17 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-03-20 09:15:00 | 246.17 | 254.27 | 246.17 | SL hit qty=1.00 sl=246.17 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-03-20 09:15:00 | 246.17 | 254.27 | 246.17 | SL hit qty=1.00 sl=246.17 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-03-20 09:15:00 | 246.17 | 254.27 | 246.17 | SL hit qty=1.00 sl=246.17 alert=retest1 |
| Cross detected — sustain check pending | 2024-03-21 09:15:00 | 251.80 | 253.84 | 246.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-21 10:15:00 | 251.80 | 253.82 | 246.26 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-03-22 09:15:00 | 244.82 | 253.54 | 246.34 | SL hit qty=1.00 sl=244.82 alert=retest2 |
| CROSSOVER_SKIP | 2024-04-18 12:15:00 | 225.00 | 242.68 | 242.76 | HTF filter: close above htf_sma |

### Cycle 2 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 248.38 | 235.04 | 235.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 11:15:00 | 253.18 | 238.18 | 236.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-22 09:15:00 | 255.40 | 261.20 | 251.52 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 12:15:00 | 251.60 | 260.98 | 251.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 251.60 | 260.98 | 251.55 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-07-26 09:15:00 | 257.92 | 258.91 | 251.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 10:15:00 | 259.33 | 258.92 | 251.59 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-08-02 14:15:00 | 250.77 | 259.38 | 253.17 | SL hit qty=1.00 sl=250.77 alert=retest2 |
| Cross detected — sustain check pending | 2024-08-16 11:15:00 | 255.57 | 253.50 | 251.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 12:15:00 | 256.55 | 253.53 | 251.40 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-11-26 15:15:00 | 295.03 | 279.41 | 273.70 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2025-03-06 15:15:00 | 285.20 | 299.05 | 299.11 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 256.55 | 287.32 | 292.64 | SL hit qty=0.50 sl=256.55 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-12 12:15:00 | 255.81 | 249.15 | 261.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 13:15:00 | 257.70 | 249.24 | 261.14 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-13 14:15:00 | 250.77 | 249.58 | 260.84 | SL hit qty=1.00 sl=250.77 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-15 13:15:00 | 255.88 | 249.94 | 260.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:15:00 | 256.39 | 250.01 | 260.30 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 256.08 | 250.07 | 260.28 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 250.77 | 250.54 | 259.61 | SL hit qty=1.00 sl=250.77 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-11 12:15:00 | 259.44 | 249.93 | 255.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 13:15:00 | 259.60 | 250.03 | 255.30 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 255.97 | 250.93 | 255.51 | SL hit qty=1.00 sl=255.97 alert=retest2 |
| CROSSOVER_SKIP | 2025-06-27 14:15:00 | 265.01 | 258.50 | 258.48 | HTF filter: close below htf_sma |
| Cross detected — sustain check pending | 2025-07-15 10:15:00 | 258.85 | 261.63 | 260.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 11:15:00 | 258.60 | 261.60 | 260.47 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-25 12:15:00 | 257.40 | 261.35 | 260.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 13:15:00 | 257.95 | 261.31 | 260.61 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 255.97 | 261.20 | 260.56 | SL hit qty=1.00 sl=255.97 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 255.97 | 261.20 | 260.56 | SL hit qty=1.00 sl=255.97 alert=retest2 |

### Cycle 3 — SELL (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 15:15:00 | 251.35 | 259.90 | 259.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 09:15:00 | 250.20 | 259.81 | 259.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 251.85 | 250.05 | 253.77 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-08-22 09:15:00 | 248.27 | 250.12 | 253.59 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 10:15:00 | 248.40 | 250.10 | 253.56 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-22 14:15:00 | 248.77 | 250.07 | 253.48 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 15:15:00 | 248.74 | 250.06 | 253.45 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 254.75 | 250.11 | 253.46 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 253.46 | 250.11 | 253.46 | SL hit qty=1.00 sl=253.46 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 253.46 | 250.11 | 253.46 | SL hit qty=1.00 sl=253.46 alert=retest1 |
| Cross detected — sustain check pending | 2025-08-28 09:15:00 | 250.88 | 250.51 | 253.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 10:15:00 | 250.19 | 250.51 | 253.43 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-28 14:15:00 | 250.73 | 250.52 | 253.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 15:15:00 | 249.98 | 250.52 | 253.36 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-01 11:15:00 | 251.48 | 250.51 | 253.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 12:15:00 | 250.65 | 250.51 | 253.21 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-02 13:15:00 | 250.70 | 250.56 | 253.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 14:15:00 | 251.00 | 250.57 | 253.12 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 255.20 | 249.40 | 252.05 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-09-10 10:15:00 | 256.35 | 249.46 | 252.07 | SL hit qty=1.00 sl=256.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 10:15:00 | 256.35 | 249.46 | 252.07 | SL hit qty=1.00 sl=256.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 10:15:00 | 256.35 | 249.46 | 252.07 | SL hit qty=1.00 sl=256.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 10:15:00 | 256.35 | 249.46 | 252.07 | SL hit qty=1.00 sl=256.35 alert=retest2 |
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
| ALERT3_SKIP | 2025-11-12 10:15:00 | 244.88 | 243.03 | 245.71 | max_alert3_locks_per_cycle=2 reached — end cycle |

### Cycle 4 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 257.33 | 246.72 | 246.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 258.71 | 247.24 | 246.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 10:15:00 | 261.40 | 261.79 | 257.19 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-01-12 12:15:00 | 263.70 | 261.82 | 257.24 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 13:15:00 | 263.65 | 261.83 | 257.28 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-13 13:15:00 | 262.90 | 261.91 | 257.47 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-13 14:15:00 | 264.25 | 261.94 | 257.51 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 09:15:00 | 268.95 | 261.96 | 257.72 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:15:00 | 267.75 | 262.02 | 257.77 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 250.00 | 262.18 | 257.97 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 257.97 | 262.18 | 257.97 | SL hit qty=1.00 sl=257.97 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 257.97 | 262.18 | 257.97 | SL hit qty=1.00 sl=257.97 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 257.97 | 262.18 | 257.97 | SL hit qty=1.00 sl=257.97 alert=retest1 |

### Cycle 5 — SELL (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 15:15:00 | 235.55 | 254.62 | 254.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 09:15:00 | 233.70 | 250.08 | 252.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 09:15:00 | 202.49 | 199.79 | 212.63 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 209.45 | 201.40 | 211.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 209.45 | 201.40 | 211.07 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-16 11:15:00 | 208.93 | 201.55 | 211.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-16 12:15:00 | 209.53 | 201.63 | 211.05 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-17 09:15:00 | 204.31 | 201.91 | 211.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 10:15:00 | 204.17 | 201.93 | 210.97 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-03-07 09:15:00 | 256.60 | 2024-03-20 09:15:00 | 246.17 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest1 | 2024-03-11 12:15:00 | 258.42 | 2024-03-20 09:15:00 | 246.17 | STOP_HIT | 1.00 | -4.74% |
| BUY | retest1 | 2024-03-14 15:15:00 | 258.98 | 2024-03-20 09:15:00 | 246.17 | STOP_HIT | 1.00 | -4.95% |
| BUY | retest1 | 2024-03-15 13:15:00 | 257.25 | 2024-03-20 09:15:00 | 246.17 | STOP_HIT | 1.00 | -4.31% |
| BUY | retest2 | 2024-03-21 10:15:00 | 251.80 | 2024-03-22 09:15:00 | 244.82 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2024-07-26 10:15:00 | 259.33 | 2024-08-02 14:15:00 | 250.77 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2024-08-16 12:15:00 | 256.55 | 2024-11-26 15:15:00 | 295.03 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-08-16 12:15:00 | 256.55 | 2025-03-19 09:15:00 | 256.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2025-05-12 13:15:00 | 257.70 | 2025-05-13 14:15:00 | 250.77 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-05-15 14:15:00 | 256.39 | 2025-05-20 13:15:00 | 250.77 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-06-11 13:15:00 | 259.60 | 2025-06-13 09:15:00 | 255.97 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-07-15 11:15:00 | 258.60 | 2025-07-28 09:15:00 | 255.97 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-07-25 13:15:00 | 257.95 | 2025-07-28 09:15:00 | 255.97 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest1 | 2025-08-22 10:15:00 | 248.40 | 2025-08-25 09:15:00 | 253.46 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest1 | 2025-08-22 15:15:00 | 248.74 | 2025-08-25 09:15:00 | 253.46 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-08-28 10:15:00 | 250.19 | 2025-09-10 10:15:00 | 256.35 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-08-28 15:15:00 | 249.98 | 2025-09-10 10:15:00 | 256.35 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-09-01 12:15:00 | 250.65 | 2025-09-10 10:15:00 | 256.35 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-09-02 14:15:00 | 251.00 | 2025-09-10 10:15:00 | 256.35 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest1 | 2026-01-12 13:15:00 | 263.65 | 2026-01-19 09:15:00 | 257.97 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest1 | 2026-01-13 14:15:00 | 264.25 | 2026-01-19 09:15:00 | 257.97 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest1 | 2026-01-16 10:15:00 | 267.75 | 2026-01-19 09:15:00 | 257.97 | STOP_HIT | 1.00 | -3.65% |
