# ONGC (ONGC)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 279.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 7 |
| PENDING | 32 |
| PENDING_CANCEL | 11 |
| ENTRY1 | 8 |
| ENTRY2 | 13 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 7 / 14
- **Target hits / Stop hits / Partials:** 0 / 19 / 2
- **Avg / median % per leg:** 0.26% / -1.60%
- **Sum % (uncompounded):** 5.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 0 | 4 | 1 | 2.16% | 10.8% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.86% | -8.6% |
| BUY @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 0 | 1 | 1 | 9.69% | 19.4% |
| SELL (all) | 16 | 5 | 31.2% | 0 | 15 | 1 | -0.33% | -5.3% |
| SELL @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 5 | 0 | -0.33% | -1.6% |
| SELL @ 3rd Alert (retest2) | 11 | 2 | 18.2% | 0 | 10 | 1 | -0.33% | -3.6% |
| retest1 (combined) | 8 | 3 | 37.5% | 0 | 8 | 0 | -1.28% | -10.2% |
| retest2 (combined) | 13 | 4 | 30.8% | 0 | 11 | 2 | 1.21% | 15.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 09:15:00 | 292.00 | 305.92 | 305.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 284.35 | 300.97 | 303.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 10:15:00 | 263.60 | 261.82 | 273.09 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-12-04 11:15:00 | 261.30 | 261.88 | 272.68 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-04 12:15:00 | 261.60 | 261.88 | 272.63 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-04 14:15:00 | 260.75 | 261.87 | 272.52 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 15:15:00 | 260.70 | 261.86 | 272.46 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-05 15:15:00 | 261.20 | 261.76 | 272.04 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 09:15:00 | 261.20 | 261.75 | 271.99 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 1080m) |
| Cross detected — sustain check pending | 2024-12-06 11:15:00 | 260.85 | 261.75 | 271.88 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-06 12:15:00 | 261.55 | 261.75 | 271.83 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-06 13:15:00 | 260.70 | 261.74 | 271.78 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 14:15:00 | 260.05 | 261.72 | 271.72 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 259.80 | 248.83 | 258.87 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-03 12:15:00 | 259.80 | 248.83 | 258.87 | SL hit (close>ema400) qty=1.00 sl=258.87 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-01-03 12:15:00 | 259.80 | 248.83 | 258.87 | SL hit (close>ema400) qty=1.00 sl=258.87 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-01-03 12:15:00 | 259.80 | 248.83 | 258.87 | SL hit (close>ema400) qty=1.00 sl=258.87 alert=retest1 |
| Cross detected — sustain check pending | 2025-01-06 09:15:00 | 255.60 | 249.20 | 258.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 10:15:00 | 252.25 | 249.23 | 258.82 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-07 09:15:00 | 264.12 | 249.61 | 258.73 | SL hit (close>static) qty=1.00 sl=260.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-13 13:15:00 | 255.80 | 253.65 | 259.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 14:15:00 | 255.98 | 253.67 | 259.57 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 261.80 | 253.82 | 259.55 | SL hit (close>static) qty=1.00 sl=260.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-24 14:15:00 | 255.95 | 258.19 | 260.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-24 15:15:00 | 258.00 | 258.18 | 260.57 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-27 09:15:00 | 254.05 | 258.14 | 260.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 10:15:00 | 251.77 | 258.08 | 260.49 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-30 13:15:00 | 254.71 | 256.78 | 259.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-30 14:15:00 | 257.36 | 256.79 | 259.51 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-01-31 14:15:00 | 263.18 | 256.97 | 259.51 | SL hit (close>static) qty=1.00 sl=260.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-03 09:15:00 | 247.30 | 256.93 | 259.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 10:15:00 | 249.00 | 256.85 | 259.41 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 259.15 | 256.34 | 258.98 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-05 13:15:00 | 261.10 | 256.46 | 258.99 | SL hit (close>static) qty=1.00 sl=260.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-06 11:15:00 | 256.30 | 256.60 | 259.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 12:15:00 | 255.60 | 256.59 | 258.98 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 09:15:00 | 217.26 | 241.19 | 248.43 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-20 10:15:00 | 235.45 | 234.58 | 242.11 | SL hit (close>ema200) qty=0.50 sl=234.58 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 251.84 | 241.88 | 241.85 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 11:15:00 | 236.99 | 242.10 | 242.12 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 14:15:00 | 247.57 | 242.11 | 242.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 09:15:00 | 251.70 | 242.26 | 242.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 10:15:00 | 244.03 | 246.21 | 244.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 10:15:00 | 244.03 | 246.21 | 244.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 244.03 | 246.21 | 244.45 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 15:15:00 | 241.10 | 243.84 | 243.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 09:15:00 | 240.41 | 243.80 | 243.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 12:15:00 | 239.81 | 239.52 | 241.33 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-08-13 13:15:00 | 238.93 | 239.51 | 241.32 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 14:15:00 | 238.66 | 239.50 | 241.31 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 240.43 | 239.03 | 240.80 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-08-22 11:15:00 | 236.75 | 238.99 | 240.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:15:00 | 236.62 | 238.97 | 240.68 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 241.35 | 237.77 | 239.72 | SL hit (close>ema400) qty=1.00 sl=239.72 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 241.35 | 237.77 | 239.72 | SL hit (close>static) qty=1.00 sl=240.95 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-04 09:15:00 | 236.71 | 238.03 | 239.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 10:15:00 | 236.88 | 238.02 | 239.71 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-23 09:15:00 | 236.28 | 236.00 | 237.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 10:15:00 | 236.18 | 236.01 | 237.81 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 241.52 | 236.22 | 237.80 | SL hit (close>static) qty=1.00 sl=240.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 241.52 | 236.22 | 237.80 | SL hit (close>static) qty=1.00 sl=240.95 alert=retest2 |

### Cycle 6 — BUY (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 10:15:00 | 243.74 | 238.90 | 238.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 246.43 | 239.42 | 239.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 10:15:00 | 247.30 | 248.70 | 245.23 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-11-11 14:15:00 | 249.50 | 248.68 | 245.29 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 15:15:00 | 249.45 | 248.69 | 245.31 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 247.75 | 249.12 | 245.78 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-11-19 13:15:00 | 249.85 | 248.88 | 246.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-19 14:15:00 | 249.10 | 248.88 | 246.06 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-20 10:15:00 | 250.15 | 248.90 | 246.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-20 11:15:00 | 249.45 | 248.90 | 246.13 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 245.90 | 248.74 | 246.22 | SL hit (close<ema400) qty=1.00 sl=246.22 alert=retest1 |

### Cycle 7 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 240.15 | 244.87 | 244.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 13:15:00 | 238.71 | 244.65 | 244.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 13:15:00 | 239.61 | 238.70 | 241.06 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-01 12:15:00 | 238.02 | 238.74 | 241.01 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 13:15:00 | 238.00 | 238.73 | 240.99 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 241.52 | 238.76 | 240.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 241.52 | 238.76 | 240.96 | SL hit (close>ema400) qty=1.00 sl=240.96 alert=retest1 |
| Cross detected — sustain check pending | 2026-01-02 11:15:00 | 240.36 | 238.77 | 240.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-02 12:15:00 | 240.54 | 238.79 | 240.95 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-05 09:15:00 | 236.85 | 238.85 | 240.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 10:15:00 | 238.20 | 238.84 | 240.93 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-06 10:15:00 | 239.32 | 238.82 | 240.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-06 11:15:00 | 240.89 | 238.84 | 240.84 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-07 11:15:00 | 240.18 | 239.00 | 240.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:15:00 | 239.93 | 239.01 | 240.85 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-13 13:15:00 | 243.76 | 238.21 | 240.17 | SL hit (close>static) qty=1.00 sl=242.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 13:15:00 | 243.76 | 238.21 | 240.17 | SL hit (close>static) qty=1.00 sl=242.55 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-20 14:15:00 | 240.29 | 239.91 | 240.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-20 15:15:00 | 240.64 | 239.91 | 240.82 | ENTRY2 sustain failed after 60m |

### Cycle 8 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 263.31 | 241.59 | 241.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 268.00 | 241.85 | 241.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 09:15:00 | 269.30 | 269.54 | 261.14 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-10 10:15:00 | 269.70 | 269.54 | 261.18 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-10 11:15:00 | 268.30 | 269.52 | 261.22 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-10 12:15:00 | 270.05 | 269.53 | 261.26 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 13:15:00 | 269.95 | 269.53 | 261.31 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-10 15:15:00 | 270.20 | 269.54 | 261.39 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-11 09:15:00 | 269.05 | 269.53 | 261.43 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-03-11 11:15:00 | 272.00 | 269.55 | 261.52 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 12:15:00 | 271.55 | 269.57 | 261.57 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 264.90 | 269.33 | 262.14 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 261.05 | 269.18 | 262.13 | SL hit (close<ema400) qty=1.00 sl=262.13 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 261.05 | 269.18 | 262.13 | SL hit (close<ema400) qty=1.00 sl=262.13 alert=retest1 |
| Cross detected — sustain check pending | 2026-03-19 10:15:00 | 269.50 | 268.12 | 262.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:15:00 | 269.25 | 268.13 | 262.28 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-23 09:15:00 | 267.95 | 268.13 | 262.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 10:15:00 | 266.90 | 268.12 | 262.64 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-24 09:15:00 | 270.10 | 268.04 | 262.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 271.20 | 268.08 | 262.80 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:15:00 | 306.93 | 282.12 | 274.77 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 12:15:00 | 283.75 | 285.22 | 277.50 | SL hit (close<ema200) qty=0.50 sl=285.22 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-12-04 15:15:00 | 260.70 | 2025-01-03 12:15:00 | 259.80 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest1 | 2024-12-06 09:15:00 | 261.20 | 2025-01-03 12:15:00 | 259.80 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest1 | 2024-12-06 14:15:00 | 260.05 | 2025-01-03 12:15:00 | 259.80 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-01-06 10:15:00 | 252.25 | 2025-01-07 09:15:00 | 264.12 | STOP_HIT | 1.00 | -4.71% |
| SELL | retest2 | 2025-01-13 14:15:00 | 255.98 | 2025-01-14 10:15:00 | 261.80 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-01-27 10:15:00 | 251.77 | 2025-01-31 14:15:00 | 263.18 | STOP_HIT | 1.00 | -4.53% |
| SELL | retest2 | 2025-02-03 10:15:00 | 249.00 | 2025-02-05 13:15:00 | 261.10 | STOP_HIT | 1.00 | -4.86% |
| SELL | retest2 | 2025-02-06 12:15:00 | 255.60 | 2025-03-04 09:15:00 | 217.26 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-02-06 12:15:00 | 255.60 | 2025-03-20 10:15:00 | 235.45 | STOP_HIT | 0.50 | 7.88% |
| SELL | retest1 | 2025-08-13 14:15:00 | 238.66 | 2025-09-02 09:15:00 | 241.35 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-08-22 12:15:00 | 236.62 | 2025-09-02 09:15:00 | 241.35 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-09-04 10:15:00 | 236.88 | 2025-09-25 09:15:00 | 241.52 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-09-23 10:15:00 | 236.18 | 2025-09-25 09:15:00 | 241.52 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest1 | 2025-11-11 15:15:00 | 249.45 | 2025-11-24 10:15:00 | 245.90 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest1 | 2026-01-01 13:15:00 | 238.00 | 2026-01-02 10:15:00 | 241.52 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-01-05 10:15:00 | 238.20 | 2026-01-13 13:15:00 | 243.76 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2026-01-07 12:15:00 | 239.93 | 2026-01-13 13:15:00 | 243.76 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest1 | 2026-03-10 13:15:00 | 269.95 | 2026-03-16 11:15:00 | 261.05 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest1 | 2026-03-11 12:15:00 | 271.55 | 2026-03-16 11:15:00 | 261.05 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2026-03-19 11:15:00 | 269.25 | 2026-04-29 10:15:00 | 306.93 | PARTIAL | 0.50 | 14.00% |
| BUY | retest2 | 2026-03-19 11:15:00 | 269.25 | 2026-05-06 12:15:00 | 283.75 | STOP_HIT | 0.50 | 5.39% |
