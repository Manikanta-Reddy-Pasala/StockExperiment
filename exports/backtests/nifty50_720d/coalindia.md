# COALINDIA (COALINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4996 bars)
- **Last close:** 470.20
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 6 |
| ALERT3 | 10 |
| PENDING | 32 |
| PENDING_CANCEL | 9 |
| ENTRY1 | 3 |
| ENTRY2 | 20 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 5 / 19
- **Target hits / Stop hits / Partials:** 0 / 20 / 4
- **Avg / median % per leg:** 2.37% / -0.84%
- **Sum % (uncompounded):** 56.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 3 | 21.4% | 0 | 11 | 3 | 2.31% | 32.4% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.38% | -4.8% |
| BUY @ 3rd Alert (retest2) | 12 | 3 | 25.0% | 0 | 9 | 3 | 3.09% | 37.1% |
| SELL (all) | 10 | 2 | 20.0% | 0 | 9 | 1 | 2.44% | 24.4% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.14% | -1.1% |
| SELL @ 3rd Alert (retest2) | 9 | 2 | 22.2% | 0 | 8 | 1 | 2.84% | 25.5% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.97% | -5.9% |
| retest2 (combined) | 21 | 5 | 23.8% | 0 | 17 | 4 | 2.98% | 62.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 09:15:00 | 489.95 | 500.35 | 500.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 11:15:00 | 488.45 | 500.14 | 500.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 09:15:00 | 501.55 | 499.20 | 499.77 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 501.55 | 499.20 | 499.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 501.55 | 499.20 | 499.77 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-10-15 13:15:00 | 490.80 | 499.01 | 499.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-15 14:15:00 | 494.15 | 498.96 | 499.62 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-17 09:15:00 | 490.45 | 498.65 | 499.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-17 10:15:00 | 494.30 | 498.60 | 499.40 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-17 12:15:00 | 491.80 | 498.48 | 499.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 13:15:00 | 487.85 | 498.38 | 499.28 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-11-12 14:15:00 | 414.67 | 460.81 | 476.19 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-03 10:15:00 | 397.70 | 387.01 | 386.98 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 10:15:00 | 397.70 | 387.01 | 386.98 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 11:15:00 | 369.75 | 386.86 | 386.92 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 12:15:00 | 398.80 | 386.88 | 386.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 13:15:00 | 399.75 | 387.01 | 386.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 11:15:00 | 391.45 | 391.58 | 389.56 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 12:15:00 | 390.45 | 391.57 | 389.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 390.45 | 391.57 | 389.57 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 392.55 | 388.53 | 388.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 392.55 | 388.57 | 388.35 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-03 13:15:00 | 392.40 | 396.92 | 393.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 14:15:00 | 393.05 | 396.88 | 393.91 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-04 11:15:00 | 393.85 | 396.70 | 393.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 12:15:00 | 394.10 | 396.68 | 393.88 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 389.00 | 397.57 | 394.95 | SL hit qty=1.00 sl=389.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 389.00 | 397.57 | 394.95 | SL hit qty=1.00 sl=389.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 389.00 | 397.57 | 394.95 | SL hit qty=1.00 sl=389.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-16 12:15:00 | 393.70 | 396.94 | 394.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 13:15:00 | 393.75 | 396.91 | 394.75 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 394.45 | 396.89 | 394.75 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 389.00 | 396.37 | 394.60 | SL hit qty=1.00 sl=389.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-27 09:15:00 | 397.05 | 394.33 | 393.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:15:00 | 395.95 | 394.34 | 393.81 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-27 13:15:00 | 393.40 | 394.35 | 393.82 | SL hit qty=1.00 sl=393.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-27 14:15:00 | 394.70 | 394.35 | 393.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 15:15:00 | 396.00 | 394.37 | 393.84 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-30 10:15:00 | 393.40 | 394.36 | 393.84 | SL hit qty=1.00 sl=393.40 alert=retest2 |

### Cycle 5 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 11:15:00 | 387.05 | 393.35 | 393.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 10:15:00 | 386.00 | 392.95 | 393.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 388.80 | 388.77 | 390.60 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-07-21 09:15:00 | 386.90 | 388.75 | 390.56 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-21 10:15:00 | 386.00 | 388.72 | 390.54 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 389.85 | 388.59 | 390.40 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-22 12:15:00 | 390.40 | 388.59 | 390.40 | SL hit qty=1.00 sl=390.40 alert=retest1 |
| Cross detected — sustain check pending | 2025-07-24 10:15:00 | 386.60 | 388.73 | 390.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 11:15:00 | 384.85 | 388.69 | 390.34 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-19 09:15:00 | 387.05 | 383.46 | 386.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-19 10:15:00 | 387.50 | 383.50 | 386.09 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-19 13:15:00 | 384.75 | 383.60 | 386.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 14:15:00 | 385.20 | 383.61 | 386.09 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-03 12:15:00 | 391.70 | 381.06 | 383.89 | SL hit qty=1.00 sl=391.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 12:15:00 | 391.70 | 381.06 | 383.89 | SL hit qty=1.00 sl=391.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-08 14:15:00 | 386.95 | 383.02 | 384.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:15:00 | 387.05 | 383.06 | 384.63 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-09 11:15:00 | 387.35 | 383.19 | 384.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-09 12:15:00 | 388.70 | 383.24 | 384.69 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 391.70 | 383.62 | 384.84 | SL hit qty=1.00 sl=391.70 alert=retest2 |

### Cycle 6 — BUY (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 15:15:00 | 394.75 | 385.98 | 385.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 396.50 | 386.08 | 385.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 389.50 | 389.80 | 388.19 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 15:15:00 | 389.10 | 389.78 | 388.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 389.10 | 389.78 | 388.20 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-09-29 09:15:00 | 390.85 | 389.79 | 388.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 10:15:00 | 391.35 | 389.81 | 388.23 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-29 12:15:00 | 388.05 | 389.77 | 388.22 | SL hit qty=1.00 sl=388.05 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-29 13:15:00 | 389.70 | 389.77 | 388.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-29 14:15:00 | 388.00 | 389.76 | 388.23 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-30 09:15:00 | 389.85 | 389.75 | 388.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 10:15:00 | 391.20 | 389.76 | 388.25 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-30 13:15:00 | 388.05 | 389.73 | 388.26 | SL hit qty=1.00 sl=388.05 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-30 14:15:00 | 390.05 | 389.73 | 388.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 15:15:00 | 389.95 | 389.73 | 388.28 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 388.05 | 389.79 | 388.35 | SL hit qty=1.00 sl=388.05 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-01 15:15:00 | 389.80 | 389.79 | 388.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-03 09:15:00 | 383.00 | 389.72 | 388.33 | ENTRY2 sustain failed after 2520m |

### Cycle 7 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 382.70 | 387.25 | 387.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 382.05 | 387.19 | 387.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 12:15:00 | 387.40 | 386.75 | 386.98 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 12:15:00 | 387.40 | 386.75 | 386.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 387.40 | 386.75 | 386.98 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-17 13:15:00 | 385.90 | 386.78 | 386.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-17 14:15:00 | 388.60 | 386.80 | 387.00 | ENTRY2 sustain failed after 60m |

### Cycle 8 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 392.15 | 387.20 | 387.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 393.65 | 387.31 | 387.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | EMA400 retest candle locked |

### Cycle 9 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 374.35 | 387.52 | 387.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 372.90 | 387.38 | 387.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 12:15:00 | 386.80 | 385.72 | 386.56 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 12:15:00 | 386.80 | 385.72 | 386.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 386.80 | 385.72 | 386.56 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-11-13 13:15:00 | 383.85 | 385.73 | 386.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 14:15:00 | 383.40 | 385.71 | 386.51 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 387.00 | 385.67 | 386.48 | SL hit qty=1.00 sl=387.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-18 14:15:00 | 384.10 | 385.83 | 386.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 15:15:00 | 383.95 | 385.81 | 386.48 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-12 10:15:00 | 384.45 | 380.69 | 382.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:15:00 | 383.60 | 380.72 | 382.74 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-17 15:15:00 | 384.40 | 381.25 | 382.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 384.50 | 381.28 | 382.79 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 385.10 | 381.32 | 382.80 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-19 09:15:00 | 384.00 | 381.53 | 382.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 387.00 | 381.53 | 382.87 | SL hit qty=1.00 sl=387.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 387.00 | 381.53 | 382.87 | SL hit qty=1.00 sl=387.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 387.00 | 381.53 | 382.87 | SL hit qty=1.00 sl=387.00 alert=retest2 |
| Sustain check cancelled (price retraced) | 2025-12-19 10:15:00 | 384.70 | 381.56 | 382.88 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-19 12:15:00 | 384.20 | 381.63 | 382.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-19 13:15:00 | 385.10 | 381.66 | 382.90 | ENTRY2 sustain failed after 60m |

### Cycle 10 — BUY (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 12:15:00 | 402.70 | 384.05 | 384.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 405.35 | 384.79 | 384.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 416.35 | 418.98 | 407.37 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-02-02 14:15:00 | 423.85 | 418.91 | 407.63 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 15:15:00 | 422.80 | 418.95 | 407.70 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-11 13:15:00 | 421.45 | 422.94 | 412.30 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 14:15:00 | 422.65 | 422.93 | 412.35 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 412.25 | 422.60 | 412.65 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 412.65 | 422.60 | 412.65 | SL hit qty=1.00 sl=412.65 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 412.65 | 422.60 | 412.65 | SL hit qty=1.00 sl=412.65 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-16 09:15:00 | 419.60 | 421.82 | 412.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 10:15:00 | 420.15 | 421.80 | 412.63 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-18 14:15:00 | 418.20 | 421.42 | 413.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 15:15:00 | 418.20 | 421.39 | 413.25 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-20 09:15:00 | 424.00 | 421.08 | 413.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:15:00 | 420.85 | 421.08 | 413.45 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-29 10:15:00 | 483.17 | 447.69 | 440.28 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-29 10:15:00 | 480.93 | 447.69 | 440.28 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-29 10:15:00 | 483.98 | 447.69 | 440.28 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-10-17 13:15:00 | 487.85 | 2024-11-12 14:15:00 | 414.67 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-10-17 13:15:00 | 487.85 | 2025-04-03 10:15:00 | 397.70 | STOP_HIT | 0.50 | 18.48% |
| BUY | retest2 | 2025-05-12 10:15:00 | 392.55 | 2025-06-13 09:15:00 | 389.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-06-03 14:15:00 | 393.05 | 2025-06-13 09:15:00 | 389.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-06-04 12:15:00 | 394.10 | 2025-06-13 09:15:00 | 389.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-06-16 13:15:00 | 393.75 | 2025-06-18 11:15:00 | 389.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-06-27 10:15:00 | 395.95 | 2025-06-27 13:15:00 | 393.40 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-06-27 15:15:00 | 396.00 | 2025-06-30 10:15:00 | 393.40 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest1 | 2025-07-21 10:15:00 | 386.00 | 2025-07-22 12:15:00 | 390.40 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-07-24 11:15:00 | 384.85 | 2025-09-03 12:15:00 | 391.70 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-08-19 14:15:00 | 385.20 | 2025-09-03 12:15:00 | 391.70 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-09-08 15:15:00 | 387.05 | 2025-09-10 11:15:00 | 391.70 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-09-29 10:15:00 | 391.35 | 2025-09-29 12:15:00 | 388.05 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-09-30 10:15:00 | 391.20 | 2025-09-30 13:15:00 | 388.05 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-30 15:15:00 | 389.95 | 2025-10-01 14:15:00 | 388.05 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-11-13 14:15:00 | 383.40 | 2025-11-14 10:15:00 | 387.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-11-18 15:15:00 | 383.95 | 2025-12-19 09:15:00 | 387.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-12-12 11:15:00 | 383.60 | 2025-12-19 09:15:00 | 387.00 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-12-18 09:15:00 | 384.50 | 2025-12-19 09:15:00 | 387.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest1 | 2026-02-02 15:15:00 | 422.80 | 2026-02-13 09:15:00 | 412.65 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest1 | 2026-02-11 14:15:00 | 422.65 | 2026-02-13 09:15:00 | 412.65 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2026-02-16 10:15:00 | 420.15 | 2026-04-29 10:15:00 | 483.17 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2026-02-18 15:15:00 | 418.20 | 2026-04-29 10:15:00 | 480.93 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2026-02-20 10:15:00 | 420.85 | 2026-04-29 10:15:00 | 483.98 | PARTIAL | 0.50 | 15.00% |
