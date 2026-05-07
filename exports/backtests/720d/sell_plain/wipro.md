# WIPRO (WIPRO)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 197.39
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 5 |
| PENDING | 18 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 10
- **Target hits / Stop hits / Partials:** 0 / 11 / 0
- **Avg / median % per leg:** -2.43% / -3.03%
- **Sum % (uncompounded):** -26.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 1 | 9.1% | 0 | 11 | 0 | -2.43% | -26.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 11 | 1 | 9.1% | 0 | 11 | 0 | -2.43% | -26.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 1 | 9.1% | 0 | 11 | 0 | -2.43% | -26.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-06 13:15:00 | 286.35 | 299.36 | 299.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 11:15:00 | 283.25 | 298.69 | 299.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 250.81 | 249.03 | 261.31 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 11:15:00 | 254.80 | 249.45 | 255.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 254.80 | 249.45 | 255.28 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-14 11:15:00 | 252.50 | 262.07 | 260.67 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:15:00 | 253.40 | 261.88 | 260.59 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 258.80 | 261.63 | 260.49 | SL hit (close>static) qty=1.00 sl=255.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-28 09:15:00 | 252.60 | 261.20 | 260.57 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:15:00 | 251.15 | 261.00 | 260.47 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-07-29 15:15:00 | 251.35 | 259.90 | 259.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 15:15:00 | 251.35 | 259.90 | 259.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 09:15:00 | 250.20 | 259.81 | 259.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 251.85 | 250.05 | 253.77 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-08-22 09:15:00 | 248.29 | 250.12 | 253.59 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-22 11:15:00 | 249.67 | 250.10 | 253.54 | ENTRY1 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-08-22 14:15:00 | 248.75 | 250.08 | 253.48 | ENTRY1 cross detected — sustain check pending (75m) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 254.69 | 250.11 | 253.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 254.69 | 250.11 | 253.46 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-28 10:15:00 | 250.19 | 250.51 | 253.43 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-28 11:15:00 | 250.66 | 250.51 | 253.42 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-28 15:15:00 | 249.98 | 250.52 | 253.37 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:15:00 | 250.34 | 250.52 | 253.35 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-08-29 11:15:00 | 250.23 | 250.53 | 253.32 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:15:00 | 249.28 | 250.50 | 253.28 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-09-01 13:15:00 | 250.36 | 250.51 | 253.19 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 15:15:00 | 250.50 | 250.51 | 253.17 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-09-03 10:15:00 | 249.70 | 250.57 | 253.08 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 12:15:00 | 249.15 | 250.55 | 253.05 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 255.20 | 249.40 | 252.05 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 257.00 | 249.55 | 252.10 | SL hit (close>static) qty=1.00 sl=256.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 257.00 | 249.55 | 252.10 | SL hit (close>static) qty=1.00 sl=256.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 257.00 | 249.55 | 252.10 | SL hit (close>static) qty=1.00 sl=256.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 257.00 | 249.55 | 252.10 | SL hit (close>static) qty=1.00 sl=256.35 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-23 09:15:00 | 249.15 | 251.30 | 252.44 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-09-23 10:15:00 | 249.74 | 251.28 | 252.42 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-24 09:15:00 | 246.29 | 251.18 | 252.34 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:15:00 | 245.10 | 251.06 | 252.27 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-10-10 13:15:00 | 248.70 | 246.44 | 249.03 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 15:15:00 | 248.98 | 246.49 | 249.03 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-10-15 10:15:00 | 248.88 | 246.56 | 248.87 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-10-15 11:15:00 | 249.60 | 246.59 | 248.87 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-17 09:15:00 | 243.00 | 247.07 | 248.98 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:15:00 | 241.45 | 246.97 | 248.91 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-11-27 11:15:00 | 248.92 | 244.59 | 245.72 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:15:00 | 248.20 | 244.66 | 245.75 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 249.63 | 244.71 | 245.77 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 256.52 | 245.93 | 246.29 | SL hit (close>static) qty=1.00 sl=256.18 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 256.52 | 245.93 | 246.29 | SL hit (close>static) qty=1.00 sl=256.18 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 256.52 | 245.93 | 246.29 | SL hit (close>static) qty=1.00 sl=256.18 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 256.52 | 245.93 | 246.29 | SL hit (close>static) qty=1.00 sl=256.18 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-19 11:15:00 | 248.15 | 261.91 | 257.88 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 13:15:00 | 246.15 | 261.61 | 257.77 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 235.55 | 254.63 | 254.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 15:15:00 | 235.55 | 254.63 | 254.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 09:15:00 | 233.70 | 249.46 | 251.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 09:15:00 | 202.49 | 199.74 | 212.53 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 209.46 | 201.37 | 210.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 209.46 | 201.37 | 210.99 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-16 11:15:00 | 208.93 | 201.53 | 210.98 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-16 12:15:00 | 209.50 | 201.61 | 210.97 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-17 09:15:00 | 204.33 | 201.89 | 210.92 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 11:15:00 | 204.76 | 201.94 | 210.86 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-07-14 13:15:00 | 253.40 | 2025-07-15 10:15:00 | 258.80 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-07-28 11:15:00 | 251.15 | 2025-07-29 15:15:00 | 251.35 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-08-29 09:15:00 | 250.34 | 2025-09-10 11:15:00 | 257.00 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-08-29 13:15:00 | 249.28 | 2025-09-10 11:15:00 | 257.00 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-09-01 15:15:00 | 250.50 | 2025-09-10 11:15:00 | 257.00 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-09-03 12:15:00 | 249.15 | 2025-09-10 11:15:00 | 257.00 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2025-09-24 11:15:00 | 245.10 | 2025-12-03 10:15:00 | 256.52 | STOP_HIT | 1.00 | -4.66% |
| SELL | retest2 | 2025-10-10 15:15:00 | 248.98 | 2025-12-03 10:15:00 | 256.52 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-10-17 11:15:00 | 241.45 | 2025-12-03 10:15:00 | 256.52 | STOP_HIT | 1.00 | -6.24% |
| SELL | retest2 | 2025-11-27 13:15:00 | 248.20 | 2025-12-03 10:15:00 | 256.52 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2026-01-19 13:15:00 | 246.15 | 2026-01-27 15:15:00 | 235.55 | STOP_HIT | 1.00 | 4.31% |
