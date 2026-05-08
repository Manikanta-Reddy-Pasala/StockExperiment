# WIPRO (WIPRO)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 197.88
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 5 |
| PENDING | 20 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 13 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 14
- **Target hits / Stop hits / Partials:** 0 / 15 / 1
- **Avg / median % per leg:** -2.22% / -1.87%
- **Sum % (uncompounded):** -35.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.73% | -17.2% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.73% | -17.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 2 | 15.4% | 0 | 12 | 1 | -1.41% | -18.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 2 | 15.4% | 0 | 12 | 1 | -1.41% | -18.3% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.73% | -17.2% |
| retest2 (combined) | 13 | 2 | 15.4% | 0 | 12 | 1 | -1.41% | -18.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 242.76 | 255.65 | 255.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 241.19 | 255.51 | 255.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 251.85 | 249.99 | 252.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 14:15:00 | 250.71 | 250.03 | 252.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 250.71 | 250.03 | 252.30 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-08-21 09:15:00 | 250.50 | 250.05 | 252.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-21 10:15:00 | 251.06 | 250.06 | 252.27 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-21 14:15:00 | 249.81 | 250.08 | 252.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 15:15:00 | 250.40 | 250.09 | 252.24 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 254.69 | 250.06 | 252.14 | SL hit (close>static) qty=1.00 sl=252.89 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-28 10:15:00 | 250.19 | 250.47 | 252.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-28 11:15:00 | 250.66 | 250.47 | 252.19 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-28 15:15:00 | 249.98 | 250.48 | 252.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:15:00 | 250.34 | 250.48 | 252.16 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-08-29 11:15:00 | 250.23 | 250.49 | 252.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 249.33 | 250.47 | 252.13 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-01 13:15:00 | 250.36 | 250.47 | 252.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 14:15:00 | 250.52 | 250.48 | 252.06 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 252.14 | 250.51 | 252.04 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-09-02 13:15:00 | 250.70 | 250.53 | 252.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 14:15:00 | 251.00 | 250.54 | 252.03 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 255.20 | 249.38 | 251.15 | SL hit (close>static) qty=1.00 sl=252.89 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 255.20 | 249.38 | 251.15 | SL hit (close>static) qty=1.00 sl=252.89 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 255.20 | 249.38 | 251.15 | SL hit (close>static) qty=1.00 sl=252.89 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 255.20 | 249.38 | 251.15 | SL hit (close>static) qty=1.00 sl=253.15 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-15 09:15:00 | 250.50 | 250.14 | 251.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:15:00 | 250.51 | 250.14 | 251.38 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-15 13:15:00 | 250.85 | 250.17 | 251.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 14:15:00 | 251.22 | 250.18 | 251.37 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 254.20 | 250.32 | 251.40 | SL hit (close>static) qty=1.00 sl=253.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 254.20 | 250.32 | 251.40 | SL hit (close>static) qty=1.00 sl=253.15 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-22 09:15:00 | 250.73 | 251.36 | 251.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 10:15:00 | 250.10 | 251.35 | 251.82 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 249.15 | 251.28 | 251.78 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-09-24 09:15:00 | 246.29 | 251.17 | 251.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:15:00 | 245.29 | 251.11 | 251.67 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 237.59 | 250.13 | 251.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-09 14:15:00 | 246.49 | 246.27 | 248.59 | SL hit (close>ema200) qty=0.50 sl=246.27 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-10 13:15:00 | 248.70 | 246.43 | 248.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 14:15:00 | 248.80 | 246.46 | 248.60 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-15 10:15:00 | 248.88 | 246.56 | 248.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-15 11:15:00 | 249.60 | 246.59 | 248.49 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-10-16 11:15:00 | 252.74 | 246.85 | 248.55 | SL hit (close>static) qty=1.00 sl=252.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-16 11:15:00 | 252.74 | 246.85 | 248.55 | SL hit (close>static) qty=1.00 sl=252.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-17 09:15:00 | 243.00 | 247.07 | 248.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:15:00 | 242.33 | 247.02 | 248.59 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-27 11:15:00 | 248.92 | 244.59 | 245.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 12:15:00 | 248.19 | 244.63 | 245.59 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 254.90 | 245.83 | 246.11 | SL hit (close>static) qty=1.00 sl=252.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 254.90 | 245.83 | 246.11 | SL hit (close>static) qty=1.00 sl=252.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 258.36 | 246.51 | 246.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 258.71 | 247.24 | 246.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 10:15:00 | 261.40 | 261.80 | 257.15 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-12 12:15:00 | 263.70 | 261.82 | 257.20 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 13:15:00 | 263.55 | 261.84 | 257.23 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-13 14:15:00 | 264.25 | 261.94 | 257.47 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-13 15:15:00 | 264.40 | 261.97 | 257.50 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 09:15:00 | 268.95 | 261.97 | 257.68 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:15:00 | 267.70 | 262.02 | 257.73 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 250.00 | 262.18 | 257.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 250.00 | 262.18 | 257.94 | SL hit (close<ema400) qty=1.00 sl=257.94 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 250.00 | 262.18 | 257.94 | SL hit (close<ema400) qty=1.00 sl=257.94 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 250.00 | 262.18 | 257.94 | SL hit (close<ema400) qty=1.00 sl=257.94 alert=retest1 |

### Cycle 3 — SELL (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 09:15:00 | 236.25 | 254.45 | 254.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 09:15:00 | 233.70 | 249.46 | 251.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 09:15:00 | 202.49 | 199.74 | 212.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 209.46 | 201.37 | 210.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 209.46 | 201.37 | 210.99 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-16 11:15:00 | 208.93 | 201.53 | 210.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-16 12:15:00 | 209.50 | 201.61 | 210.96 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-17 09:15:00 | 204.33 | 201.89 | 210.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 10:15:00 | 204.17 | 201.91 | 210.89 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-21 15:15:00 | 250.40 | 2025-08-25 09:15:00 | 254.69 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-08-29 09:15:00 | 250.34 | 2025-09-10 09:15:00 | 255.20 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-08-29 12:15:00 | 249.33 | 2025-09-10 09:15:00 | 255.20 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-09-01 14:15:00 | 250.52 | 2025-09-10 09:15:00 | 255.20 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-09-02 14:15:00 | 251.00 | 2025-09-10 09:15:00 | 255.20 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-09-15 10:15:00 | 250.51 | 2025-09-16 14:15:00 | 254.20 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-09-15 14:15:00 | 251.22 | 2025-09-16 14:15:00 | 254.20 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-09-22 10:15:00 | 250.10 | 2025-09-26 09:15:00 | 237.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 10:15:00 | 250.10 | 2025-10-09 14:15:00 | 246.49 | STOP_HIT | 0.50 | 1.44% |
| SELL | retest2 | 2025-09-24 10:15:00 | 245.29 | 2025-10-16 11:15:00 | 252.74 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-10-10 14:15:00 | 248.80 | 2025-10-16 11:15:00 | 252.74 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-10-17 10:15:00 | 242.33 | 2025-12-03 09:15:00 | 254.90 | STOP_HIT | 1.00 | -5.19% |
| SELL | retest2 | 2025-11-27 12:15:00 | 248.19 | 2025-12-03 09:15:00 | 254.90 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest1 | 2026-01-12 13:15:00 | 263.55 | 2026-01-19 09:15:00 | 250.00 | STOP_HIT | 1.00 | -5.14% |
| BUY | retest1 | 2026-01-13 15:15:00 | 264.40 | 2026-01-19 09:15:00 | 250.00 | STOP_HIT | 1.00 | -5.45% |
| BUY | retest1 | 2026-01-16 10:15:00 | 267.70 | 2026-01-19 09:15:00 | 250.00 | STOP_HIT | 1.00 | -6.61% |
