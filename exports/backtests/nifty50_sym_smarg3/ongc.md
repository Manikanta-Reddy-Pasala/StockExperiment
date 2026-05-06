# ONGC (ONGC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 280.80
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 3 |
| PENDING | 16 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 7 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 1 / 8
- **Target hits / Stop hits / Partials:** 0 / 8 / 1
- **Avg / median % per leg:** -0.24% / -1.83%
- **Sum % (uncompounded):** -2.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | 0.86% | 4.3% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.61% | -7.8% |
| BUY @ 3rd Alert (retest2) | 2 | 1 | 50.0% | 0 | 1 | 1 | 6.07% | 12.1% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.62% | -6.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.90% | -0.9% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.86% | -5.6% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.18% | -8.7% |
| retest2 (combined) | 5 | 1 | 20.0% | 0 | 4 | 1 | 1.31% | 6.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 11:15:00 | 236.99 | 242.10 | 242.12 | EMA200 below EMA400 |
| CROSSOVER_SKIP | 2025-06-11 14:15:00 | 247.57 | 242.11 | 242.09 | HTF filter: close below htf_sma |

### Cycle 2 — SELL (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 15:15:00 | 241.10 | 243.84 | 243.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 09:15:00 | 240.41 | 243.80 | 243.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 12:15:00 | 239.81 | 239.52 | 241.33 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-08-13 13:15:00 | 238.93 | 239.51 | 241.32 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 14:15:00 | 238.66 | 239.50 | 241.31 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 240.43 | 239.03 | 240.80 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-08-21 09:15:00 | 240.80 | 239.03 | 240.80 | SL hit qty=1.00 sl=240.80 alert=retest1 |
| Cross detected — sustain check pending | 2025-08-22 11:15:00 | 236.75 | 238.99 | 240.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:15:00 | 236.62 | 238.97 | 240.68 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 240.95 | 237.77 | 239.72 | SL hit qty=1.00 sl=240.95 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-04 09:15:00 | 236.71 | 238.03 | 239.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 10:15:00 | 236.88 | 238.02 | 239.71 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-23 09:15:00 | 236.28 | 236.00 | 237.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 10:15:00 | 236.18 | 236.01 | 237.81 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 240.95 | 236.22 | 237.80 | SL hit qty=1.00 sl=240.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 240.95 | 236.22 | 237.80 | SL hit qty=1.00 sl=240.95 alert=retest2 |

### Cycle 3 — BUY (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 10:15:00 | 243.74 | 238.90 | 238.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 246.43 | 239.42 | 239.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 10:15:00 | 247.30 | 248.70 | 245.23 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-11-11 14:15:00 | 249.50 | 248.68 | 245.29 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 15:15:00 | 249.45 | 248.69 | 245.31 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 247.75 | 249.12 | 245.78 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-11-14 09:15:00 | 245.78 | 249.12 | 245.78 | SL hit qty=1.00 sl=245.78 alert=retest1 |
| Cross detected — sustain check pending | 2025-11-19 13:15:00 | 249.85 | 248.88 | 246.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-19 14:15:00 | 249.10 | 248.88 | 246.06 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-20 10:15:00 | 250.15 | 248.90 | 246.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-20 11:15:00 | 249.45 | 248.90 | 246.13 | ENTRY2 sustain failed after 60m |
| CROSSOVER_SKIP | 2025-12-11 09:15:00 | 240.15 | 244.87 | 244.88 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2026-01-14 10:15:00 | 249.82 | 238.53 | 240.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 11:15:00 | 250.16 | 238.65 | 240.34 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 245.50 | 239.12 | 240.54 | SL hit qty=1.00 sl=245.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-28 09:15:00 | 263.56 | 241.37 | 241.47 | ENTRY2 cross detected — sustain check pending (15m) |

### Cycle 4 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 263.31 | 241.59 | 241.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 268.00 | 241.85 | 241.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 09:15:00 | 269.30 | 269.54 | 261.14 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-03-10 10:15:00 | 269.70 | 269.54 | 261.18 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-10 11:15:00 | 268.30 | 269.52 | 261.22 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-10 12:15:00 | 270.05 | 269.53 | 261.26 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 13:15:00 | 269.95 | 269.53 | 261.31 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-10 15:15:00 | 270.20 | 269.54 | 261.39 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-11 09:15:00 | 269.05 | 269.53 | 261.43 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-03-11 11:15:00 | 272.00 | 269.55 | 261.52 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 12:15:00 | 271.55 | 269.57 | 261.57 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 264.90 | 269.33 | 262.14 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-03-16 09:15:00 | 262.14 | 269.33 | 262.14 | SL hit qty=1.00 sl=262.14 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-16 09:15:00 | 262.14 | 269.33 | 262.14 | SL hit qty=1.00 sl=262.14 alert=retest1 |
| Cross detected — sustain check pending | 2026-03-19 10:15:00 | 269.50 | 268.12 | 262.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:15:00 | 269.25 | 268.13 | 262.28 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-23 09:15:00 | 267.95 | 268.13 | 262.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 10:15:00 | 266.90 | 268.12 | 262.64 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-24 09:15:00 | 270.10 | 268.04 | 262.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 271.20 | 268.08 | 262.80 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-29 10:15:00 | 306.93 | 282.12 | 274.77 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-08-13 14:15:00 | 238.66 | 2025-08-21 09:15:00 | 240.80 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-08-22 12:15:00 | 236.62 | 2025-09-02 09:15:00 | 240.95 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-09-04 10:15:00 | 236.88 | 2025-09-25 09:15:00 | 240.95 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-09-23 10:15:00 | 236.18 | 2025-09-25 09:15:00 | 240.95 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest1 | 2025-11-11 15:15:00 | 249.45 | 2025-11-14 09:15:00 | 245.78 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-01-14 11:15:00 | 250.16 | 2026-01-16 09:15:00 | 245.50 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest1 | 2026-03-10 13:15:00 | 269.95 | 2026-03-16 09:15:00 | 262.14 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest1 | 2026-03-11 12:15:00 | 271.55 | 2026-03-16 09:15:00 | 262.14 | STOP_HIT | 1.00 | -3.47% |
| BUY | retest2 | 2026-03-19 11:15:00 | 269.25 | 2026-04-29 10:15:00 | 306.93 | PARTIAL | 0.50 | 14.00% |
