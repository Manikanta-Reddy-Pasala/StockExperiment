# ONGC (ONGC)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 283.60
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
| ALERT2_SKIP | 1 |
| ALERT3 | 3 |
| PENDING | 11 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 4 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 0 / 5 / 1
- **Avg / median % per leg:** 1.18% / -2.06%
- **Sum % (uncompounded):** 7.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 0 | 5 | 1 | 1.18% | 7.1% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.35% | -10.0% |
| BUY @ 3rd Alert (retest2) | 3 | 2 | 66.7% | 0 | 2 | 1 | 5.71% | 17.1% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.35% | -10.0% |
| retest2 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 5.71% | 17.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 251.84 | 241.87 | 241.84 | EMA200 above EMA400 |

### Cycle 2 — BUY (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 13:15:00 | 247.86 | 242.07 | 242.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 09:15:00 | 251.65 | 242.27 | 242.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 10:15:00 | 244.03 | 246.21 | 244.45 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 10:15:00 | 244.03 | 246.21 | 244.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 244.03 | 246.21 | 244.45 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-07 09:15:00 | 247.58 | 238.34 | 238.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-07 10:15:00 | 246.48 | 238.43 | 238.66 | ENTRY2 sustain failed after 60m |

### Cycle 3 — BUY (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 10:15:00 | 243.74 | 238.90 | 238.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 246.35 | 239.42 | 239.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 10:15:00 | 247.20 | 248.70 | 245.24 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-11-11 15:15:00 | 249.90 | 248.69 | 245.32 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:15:00 | 253.00 | 248.74 | 245.35 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 247.75 | 249.14 | 245.79 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-11-19 13:15:00 | 249.80 | 248.89 | 246.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-19 14:15:00 | 249.05 | 248.89 | 246.07 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-20 10:15:00 | 250.10 | 248.90 | 246.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-20 11:15:00 | 249.40 | 248.91 | 246.13 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 245.90 | 248.74 | 246.22 | SL hit (close<ema400) qty=1.00 sl=246.22 alert=retest1 |
| Cross detected — sustain check pending | 2026-01-14 10:15:00 | 249.82 | 238.52 | 240.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 11:15:00 | 250.16 | 238.63 | 240.33 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-16 13:15:00 | 245.00 | 239.36 | 240.63 | SL hit (close<static) qty=1.00 sl=245.10 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-28 09:15:00 | 263.56 | 241.36 | 241.46 | ENTRY2 cross detected — sustain check pending (15m) |

### Cycle 4 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 263.31 | 241.57 | 241.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 268.08 | 241.84 | 241.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 09:15:00 | 269.30 | 269.70 | 261.39 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-03-11 11:15:00 | 272.00 | 269.71 | 261.76 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 12:15:00 | 271.50 | 269.72 | 261.81 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-12 09:15:00 | 271.05 | 269.77 | 261.99 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:15:00 | 272.60 | 269.80 | 262.04 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 264.90 | 269.46 | 262.36 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 262.20 | 269.39 | 262.36 | SL hit (close<ema400) qty=1.00 sl=262.36 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 262.20 | 269.39 | 262.36 | SL hit (close<ema400) qty=1.00 sl=262.36 alert=retest1 |
| Cross detected — sustain check pending | 2026-03-19 10:15:00 | 269.50 | 268.22 | 262.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:15:00 | 269.40 | 268.23 | 262.47 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-23 09:15:00 | 267.90 | 268.22 | 262.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 10:15:00 | 266.80 | 268.20 | 262.82 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-24 09:15:00 | 270.20 | 268.12 | 262.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 271.30 | 268.15 | 262.98 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-29 10:15:00 | 306.82 | 282.27 | 275.05 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 12:15:00 | 283.70 | 285.33 | 277.74 | SL hit (close<ema200) qty=0.50 sl=285.33 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-11-12 09:15:00 | 253.00 | 2025-11-24 10:15:00 | 245.90 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2026-01-14 11:15:00 | 250.16 | 2026-01-16 13:15:00 | 245.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest1 | 2026-03-11 12:15:00 | 271.50 | 2026-03-16 10:15:00 | 262.20 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest1 | 2026-03-12 10:15:00 | 272.60 | 2026-03-16 10:15:00 | 262.20 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest2 | 2026-03-19 11:15:00 | 269.40 | 2026-04-29 10:15:00 | 306.82 | PARTIAL | 0.50 | 13.89% |
| BUY | retest2 | 2026-03-19 11:15:00 | 269.40 | 2026-05-06 12:15:00 | 283.70 | STOP_HIT | 0.50 | 5.31% |
