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
| ALERT3 | 4 |
| PENDING | 22 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 4 |
| ENTRY2 | 11 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 12
- **Target hits / Stop hits / Partials:** 0 / 15 / 1
- **Avg / median % per leg:** -0.63% / -2.01%
- **Sum % (uncompounded):** -10.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 4 | 25.0% | 0 | 15 | 1 | -0.63% | -10.1% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 4 | 0 | -0.23% | -0.9% |
| SELL @ 3rd Alert (retest2) | 12 | 2 | 16.7% | 0 | 11 | 1 | -0.76% | -9.2% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 4 | 0 | -0.23% | -0.9% |
| retest2 (combined) | 12 | 2 | 16.7% | 0 | 11 | 1 | -0.76% | -9.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 10:15:00 | 287.05 | 307.07 | 307.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 284.30 | 300.97 | 303.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 10:15:00 | 263.60 | 261.78 | 273.11 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-12-04 11:15:00 | 261.20 | 261.84 | 272.70 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-04 12:15:00 | 261.60 | 261.84 | 272.64 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-04 14:15:00 | 260.80 | 261.83 | 272.53 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 09:15:00 | 258.60 | 261.78 | 272.40 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 1140m) |
| Cross detected — sustain check pending | 2024-12-05 15:15:00 | 261.25 | 261.72 | 272.06 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 09:15:00 | 261.20 | 261.72 | 272.00 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 1080m) |
| Cross detected — sustain check pending | 2024-12-06 11:15:00 | 260.85 | 261.71 | 271.90 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-06 12:15:00 | 261.60 | 261.71 | 271.85 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-06 13:15:00 | 260.70 | 261.70 | 271.79 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 15:15:00 | 260.15 | 261.67 | 271.68 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 259.73 | 248.83 | 258.88 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-01-03 12:15:00 | 259.73 | 248.83 | 258.88 | SL hit (close>ema400) qty=1.00 sl=258.88 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-01-03 12:15:00 | 259.73 | 248.83 | 258.88 | SL hit (close>ema400) qty=1.00 sl=258.88 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-01-03 12:15:00 | 259.73 | 248.83 | 258.88 | SL hit (close>ema400) qty=1.00 sl=258.88 alert=retest1 |
| Cross detected — sustain check pending | 2025-01-06 09:15:00 | 255.64 | 249.20 | 258.87 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 11:15:00 | 252.83 | 249.26 | 258.81 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-01-07 09:15:00 | 264.18 | 249.62 | 258.75 | SL hit (close>static) qty=1.00 sl=260.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-13 13:15:00 | 255.80 | 253.66 | 259.60 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 15:15:00 | 255.74 | 253.70 | 259.56 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 261.80 | 253.83 | 259.57 | SL hit (close>static) qty=1.00 sl=260.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-24 14:15:00 | 255.95 | 258.19 | 260.59 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-24 15:15:00 | 258.25 | 258.19 | 260.58 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-27 09:15:00 | 254.05 | 258.15 | 260.55 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 11:15:00 | 251.20 | 258.02 | 260.46 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-01-30 13:15:00 | 254.67 | 256.79 | 259.54 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-30 14:15:00 | 257.36 | 256.80 | 259.53 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-01-31 14:15:00 | 263.18 | 256.98 | 259.53 | SL hit (close>static) qty=1.00 sl=260.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-01 12:15:00 | 256.05 | 257.12 | 259.53 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-02-01 13:15:00 | 257.55 | 257.12 | 259.52 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-03 09:15:00 | 247.35 | 257.04 | 259.44 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 11:15:00 | 247.15 | 256.86 | 259.33 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 259.15 | 256.43 | 258.96 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-02-05 13:15:00 | 261.10 | 256.55 | 258.97 | SL hit (close>static) qty=1.00 sl=260.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-06 11:15:00 | 256.30 | 256.69 | 258.98 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 13:15:00 | 254.75 | 256.66 | 258.94 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-03-04 09:15:00 | 216.54 | 241.22 | 248.42 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-20 10:15:00 | 235.58 | 234.59 | 242.11 | SL hit (close>ema200) qty=0.50 sl=234.59 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 11:15:00 | 236.99 | 242.11 | 242.12 | EMA200 below EMA400 |

### Cycle 3 — SELL (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 15:15:00 | 241.10 | 243.84 | 243.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 09:15:00 | 240.29 | 243.80 | 243.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 12:15:00 | 239.82 | 239.52 | 241.33 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-08-13 13:15:00 | 238.93 | 239.51 | 241.32 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 15:15:00 | 238.50 | 239.49 | 241.29 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 240.43 | 239.03 | 240.80 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-22 11:15:00 | 236.75 | 238.99 | 240.70 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 13:15:00 | 236.32 | 238.94 | 240.66 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 241.38 | 237.78 | 239.73 | SL hit (close>ema400) qty=1.00 sl=239.73 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 241.38 | 237.78 | 239.73 | SL hit (close>static) qty=1.00 sl=240.99 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-04 09:15:00 | 236.71 | 238.05 | 239.73 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-09-04 11:15:00 | 237.25 | 238.03 | 239.71 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-09-04 12:15:00 | 236.69 | 238.01 | 239.69 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 14:15:00 | 235.60 | 237.97 | 239.65 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-09-23 09:15:00 | 236.24 | 236.02 | 237.83 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 11:15:00 | 235.68 | 236.02 | 237.81 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 241.52 | 236.24 | 237.82 | SL hit (close>static) qty=1.00 sl=240.99 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 241.52 | 236.24 | 237.82 | SL hit (close>static) qty=1.00 sl=240.99 alert=retest2 |

### Cycle 4 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 240.20 | 244.86 | 244.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 13:15:00 | 238.73 | 244.64 | 244.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 13:15:00 | 239.61 | 238.70 | 241.05 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 240.18 | 238.74 | 241.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 240.18 | 238.74 | 241.04 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-01 12:15:00 | 237.93 | 238.73 | 241.00 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 14:15:00 | 237.72 | 238.71 | 240.97 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 241.55 | 238.75 | 240.96 | SL hit (close>static) qty=1.00 sl=241.11 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-05 09:15:00 | 236.83 | 238.84 | 240.94 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 237.11 | 238.82 | 240.91 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-01-06 14:15:00 | 241.88 | 238.91 | 240.85 | SL hit (close>static) qty=1.00 sl=241.11 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-07 13:15:00 | 238.60 | 239.00 | 240.84 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-01-07 14:15:00 | 239.14 | 239.00 | 240.83 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-08 09:15:00 | 236.20 | 238.97 | 240.80 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:15:00 | 233.07 | 238.87 | 240.73 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 241.58 | 238.11 | 240.14 | SL hit (close>static) qty=1.00 sl=241.11 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-12-05 09:15:00 | 258.60 | 2025-01-03 12:15:00 | 259.73 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-12-06 09:15:00 | 261.20 | 2025-01-03 12:15:00 | 259.73 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest1 | 2024-12-06 15:15:00 | 260.15 | 2025-01-03 12:15:00 | 259.73 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-01-06 11:15:00 | 252.83 | 2025-01-07 09:15:00 | 264.18 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest2 | 2025-01-13 15:15:00 | 255.74 | 2025-01-14 10:15:00 | 261.80 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-01-27 11:15:00 | 251.20 | 2025-01-31 14:15:00 | 263.18 | STOP_HIT | 1.00 | -4.77% |
| SELL | retest2 | 2025-02-03 11:15:00 | 247.15 | 2025-02-05 13:15:00 | 261.10 | STOP_HIT | 1.00 | -5.64% |
| SELL | retest2 | 2025-02-06 13:15:00 | 254.75 | 2025-03-04 09:15:00 | 216.54 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-02-06 13:15:00 | 254.75 | 2025-03-20 10:15:00 | 235.58 | STOP_HIT | 0.50 | 7.53% |
| SELL | retest1 | 2025-08-13 15:15:00 | 238.50 | 2025-09-02 09:15:00 | 241.38 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-08-22 13:15:00 | 236.32 | 2025-09-02 09:15:00 | 241.38 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-09-04 14:15:00 | 235.60 | 2025-09-25 09:15:00 | 241.52 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-09-23 11:15:00 | 235.68 | 2025-09-25 09:15:00 | 241.52 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2026-01-01 14:15:00 | 237.72 | 2026-01-02 10:15:00 | 241.55 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-01-05 11:15:00 | 237.11 | 2026-01-06 14:15:00 | 241.88 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2026-01-08 11:15:00 | 233.07 | 2026-01-13 11:15:00 | 241.58 | STOP_HIT | 1.00 | -3.65% |
