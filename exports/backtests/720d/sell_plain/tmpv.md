# TMPV (TMPV)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 358.95
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 6 |
| PENDING | 23 |
| PENDING_CANCEL | 8 |
| ENTRY1 | 2 |
| ENTRY2 | 13 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 9
- **Target hits / Stop hits / Partials:** 0 / 15 / 5
- **Avg / median % per leg:** 5.33% / 3.56%
- **Sum % (uncompounded):** 106.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 20 | 11 | 55.0% | 0 | 15 | 5 | 5.33% | 106.5% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 15.27% | 61.1% |
| SELL @ 3rd Alert (retest2) | 16 | 7 | 43.8% | 0 | 13 | 3 | 2.84% | 45.4% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 2 | 2 | 15.27% | 61.1% |
| retest2 (combined) | 16 | 7 | 43.8% | 0 | 13 | 3 | 2.84% | 45.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 15:15:00 | 587.85 | 625.08 | 625.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-20 09:15:00 | 583.42 | 624.66 | 624.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 14:15:00 | 495.55 | 495.34 | 524.45 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-12-09 12:15:00 | 488.21 | 495.12 | 523.62 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 14:15:00 | 484.33 | 494.92 | 523.24 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2024-12-10 11:15:00 | 486.61 | 494.63 | 522.53 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 13:15:00 | 484.12 | 494.43 | 522.15 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-02-11 12:15:00 | 411.68 | 444.78 | 463.77 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-02-11 12:15:00 | 411.50 | 444.78 | 463.77 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-18 10:15:00 | 408.94 | 407.01 | 427.98 | SL hit (close>ema200) qty=0.50 sl=407.01 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-18 10:15:00 | 408.94 | 407.01 | 427.98 | SL hit (close>ema200) qty=0.50 sl=407.01 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 425.97 | 408.80 | 426.63 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-03-27 09:15:00 | 406.42 | 413.40 | 426.99 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 11:15:00 | 405.12 | 413.27 | 426.78 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-04-07 09:15:00 | 344.35 | 407.80 | 421.46 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 390.70 | 390.56 | 407.53 | SL hit (close>ema200) qty=0.50 sl=390.56 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-08 11:15:00 | 420.42 | 396.05 | 405.57 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:15:00 | 414.15 | 396.43 | 405.66 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-05-09 10:15:00 | 421.42 | 397.30 | 405.92 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-09 11:15:00 | 427.18 | 397.60 | 406.02 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-05-09 12:15:00 | 427.64 | 397.90 | 406.13 | SL hit (close>static) qty=1.00 sl=427.27 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-14 11:15:00 | 422.27 | 403.96 | 408.54 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-14 13:15:00 | 422.58 | 404.31 | 408.67 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-06-16 09:15:00 | 412.85 | 429.01 | 423.43 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 11:15:00 | 415.06 | 428.74 | 423.34 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-07-01 15:15:00 | 414.24 | 419.45 | 419.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 414.24 | 419.45 | 419.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 14:15:00 | 413.55 | 418.95 | 419.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 417.67 | 416.37 | 417.67 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 417.67 | 416.37 | 417.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 417.67 | 416.37 | 417.67 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-30 09:15:00 | 405.42 | 416.97 | 417.80 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 11:15:00 | 405.45 | 416.75 | 417.68 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-08-18 14:15:00 | 409.64 | 405.95 | 410.72 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-19 09:15:00 | 418.76 | 406.12 | 410.75 | ENTRY2 sustain failed after 1140m |
| Stop hit — per-position SL triggered | 2025-08-19 10:15:00 | 423.21 | 406.29 | 410.81 | SL hit (close>static) qty=1.00 sl=420.06 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-28 09:15:00 | 406.00 | 409.60 | 411.83 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 11:15:00 | 408.06 | 409.56 | 411.79 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-08-28 14:15:00 | 409.06 | 409.56 | 411.76 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:15:00 | 406.85 | 409.54 | 411.72 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1140m) |
| Cross detected — sustain check pending | 2025-08-29 12:15:00 | 408.30 | 409.54 | 411.69 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:15:00 | 405.33 | 409.47 | 411.64 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 411.12 | 409.42 | 411.57 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 420.33 | 410.01 | 411.80 | SL hit (close>static) qty=1.00 sl=420.06 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 420.33 | 410.01 | 411.80 | SL hit (close>static) qty=1.00 sl=420.06 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 420.33 | 410.01 | 411.80 | SL hit (close>static) qty=1.00 sl=420.06 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-25 09:15:00 | 404.12 | 421.99 | 418.73 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:15:00 | 401.64 | 421.58 | 418.56 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-09-26 13:15:00 | 407.82 | 420.31 | 418.04 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 15:15:00 | 408.09 | 420.07 | 417.94 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-09-29 11:15:00 | 406.64 | 419.73 | 417.80 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-09-29 12:15:00 | 408.61 | 419.62 | 417.75 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-29 14:15:00 | 407.64 | 419.40 | 417.66 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-09-29 15:15:00 | 408.73 | 419.29 | 417.62 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-30 09:15:00 | 405.76 | 419.16 | 417.56 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:15:00 | 406.58 | 418.91 | 417.45 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 412.15 | 418.63 | 417.33 | SL hit (close>static) qty=1.00 sl=411.76 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 412.15 | 418.63 | 417.33 | SL hit (close>static) qty=1.00 sl=411.76 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 412.15 | 418.63 | 417.33 | SL hit (close>static) qty=1.00 sl=411.76 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-09 09:15:00 | 406.61 | 420.94 | 418.86 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-10-09 10:15:00 | 408.67 | 420.82 | 418.81 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-09 11:15:00 | 407.61 | 420.68 | 418.75 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:15:00 | 407.97 | 420.43 | 418.64 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-10-09 14:15:00 | 412.67 | 420.35 | 418.61 | SL hit (close>static) qty=1.00 sl=411.76 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 415.65 | 418.37 | 417.73 | EMA400 retest candle locked |

### Cycle 3 — SELL (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 10:15:00 | 391.60 | 417.05 | 417.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 13:15:00 | 389.75 | 416.29 | 416.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 411.85 | 411.23 | 413.72 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 411.85 | 411.23 | 413.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 411.85 | 411.23 | 413.72 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-11-04 11:15:00 | 409.30 | 411.48 | 413.43 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 13:15:00 | 405.85 | 411.41 | 413.37 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-11-11 09:15:00 | 403.15 | 410.53 | 412.68 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 11:15:00 | 404.60 | 410.39 | 412.60 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-12-09 09:15:00 | 344.97 | 374.35 | 388.61 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-12-09 09:15:00 | 343.91 | 374.35 | 388.61 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 10:15:00 | 363.75 | 361.35 | 376.59 | SL hit (close>ema200) qty=0.50 sl=361.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 10:15:00 | 363.75 | 361.35 | 376.59 | SL hit (close>ema200) qty=0.50 sl=361.35 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 331.95 | 368.39 | 368.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 12:15:00 | 328.90 | 367.64 | 368.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 330.90 | 326.70 | 341.68 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 13:15:00 | 341.60 | 328.29 | 341.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 341.60 | 328.29 | 341.22 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 339.25 | 328.68 | 341.23 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-13 10:15:00 | 344.25 | 328.84 | 341.24 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-30 12:15:00 | 337.85 | 343.36 | 345.79 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-30 13:15:00 | 340.05 | 343.33 | 345.76 | ENTRY2 sustain failed after 60m |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-12-09 14:15:00 | 484.33 | 2025-02-11 12:15:00 | 411.68 | PARTIAL | 0.50 | 15.00% |
| SELL | retest1 | 2024-12-10 13:15:00 | 484.12 | 2025-02-11 12:15:00 | 411.50 | PARTIAL | 0.50 | 15.00% |
| SELL | retest1 | 2024-12-09 14:15:00 | 484.33 | 2025-03-18 10:15:00 | 408.94 | STOP_HIT | 0.50 | 15.57% |
| SELL | retest1 | 2024-12-10 13:15:00 | 484.12 | 2025-03-18 10:15:00 | 408.94 | STOP_HIT | 0.50 | 15.53% |
| SELL | retest2 | 2025-03-27 11:15:00 | 405.12 | 2025-04-07 09:15:00 | 344.35 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-03-27 11:15:00 | 405.12 | 2025-04-23 09:15:00 | 390.70 | STOP_HIT | 0.50 | 3.56% |
| SELL | retest2 | 2025-05-08 13:15:00 | 414.15 | 2025-05-09 12:15:00 | 427.64 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2025-06-16 11:15:00 | 415.06 | 2025-07-01 15:15:00 | 414.24 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-07-30 11:15:00 | 405.45 | 2025-08-19 10:15:00 | 423.21 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2025-08-28 11:15:00 | 408.06 | 2025-09-02 11:15:00 | 420.33 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2025-08-29 09:15:00 | 406.85 | 2025-09-02 11:15:00 | 420.33 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2025-08-29 14:15:00 | 405.33 | 2025-09-02 11:15:00 | 420.33 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2025-09-25 11:15:00 | 401.64 | 2025-09-30 14:15:00 | 412.15 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-09-26 15:15:00 | 408.09 | 2025-09-30 14:15:00 | 412.15 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-09-30 11:15:00 | 406.58 | 2025-09-30 14:15:00 | 412.15 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-10-09 13:15:00 | 407.97 | 2025-10-09 14:15:00 | 412.67 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-11-04 13:15:00 | 405.85 | 2025-12-09 09:15:00 | 344.97 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-11-11 11:15:00 | 404.60 | 2025-12-09 09:15:00 | 343.91 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-11-04 13:15:00 | 405.85 | 2025-12-23 10:15:00 | 363.75 | STOP_HIT | 0.50 | 10.37% |
| SELL | retest2 | 2025-11-11 11:15:00 | 404.60 | 2025-12-23 10:15:00 | 363.75 | STOP_HIT | 0.50 | 10.10% |
