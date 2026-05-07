# ITC (ITC)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 307.55
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
| ALERT2_SKIP | 1 |
| ALERT3 | 4 |
| PENDING | 14 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 7 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 8
- **Target hits / Stop hits / Partials:** 0 / 11 / 1
- **Avg / median % per leg:** 2.85% / -0.80%
- **Sum % (uncompounded):** 34.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 4 | 33.3% | 0 | 11 | 1 | 2.85% | 34.2% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.12% | -4.5% |
| SELL @ 3rd Alert (retest2) | 8 | 4 | 50.0% | 0 | 7 | 1 | 4.83% | 38.7% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.12% | -4.5% |
| retest2 (combined) | 8 | 4 | 50.0% | 0 | 7 | 1 | 4.83% | 38.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 11:15:00 | 478.55 | 492.84 | 492.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 09:15:00 | 476.80 | 491.43 | 492.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 480.95 | 479.78 | 484.78 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-11-28 10:15:00 | 475.40 | 479.74 | 484.73 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 12:15:00 | 473.75 | 479.63 | 484.62 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2024-12-02 09:15:00 | 475.85 | 479.25 | 484.16 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-02 11:15:00 | 478.10 | 479.20 | 484.09 | ENTRY1 sustain failed after 120m |
| Cross detected — sustain check pending | 2024-12-03 09:15:00 | 468.40 | 479.02 | 483.88 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 11:15:00 | 469.85 | 478.83 | 483.73 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 472.65 | 471.86 | 477.50 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-23 13:15:00 | 471.80 | 471.86 | 477.48 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-23 14:15:00 | 474.45 | 471.89 | 477.46 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-12-24 10:15:00 | 478.80 | 472.01 | 477.44 | SL hit (close>ema400) qty=1.00 sl=477.44 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-12-24 10:15:00 | 478.80 | 472.01 | 477.44 | SL hit (close>ema400) qty=1.00 sl=477.44 alert=retest1 |
| Cross detected — sustain check pending | 2025-01-06 09:15:00 | 461.65 | 475.84 | 478.28 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 11:15:00 | 449.50 | 475.32 | 477.99 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-06-05 13:15:00 | 417.95 | 425.26 | 425.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 417.95 | 425.26 | 425.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 417.15 | 424.64 | 424.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 11:15:00 | 419.90 | 418.57 | 420.87 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-07-10 10:15:00 | 417.20 | 418.59 | 420.81 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 12:15:00 | 417.65 | 418.56 | 420.78 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2025-07-10 14:15:00 | 416.80 | 418.54 | 420.74 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 09:15:00 | 417.25 | 418.51 | 420.71 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 1140m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 420.60 | 418.47 | 420.54 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 420.60 | 418.47 | 420.54 | SL hit (close>ema400) qty=1.00 sl=420.54 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 420.60 | 418.47 | 420.54 | SL hit (close>ema400) qty=1.00 sl=420.54 alert=retest1 |
| Cross detected — sustain check pending | 2025-07-21 13:15:00 | 418.95 | 419.61 | 420.85 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-07-21 14:15:00 | 420.30 | 419.62 | 420.85 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-22 09:15:00 | 418.65 | 419.62 | 420.84 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:15:00 | 417.70 | 419.58 | 420.81 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-08-04 10:15:00 | 417.95 | 416.04 | 418.42 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 12:15:00 | 418.05 | 416.08 | 418.42 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-10-23 10:15:00 | 418.75 | 405.66 | 408.08 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 12:15:00 | 417.20 | 405.90 | 408.18 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-10-27 11:15:00 | 422.30 | 407.25 | 408.73 | SL hit (close>static) qty=1.00 sl=420.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 11:15:00 | 422.30 | 407.25 | 408.73 | SL hit (close>static) qty=1.00 sl=420.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 11:15:00 | 422.30 | 407.25 | 408.73 | SL hit (close>static) qty=1.00 sl=420.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-28 10:15:00 | 417.60 | 408.01 | 409.07 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:15:00 | 418.00 | 408.20 | 409.15 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 417.05 | 408.28 | 409.19 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-10-29 14:15:00 | 421.65 | 409.10 | 409.58 | SL hit (close>static) qty=1.00 sl=420.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-03 11:15:00 | 413.95 | 410.90 | 410.47 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 13:15:00 | 414.15 | 410.96 | 410.51 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-11-10 14:15:00 | 405.35 | 410.15 | 410.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 14:15:00 | 405.35 | 410.15 | 410.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 403.95 | 409.38 | 409.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 409.40 | 409.22 | 409.65 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 409.40 | 409.22 | 409.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 409.40 | 409.22 | 409.65 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-11-17 10:15:00 | 407.10 | 409.20 | 409.64 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 12:15:00 | 406.70 | 409.16 | 409.61 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-01-02 09:15:00 | 345.69 | 400.76 | 403.71 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 330.45 | 330.27 | 349.69 | SL hit (close>ema200) qty=0.50 sl=330.27 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-11-28 12:15:00 | 473.75 | 2024-12-24 10:15:00 | 478.80 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest1 | 2024-12-03 11:15:00 | 469.85 | 2024-12-24 10:15:00 | 478.80 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-01-06 11:15:00 | 449.50 | 2025-06-05 13:15:00 | 417.95 | STOP_HIT | 1.00 | 7.02% |
| SELL | retest1 | 2025-07-10 12:15:00 | 417.65 | 2025-07-15 09:15:00 | 420.60 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest1 | 2025-07-11 09:15:00 | 417.25 | 2025-07-15 09:15:00 | 420.60 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-07-22 11:15:00 | 417.70 | 2025-10-27 11:15:00 | 422.30 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-08-04 12:15:00 | 418.05 | 2025-10-27 11:15:00 | 422.30 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-10-23 12:15:00 | 417.20 | 2025-10-27 11:15:00 | 422.30 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-10-28 12:15:00 | 418.00 | 2025-10-29 14:15:00 | 421.65 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-11-03 13:15:00 | 414.15 | 2025-11-10 14:15:00 | 405.35 | STOP_HIT | 1.00 | 2.12% |
| SELL | retest2 | 2025-11-17 12:15:00 | 406.70 | 2026-01-02 09:15:00 | 345.69 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-11-17 12:15:00 | 406.70 | 2026-02-18 11:15:00 | 330.45 | STOP_HIT | 0.50 | 18.75% |
