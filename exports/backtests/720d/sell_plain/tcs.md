# TCS (TCS)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 2403.00
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
| ALERT2_SKIP | 2 |
| ALERT3 | 2 |
| PENDING | 10 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 5 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 0 / 5 / 2
- **Avg / median % per leg:** 5.91% / 10.92%
- **Sum % (uncompounded):** 41.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 4 | 57.1% | 0 | 5 | 2 | 5.91% | 41.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 4 | 57.1% | 0 | 5 | 2 | 5.91% | 41.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 4 | 57.1% | 0 | 5 | 2 | 5.91% | 41.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 15:15:00 | 4094.00 | 4274.45 | 4274.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 12:15:00 | 4088.45 | 4247.96 | 4260.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 4139.00 | 4133.47 | 4188.90 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 4101.35 | 4133.31 | 4187.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 4101.35 | 4133.31 | 4187.99 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-11-18 09:15:00 | 4040.45 | 4141.74 | 4182.36 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 11:15:00 | 4023.20 | 4139.24 | 4180.70 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-11-19 12:15:00 | 4097.10 | 4133.49 | 4176.12 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:15:00 | 4030.90 | 4132.04 | 4174.97 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-11-22 13:15:00 | 4216.10 | 4127.91 | 4170.08 | SL hit (close>static) qty=1.00 sl=4205.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-22 13:15:00 | 4216.10 | 4127.91 | 4170.08 | SL hit (close>static) qty=1.00 sl=4205.80 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-31 09:15:00 | 4033.60 | 4258.14 | 4247.61 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 11:15:00 | 4073.30 | 4254.51 | 4245.89 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-12-31 14:15:00 | 4092.45 | 4249.57 | 4243.53 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-31 15:15:00 | 4106.00 | 4248.14 | 4242.84 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-01-02 10:15:00 | 4148.65 | 4237.00 | 4237.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 10:15:00 | 4148.65 | 4237.00 | 4237.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 11:15:00 | 4129.65 | 4235.93 | 4236.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 09:15:00 | 4199.00 | 4185.77 | 4209.32 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 09:15:00 | 4199.00 | 4185.77 | 4209.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 4199.00 | 4185.77 | 4209.32 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-17 09:15:00 | 4141.15 | 4203.44 | 4215.38 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 11:15:00 | 4126.55 | 4202.05 | 4214.56 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-01-23 10:15:00 | 4157.40 | 4179.51 | 4200.88 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 12:15:00 | 4161.00 | 4179.18 | 4200.50 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-02-28 09:15:00 | 3507.57 | 3930.12 | 4034.20 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-02-28 09:15:00 | 3536.85 | 3930.12 | 4034.20 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-25 09:15:00 | 3675.95 | 3674.88 | 3826.14 | SL hit (close>ema200) qty=0.50 sl=3674.88 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-25 09:15:00 | 3675.95 | 3674.88 | 3826.14 | SL hit (close>ema200) qty=0.50 sl=3674.88 alert=retest2 |

### Cycle 3 — SELL (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 12:15:00 | 2991.90 | 3167.66 | 3168.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 2933.90 | 3160.19 | 3164.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 11:15:00 | 2536.20 | 2525.99 | 2688.83 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 2511.00 | 2531.89 | 2677.14 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 11:15:00 | 2511.00 | 2531.45 | 2675.47 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 2480.60 | 2530.50 | 2671.43 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 11:15:00 | 2482.00 | 2529.59 | 2669.57 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-22 11:15:00 | 2517.60 | 2542.44 | 2649.91 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-22 12:15:00 | 2523.30 | 2542.25 | 2649.28 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-23 15:15:00 | 2518.20 | 2541.71 | 2643.79 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:15:00 | 2460.20 | 2540.90 | 2642.87 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 1080m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-18 11:15:00 | 4023.20 | 2024-11-22 13:15:00 | 4216.10 | STOP_HIT | 1.00 | -4.79% |
| SELL | retest2 | 2024-11-19 14:15:00 | 4030.90 | 2024-11-22 13:15:00 | 4216.10 | STOP_HIT | 1.00 | -4.59% |
| SELL | retest2 | 2024-12-31 11:15:00 | 4073.30 | 2025-01-02 10:15:00 | 4148.65 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-01-17 11:15:00 | 4126.55 | 2025-02-28 09:15:00 | 3507.57 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-01-23 12:15:00 | 4161.00 | 2025-02-28 09:15:00 | 3536.85 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-01-17 11:15:00 | 4126.55 | 2025-03-25 09:15:00 | 3675.95 | STOP_HIT | 0.50 | 10.92% |
| SELL | retest2 | 2025-01-23 12:15:00 | 4161.00 | 2025-03-25 09:15:00 | 3675.95 | STOP_HIT | 0.50 | 11.66% |
