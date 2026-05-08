# TCS (TCS)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 2397.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 1 |
| PENDING | 13 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 1 / 6
- **Target hits / Stop hits / Partials:** 0 / 6 / 1
- **Avg / median % per leg:** -1.07% / -1.31%
- **Sum % (uncompounded):** -7.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.08% | -12.5% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.39% | -4.2% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.78% | -8.3% |
| SELL (all) | 1 | 1 | 100.0% | 0 | 0 | 1 | 5.00% | 5.0% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 0 | 1 | 5.00% | 5.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.21% | 0.8% |
| retest2 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.78% | -8.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 13:15:00 | 3127.70 | 3086.00 | 3085.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 14:15:00 | 3135.00 | 3086.48 | 3086.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 14:15:00 | 3204.30 | 3212.34 | 3169.11 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-01 10:15:00 | 3221.00 | 3212.38 | 3169.78 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:15:00 | 3224.80 | 3212.50 | 3170.05 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-05 10:15:00 | 3223.10 | 3215.10 | 3174.06 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 11:15:00 | 3229.60 | 3215.25 | 3174.34 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-06 09:15:00 | 3238.10 | 3215.64 | 3175.55 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 10:15:00 | 3235.00 | 3215.83 | 3175.84 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-08 11:15:00 | 3221.20 | 3221.72 | 3181.79 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-08 12:15:00 | 3218.70 | 3221.69 | 3181.98 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 3192.10 | 3219.83 | 3183.34 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-12 12:15:00 | 3213.20 | 3219.57 | 3183.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 13:15:00 | 3211.40 | 3219.49 | 3183.71 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 09:15:00 | 3208.50 | 3220.88 | 3187.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-16 10:15:00 | 3200.10 | 3220.67 | 3187.45 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-16 11:15:00 | 3206.80 | 3220.54 | 3187.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 3209.90 | 3220.43 | 3187.66 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 15:15:00 | 3209.00 | 3220.02 | 3187.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-19 09:15:00 | 3185.00 | 3219.67 | 3187.92 | ENTRY2 sustain failed after 3960m |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 3185.00 | 3219.67 | 3187.92 | SL hit (close<ema400) qty=1.00 sl=3187.92 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 3185.00 | 3219.67 | 3187.92 | SL hit (close<ema400) qty=1.00 sl=3187.92 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 3185.00 | 3219.67 | 3187.92 | SL hit (close<ema400) qty=1.00 sl=3187.92 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 3169.30 | 3219.17 | 3187.83 | SL hit (close<static) qty=1.00 sl=3181.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 3169.30 | 3219.17 | 3187.83 | SL hit (close<static) qty=1.00 sl=3181.20 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-03 10:15:00 | 3226.90 | 3183.94 | 3176.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 11:15:00 | 3235.00 | 3184.45 | 3176.85 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 3048.70 | 3184.63 | 3177.14 | SL hit (close<static) qty=1.00 sl=3181.20 alert=retest2 |

### Cycle 2 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 2996.80 | 3169.43 | 3169.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 2933.90 | 3160.19 | 3165.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 11:15:00 | 2536.20 | 2525.99 | 2688.92 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 2511.00 | 2531.89 | 2677.22 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:15:00 | 2507.60 | 2531.65 | 2676.37 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 2480.60 | 2530.50 | 2671.51 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2486.30 | 2530.06 | 2670.59 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-22 11:15:00 | 2517.60 | 2542.44 | 2649.97 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-22 12:15:00 | 2523.30 | 2542.25 | 2649.34 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-23 15:15:00 | 2518.20 | 2541.71 | 2643.85 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:15:00 | 2460.20 | 2540.90 | 2642.93 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 1080m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:15:00 | 2382.22 | 2491.22 | 2586.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-01-01 11:15:00 | 3224.80 | 2026-01-19 09:15:00 | 3185.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest1 | 2026-01-05 11:15:00 | 3229.60 | 2026-01-19 09:15:00 | 3185.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest1 | 2026-01-06 10:15:00 | 3235.00 | 2026-01-19 09:15:00 | 3185.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2026-01-12 13:15:00 | 3211.40 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-01-16 12:15:00 | 3209.90 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-02-03 11:15:00 | 3235.00 | 2026-02-04 09:15:00 | 3048.70 | STOP_HIT | 1.00 | -5.76% |
| SELL | retest1 | 2026-04-10 10:15:00 | 2507.60 | 2026-05-08 10:15:00 | 2382.22 | PARTIAL | 0.50 | 5.00% |
