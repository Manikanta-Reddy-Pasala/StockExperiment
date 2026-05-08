# TCS (TCS)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:15:00 (1237 bars)
- **Last close:** 2394.40
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
| PENDING | 12 |
| PENDING_CANCEL | 3 |
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
- **Avg / median % per leg:** -1.23% / -1.72%
- **Sum % (uncompounded):** -8.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.27% | -13.6% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.86% | -5.6% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.67% | -8.0% |
| SELL (all) | 1 | 1 | 100.0% | 0 | 0 | 1 | 5.00% | 5.0% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 0 | 1 | 5.00% | 5.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.14% | -0.6% |
| retest2 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.67% | -8.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 14:15:00 | 3148.90 | 3060.10 | 3059.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 13:15:00 | 3165.10 | 3065.50 | 3062.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 14:15:00 | 3204.30 | 3212.23 | 3163.70 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-01 10:15:00 | 3221.00 | 3212.27 | 3164.45 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:15:00 | 3224.80 | 3212.39 | 3164.75 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-05 10:15:00 | 3223.10 | 3215.04 | 3169.11 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 11:15:00 | 3228.30 | 3215.17 | 3169.40 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-06 09:15:00 | 3238.10 | 3215.57 | 3170.73 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 10:15:00 | 3235.00 | 3215.76 | 3171.05 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-08 11:15:00 | 3221.20 | 3221.66 | 3177.35 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-08 12:15:00 | 3218.80 | 3221.63 | 3177.56 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 11:15:00 | 3199.90 | 3219.53 | 3179.25 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-12 12:15:00 | 3213.20 | 3219.47 | 3179.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 13:15:00 | 3211.40 | 3219.39 | 3179.58 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 09:15:00 | 3208.50 | 3220.75 | 3183.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 10:15:00 | 3200.10 | 3220.54 | 3183.65 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 3169.30 | 3219.04 | 3184.16 | SL hit (close<ema400) qty=1.00 sl=3184.16 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 3169.30 | 3219.04 | 3184.16 | SL hit (close<ema400) qty=1.00 sl=3184.16 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 3169.30 | 3219.04 | 3184.16 | SL hit (close<ema400) qty=1.00 sl=3184.16 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 3169.30 | 3219.04 | 3184.16 | SL hit (close<static) qty=1.00 sl=3174.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 3169.30 | 3219.04 | 3184.16 | SL hit (close<static) qty=1.00 sl=3174.60 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-28 15:15:00 | 3200.10 | 3194.59 | 3177.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-29 09:15:00 | 3136.10 | 3194.01 | 3177.37 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-02-03 10:15:00 | 3226.90 | 3185.19 | 3174.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 11:15:00 | 3234.70 | 3185.69 | 3174.68 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 3048.70 | 3185.82 | 3175.03 | SL hit (close<static) qty=1.00 sl=3174.60 alert=retest2 |

### Cycle 2 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 2991.70 | 3163.51 | 3164.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 2934.00 | 3161.22 | 3163.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 11:15:00 | 2536.00 | 2525.99 | 2688.36 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 2510.60 | 2531.89 | 2676.71 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:15:00 | 2507.60 | 2531.65 | 2675.86 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 2480.60 | 2530.49 | 2671.02 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2486.70 | 2530.05 | 2670.10 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-22 11:15:00 | 2517.90 | 2540.80 | 2650.93 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-22 12:15:00 | 2523.30 | 2540.62 | 2650.29 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-23 15:15:00 | 2517.00 | 2540.25 | 2644.76 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:15:00 | 2460.20 | 2539.46 | 2643.84 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 1080m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:15:00 | 2382.22 | 2490.46 | 2587.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-01-01 11:15:00 | 3224.80 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest1 | 2026-01-05 11:15:00 | 3228.30 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest1 | 2026-01-06 10:15:00 | 3235.00 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2026-01-12 13:15:00 | 3211.40 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-01-16 10:15:00 | 3200.10 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2026-02-03 11:15:00 | 3234.70 | 2026-02-04 09:15:00 | 3048.70 | STOP_HIT | 1.00 | -5.75% |
| SELL | retest1 | 2026-04-10 10:15:00 | 2507.60 | 2026-05-08 10:15:00 | 2382.22 | PARTIAL | 0.50 | 5.00% |
