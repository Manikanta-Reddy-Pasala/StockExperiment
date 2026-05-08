# TCS (TCS)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:15:00 (1237 bars)
- **Last close:** 2394.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty @ 15% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 15%, trail SL → EMA200
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 0 / 6
- **Target hits / Stop hits / Partials:** 0 / 6 / 0
- **Avg / median % per leg:** -2.27% / -1.72%
- **Sum % (uncompounded):** -13.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.27% | -13.6% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.86% | -5.6% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.67% | -8.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.86% | -5.6% |
| retest2 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.67% | -8.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 11:15:00 | 3101.00 | 3038.10 | 3037.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 15:15:00 | 3108.30 | 3040.37 | 3038.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 14:15:00 | 3204.30 | 3212.17 | 3159.72 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-01 10:15:00 | 3221.00 | 3212.22 | 3160.52 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:15:00 | 3224.80 | 3212.34 | 3160.84 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-05 10:15:00 | 3223.10 | 3214.99 | 3165.45 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 11:15:00 | 3228.30 | 3215.13 | 3165.76 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-06 09:15:00 | 3238.10 | 3215.53 | 3167.18 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 10:15:00 | 3235.00 | 3215.72 | 3167.52 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-08 11:15:00 | 3221.20 | 3221.63 | 3174.07 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-08 12:15:00 | 3218.80 | 3221.60 | 3174.30 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 11:15:00 | 3199.90 | 3219.50 | 3176.20 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-12 12:15:00 | 3213.20 | 3219.44 | 3176.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 13:15:00 | 3211.40 | 3219.36 | 3176.55 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 09:15:00 | 3208.50 | 3220.72 | 3180.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 10:15:00 | 3200.10 | 3220.52 | 3180.89 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 3169.30 | 3219.01 | 3181.49 | SL hit (close<ema400) qty=1.00 sl=3181.49 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 3169.30 | 3219.01 | 3181.49 | SL hit (close<ema400) qty=1.00 sl=3181.49 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 3169.30 | 3219.01 | 3181.49 | SL hit (close<ema400) qty=1.00 sl=3181.49 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 3169.30 | 3219.01 | 3181.49 | SL hit (close<static) qty=1.00 sl=3174.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 3169.30 | 3219.01 | 3181.49 | SL hit (close<static) qty=1.00 sl=3174.60 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-28 15:15:00 | 3200.10 | 3194.58 | 3175.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-29 09:15:00 | 3136.10 | 3194.00 | 3175.27 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-02-03 10:15:00 | 3226.90 | 3185.18 | 3172.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 11:15:00 | 3234.70 | 3185.68 | 3172.81 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 3048.70 | 3185.81 | 3173.21 | SL hit (close<static) qty=1.00 sl=3174.60 alert=retest2 |

### Cycle 2 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 2934.00 | 3161.21 | 3161.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 10:15:00 | 2917.30 | 3158.79 | 3160.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 11:15:00 | 2536.00 | 2525.99 | 2687.92 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 2510.60 | 2531.89 | 2676.30 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:15:00 | 2507.60 | 2531.65 | 2675.46 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 2480.60 | 2530.49 | 2670.62 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2486.70 | 2530.05 | 2669.71 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-22 11:15:00 | 2517.90 | 2540.80 | 2650.61 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-22 12:15:00 | 2523.30 | 2540.62 | 2649.98 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-23 15:15:00 | 2517.00 | 2540.25 | 2644.46 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:15:00 | 2460.20 | 2539.46 | 2643.54 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 1080m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-01-01 11:15:00 | 3224.80 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest1 | 2026-01-05 11:15:00 | 3228.30 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest1 | 2026-01-06 10:15:00 | 3235.00 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2026-01-12 13:15:00 | 3211.40 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-01-16 10:15:00 | 3200.10 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2026-02-03 11:15:00 | 3234.70 | 2026-02-04 09:15:00 | 3048.70 | STOP_HIT | 1.00 | -5.75% |
