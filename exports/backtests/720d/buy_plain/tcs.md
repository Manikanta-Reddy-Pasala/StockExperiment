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
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 2 |
| PENDING | 11 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 4 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 8
- **Target hits / Stop hits / Partials:** 0 / 8 / 0
- **Avg / median % per leg:** -1.90% / -1.31%
- **Sum % (uncompounded):** -15.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.90% | -15.2% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.44% | -5.8% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.36% | -9.5% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.44% | -5.8% |
| retest2 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.36% | -9.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 13:15:00 | 4376.65 | 4202.39 | 4202.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 09:15:00 | 4382.60 | 4207.03 | 4204.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 10:15:00 | 4304.30 | 4308.99 | 4264.86 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-12-18 11:15:00 | 4329.85 | 4309.19 | 4265.19 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 12:15:00 | 4336.00 | 4309.46 | 4265.54 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 4284.60 | 4310.45 | 4267.56 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 4266.50 | 4308.86 | 4267.82 | SL hit (close<ema400) qty=1.00 sl=4267.82 alert=retest1 |
| Cross detected — sustain check pending | 2025-01-13 11:15:00 | 4308.50 | 4193.61 | 4212.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-13 12:15:00 | 4311.00 | 4194.78 | 4212.77 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-14 11:15:00 | 4262.55 | 4200.04 | 4214.91 | SL hit (close<static) qty=1.00 sl=4264.95 alert=retest2 |

### Cycle 2 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 3160.90 | 3082.17 | 3082.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 10:15:00 | 3190.40 | 3097.87 | 3090.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 14:15:00 | 3204.30 | 3212.34 | 3168.29 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-01-01 10:15:00 | 3221.00 | 3212.38 | 3168.97 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:15:00 | 3224.80 | 3212.50 | 3169.25 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-05 10:15:00 | 3223.10 | 3215.10 | 3173.31 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 11:15:00 | 3229.60 | 3215.25 | 3173.59 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-06 09:15:00 | 3238.10 | 3215.63 | 3174.82 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 10:15:00 | 3235.00 | 3215.83 | 3175.12 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-08 11:15:00 | 3221.20 | 3221.72 | 3181.12 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-08 12:15:00 | 3218.70 | 3221.69 | 3181.31 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 3192.10 | 3219.83 | 3182.70 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-12 12:15:00 | 3213.20 | 3219.57 | 3182.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 13:15:00 | 3211.40 | 3219.49 | 3183.08 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 09:15:00 | 3208.50 | 3220.88 | 3186.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-16 10:15:00 | 3200.10 | 3220.67 | 3186.88 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-16 11:15:00 | 3206.80 | 3220.54 | 3186.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 3209.90 | 3220.43 | 3187.09 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 15:15:00 | 3209.00 | 3220.02 | 3187.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-19 09:15:00 | 3185.00 | 3219.67 | 3187.37 | ENTRY2 sustain failed after 3960m |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 3185.00 | 3219.67 | 3187.37 | SL hit (close<ema400) qty=1.00 sl=3187.37 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 3185.00 | 3219.67 | 3187.37 | SL hit (close<ema400) qty=1.00 sl=3187.37 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 3185.00 | 3219.67 | 3187.37 | SL hit (close<ema400) qty=1.00 sl=3187.37 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 3169.30 | 3219.17 | 3187.28 | SL hit (close<static) qty=1.00 sl=3181.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 3169.30 | 3219.17 | 3187.28 | SL hit (close<static) qty=1.00 sl=3181.20 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-03 10:15:00 | 3226.90 | 3183.94 | 3176.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 11:15:00 | 3235.00 | 3184.45 | 3176.48 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 3048.70 | 3184.63 | 3176.78 | SL hit (close<static) qty=1.00 sl=3181.20 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-12-18 12:15:00 | 4336.00 | 2024-12-20 10:15:00 | 4266.50 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-01-13 12:15:00 | 4311.00 | 2025-01-14 11:15:00 | 4262.55 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest1 | 2026-01-01 11:15:00 | 3224.80 | 2026-01-19 09:15:00 | 3185.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest1 | 2026-01-05 11:15:00 | 3229.60 | 2026-01-19 09:15:00 | 3185.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest1 | 2026-01-06 10:15:00 | 3235.00 | 2026-01-19 09:15:00 | 3185.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2026-01-12 13:15:00 | 3211.40 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-01-16 12:15:00 | 3209.90 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-02-03 11:15:00 | 3235.00 | 2026-02-04 09:15:00 | 3048.70 | STOP_HIT | 1.00 | -5.76% |
