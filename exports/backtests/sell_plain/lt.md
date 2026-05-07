# LT (LT)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2025-11-10 09:15:00 → 2026-05-07 15:15:00 (847 bars)
- **Last close:** 4023.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 1 |
| PENDING | 3 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 0 |
| ENTRY2 | 2 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 2 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 0
- **Avg / median % per leg:** -3.73% / -2.88%
- **Sum % (uncompounded):** -7.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.73% | -7.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.73% | -7.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.73% | -7.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 3588.60 | 4038.46 | 4039.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 3556.10 | 4033.66 | 4036.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 3732.90 | 3727.17 | 3843.91 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 3971.00 | 3728.64 | 3839.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3971.00 | 3728.64 | 3839.50 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-09 10:15:00 | 3919.80 | 3748.16 | 3845.12 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 12:15:00 | 3902.80 | 3751.49 | 3845.83 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 3923.00 | 3770.64 | 3850.60 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-13 11:15:00 | 3941.80 | 3773.90 | 3851.45 | ENTRY2 sustain failed after 120m |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 4081.60 | 3783.91 | 3854.59 | SL hit (close>static) qty=1.00 sl=4023.40 alert=retest2 |
| Cross detected — sustain check pending | 2026-05-06 09:15:00 | 3919.50 | 3956.55 | 3934.78 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:15:00 | 3920.60 | 3955.93 | 3934.68 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-05-07 12:15:00 | 4033.50 | 3958.81 | 3936.97 | SL hit (close>static) qty=1.00 sl=4023.40 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-04-09 12:15:00 | 3902.80 | 2026-04-15 09:15:00 | 4081.60 | STOP_HIT | 1.00 | -4.58% |
| SELL | retest2 | 2026-05-06 11:15:00 | 3920.60 | 2026-05-07 12:15:00 | 4033.50 | STOP_HIT | 1.00 | -2.88% |
