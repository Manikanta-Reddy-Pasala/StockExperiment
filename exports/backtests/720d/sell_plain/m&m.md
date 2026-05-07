# M&M (M&M)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 3366.00
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
| ALERT2 | 2 |
| ALERT2_SKIP | 2 |
| ALERT3 | 2 |
| PENDING | 3 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 3 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 0 / 3 / 2
- **Avg / median % per leg:** 7.82% / 7.26%
- **Sum % (uncompounded):** 39.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 4 | 80.0% | 0 | 3 | 2 | 7.82% | 39.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 4 | 80.0% | 0 | 3 | 2 | 7.82% | 39.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 5 | 4 | 80.0% | 0 | 3 | 2 | 7.82% | 39.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 2791.15 | 2977.60 | 2977.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 09:15:00 | 2751.40 | 2997.36 | 2993.86 | Break + close below crossover candle low |

### Cycle 2 — SELL (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 12:15:00 | 2755.95 | 2990.33 | 2990.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 09:15:00 | 2694.05 | 2970.13 | 2979.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 13:15:00 | 2788.40 | 2782.65 | 2857.03 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 09:15:00 | 2859.20 | 2788.09 | 2853.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 2859.20 | 2788.09 | 2853.78 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-03-21 14:15:00 | 2801.85 | 2790.89 | 2853.58 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 2739.20 | 2790.34 | 2852.68 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 4020m) |
| Stop hit — per-position SL triggered | 2025-04-23 10:15:00 | 2869.50 | 2697.65 | 2761.80 | SL hit (close>static) qty=1.00 sl=2863.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 3380.00 | 3618.61 | 3619.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 11:15:00 | 3333.20 | 3613.12 | 3616.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 3592.10 | 3566.84 | 3590.71 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 3592.10 | 3566.84 | 3590.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 3592.10 | 3566.84 | 3590.71 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-16 10:15:00 | 3510.60 | 3584.60 | 3594.84 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 12:15:00 | 3473.80 | 3582.73 | 3593.80 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-02-19 09:15:00 | 3502.90 | 3568.25 | 3585.26 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:15:00 | 3449.10 | 3566.18 | 3584.05 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-13 12:15:00 | 2952.73 | 3380.48 | 3467.73 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-13 13:15:00 | 2931.73 | 3376.12 | 3465.10 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 3221.60 | 3160.50 | 3296.76 | SL hit (close>ema200) qty=0.50 sl=3160.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 3221.60 | 3160.50 | 3296.76 | SL hit (close>ema200) qty=0.50 sl=3160.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-03-24 09:15:00 | 2739.20 | 2025-04-23 10:15:00 | 2869.50 | STOP_HIT | 1.00 | -4.76% |
| SELL | retest2 | 2026-02-16 12:15:00 | 3473.80 | 2026-03-13 12:15:00 | 2952.73 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-02-19 11:15:00 | 3449.10 | 2026-03-13 13:15:00 | 2931.73 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-02-16 12:15:00 | 3473.80 | 2026-04-08 09:15:00 | 3221.60 | STOP_HIT | 0.50 | 7.26% |
| SELL | retest2 | 2026-02-19 11:15:00 | 3449.10 | 2026-04-08 09:15:00 | 3221.60 | STOP_HIT | 0.50 | 6.60% |
