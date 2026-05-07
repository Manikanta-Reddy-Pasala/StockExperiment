# SHRIRAMFIN (SHRIRAMFIN)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1014.50
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 2 |
| ALERT3 | 3 |
| PENDING | 11 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 8
- **Target hits / Stop hits / Partials:** 0 / 9 / 1
- **Avg / median % per leg:** -0.78% / -2.23%
- **Sum % (uncompounded):** -7.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 2 | 20.0% | 0 | 9 | 1 | -0.78% | -7.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 2 | 20.0% | 0 | 9 | 1 | -0.78% | -7.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 10 | 2 | 20.0% | 0 | 9 | 1 | -0.78% | -7.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 575.32 | 639.27 | 639.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 11:15:00 | 572.28 | 638.60 | 639.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 617.40 | 611.32 | 622.17 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 11:15:00 | 618.73 | 611.46 | 622.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 618.73 | 611.46 | 622.13 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-05 10:15:00 | 616.78 | 614.26 | 622.59 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-05 11:15:00 | 622.53 | 614.34 | 622.59 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-17 09:15:00 | 606.43 | 622.29 | 625.11 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 11:15:00 | 604.35 | 621.97 | 624.92 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-01-20 09:15:00 | 513.70 | 575.78 | 593.89 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 558.30 | 555.06 | 577.51 | SL hit (close>ema200) qty=0.50 sl=555.06 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-07 09:15:00 | 606.40 | 635.05 | 614.80 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 11:15:00 | 591.90 | 634.33 | 614.64 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-04-08 09:15:00 | 644.40 | 633.50 | 614.71 | SL hit (close>static) qty=1.00 sl=622.97 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-28 09:15:00 | 606.90 | 656.81 | 634.04 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-28 10:15:00 | 620.35 | 656.45 | 633.98 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-29 09:15:00 | 616.55 | 654.48 | 633.64 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:15:00 | 611.95 | 653.68 | 633.45 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-04-30 10:15:00 | 614.50 | 651.44 | 632.91 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 12:15:00 | 614.25 | 650.69 | 632.72 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-05-05 10:15:00 | 627.15 | 646.10 | 631.40 | SL hit (close>static) qty=1.00 sl=622.97 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-05 10:15:00 | 627.15 | 646.10 | 631.40 | SL hit (close>static) qty=1.00 sl=622.97 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 620.50 | 644.94 | 631.25 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-08 14:15:00 | 616.40 | 641.93 | 630.91 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 09:15:00 | 603.35 | 641.23 | 630.66 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1140m) |
| Stop hit — per-position SL triggered | 2025-05-12 10:15:00 | 634.00 | 638.78 | 629.82 | SL hit (close>static) qty=1.00 sl=631.85 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-25 09:15:00 | 613.05 | 664.17 | 661.11 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 11:15:00 | 605.85 | 663.02 | 660.57 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 637.25 | 660.72 | 659.46 | SL hit (close>static) qty=1.00 sl=631.85 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 12:15:00 | 637.50 | 658.16 | 658.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 09:15:00 | 631.25 | 657.32 | 657.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 610.80 | 609.79 | 624.69 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 626.85 | 610.58 | 624.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 626.85 | 610.58 | 624.58 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-09-16 11:15:00 | 617.10 | 612.92 | 624.73 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:15:00 | 615.50 | 612.97 | 624.63 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-09-18 10:15:00 | 629.20 | 613.99 | 624.53 | SL hit (close>static) qty=1.00 sl=627.65 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-25 10:15:00 | 618.50 | 618.39 | 625.25 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:15:00 | 615.40 | 618.35 | 625.16 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-09-30 11:15:00 | 618.70 | 617.27 | 623.94 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:15:00 | 612.90 | 617.20 | 623.84 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 627.95 | 617.30 | 623.79 | SL hit (close>static) qty=1.00 sl=627.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 627.95 | 617.30 | 623.79 | SL hit (close>static) qty=1.00 sl=627.65 alert=retest2 |

### Cycle 3 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 870.90 | 975.53 | 975.58 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-12-17 11:15:00 | 604.35 | 2025-01-20 09:15:00 | 513.70 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-12-17 11:15:00 | 604.35 | 2025-01-30 09:15:00 | 558.30 | STOP_HIT | 0.50 | 7.62% |
| SELL | retest2 | 2025-04-07 11:15:00 | 591.90 | 2025-04-08 09:15:00 | 644.40 | STOP_HIT | 1.00 | -8.87% |
| SELL | retest2 | 2025-04-29 11:15:00 | 611.95 | 2025-05-05 10:15:00 | 627.15 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-04-30 12:15:00 | 614.25 | 2025-05-05 10:15:00 | 627.15 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-05-09 09:15:00 | 603.35 | 2025-05-12 10:15:00 | 634.00 | STOP_HIT | 1.00 | -5.08% |
| SELL | retest2 | 2025-07-25 11:15:00 | 605.85 | 2025-07-28 09:15:00 | 637.25 | STOP_HIT | 1.00 | -5.18% |
| SELL | retest2 | 2025-09-16 13:15:00 | 615.50 | 2025-09-18 10:15:00 | 629.20 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-09-25 12:15:00 | 615.40 | 2025-10-01 09:15:00 | 627.95 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-09-30 13:15:00 | 612.90 | 2025-10-01 09:15:00 | 627.95 | STOP_HIT | 1.00 | -2.46% |
