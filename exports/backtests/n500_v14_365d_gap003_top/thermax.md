# Thermax Ltd. (THERMAX)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 4707.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 1 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 2 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 0
- **Target hits / Stop hits / Partials:** 0 / 2 / 2
- **Avg / median % per leg:** 4.64% / 5.00%
- **Sum % (uncompounded):** 18.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 4 | 100.0% | 0 | 2 | 2 | 4.64% | 18.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 0 | 2 | 2 | 4.64% | 18.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 4 | 4 | 100.0% | 0 | 2 | 2 | 4.64% | 18.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 3271.10 | 3535.76 | 3537.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 12:15:00 | 3261.60 | 3486.36 | 3510.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 3361.00 | 3360.79 | 3425.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 13:00:00 | 3361.00 | 3360.79 | 3425.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 3403.00 | 3358.85 | 3421.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:15:00 | 3377.90 | 3359.08 | 3421.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:00:00 | 3374.00 | 3359.50 | 3420.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 3209.01 | 3335.00 | 3386.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 3205.30 | 3335.00 | 3386.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 3231.60 | 3221.63 | 3291.11 | SL hit (close>ema200) qty=0.50 sl=3221.63 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 3231.60 | 3221.63 | 3291.11 | SL hit (close>ema200) qty=0.50 sl=3221.63 alert=retest2 |
| CROSSOVER_SKIP | 2026-02-25 12:15:00 | 3185.60 | 2983.29 | 2982.50 | min_gap filter: gap=0.025% < 0.030% |
| TREND_RESET | 2026-02-25 12:15:00 | 3185.60 | 2983.29 | 2982.50 | EMA inversion without crossover edge (EMA200=2983.29 EMA400=2982.50) — end cycle |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-10 11:15:00 | 3377.90 | 2025-09-26 09:15:00 | 3209.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-10 13:00:00 | 3374.00 | 2025-09-26 09:15:00 | 3205.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-10 11:15:00 | 3377.90 | 2025-10-23 10:15:00 | 3231.60 | STOP_HIT | 0.50 | 4.33% |
| SELL | retest2 | 2025-09-10 13:00:00 | 3374.00 | 2025-10-23 10:15:00 | 3231.60 | STOP_HIT | 0.50 | 4.22% |
