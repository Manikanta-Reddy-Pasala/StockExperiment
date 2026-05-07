# MARUTI (MARUTI)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 13770.00
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
| ALERT3 | 3 |
| PENDING | 19 |
| PENDING_CANCEL | 8 |
| ENTRY1 | 7 |
| ENTRY2 | 4 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 0 / 9
- **Target hits / Stop hits / Partials:** 0 / 9 / 0
- **Avg / median % per leg:** -3.26% / -2.61%
- **Sum % (uncompounded):** -29.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -3.26% | -29.4% |
| SELL @ 2nd Alert (retest1) | 7 | 0 | 0.0% | 0 | 7 | 0 | -3.66% | -25.6% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.85% | -3.7% |
| retest1 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -3.66% | -25.6% |
| retest2 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.85% | -3.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 12:15:00 | 11985.70 | 12502.67 | 12504.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 13:15:00 | 11966.30 | 12497.33 | 12502.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 12:15:00 | 11331.15 | 11317.54 | 11640.54 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-12-10 11:15:00 | 11235.30 | 11313.82 | 11618.32 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 13:15:00 | 11212.75 | 11311.86 | 11614.30 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2024-12-12 09:15:00 | 11185.40 | 11306.99 | 11597.05 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 11:15:00 | 11140.75 | 11303.50 | 11592.41 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2024-12-16 10:15:00 | 11246.10 | 11291.52 | 11567.98 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-16 11:15:00 | 11268.15 | 11291.29 | 11566.49 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-16 13:15:00 | 11248.35 | 11290.71 | 11563.46 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-16 14:15:00 | 11267.70 | 11290.48 | 11561.98 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-17 09:15:00 | 11196.75 | 11289.30 | 11558.69 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 11:15:00 | 11140.00 | 11286.72 | 11554.72 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 11431.00 | 11099.95 | 11360.55 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-01-02 09:15:00 | 11431.00 | 11099.95 | 11360.55 | SL hit (close>ema400) qty=1.00 sl=11360.55 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-01-02 09:15:00 | 11431.00 | 11099.95 | 11360.55 | SL hit (close>ema400) qty=1.00 sl=11360.55 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-01-02 09:15:00 | 11431.00 | 11099.95 | 11360.55 | SL hit (close>ema400) qty=1.00 sl=11360.55 alert=retest1 |

### Cycle 2 — SELL (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-18 09:15:00 | 11624.80 | 12001.93 | 12003.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 13:15:00 | 11454.35 | 11891.92 | 11939.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 11807.00 | 11721.66 | 11830.03 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-15 09:15:00 | 11807.00 | 11721.66 | 11830.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 11807.00 | 11721.66 | 11830.03 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-16 09:15:00 | 11678.00 | 11730.18 | 11830.66 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-16 10:15:00 | 11701.00 | 11729.89 | 11830.01 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-16 11:15:00 | 11666.00 | 11729.26 | 11829.19 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-16 13:15:00 | 11679.00 | 11727.89 | 11827.51 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-04-17 15:15:00 | 11665.00 | 11721.85 | 11820.03 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 11664.00 | 11721.27 | 11819.25 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 5400m) |
| Cross detected — sustain check pending | 2025-04-22 09:15:00 | 11631.00 | 11720.96 | 11815.72 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-22 11:15:00 | 11694.00 | 11720.07 | 11814.34 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-04-22 12:15:00 | 11676.00 | 11719.63 | 11813.65 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-22 13:15:00 | 11740.00 | 11719.84 | 11813.28 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-04-23 14:15:00 | 11888.00 | 11725.64 | 11812.56 | SL hit (close>static) qty=1.00 sl=11862.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-23 14:15:00 | 11888.00 | 11725.64 | 11812.56 | SL hit (close>static) qty=1.00 sl=11862.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-25 15:15:00 | 11650.00 | 11743.02 | 11815.41 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-28 09:15:00 | 11692.00 | 11742.52 | 11814.79 | ENTRY2 sustain failed after 3960m |

### Cycle 3 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 14511.00 | 16031.42 | 16031.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 09:15:00 | 14425.00 | 15985.22 | 16008.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 13626.00 | 13304.14 | 14062.68 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 13180.00 | 13365.71 | 14019.53 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 11:15:00 | 13174.00 | 13361.50 | 14010.91 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-23 10:15:00 | 13183.00 | 13370.74 | 13876.78 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-23 11:15:00 | 13219.00 | 13369.23 | 13873.50 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-23 12:15:00 | 13163.00 | 13367.18 | 13869.96 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 14:15:00 | 13150.00 | 13362.70 | 13862.70 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-28 11:15:00 | 13136.00 | 13325.55 | 13800.12 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 13:15:00 | 13176.00 | 13322.06 | 13793.64 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-30 09:15:00 | 13015.00 | 13316.07 | 13767.55 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 11:15:00 | 13034.00 | 13310.55 | 13760.29 | SELL ENTRY1 attempt 4/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 13748.00 | 13312.84 | 13750.31 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-05-04 11:15:00 | 13525.00 | 13318.99 | 13749.05 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-05-04 13:15:00 | 13549.00 | 13323.08 | 13746.82 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2026-05-05 09:15:00 | 13448.00 | 13329.29 | 13743.65 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 11:15:00 | 13482.00 | 13331.45 | 13740.61 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-05-06 10:15:00 | 13433.00 | 13340.03 | 13732.87 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:15:00 | 13476.00 | 13343.13 | 13730.52 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 13740.00 | 13351.78 | 13729.11 | SL hit (close>ema400) qty=1.00 sl=13729.11 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 13740.00 | 13351.78 | 13729.11 | SL hit (close>ema400) qty=1.00 sl=13729.11 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 13740.00 | 13351.78 | 13729.11 | SL hit (close>ema400) qty=1.00 sl=13729.11 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 13740.00 | 13351.78 | 13729.11 | SL hit (close>ema400) qty=1.00 sl=13729.11 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-12-10 13:15:00 | 11212.75 | 2025-01-02 09:15:00 | 11431.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest1 | 2024-12-12 11:15:00 | 11140.75 | 2025-01-02 09:15:00 | 11431.00 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest1 | 2024-12-17 11:15:00 | 11140.00 | 2025-01-02 09:15:00 | 11431.00 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2025-04-16 13:15:00 | 11679.00 | 2025-04-23 14:15:00 | 11888.00 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-04-21 09:15:00 | 11664.00 | 2025-04-23 14:15:00 | 11888.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest1 | 2026-04-13 11:15:00 | 13174.00 | 2026-05-06 15:15:00 | 13740.00 | STOP_HIT | 1.00 | -4.30% |
| SELL | retest1 | 2026-04-23 14:15:00 | 13150.00 | 2026-05-06 15:15:00 | 13740.00 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest1 | 2026-04-28 13:15:00 | 13176.00 | 2026-05-06 15:15:00 | 13740.00 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest1 | 2026-04-30 11:15:00 | 13034.00 | 2026-05-06 15:15:00 | 13740.00 | STOP_HIT | 1.00 | -5.42% |
