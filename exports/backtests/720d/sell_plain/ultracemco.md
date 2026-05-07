# ULTRACEMCO (ULTRACEMCO)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 12121.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 6 |
| ALERT3 | 6 |
| PENDING | 14 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 7
- **Target hits / Stop hits / Partials:** 0 / 10 / 0
- **Avg / median % per leg:** 0.17% / -1.91%
- **Sum % (uncompounded):** 1.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 3 | 30.0% | 0 | 10 | 0 | 0.17% | 1.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 3 | 30.0% | 0 | 10 | 0 | 0.17% | 1.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 10 | 3 | 30.0% | 0 | 10 | 0 | 0.17% | 1.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 09:15:00 | 10758.55 | 11388.89 | 11391.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 10:15:00 | 10611.50 | 11111.87 | 11218.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 11151.90 | 11058.57 | 11179.76 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 10:15:00 | 11187.00 | 11059.85 | 11179.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 11187.00 | 11059.85 | 11179.80 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-11-27 09:15:00 | 11043.25 | 11105.33 | 11192.56 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 11:15:00 | 10960.00 | 11102.85 | 11190.44 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-11-28 10:15:00 | 11060.00 | 11103.50 | 11188.18 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 12:15:00 | 11020.95 | 11102.69 | 11186.94 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-11-29 14:15:00 | 11203.65 | 11102.65 | 11183.17 | SL hit (close>static) qty=1.00 sl=11198.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-29 14:15:00 | 11203.65 | 11102.65 | 11183.17 | SL hit (close>static) qty=1.00 sl=11198.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-10 09:15:00 | 11029.75 | 11506.92 | 11449.68 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 11:15:00 | 11016.10 | 11497.65 | 11445.60 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-01-14 11:15:00 | 10560.75 | 11396.74 | 11397.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 11:15:00 | 10560.75 | 11396.74 | 11397.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 12:15:00 | 10525.35 | 11388.07 | 11392.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 13:15:00 | 11249.85 | 11111.81 | 11234.70 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 13:15:00 | 11249.85 | 11111.81 | 11234.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 11249.85 | 11111.81 | 11234.70 | EMA400 retest candle locked |

### Cycle 3 — SELL (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 12:15:00 | 11127.10 | 11309.08 | 11309.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 11043.25 | 11306.44 | 11308.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 11:15:00 | 10832.10 | 10797.50 | 10987.31 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 10884.00 | 10802.38 | 10985.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 10884.00 | 10802.38 | 10985.09 | EMA400 retest candle locked |

### Cycle 4 — SELL (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 14:15:00 | 11259.00 | 11430.13 | 11430.21 | EMA200 below EMA400 |

### Cycle 5 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 11321.00 | 11429.41 | 11429.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 11143.00 | 11425.27 | 11427.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 11437.00 | 11410.21 | 11419.89 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 11:15:00 | 11437.00 | 11410.21 | 11419.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 11437.00 | 11410.21 | 11419.89 | EMA400 retest candle locked |

### Cycle 6 — SELL (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 15:15:00 | 12163.00 | 12306.12 | 12306.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 12136.00 | 12294.56 | 12300.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 13:15:00 | 12288.00 | 12278.51 | 12292.10 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 13:15:00 | 12288.00 | 12278.51 | 12292.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 12288.00 | 12278.51 | 12292.10 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-23 13:15:00 | 12190.00 | 12290.03 | 12296.51 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 15:15:00 | 12112.00 | 12286.65 | 12294.75 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-01-14 15:15:00 | 12242.00 | 11868.89 | 11871.77 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-01-16 09:15:00 | 12284.00 | 11873.02 | 11873.83 | ENTRY2 sustain failed after 2520m |
| Stop hit — per-position SL triggered | 2026-01-16 11:15:00 | 12343.00 | 11881.94 | 11878.30 | SL hit (close>static) qty=1.00 sl=12305.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-19 15:15:00 | 12261.00 | 11929.85 | 11903.13 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-01-20 09:15:00 | 12360.00 | 11934.13 | 11905.41 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-01-20 12:15:00 | 12235.00 | 11944.16 | 11910.89 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 14:15:00 | 12045.00 | 11947.29 | 11912.80 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 12375.00 | 11967.79 | 11924.75 | SL hit (close>static) qty=1.00 sl=12305.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-02 10:15:00 | 12212.00 | 12202.83 | 12065.97 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-02-02 11:15:00 | 12348.00 | 12204.28 | 12067.38 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-04 09:15:00 | 11979.00 | 12684.92 | 12465.65 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 11:15:00 | 11920.00 | 12669.95 | 12460.31 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-03-05 15:15:00 | 12260.00 | 12613.18 | 12442.31 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 12154.00 | 12608.61 | 12440.87 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2026-03-12 15:15:00 | 11118.00 | 12296.55 | 12299.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 15:15:00 | 11118.00 | 12296.55 | 12299.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 15:15:00 | 11118.00 | 12296.55 | 12299.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 10762.00 | 12281.28 | 12291.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 11620.00 | 11358.66 | 11710.19 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 10:15:00 | 11736.00 | 11362.41 | 11710.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 11736.00 | 11362.41 | 11710.32 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-08 14:15:00 | 11576.00 | 11374.95 | 11709.76 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-08 15:15:00 | 11590.00 | 11377.09 | 11709.16 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 11452.00 | 11377.84 | 11707.88 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 11:15:00 | 11478.00 | 11379.78 | 11705.57 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-10 15:15:00 | 11562.00 | 11394.71 | 11695.76 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 11325.00 | 11394.02 | 11693.91 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2026-04-15 12:15:00 | 11808.00 | 11413.15 | 11689.00 | SL hit (close>static) qty=1.00 sl=11768.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 12:15:00 | 11808.00 | 11413.15 | 11689.00 | SL hit (close>static) qty=1.00 sl=11768.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-30 09:15:00 | 11528.00 | 11703.07 | 11774.58 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 11:15:00 | 11549.00 | 11699.76 | 11772.21 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 11779.00 | 11696.99 | 11766.50 | SL hit (close>static) qty=1.00 sl=11768.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-27 11:15:00 | 10960.00 | 2024-11-29 14:15:00 | 11203.65 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2024-11-28 12:15:00 | 11020.95 | 2024-11-29 14:15:00 | 11203.65 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-01-10 11:15:00 | 11016.10 | 2025-01-14 11:15:00 | 10560.75 | STOP_HIT | 1.00 | 4.13% |
| SELL | retest2 | 2025-10-23 15:15:00 | 12112.00 | 2026-01-16 11:15:00 | 12343.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2026-01-20 14:15:00 | 12045.00 | 2026-01-22 09:15:00 | 12375.00 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2026-03-04 11:15:00 | 11920.00 | 2026-03-12 15:15:00 | 11118.00 | STOP_HIT | 1.00 | 6.73% |
| SELL | retest2 | 2026-03-06 09:15:00 | 12154.00 | 2026-03-12 15:15:00 | 11118.00 | STOP_HIT | 1.00 | 8.52% |
| SELL | retest2 | 2026-04-09 11:15:00 | 11478.00 | 2026-04-15 12:15:00 | 11808.00 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2026-04-13 09:15:00 | 11325.00 | 2026-04-15 12:15:00 | 11808.00 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2026-04-30 11:15:00 | 11549.00 | 2026-05-05 09:15:00 | 11779.00 | STOP_HIT | 1.00 | -1.99% |
