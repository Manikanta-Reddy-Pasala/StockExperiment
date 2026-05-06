# ULTRACEMCO (ULTRACEMCO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 12093.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 6 |
| ALERT3 | 9 |
| PENDING | 30 |
| PENDING_CANCEL | 13 |
| ENTRY1 | 7 |
| ENTRY2 | 10 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 12
- **Target hits / Stop hits / Partials:** 0 / 17 / 1
- **Avg / median % per leg:** 0.84% / -0.93%
- **Sum % (uncompounded):** 15.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 6 | 33.3% | 0 | 17 | 1 | 0.84% | 15.1% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 0 | 7 | 0 | -0.72% | -5.0% |
| BUY @ 3rd Alert (retest2) | 11 | 2 | 18.2% | 0 | 10 | 1 | 1.83% | 20.1% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 0 | 7 | 0 | -0.72% | -5.0% |
| retest2 (combined) | 11 | 2 | 18.2% | 0 | 10 | 1 | 1.83% | 20.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 13:15:00 | 9993.70 | 9734.84 | 9734.36 | EMA200 above EMA400 |
| CROSSOVER_SKIP | 2024-05-08 14:15:00 | 9514.15 | 9734.36 | 9734.91 | HTF filter: close above htf_sma |
| First Alert — break + close above crossover candle high | 2024-05-23 13:15:00 | 10051.25 | 9720.86 | 9722.24 | Break + close above crossover candle high |

### Cycle 2 — BUY (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 14:15:00 | 10165.75 | 9725.28 | 9724.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 09:15:00 | 10226.25 | 9734.65 | 9729.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 9842.95 | 9900.58 | 9824.15 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 12:15:00 | 10020.10 | 9901.77 | 9825.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 10020.10 | 9901.77 | 9825.12 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-06-06 10:15:00 | 10145.75 | 9911.36 | 9834.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-06 11:15:00 | 10034.30 | 9912.58 | 9835.48 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-07 09:15:00 | 10126.45 | 9919.49 | 9840.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 10:15:00 | 10220.90 | 9922.49 | 9842.77 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-06-27 09:15:00 | 11754.04 | 10542.69 | 10248.55 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2024-10-23 10:15:00 | 10776.45 | 11382.30 | 11384.56 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2024-12-05 13:15:00 | 11890.60 | 11257.79 | 11254.65 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2024-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 13:15:00 | 11890.60 | 11257.79 | 11254.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 14:15:00 | 11977.85 | 11264.96 | 11258.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 09:15:00 | 11495.85 | 11551.06 | 11430.23 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 12:15:00 | 11444.40 | 11549.17 | 11431.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 11444.40 | 11549.17 | 11431.08 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-23 10:15:00 | 11519.25 | 11544.34 | 11431.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-23 11:15:00 | 11453.70 | 11543.44 | 11431.66 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-02 09:15:00 | 11517.60 | 11496.08 | 11428.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 10:15:00 | 11563.05 | 11496.75 | 11429.37 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-07 09:15:00 | 11623.25 | 11531.56 | 11454.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 10:15:00 | 11583.25 | 11532.07 | 11454.93 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-08 12:15:00 | 11427.55 | 11533.86 | 11459.27 | SL hit qty=1.00 sl=11427.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-08 12:15:00 | 11427.55 | 11533.86 | 11459.27 | SL hit qty=1.00 sl=11427.55 alert=retest2 |
| CROSSOVER_SKIP | 2025-01-14 11:15:00 | 10561.45 | 11396.93 | 11397.00 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2025-01-29 11:15:00 | 11515.00 | 11155.18 | 11242.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-29 12:15:00 | 11534.65 | 11158.95 | 11244.16 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-31 09:15:00 | 11427.55 | 11195.52 | 11258.41 | SL hit qty=1.00 sl=11427.55 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-31 10:15:00 | 11521.60 | 11198.77 | 11259.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-31 11:15:00 | 11460.80 | 11201.38 | 11260.72 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-31 13:15:00 | 11520.00 | 11207.59 | 11263.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-31 14:15:00 | 11473.80 | 11210.24 | 11264.31 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-05 11:15:00 | 11524.35 | 11228.25 | 11268.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 12:15:00 | 11565.80 | 11231.61 | 11270.24 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-06 09:15:00 | 11427.55 | 11243.78 | 11275.63 | SL hit qty=1.00 sl=11427.55 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 12:15:00 | 11274.10 | 11303.38 | 11303.71 | EMA400 retest candle locked |

### Cycle 4 — BUY (started 2025-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 12:15:00 | 11559.00 | 11304.75 | 11304.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 09:15:00 | 11590.00 | 11313.09 | 11308.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 10:15:00 | 11287.50 | 11328.66 | 11316.75 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 10:15:00 | 11287.50 | 11328.66 | 11316.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 11287.50 | 11328.66 | 11316.75 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-17 14:15:00 | 11497.00 | 11325.46 | 11315.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-17 15:15:00 | 11470.30 | 11326.90 | 11316.45 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-18 13:15:00 | 11269.45 | 11326.98 | 11316.76 | SL hit qty=1.00 sl=11269.45 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-19 10:15:00 | 11450.00 | 11328.90 | 11317.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-19 11:15:00 | 11395.60 | 11329.57 | 11318.31 | ENTRY2 sustain failed after 60m |

### Cycle 5 — SELL (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 11:15:00 | 11145.40 | 11307.19 | 11307.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 11043.25 | 11302.79 | 11305.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 11:15:00 | 10832.10 | 10797.16 | 10986.36 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 09:15:00 | 10921.85 | 10808.59 | 10981.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 10921.85 | 10808.59 | 10981.21 | EMA400 retest candle locked |

### Cycle 6 — BUY (started 2025-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 09:15:00 | 11395.10 | 11098.50 | 11097.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 11620.15 | 11131.19 | 11114.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 11:15:00 | 11567.00 | 11572.00 | 11385.11 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-05-05 09:15:00 | 11679.00 | 11574.67 | 11391.07 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 10:15:00 | 11693.00 | 11575.85 | 11392.57 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-06 14:15:00 | 11697.00 | 11583.79 | 11406.48 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 15:15:00 | 11660.00 | 11584.55 | 11407.74 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-07 11:15:00 | 11663.00 | 11585.47 | 11410.83 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-07 12:15:00 | 11650.00 | 11586.11 | 11412.03 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-07 14:15:00 | 11664.00 | 11587.31 | 11414.36 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-07 15:15:00 | 11658.00 | 11588.02 | 11415.58 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-08 12:15:00 | 11671.00 | 11590.78 | 11420.39 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-08 13:15:00 | 11621.00 | 11591.08 | 11421.39 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 11369.00 | 11589.82 | 11423.28 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 11423.28 | 11589.82 | 11423.28 | SL hit qty=1.00 sl=11423.28 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 11423.28 | 11589.82 | 11423.28 | SL hit qty=1.00 sl=11423.28 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 11423.28 | 11589.82 | 11423.28 | SL hit qty=1.00 sl=11423.28 alert=retest1 |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 11603.00 | 11577.08 | 11422.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 11625.00 | 11577.55 | 11423.52 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-28 11:15:00 | 11330.00 | 11647.05 | 11522.40 | SL hit qty=1.00 sl=11330.00 alert=retest2 |
| CROSSOVER_SKIP | 2025-06-09 15:15:00 | 11266.00 | 11429.40 | 11429.88 | slope filter: EMA200 not falling 0.50% over 350 bars |

### Cycle 7 — BUY (started 2025-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 11:15:00 | 11493.00 | 11430.46 | 11430.39 | EMA200 above EMA400 |
| CROSSOVER_SKIP | 2025-06-12 14:15:00 | 11321.00 | 11430.14 | 11430.26 | slope filter: EMA200 not falling 0.50% over 350 bars |
| First Alert — break + close above crossover candle high | 2025-06-16 14:15:00 | 11510.00 | 11413.03 | 11421.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 09:15:00 | 11409.00 | 11413.80 | 11421.55 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 11409.00 | 11413.80 | 11421.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 11409.00 | 11413.80 | 11421.55 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-19 09:15:00 | 11461.00 | 11412.68 | 11420.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-19 10:15:00 | 11447.00 | 11413.02 | 11420.57 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-20 09:15:00 | 11472.00 | 11414.60 | 11421.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:15:00 | 11493.00 | 11415.38 | 11421.50 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-20 15:15:00 | 11458.00 | 11416.53 | 11421.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-23 09:15:00 | 11350.00 | 11415.87 | 11421.57 | ENTRY2 sustain failed after 3960m |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 11361.00 | 11415.87 | 11421.57 | SL hit qty=1.00 sl=11361.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-23 13:15:00 | 11474.00 | 11415.47 | 11421.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-23 14:15:00 | 11438.00 | 11415.69 | 11421.34 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-24 09:15:00 | 11731.00 | 11419.16 | 11423.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 10:15:00 | 11802.00 | 11422.97 | 11424.91 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 11765.00 | 11430.17 | 11428.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 11765.00 | 11430.17 | 11428.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 11893.00 | 11459.16 | 11443.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 09:15:00 | 12186.00 | 12207.48 | 11966.53 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-07-30 09:15:00 | 12291.00 | 12208.63 | 11975.39 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 10:15:00 | 12295.00 | 12209.49 | 11976.99 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-31 12:15:00 | 12295.00 | 12215.54 | 11990.33 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 13:15:00 | 12327.00 | 12216.65 | 11992.01 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-04 13:15:00 | 12289.00 | 12212.63 | 12005.13 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-04 14:15:00 | 12245.00 | 12212.96 | 12006.33 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-05 09:15:00 | 12327.00 | 12214.52 | 12009.17 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-05 10:15:00 | 12322.00 | 12215.59 | 12010.73 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-07 14:15:00 | 12271.00 | 12221.82 | 12031.71 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-07 15:15:00 | 12309.00 | 12222.69 | 12033.10 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 12391.00 | 12535.50 | 12354.00 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 12354.00 | 12535.50 | 12354.00 | SL hit qty=1.00 sl=12354.00 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 12354.00 | 12535.50 | 12354.00 | SL hit qty=1.00 sl=12354.00 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 12354.00 | 12535.50 | 12354.00 | SL hit qty=1.00 sl=12354.00 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 12354.00 | 12535.50 | 12354.00 | SL hit qty=1.00 sl=12354.00 alert=retest1 |
| Cross detected — sustain check pending | 2025-09-12 11:15:00 | 12466.00 | 12533.67 | 12354.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-12 12:15:00 | 12420.00 | 12532.54 | 12355.21 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-15 12:15:00 | 12455.00 | 12522.12 | 12356.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 13:15:00 | 12454.00 | 12521.44 | 12356.49 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-23 12:15:00 | 12433.00 | 12544.71 | 12400.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-23 13:15:00 | 12419.00 | 12543.46 | 12400.17 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 12346.00 | 12538.67 | 12399.89 | SL hit qty=1.00 sl=12346.00 alert=retest2 |
| CROSSOVER_SKIP | 2025-10-09 14:15:00 | 12177.00 | 12307.18 | 12307.32 | HTF filter: close above htf_sma |

### Cycle 9 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 12304.00 | 11877.28 | 11875.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 12343.00 | 11881.91 | 11877.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 12662.00 | 12704.29 | 12453.58 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 12469.00 | 12701.44 | 12460.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 12469.00 | 12701.44 | 12460.78 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2026-03-12 15:15:00 | 11089.00 | 12293.72 | 12293.83 | slope filter: EMA200 not falling 0.50% over 350 bars |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-07 10:15:00 | 10220.90 | 2024-06-27 09:15:00 | 11754.04 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-06-07 10:15:00 | 10220.90 | 2024-12-05 13:15:00 | 11890.60 | STOP_HIT | 0.50 | 16.34% |
| BUY | retest2 | 2025-01-02 10:15:00 | 11563.05 | 2025-01-08 12:15:00 | 11427.55 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-01-07 10:15:00 | 11583.25 | 2025-01-08 12:15:00 | 11427.55 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-01-29 12:15:00 | 11534.65 | 2025-01-31 09:15:00 | 11427.55 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-02-05 12:15:00 | 11565.80 | 2025-02-06 09:15:00 | 11427.55 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-02-17 15:15:00 | 11470.30 | 2025-02-18 13:15:00 | 11269.45 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest1 | 2025-05-05 10:15:00 | 11693.00 | 2025-05-09 09:15:00 | 11423.28 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest1 | 2025-05-06 15:15:00 | 11660.00 | 2025-05-09 09:15:00 | 11423.28 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest1 | 2025-05-07 15:15:00 | 11658.00 | 2025-05-09 09:15:00 | 11423.28 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-05-12 10:15:00 | 11625.00 | 2025-05-28 11:15:00 | 11330.00 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-06-20 10:15:00 | 11493.00 | 2025-06-23 09:15:00 | 11361.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-06-24 10:15:00 | 11802.00 | 2025-06-24 12:15:00 | 11765.00 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-07-30 10:15:00 | 12295.00 | 2025-09-12 09:15:00 | 12354.00 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest1 | 2025-07-31 13:15:00 | 12327.00 | 2025-09-12 09:15:00 | 12354.00 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest1 | 2025-08-05 10:15:00 | 12322.00 | 2025-09-12 09:15:00 | 12354.00 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest1 | 2025-08-07 15:15:00 | 12309.00 | 2025-09-12 09:15:00 | 12354.00 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2025-09-15 13:15:00 | 12454.00 | 2025-09-24 09:15:00 | 12346.00 | STOP_HIT | 1.00 | -0.87% |
