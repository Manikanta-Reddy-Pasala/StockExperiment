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
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 4 |
| ALERT3 | 8 |
| PENDING | 35 |
| PENDING_CANCEL | 13 |
| ENTRY1 | 6 |
| ENTRY2 | 15 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 17
- **Target hits / Stop hits / Partials:** 0 / 21 / 0
- **Avg / median % per leg:** -1.49% / -1.51%
- **Sum % (uncompounded):** -31.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 4 | 19.0% | 0 | 21 | 0 | -1.49% | -31.4% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 0 | 6 | 0 | -0.66% | -4.0% |
| BUY @ 3rd Alert (retest2) | 15 | 0 | 0.0% | 0 | 15 | 0 | -1.83% | -27.4% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 0 | 6 | 0 | -0.66% | -4.0% |
| retest2 (combined) | 15 | 0 | 0.0% | 0 | 15 | 0 | -1.83% | -27.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 13:15:00 | 11890.60 | 11257.22 | 11254.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 14:15:00 | 11976.20 | 11264.37 | 11258.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 09:15:00 | 11495.85 | 11550.54 | 11430.36 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 12:15:00 | 11444.40 | 11548.62 | 11431.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 11444.40 | 11548.62 | 11431.19 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-23 10:15:00 | 11519.10 | 11544.20 | 11431.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-23 11:15:00 | 11451.85 | 11543.28 | 11431.94 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-02 09:15:00 | 11515.00 | 11495.88 | 11428.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 10:15:00 | 11563.05 | 11496.55 | 11429.53 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-07 09:15:00 | 11623.25 | 11531.33 | 11454.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 10:15:00 | 11584.00 | 11531.86 | 11455.05 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-08 14:15:00 | 11408.80 | 11531.40 | 11458.98 | SL hit (close<static) qty=1.00 sl=11427.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-08 14:15:00 | 11408.80 | 11531.40 | 11458.98 | SL hit (close<static) qty=1.00 sl=11427.55 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-29 11:15:00 | 11515.00 | 11155.02 | 11242.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-29 12:15:00 | 11529.25 | 11158.74 | 11244.19 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-30 15:15:00 | 11527.90 | 11192.89 | 11257.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-31 09:15:00 | 11488.85 | 11195.84 | 11258.69 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-01-31 10:15:00 | 11521.60 | 11199.08 | 11260.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-31 11:15:00 | 11460.80 | 11201.68 | 11261.01 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-31 13:15:00 | 11523.20 | 11207.92 | 11263.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-31 14:15:00 | 11473.80 | 11210.56 | 11264.60 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-31 15:15:00 | 11599.00 | 11214.43 | 11266.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 09:15:00 | 11713.05 | 11219.39 | 11268.49 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 11113.60 | 11226.05 | 11271.13 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-02-01 12:15:00 | 11113.60 | 11226.05 | 11271.13 | SL hit (close<static) qty=1.00 sl=11427.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-01 12:15:00 | 11113.60 | 11226.05 | 11271.13 | SL hit (close<static) qty=1.00 sl=11427.55 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-05 12:15:00 | 11570.20 | 11243.42 | 11275.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 13:15:00 | 11581.50 | 11246.79 | 11276.89 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-07 09:15:00 | 11676.90 | 11270.40 | 11287.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:15:00 | 11604.60 | 11273.73 | 11289.09 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-10 10:15:00 | 11538.35 | 11297.23 | 11300.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 11:15:00 | 11568.80 | 11299.93 | 11301.88 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-10 13:15:00 | 11537.95 | 11304.75 | 11304.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-10 13:15:00 | 11537.95 | 11304.75 | 11304.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-10 13:15:00 | 11537.95 | 11304.75 | 11304.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 13:15:00 | 11537.95 | 11304.75 | 11304.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-12 12:15:00 | 11559.00 | 11312.89 | 11308.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 10:15:00 | 11287.50 | 11335.36 | 11320.42 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 10:15:00 | 11287.50 | 11335.36 | 11320.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 11287.50 | 11335.36 | 11320.42 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-17 14:15:00 | 11497.00 | 11331.28 | 11319.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-17 15:15:00 | 11464.15 | 11332.60 | 11319.79 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-19 10:15:00 | 11450.00 | 11333.91 | 11321.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-19 11:15:00 | 11395.55 | 11334.52 | 11321.38 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-02-20 15:15:00 | 11256.00 | 11330.96 | 11320.26 | SL hit (close<static) qty=1.00 sl=11265.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-25 11:15:00 | 11454.50 | 10849.83 | 10989.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 12:15:00 | 11440.35 | 10855.70 | 10992.09 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-26 09:15:00 | 11419.75 | 10878.08 | 11000.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:15:00 | 11434.80 | 10883.62 | 11002.86 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-27 09:15:00 | 11436.70 | 10915.41 | 11015.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 10:15:00 | 11526.15 | 10921.48 | 11018.01 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-02 09:15:00 | 11200.05 | 11021.35 | 11061.50 | SL hit (close<static) qty=1.00 sl=11265.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-02 09:15:00 | 11200.05 | 11021.35 | 11061.50 | SL hit (close<static) qty=1.00 sl=11265.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-02 09:15:00 | 11200.05 | 11021.35 | 11061.50 | SL hit (close<static) qty=1.00 sl=11265.10 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 11040.50 | 11093.52 | 11095.32 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-07 15:15:00 | 11288.90 | 11096.42 | 11096.72 | ENTRY2 cross detected — sustain check pending (15m) |

### Cycle 3 — BUY (started 2025-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 09:15:00 | 11395.10 | 11099.39 | 11098.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 11620.20 | 11131.71 | 11114.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 11:15:00 | 11567.00 | 11572.00 | 11385.45 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-05-05 09:15:00 | 11679.00 | 11574.80 | 11391.47 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 10:15:00 | 11694.00 | 11575.98 | 11392.98 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-06 13:15:00 | 11659.00 | 11582.64 | 11405.34 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 14:15:00 | 11697.00 | 11583.78 | 11406.79 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-07 11:15:00 | 11663.00 | 11585.46 | 11411.15 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-07 12:15:00 | 11650.00 | 11586.11 | 11412.34 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-07 14:15:00 | 11664.00 | 11587.30 | 11414.67 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-07 15:15:00 | 11650.00 | 11587.93 | 11415.84 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-08 09:15:00 | 11673.00 | 11588.77 | 11417.12 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-08 10:15:00 | 11636.00 | 11589.24 | 11418.22 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-08 12:15:00 | 11669.00 | 11590.70 | 11420.65 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-08 13:15:00 | 11621.00 | 11591.01 | 11421.65 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 11369.00 | 11588.84 | 11423.09 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 11369.00 | 11588.84 | 11423.09 | SL hit (close<ema400) qty=1.00 sl=11423.09 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 11369.00 | 11588.84 | 11423.09 | SL hit (close<ema400) qty=1.00 sl=11423.09 alert=retest1 |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 11603.00 | 11576.25 | 11422.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 11625.00 | 11576.74 | 11423.38 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-28 11:15:00 | 11296.00 | 11646.41 | 11522.19 | SL hit (close<static) qty=1.00 sl=11327.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 13:15:00 | 11453.00 | 11430.33 | 11430.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 13:15:00 | 11493.00 | 11411.31 | 11420.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 09:15:00 | 11409.00 | 11412.86 | 11420.99 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 11409.00 | 11412.86 | 11420.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 11409.00 | 11412.86 | 11420.99 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-24 09:15:00 | 11731.00 | 11418.06 | 11422.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 10:15:00 | 11802.00 | 11421.88 | 11424.20 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 11767.00 | 11429.13 | 11427.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 11767.00 | 11429.13 | 11427.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 11893.00 | 11458.38 | 11442.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 09:15:00 | 12186.00 | 12206.86 | 11965.91 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-07-30 09:15:00 | 12291.00 | 12208.15 | 11974.84 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 10:15:00 | 12298.00 | 12209.04 | 11976.45 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-31 11:15:00 | 12263.00 | 12214.18 | 11988.21 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 12:15:00 | 12298.00 | 12215.01 | 11989.76 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-04 13:15:00 | 12289.00 | 12212.18 | 12004.60 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-04 14:15:00 | 12245.00 | 12212.51 | 12005.80 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-05 09:15:00 | 12327.00 | 12214.02 | 12008.61 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-05 10:15:00 | 12324.00 | 12215.11 | 12010.19 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-07 14:15:00 | 12269.00 | 12221.22 | 12031.11 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-07 15:15:00 | 12292.00 | 12221.92 | 12032.41 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 12390.00 | 12534.92 | 12353.43 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-09-12 11:15:00 | 12465.00 | 12533.09 | 12354.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-12 12:15:00 | 12420.00 | 12531.96 | 12354.65 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-09-12 13:15:00 | 12353.00 | 12530.18 | 12354.64 | SL hit (close<ema400) qty=1.00 sl=12354.64 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-12 13:15:00 | 12353.00 | 12530.18 | 12354.64 | SL hit (close<ema400) qty=1.00 sl=12354.64 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-12 13:15:00 | 12353.00 | 12530.18 | 12354.64 | SL hit (close<ema400) qty=1.00 sl=12354.64 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-12 13:15:00 | 12353.00 | 12530.18 | 12354.64 | SL hit (close<ema400) qty=1.00 sl=12354.64 alert=retest1 |
| Cross detected — sustain check pending | 2025-09-15 12:15:00 | 12456.00 | 12521.43 | 12355.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 13:15:00 | 12454.00 | 12520.76 | 12355.87 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-16 09:15:00 | 12520.00 | 12518.87 | 12357.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 10:15:00 | 12515.00 | 12518.84 | 12358.16 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-23 12:15:00 | 12433.00 | 12544.49 | 12399.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-23 13:15:00 | 12419.00 | 12543.24 | 12399.79 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-23 15:15:00 | 12470.00 | 12541.10 | 12400.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-24 09:15:00 | 12354.00 | 12539.24 | 12399.91 | ENTRY2 sustain failed after 1080m |
| Stop hit — per-position SL triggered | 2025-09-24 10:15:00 | 12315.00 | 12537.01 | 12399.49 | SL hit (close<static) qty=1.00 sl=12343.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 10:15:00 | 12315.00 | 12537.01 | 12399.49 | SL hit (close<static) qty=1.00 sl=12343.00 alert=retest2 |

### Cycle 6 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 12304.00 | 11877.31 | 11875.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 12343.00 | 11881.94 | 11878.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 12669.00 | 12709.64 | 12461.45 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 12469.00 | 12706.33 | 12468.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 12469.00 | 12706.33 | 12468.32 | EMA400 retest candle locked |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-01-02 10:15:00 | 11563.05 | 2025-01-08 14:15:00 | 11408.80 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-01-07 10:15:00 | 11584.00 | 2025-01-08 14:15:00 | 11408.80 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-01-29 12:15:00 | 11529.25 | 2025-02-01 12:15:00 | 11113.60 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2025-02-01 09:15:00 | 11713.05 | 2025-02-01 12:15:00 | 11113.60 | STOP_HIT | 1.00 | -5.12% |
| BUY | retest2 | 2025-02-05 13:15:00 | 11581.50 | 2025-02-10 13:15:00 | 11537.95 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-02-07 10:15:00 | 11604.60 | 2025-02-10 13:15:00 | 11537.95 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-02-10 11:15:00 | 11568.80 | 2025-02-10 13:15:00 | 11537.95 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-02-17 15:15:00 | 11464.15 | 2025-02-20 15:15:00 | 11256.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-03-25 12:15:00 | 11440.35 | 2025-04-02 09:15:00 | 11200.05 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-03-26 10:15:00 | 11434.80 | 2025-04-02 09:15:00 | 11200.05 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-03-27 10:15:00 | 11526.15 | 2025-04-02 09:15:00 | 11200.05 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest1 | 2025-05-05 10:15:00 | 11694.00 | 2025-05-09 09:15:00 | 11369.00 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest1 | 2025-05-06 14:15:00 | 11697.00 | 2025-05-09 09:15:00 | 11369.00 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2025-05-12 10:15:00 | 11625.00 | 2025-05-28 11:15:00 | 11296.00 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2025-06-24 10:15:00 | 11802.00 | 2025-06-24 12:15:00 | 11767.00 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-07-30 10:15:00 | 12298.00 | 2025-09-12 13:15:00 | 12353.00 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest1 | 2025-07-31 12:15:00 | 12298.00 | 2025-09-12 13:15:00 | 12353.00 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest1 | 2025-08-05 10:15:00 | 12324.00 | 2025-09-12 13:15:00 | 12353.00 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest1 | 2025-08-07 15:15:00 | 12292.00 | 2025-09-12 13:15:00 | 12353.00 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2025-09-15 13:15:00 | 12454.00 | 2025-09-24 10:15:00 | 12315.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-09-16 10:15:00 | 12515.00 | 2025-09-24 10:15:00 | 12315.00 | STOP_HIT | 1.00 | -1.60% |
