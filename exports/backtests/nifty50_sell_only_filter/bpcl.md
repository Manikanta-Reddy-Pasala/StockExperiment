# BPCL (BPCL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4996 bars)
- **Last close:** 314.05
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
| ALERT2_SKIP | 0 |
| ALERT3 | 6 |
| PENDING | 28 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 18 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 15
- **Target hits / Stop hits / Partials:** 1 / 22 / 5
- **Avg / median % per leg:** 3.26% / 0.00%
- **Sum % (uncompounded):** 91.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 13 | 46.4% | 1 | 22 | 5 | 3.26% | 91.4% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.63% | -18.2% |
| BUY @ 3rd Alert (retest2) | 23 | 13 | 56.5% | 1 | 17 | 5 | 4.76% | 109.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.63% | -18.2% |
| retest2 (combined) | 23 | 13 | 56.5% | 1 | 17 | 5 | 4.76% | 109.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 10:15:00 | 193.32 | 177.44 | 177.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-13 10:15:00 | 194.27 | 179.39 | 178.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 14:15:00 | 297.10 | 300.72 | 277.79 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-03-14 14:15:00 | 304.30 | 300.78 | 278.60 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-14 15:15:00 | 304.00 | 300.81 | 278.73 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 14:15:00 | 278.60 | 298.30 | 279.52 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-03-19 14:15:00 | 279.52 | 298.30 | 279.52 | SL hit qty=1.00 sl=279.52 alert=retest1 |
| Cross detected — sustain check pending | 2024-03-20 09:15:00 | 281.62 | 297.94 | 279.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 10:15:00 | 281.75 | 297.78 | 279.54 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-05-21 12:15:00 | 324.01 | 305.10 | 297.73 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 281.75 | 312.19 | 304.37 | SL hit qty=0.50 sl=281.75 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-05 10:15:00 | 282.02 | 311.89 | 304.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 11:15:00 | 281.55 | 311.59 | 304.15 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-07-25 10:15:00 | 323.78 | 307.62 | 305.75 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2024-09-02 10:15:00 | 366.01 | 339.63 | 328.81 | Target hit (30%) qty=0.50 alert=retest2 |
| CROSSOVER_SKIP | 2024-10-28 11:15:00 | 310.85 | 336.28 | 336.40 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2025-01-09 15:15:00 | 281.00 | 294.29 | 302.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-10 09:15:00 | 278.00 | 294.13 | 302.15 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-01-10 12:15:00 | 281.00 | 293.73 | 301.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-10 13:15:00 | 280.20 | 293.59 | 301.72 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-21 09:15:00 | 285.50 | 285.58 | 295.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 10:15:00 | 284.85 | 285.57 | 295.55 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-22 09:15:00 | 278.20 | 285.29 | 295.12 | SL hit qty=1.00 sl=278.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-21 13:15:00 | 282.28 | 260.96 | 267.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-03-21 14:15:00 | 279.37 | 261.14 | 267.30 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-03-24 10:15:00 | 282.49 | 261.73 | 267.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 11:15:00 | 283.26 | 261.94 | 267.58 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-25 09:15:00 | 278.20 | 262.85 | 267.90 | SL hit qty=1.00 sl=278.20 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 272.10 | 270.67 | 271.09 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-07 10:15:00 | 274.90 | 270.71 | 271.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 11:15:00 | 273.95 | 270.74 | 271.12 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-08 14:15:00 | 286.05 | 271.60 | 271.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 14:15:00 | 286.05 | 271.60 | 271.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 11:15:00 | 289.50 | 272.14 | 271.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 309.90 | 310.83 | 300.80 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-06-04 11:15:00 | 311.75 | 310.80 | 300.99 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 12:15:00 | 311.30 | 310.81 | 301.04 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-06 09:15:00 | 313.60 | 310.90 | 301.61 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:15:00 | 315.75 | 310.95 | 301.68 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 309.45 | 314.25 | 304.91 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 304.91 | 314.25 | 304.91 | SL hit qty=1.00 sl=304.91 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 304.91 | 314.25 | 304.91 | SL hit qty=1.00 sl=304.91 alert=retest1 |
| Cross detected — sustain check pending | 2025-06-13 11:15:00 | 313.15 | 314.19 | 304.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 12:15:00 | 311.80 | 314.17 | 305.01 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-16 12:15:00 | 311.50 | 313.98 | 305.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 13:15:00 | 316.25 | 314.00 | 305.28 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-23 10:15:00 | 310.75 | 313.90 | 306.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:15:00 | 311.10 | 313.87 | 306.55 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-07-08 12:15:00 | 358.57 | 324.89 | 315.13 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-07-08 12:15:00 | 357.77 | 324.89 | 315.13 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 311.80 | 333.36 | 326.51 | SL hit qty=0.50 sl=311.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 311.10 | 333.36 | 326.51 | SL hit qty=0.50 sl=311.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-07 14:15:00 | 310.35 | 329.96 | 325.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-07 15:15:00 | 310.10 | 329.77 | 325.28 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-08 09:15:00 | 316.80 | 329.64 | 325.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 10:15:00 | 317.95 | 329.52 | 325.20 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 318.40 | 329.41 | 325.17 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-11 14:15:00 | 321.10 | 328.50 | 324.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 15:15:00 | 320.95 | 328.42 | 324.89 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-14 13:15:00 | 317.70 | 327.37 | 324.66 | SL hit qty=1.00 sl=317.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-19 14:15:00 | 320.95 | 325.84 | 324.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 15:15:00 | 321.90 | 325.80 | 324.05 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-20 12:15:00 | 322.10 | 325.60 | 323.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-20 13:15:00 | 320.90 | 325.55 | 323.97 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-21 09:15:00 | 321.85 | 325.40 | 323.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:15:00 | 322.95 | 325.38 | 323.91 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 317.70 | 325.05 | 323.79 | SL hit qty=1.00 sl=317.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 317.70 | 325.05 | 323.79 | SL hit qty=1.00 sl=317.70 alert=retest2 |
| CROSSOVER_SKIP | 2025-08-28 14:15:00 | 310.55 | 322.58 | 322.64 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2025-09-11 09:15:00 | 322.90 | 318.94 | 320.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 10:15:00 | 324.60 | 319.00 | 320.48 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 319.65 | 319.07 | 320.50 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-09-12 14:15:00 | 317.70 | 319.12 | 320.47 | SL hit qty=1.00 sl=317.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-17 09:15:00 | 323.10 | 319.02 | 320.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:15:00 | 323.80 | 319.07 | 320.33 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-17 14:15:00 | 324.05 | 319.20 | 320.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 15:15:00 | 323.55 | 319.24 | 320.39 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-18 13:15:00 | 325.00 | 319.44 | 320.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:15:00 | 325.50 | 319.50 | 320.48 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 330.65 | 321.42 | 321.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 330.65 | 321.42 | 321.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 330.65 | 321.42 | 321.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 330.65 | 321.42 | 321.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 330.65 | 321.42 | 321.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 14:15:00 | 330.65 | 321.42 | 321.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 09:15:00 | 335.20 | 322.85 | 322.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 13:15:00 | 331.55 | 332.17 | 327.90 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-10-15 10:15:00 | 338.95 | 332.25 | 328.02 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 11:15:00 | 338.40 | 332.31 | 328.07 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-17 13:15:00 | 335.65 | 332.84 | 328.67 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 14:15:00 | 335.80 | 332.87 | 328.71 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 334.00 | 333.32 | 329.31 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 329.31 | 333.32 | 329.31 | SL hit qty=1.00 sl=329.31 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 329.31 | 333.32 | 329.31 | SL hit qty=1.00 sl=329.31 alert=retest1 |
| Cross detected — sustain check pending | 2025-10-27 09:15:00 | 339.85 | 333.28 | 329.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 10:15:00 | 337.85 | 333.33 | 329.47 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-02-05 09:15:00 | 388.53 | 363.57 | 360.68 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 337.85 | 369.82 | 367.18 | SL hit qty=0.50 sl=337.85 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-09 11:15:00 | 334.55 | 369.09 | 366.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 12:15:00 | 334.45 | 368.75 | 366.68 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 327.40 | 367.23 | 365.95 | SL hit qty=1.00 sl=327.40 alert=retest2 |
| CROSSOVER_SKIP | 2026-03-11 09:15:00 | 325.15 | 364.42 | 364.57 | slope filter: EMA200 not falling 0.50% over 350 bars |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-03-14 15:15:00 | 304.00 | 2024-03-19 14:15:00 | 279.52 | STOP_HIT | 1.00 | -8.05% |
| BUY | retest2 | 2024-03-20 10:15:00 | 281.75 | 2024-05-21 12:15:00 | 324.01 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-03-20 10:15:00 | 281.75 | 2024-06-05 09:15:00 | 281.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2024-06-05 11:15:00 | 281.55 | 2024-07-25 10:15:00 | 323.78 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-06-05 11:15:00 | 281.55 | 2024-09-02 10:15:00 | 366.01 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2025-01-21 10:15:00 | 284.85 | 2025-01-22 09:15:00 | 278.20 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-03-24 11:15:00 | 283.26 | 2025-03-25 09:15:00 | 278.20 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-04-07 11:15:00 | 273.95 | 2025-04-08 14:15:00 | 286.05 | STOP_HIT | 1.00 | 4.42% |
| BUY | retest1 | 2025-06-04 12:15:00 | 311.30 | 2025-06-13 09:15:00 | 304.91 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest1 | 2025-06-06 10:15:00 | 315.75 | 2025-06-13 09:15:00 | 304.91 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest2 | 2025-06-13 12:15:00 | 311.80 | 2025-07-08 12:15:00 | 358.57 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-06-16 13:15:00 | 316.25 | 2025-07-08 12:15:00 | 357.77 | PARTIAL | 0.50 | 13.13% |
| BUY | retest2 | 2025-06-13 12:15:00 | 311.80 | 2025-08-05 09:15:00 | 311.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2025-06-16 13:15:00 | 316.25 | 2025-08-05 09:15:00 | 311.10 | STOP_HIT | 0.50 | -1.63% |
| BUY | retest2 | 2025-06-23 11:15:00 | 311.10 | 2025-08-14 13:15:00 | 317.70 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest2 | 2025-08-08 10:15:00 | 317.95 | 2025-08-22 09:15:00 | 317.70 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-08-11 15:15:00 | 320.95 | 2025-08-22 09:15:00 | 317.70 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-08-19 15:15:00 | 321.90 | 2025-09-12 14:15:00 | 317.70 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-08-21 10:15:00 | 322.95 | 2025-09-23 14:15:00 | 330.65 | STOP_HIT | 1.00 | 2.38% |
| BUY | retest2 | 2025-09-11 10:15:00 | 324.60 | 2025-09-23 14:15:00 | 330.65 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2025-09-17 10:15:00 | 323.80 | 2025-09-23 14:15:00 | 330.65 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest2 | 2025-09-17 15:15:00 | 323.55 | 2025-09-23 14:15:00 | 330.65 | STOP_HIT | 1.00 | 2.19% |
| BUY | retest2 | 2025-09-18 14:15:00 | 325.50 | 2025-09-23 14:15:00 | 330.65 | STOP_HIT | 1.00 | 1.58% |
| BUY | retest1 | 2025-10-15 11:15:00 | 338.40 | 2025-10-24 09:15:00 | 329.31 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest1 | 2025-10-17 14:15:00 | 335.80 | 2025-10-24 09:15:00 | 329.31 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-10-27 10:15:00 | 337.85 | 2026-02-05 09:15:00 | 388.53 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-10-27 10:15:00 | 337.85 | 2026-03-09 09:15:00 | 337.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2026-03-09 12:15:00 | 334.45 | 2026-03-10 09:15:00 | 327.40 | STOP_HIT | 1.00 | -2.11% |
