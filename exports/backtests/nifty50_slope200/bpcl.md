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
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 3 |
| PENDING | 13 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 7 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 8
- **Target hits / Stop hits / Partials:** 2 / 8 / 4
- **Avg / median % per leg:** 7.22% / 0.00%
- **Sum % (uncompounded):** 101.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 6 | 42.9% | 2 | 8 | 4 | 7.22% | 101.1% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.22% | -12.7% |
| BUY @ 3rd Alert (retest2) | 11 | 6 | 54.5% | 2 | 5 | 4 | 10.34% | 113.8% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.22% | -12.7% |
| retest2 (combined) | 11 | 6 | 54.5% | 2 | 5 | 4 | 10.34% | 113.8% |

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
| CROSSOVER_SKIP | 2024-10-28 11:15:00 | 310.85 | 336.28 | 336.40 | slope filter: EMA200 not falling 2.00% over 1400 bars |
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
| CROSSOVER_SKIP | 2025-04-08 14:15:00 | 286.05 | 271.60 | 271.54 | HTF filter: close below htf_sma |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-04-29 09:15:00 | 315.04 | 286.92 | 280.54 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2025-07-08 10:15:00 | 356.14 | 324.24 | 314.71 | Target hit (30%) qty=0.50 alert=retest2 |
| CROSSOVER_SKIP | 2025-08-28 14:15:00 | 310.55 | 322.58 | 322.64 | HTF filter: close above htf_sma |

### Cycle 2 — BUY (started 2025-09-23 14:15:00)

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
| CROSSOVER_SKIP | 2026-03-11 09:15:00 | 325.15 | 364.42 | 364.57 | slope filter: EMA200 not falling 2.00% over 1400 bars |


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
| BUY | retest2 | 2025-04-07 11:15:00 | 273.95 | 2025-04-29 09:15:00 | 315.04 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-07 11:15:00 | 273.95 | 2025-07-08 10:15:00 | 356.14 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest1 | 2025-10-15 11:15:00 | 338.40 | 2025-10-24 09:15:00 | 329.31 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest1 | 2025-10-17 14:15:00 | 335.80 | 2025-10-24 09:15:00 | 329.31 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-10-27 10:15:00 | 337.85 | 2026-02-05 09:15:00 | 388.53 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-10-27 10:15:00 | 337.85 | 2026-03-09 09:15:00 | 337.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2026-03-09 12:15:00 | 334.45 | 2026-03-10 09:15:00 | 327.40 | STOP_HIT | 1.00 | -2.11% |
