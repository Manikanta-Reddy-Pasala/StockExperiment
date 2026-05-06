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
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 3 |
| PENDING | 13 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 6 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 8 / 8
- **Target hits / Stop hits / Partials:** 1 / 10 / 5
- **Avg / median % per leg:** 5.22% / 8.85%
- **Sum % (uncompounded):** 83.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 6.02% | 60.2% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.22% | -12.7% |
| BUY @ 3rd Alert (retest2) | 7 | 4 | 57.1% | 1 | 3 | 3 | 10.41% | 72.9% |
| SELL (all) | 6 | 4 | 66.7% | 0 | 4 | 2 | 3.88% | 23.3% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 12.11% | 48.4% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -12.57% | -25.1% |
| retest1 (combined) | 7 | 4 | 57.1% | 0 | 5 | 2 | 5.11% | 35.7% |
| retest2 (combined) | 9 | 4 | 44.4% | 1 | 5 | 3 | 5.31% | 47.8% |

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

### Cycle 2 — SELL (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 11:15:00 | 310.85 | 336.28 | 336.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 13:15:00 | 308.55 | 335.75 | 336.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 12:15:00 | 303.85 | 303.70 | 313.91 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-12-13 10:15:00 | 298.10 | 303.64 | 312.32 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 11:15:00 | 298.60 | 303.59 | 312.25 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-16 09:15:00 | 298.05 | 303.46 | 311.97 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:15:00 | 296.20 | 303.39 | 311.89 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-01-29 09:15:00 | 253.81 | 279.64 | 290.52 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-01-29 09:15:00 | 251.77 | 279.64 | 290.52 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 267.35 | 257.75 | 267.50 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-03-12 11:15:00 | 264.27 | 257.90 | 267.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 12:15:00 | 264.39 | 257.96 | 267.47 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-13 09:15:00 | 265.44 | 258.27 | 267.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:15:00 | 264.00 | 258.33 | 267.41 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-20 10:15:00 | 270.00 | 259.53 | 266.86 | SL hit qty=1.00 sl=270.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-20 10:15:00 | 270.00 | 259.53 | 266.86 | SL hit qty=1.00 sl=270.00 alert=retest2 |
| CROSSOVER_SKIP | 2025-04-08 14:15:00 | 286.05 | 271.60 | 271.54 | HTF filter: close below htf_sma |
| Stop hit — per-position SL triggered | 2025-04-16 09:15:00 | 296.20 | 275.61 | 273.66 | SL hit qty=0.50 sl=296.20 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-04-16 11:15:00 | 298.60 | 276.04 | 273.89 | SL hit qty=0.50 sl=298.60 alert=retest1 |
| CROSSOVER_SKIP | 2025-08-28 14:15:00 | 310.55 | 322.58 | 322.64 | HTF filter: close above htf_sma |

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

### Cycle 4 — SELL (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 09:15:00 | 325.15 | 364.42 | 364.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 323.90 | 364.02 | 364.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 14:15:00 | 310.30 | 307.33 | 325.98 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-04-24 09:15:00 | 301.90 | 309.08 | 323.56 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:15:00 | 305.65 | 309.04 | 323.47 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-29 14:15:00 | 303.85 | 309.04 | 321.77 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 15:15:00 | 304.95 | 309.00 | 321.69 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-03-14 15:15:00 | 304.00 | 2024-03-19 14:15:00 | 279.52 | STOP_HIT | 1.00 | -8.05% |
| BUY | retest2 | 2024-03-20 10:15:00 | 281.75 | 2024-05-21 12:15:00 | 324.01 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-03-20 10:15:00 | 281.75 | 2024-06-05 09:15:00 | 281.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2024-06-05 11:15:00 | 281.55 | 2024-07-25 10:15:00 | 323.78 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-06-05 11:15:00 | 281.55 | 2024-09-02 10:15:00 | 366.01 | TARGET_HIT | 0.50 | 30.00% |
| SELL | retest1 | 2024-12-13 11:15:00 | 298.60 | 2025-01-29 09:15:00 | 253.81 | PARTIAL | 0.50 | 15.00% |
| SELL | retest1 | 2024-12-16 10:15:00 | 296.20 | 2025-01-29 09:15:00 | 251.77 | PARTIAL | 0.50 | 15.00% |
| SELL | retest1 | 2024-12-13 11:15:00 | 298.60 | 2025-03-20 10:15:00 | 270.00 | STOP_HIT | 0.50 | 9.58% |
| SELL | retest1 | 2024-12-16 10:15:00 | 296.20 | 2025-03-20 10:15:00 | 270.00 | STOP_HIT | 0.50 | 8.85% |
| SELL | retest2 | 2025-03-12 12:15:00 | 264.39 | 2025-04-16 09:15:00 | 296.20 | STOP_HIT | 1.00 | -12.03% |
| SELL | retest2 | 2025-03-13 10:15:00 | 264.00 | 2025-04-16 11:15:00 | 298.60 | STOP_HIT | 1.00 | -13.11% |
| BUY | retest1 | 2025-10-15 11:15:00 | 338.40 | 2025-10-24 09:15:00 | 329.31 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest1 | 2025-10-17 14:15:00 | 335.80 | 2025-10-24 09:15:00 | 329.31 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-10-27 10:15:00 | 337.85 | 2026-02-05 09:15:00 | 388.53 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-10-27 10:15:00 | 337.85 | 2026-03-09 09:15:00 | 337.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2026-03-09 12:15:00 | 334.45 | 2026-03-10 09:15:00 | 327.40 | STOP_HIT | 1.00 | -2.11% |
