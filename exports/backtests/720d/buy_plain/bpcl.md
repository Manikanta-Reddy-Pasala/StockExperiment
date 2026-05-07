# BPCL (BPCL)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 308.20
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
| ALERT3 | 4 |
| PENDING | 21 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 12 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 3
- **Target hits / Stop hits / Partials:** 0 / 16 / 4
- **Avg / median % per leg:** 5.53% / 3.69%
- **Sum % (uncompounded):** 110.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 17 | 85.0% | 0 | 16 | 4 | 5.53% | 110.6% |
| BUY @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 4 | 2 | 8.56% | 51.4% |
| BUY @ 3rd Alert (retest2) | 14 | 11 | 78.6% | 0 | 12 | 2 | 4.23% | 59.3% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 6 | 100.0% | 0 | 4 | 2 | 8.56% | 51.4% |
| retest2 (combined) | 14 | 11 | 78.6% | 0 | 12 | 2 | 4.23% | 59.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 281.40 | 271.15 | 271.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 12:15:00 | 283.00 | 271.26 | 271.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 309.90 | 310.81 | 300.74 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-06-04 11:15:00 | 311.75 | 310.79 | 300.92 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 12:15:00 | 311.30 | 310.79 | 300.97 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-06 09:15:00 | 313.60 | 310.89 | 301.55 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:15:00 | 315.75 | 310.94 | 301.62 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 309.45 | 314.25 | 304.86 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-13 11:15:00 | 313.30 | 314.19 | 304.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 12:15:00 | 311.80 | 314.17 | 304.96 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-16 10:15:00 | 310.80 | 314.04 | 305.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-16 11:15:00 | 310.20 | 314.00 | 305.15 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-16 12:15:00 | 311.45 | 313.98 | 305.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 13:15:00 | 316.25 | 314.00 | 305.23 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-23 10:15:00 | 310.75 | 313.89 | 306.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:15:00 | 311.20 | 313.87 | 306.50 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-07-08 12:15:00 | 358.00 | 324.87 | 315.10 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-07-08 12:15:00 | 358.57 | 324.87 | 315.10 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-07-08 12:15:00 | 357.88 | 324.87 | 315.10 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 336.40 | 336.69 | 325.92 | SL hit (close<ema200) qty=0.50 sl=336.69 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 336.40 | 336.69 | 325.92 | SL hit (close<ema200) qty=0.50 sl=336.69 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 336.40 | 336.69 | 325.92 | SL hit (close<ema200) qty=0.50 sl=336.69 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 326.90 | 336.28 | 327.10 | SL hit (close<ema400) qty=1.00 sl=327.10 alert=retest1 |
| Cross detected — sustain check pending | 2025-08-07 14:15:00 | 310.35 | 329.98 | 325.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-07 15:15:00 | 310.15 | 329.78 | 325.28 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-08 09:15:00 | 316.80 | 329.65 | 325.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 10:15:00 | 317.90 | 329.54 | 325.20 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 318.40 | 329.43 | 325.17 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-11 14:15:00 | 321.10 | 328.51 | 324.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-11 15:15:00 | 320.20 | 328.43 | 324.88 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-12 09:15:00 | 324.95 | 328.40 | 324.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 10:15:00 | 325.25 | 328.37 | 324.89 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 315.10 | 327.07 | 324.54 | SL hit (close<static) qty=1.00 sl=317.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-19 14:15:00 | 320.95 | 325.85 | 324.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 15:15:00 | 321.90 | 325.81 | 324.05 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-20 12:15:00 | 322.15 | 325.60 | 323.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 13:15:00 | 321.00 | 325.56 | 323.97 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-21 09:15:00 | 321.85 | 325.41 | 323.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:15:00 | 322.95 | 325.39 | 323.91 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 320.85 | 325.34 | 323.89 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 317.40 | 324.99 | 323.76 | SL hit (close<static) qty=1.00 sl=317.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 317.40 | 324.99 | 323.76 | SL hit (close<static) qty=1.00 sl=317.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 317.40 | 324.99 | 323.76 | SL hit (close<static) qty=1.00 sl=317.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-11 10:15:00 | 324.60 | 319.01 | 320.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-11 11:15:00 | 322.60 | 319.04 | 320.50 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-17 09:15:00 | 323.10 | 319.04 | 320.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:15:00 | 323.80 | 319.09 | 320.34 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-17 14:15:00 | 324.05 | 319.22 | 320.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 15:15:00 | 323.40 | 319.26 | 320.39 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-18 13:15:00 | 325.00 | 319.45 | 320.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:15:00 | 325.45 | 319.51 | 320.49 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 330.65 | 321.44 | 321.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 330.65 | 321.44 | 321.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 330.65 | 321.44 | 321.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 330.65 | 321.44 | 321.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 330.65 | 321.44 | 321.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 14:15:00 | 330.65 | 321.44 | 321.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 09:15:00 | 335.30 | 322.86 | 322.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 12:15:00 | 332.15 | 332.18 | 327.88 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-10-15 10:15:00 | 338.95 | 332.25 | 328.02 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 11:15:00 | 338.40 | 332.31 | 328.07 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-17 13:15:00 | 335.55 | 332.84 | 328.68 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 14:15:00 | 335.80 | 332.87 | 328.71 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 334.00 | 333.32 | 329.31 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-27 09:15:00 | 338.20 | 333.29 | 329.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 10:15:00 | 337.85 | 333.33 | 329.47 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 350.90 | 358.14 | 351.27 | SL hit (close<ema400) qty=1.00 sl=351.27 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 350.90 | 358.14 | 351.27 | SL hit (close<ema400) qty=1.00 sl=351.27 alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-02-05 09:15:00 | 388.53 | 363.52 | 360.71 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 370.35 | 370.71 | 365.40 | SL hit (close<ema200) qty=0.50 sl=370.71 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-09 11:15:00 | 334.55 | 369.07 | 366.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-09 12:15:00 | 334.25 | 368.72 | 366.68 | ENTRY2 sustain failed after 60m |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-04 12:15:00 | 311.30 | 2025-07-08 12:15:00 | 358.00 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-06-06 10:15:00 | 315.75 | 2025-07-08 12:15:00 | 358.57 | PARTIAL | 0.50 | 13.56% |
| BUY | retest2 | 2025-06-13 12:15:00 | 311.80 | 2025-07-08 12:15:00 | 357.88 | PARTIAL | 0.50 | 14.78% |
| BUY | retest1 | 2025-06-04 12:15:00 | 311.30 | 2025-07-25 09:15:00 | 336.40 | STOP_HIT | 0.50 | 8.06% |
| BUY | retest1 | 2025-06-06 10:15:00 | 315.75 | 2025-07-25 09:15:00 | 336.40 | STOP_HIT | 0.50 | 6.54% |
| BUY | retest2 | 2025-06-13 12:15:00 | 311.80 | 2025-07-25 09:15:00 | 336.40 | STOP_HIT | 0.50 | 7.89% |
| BUY | retest2 | 2025-06-16 13:15:00 | 316.25 | 2025-07-31 09:15:00 | 326.90 | STOP_HIT | 1.00 | 3.37% |
| BUY | retest2 | 2025-06-23 11:15:00 | 311.20 | 2025-08-18 09:15:00 | 315.10 | STOP_HIT | 1.00 | 1.25% |
| BUY | retest2 | 2025-08-08 10:15:00 | 317.90 | 2025-08-22 10:15:00 | 317.40 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-08-12 10:15:00 | 325.25 | 2025-08-22 10:15:00 | 317.40 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-08-19 15:15:00 | 321.90 | 2025-08-22 10:15:00 | 317.40 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-08-20 13:15:00 | 321.00 | 2025-09-23 14:15:00 | 330.65 | STOP_HIT | 1.00 | 3.01% |
| BUY | retest2 | 2025-08-21 10:15:00 | 322.95 | 2025-09-23 14:15:00 | 330.65 | STOP_HIT | 1.00 | 2.38% |
| BUY | retest2 | 2025-09-17 10:15:00 | 323.80 | 2025-09-23 14:15:00 | 330.65 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest2 | 2025-09-17 15:15:00 | 323.40 | 2025-09-23 14:15:00 | 330.65 | STOP_HIT | 1.00 | 2.24% |
| BUY | retest2 | 2025-09-18 14:15:00 | 325.45 | 2025-09-23 14:15:00 | 330.65 | STOP_HIT | 1.00 | 1.60% |
| BUY | retest1 | 2025-10-15 11:15:00 | 338.40 | 2025-12-11 09:15:00 | 350.90 | STOP_HIT | 1.00 | 3.69% |
| BUY | retest1 | 2025-10-17 14:15:00 | 335.80 | 2025-12-11 09:15:00 | 350.90 | STOP_HIT | 1.00 | 4.50% |
| BUY | retest2 | 2025-10-27 10:15:00 | 337.85 | 2026-02-05 09:15:00 | 388.53 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-10-27 10:15:00 | 337.85 | 2026-02-16 09:15:00 | 370.35 | STOP_HIT | 0.50 | 9.62% |
