# NTPC (NTPC)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 400.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 4 |
| PENDING | 13 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 1 |
| ENTRY2 | 11 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 8
- **Target hits / Stop hits / Partials:** 0 / 11 / 1
- **Avg / median % per leg:** 1.17% / -0.97%
- **Sum % (uncompounded):** 14.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 4 | 33.3% | 0 | 11 | 1 | 1.17% | 14.1% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 11.70% | 23.4% |
| BUY @ 3rd Alert (retest2) | 10 | 2 | 20.0% | 0 | 10 | 0 | -0.93% | -9.3% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 11.70% | 23.4% |
| retest2 (combined) | 10 | 2 | 20.0% | 0 | 10 | 0 | -0.93% | -9.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 09:15:00 | 362.50 | 332.88 | 332.86 | EMA200 above EMA400 |

### Cycle 2 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 340.95 | 339.47 | 339.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 12:15:00 | 341.70 | 336.41 | 337.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 335.80 | 336.99 | 337.78 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 14:15:00 | 335.80 | 336.99 | 337.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 335.80 | 336.99 | 337.78 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-20 09:15:00 | 339.50 | 336.89 | 337.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:15:00 | 339.15 | 336.92 | 337.71 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-21 12:15:00 | 338.35 | 337.23 | 337.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 13:15:00 | 339.10 | 337.25 | 337.84 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-25 09:15:00 | 338.75 | 337.28 | 337.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:15:00 | 338.50 | 337.29 | 337.83 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 335.20 | 337.32 | 337.82 | SL hit (close<static) qty=1.00 sl=335.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 335.20 | 337.32 | 337.82 | SL hit (close<static) qty=1.00 sl=335.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 335.20 | 337.32 | 337.82 | SL hit (close<static) qty=1.00 sl=335.65 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-19 10:15:00 | 339.10 | 333.60 | 335.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 11:15:00 | 339.15 | 333.66 | 335.12 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 340.10 | 336.35 | 336.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 09:15:00 | 340.10 | 336.35 | 336.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 10:15:00 | 341.25 | 336.40 | 336.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 337.25 | 337.46 | 336.95 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 11:15:00 | 337.45 | 337.46 | 336.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 337.45 | 337.46 | 336.95 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-07 13:15:00 | 338.90 | 337.49 | 336.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-07 14:15:00 | 338.00 | 337.49 | 336.97 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-10 09:15:00 | 339.95 | 337.14 | 336.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 340.75 | 337.18 | 336.85 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-13 10:15:00 | 338.70 | 337.34 | 336.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 11:15:00 | 339.00 | 337.35 | 336.95 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-14 12:15:00 | 335.60 | 337.52 | 337.05 | SL hit (close<static) qty=1.00 sl=336.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 12:15:00 | 335.60 | 337.52 | 337.05 | SL hit (close<static) qty=1.00 sl=336.85 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-15 09:15:00 | 339.90 | 337.52 | 337.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:15:00 | 340.50 | 337.55 | 337.08 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-28 12:15:00 | 338.80 | 339.18 | 338.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 13:15:00 | 338.50 | 339.18 | 338.12 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 339.15 | 339.18 | 338.13 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-29 09:15:00 | 343.60 | 339.22 | 338.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 10:15:00 | 346.75 | 339.29 | 338.20 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-31 10:15:00 | 337.00 | 340.07 | 338.68 | SL hit (close<static) qty=1.00 sl=337.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 336.45 | 339.96 | 338.65 | SL hit (close<static) qty=1.00 sl=336.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 336.45 | 339.96 | 338.65 | SL hit (close<static) qty=1.00 sl=336.85 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-02 09:15:00 | 340.30 | 325.70 | 328.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:15:00 | 345.40 | 325.90 | 328.16 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 348.15 | 330.27 | 330.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 348.15 | 330.27 | 330.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 351.50 | 334.39 | 332.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 336.30 | 336.95 | 334.30 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-01-27 09:15:00 | 342.40 | 337.01 | 334.35 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 10:15:00 | 342.85 | 337.07 | 334.39 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-13 11:15:00 | 394.28 | 370.85 | 360.10 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 371.65 | 373.90 | 363.74 | SL hit (close<ema200) qty=0.50 sl=373.90 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 367.15 | 374.18 | 365.61 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-08 09:15:00 | 370.60 | 371.68 | 365.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 10:15:00 | 370.70 | 371.67 | 365.34 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-20 10:15:00 | 339.15 | 2025-08-26 09:15:00 | 335.20 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-08-21 13:15:00 | 339.10 | 2025-08-26 09:15:00 | 335.20 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-08-25 10:15:00 | 338.50 | 2025-08-26 09:15:00 | 335.20 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-09-19 11:15:00 | 339.15 | 2025-09-29 09:15:00 | 340.10 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-10-10 10:15:00 | 340.75 | 2025-10-14 12:15:00 | 335.60 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-10-13 11:15:00 | 339.00 | 2025-10-14 12:15:00 | 335.60 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-10-15 10:15:00 | 340.50 | 2025-10-31 10:15:00 | 337.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-10-28 13:15:00 | 338.50 | 2025-10-31 14:15:00 | 336.45 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-10-29 10:15:00 | 346.75 | 2025-10-31 14:15:00 | 336.45 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2026-01-02 10:15:00 | 345.40 | 2026-01-07 09:15:00 | 348.15 | STOP_HIT | 1.00 | 0.80% |
| BUY | retest1 | 2026-01-27 10:15:00 | 342.85 | 2026-03-13 11:15:00 | 394.28 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2026-01-27 10:15:00 | 342.85 | 2026-03-23 09:15:00 | 371.65 | STOP_HIT | 0.50 | 8.40% |
