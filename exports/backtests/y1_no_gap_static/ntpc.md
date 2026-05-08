# NTPC (NTPC)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 402.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 5 |
| PENDING | 7 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 3 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 1
- **Avg / median % per leg:** 1.87% / -0.54%
- **Sum % (uncompounded):** 11.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.82% | 14.5% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | 2.23% | 4.5% |
| BUY @ 3rd Alert (retest2) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.07% | -3.2% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.02% | -2.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.60% | -1.2% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.81% | 2.4% |
| retest2 (combined) | 3 | 1 | 33.3% | 1 | 2 | 0 | 2.94% | 8.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 333.80 | 337.12 | 337.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 332.20 | 336.96 | 337.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 12:15:00 | 336.45 | 336.35 | 336.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 12:15:00 | 336.45 | 336.35 | 336.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 336.45 | 336.35 | 336.72 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 13:15:00 | 342.10 | 337.04 | 337.02 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 11:15:00 | 330.70 | 336.99 | 337.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 327.25 | 336.67 | 336.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 10:15:00 | 336.05 | 335.77 | 336.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 10:15:00 | 336.05 | 335.77 | 336.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 336.05 | 335.77 | 336.37 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-09-02 13:15:00 | 335.15 | 335.79 | 336.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-02 14:15:00 | 336.25 | 335.80 | 336.37 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-03 12:15:00 | 334.65 | 335.81 | 336.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 13:15:00 | 334.65 | 335.80 | 336.36 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-16 15:15:00 | 335.00 | 333.11 | 334.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-17 09:15:00 | 335.25 | 333.13 | 334.61 | ENTRY2 sustain failed after 1080m |
| Stop hit — per-position SL triggered | 2025-09-17 10:15:00 | 336.70 | 333.17 | 334.62 | SL hit (close>static) qty=1.00 sl=336.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-18 12:15:00 | 334.85 | 333.42 | 334.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 13:15:00 | 335.05 | 333.44 | 334.69 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-18 14:15:00 | 337.00 | 333.47 | 334.70 | SL hit (close>static) qty=1.00 sl=336.60 alert=retest2 |

### Cycle 4 — BUY (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 10:15:00 | 347.90 | 335.82 | 335.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 12:15:00 | 349.60 | 339.48 | 338.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 338.10 | 340.10 | 338.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 338.10 | 340.10 | 338.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 338.10 | 340.10 | 338.55 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 15:15:00 | 326.00 | 337.30 | 337.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 325.50 | 337.18 | 337.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 325.45 | 324.54 | 328.12 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-12-24 14:15:00 | 322.40 | 324.51 | 328.02 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 15:15:00 | 322.40 | 324.49 | 327.99 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 328.90 | 324.57 | 327.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 328.90 | 324.57 | 327.67 | SL hit (close>ema400) qty=1.00 sl=327.67 alert=retest1 |

### Cycle 6 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 348.15 | 330.27 | 330.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 351.50 | 334.39 | 332.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 336.30 | 336.95 | 334.28 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-27 09:15:00 | 342.40 | 337.01 | 334.33 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 10:15:00 | 342.85 | 337.07 | 334.38 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 11:15:00 | 359.99 | 338.70 | 335.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-01 15:15:00 | 341.00 | 341.03 | 336.92 | SL hit (close<ema200) qty=0.50 sl=341.03 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 367.15 | 374.18 | 365.61 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-04-08 09:15:00 | 370.60 | 371.68 | 365.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 10:15:00 | 370.70 | 371.67 | 365.34 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2026-04-27 09:15:00 | 407.77 | 384.40 | 374.81 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-03 13:15:00 | 334.65 | 2025-09-17 10:15:00 | 336.70 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-09-18 13:15:00 | 335.05 | 2025-09-18 14:15:00 | 337.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2025-12-24 15:15:00 | 322.40 | 2025-12-31 09:15:00 | 328.90 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest1 | 2026-01-27 10:15:00 | 342.85 | 2026-01-29 11:15:00 | 359.99 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-01-27 10:15:00 | 342.85 | 2026-02-01 15:15:00 | 341.00 | STOP_HIT | 0.50 | -0.54% |
| BUY | retest2 | 2026-04-08 10:15:00 | 370.70 | 2026-04-27 09:15:00 | 407.77 | TARGET_HIT | 1.00 | 10.00% |
