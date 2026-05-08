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
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 2 |
| PENDING | 2 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 1 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 1
- **Avg / median % per leg:** 4.82% / 5.00%
- **Sum % (uncompounded):** 14.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.82% | 14.5% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | 2.23% | 4.5% |
| BUY @ 3rd Alert (retest2) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 1 | 1 | 2.23% | 4.5% |
| retest2 (combined) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 10:15:00 | 347.90 | 335.82 | 335.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 12:15:00 | 349.60 | 339.48 | 338.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 338.10 | 340.10 | 338.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 338.10 | 340.10 | 338.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 338.10 | 340.10 | 338.55 | EMA400 retest candle locked (from upside) |
| CROSSOVER_SKIP | 2025-11-07 15:15:00 | 326.00 | 337.30 | 337.31 | min_gap filter: gap=0.003% < 0.010% |
| TREND_RESET | 2025-11-07 15:15:00 | 326.00 | 337.30 | 337.31 | EMA inversion without crossover edge (EMA200=337.30 EMA400=337.31) — end cycle |

### Cycle 2 — BUY (started 2026-01-07 09:15:00)

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
| BUY | retest1 | 2026-01-27 10:15:00 | 342.85 | 2026-01-29 11:15:00 | 359.99 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-01-27 10:15:00 | 342.85 | 2026-02-01 15:15:00 | 341.00 | STOP_HIT | 0.50 | -0.54% |
| BUY | retest2 | 2026-04-08 10:15:00 | 370.70 | 2026-04-27 09:15:00 | 407.77 | TARGET_HIT | 1.00 | 10.00% |
