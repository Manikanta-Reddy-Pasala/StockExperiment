# TMPV (TMPV)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 355.50
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
| ALERT2_SKIP | 5 |
| ALERT3 | 5 |
| PENDING | 7 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 0 |
| ENTRY2 | 3 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 3.11% / 5.00%
- **Sum % (uncompounded):** 12.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.11% | 12.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.11% | 12.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.11% | 12.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 13:15:00 | 433.79 | 417.61 | 417.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 434.64 | 419.05 | 418.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 422.03 | 422.51 | 420.30 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-09-23 09:15:00 | 425.45 | 422.53 | 420.33 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-23 10:15:00 | 423.03 | 422.54 | 420.34 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-23 15:15:00 | 425.58 | 422.64 | 420.45 | ENTRY1 cross detected — sustain check pending (15m) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 419.73 | 422.61 | 420.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 419.73 | 422.61 | 420.45 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 15:15:00 | 413.67 | 418.62 | 418.63 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 420.21 | 418.64 | 418.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 10:15:00 | 424.76 | 418.70 | 418.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 420.39 | 421.45 | 420.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 420.39 | 421.45 | 420.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 420.39 | 421.45 | 420.16 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 402.36 | 419.02 | 419.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 392.95 | 418.43 | 418.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 411.85 | 411.25 | 414.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 10:15:00 | 414.15 | 411.26 | 414.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 414.15 | 411.26 | 414.34 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-10-29 13:15:00 | 411.95 | 411.32 | 414.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:15:00 | 411.20 | 411.31 | 414.31 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-31 09:15:00 | 415.40 | 411.35 | 414.19 | SL hit (close>static) qty=1.00 sl=415.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-31 13:15:00 | 412.10 | 411.41 | 414.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 14:15:00 | 410.90 | 411.40 | 414.15 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-03 14:15:00 | 417.25 | 411.48 | 414.10 | SL hit (close>static) qty=1.00 sl=415.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-04 09:15:00 | 411.35 | 411.54 | 414.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:15:00 | 409.60 | 411.52 | 414.07 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 14:15:00 | 389.12 | 408.16 | 411.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-11-17 09:15:00 | 368.64 | 407.69 | 411.38 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 09:15:00 | 384.50 | 367.53 | 367.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 387.00 | 371.08 | 369.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 10:15:00 | 373.05 | 373.21 | 370.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 12:15:00 | 369.85 | 373.15 | 370.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 12:15:00 | 369.85 | 373.15 | 370.66 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 331.95 | 368.39 | 368.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 12:15:00 | 328.90 | 367.64 | 368.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 330.90 | 326.70 | 341.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 13:15:00 | 341.60 | 328.29 | 341.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 341.60 | 328.29 | 341.24 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 339.25 | 328.68 | 341.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-13 10:15:00 | 344.25 | 328.84 | 341.25 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-30 12:15:00 | 337.85 | 343.36 | 345.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-30 13:15:00 | 340.05 | 343.33 | 345.77 | ENTRY2 sustain failed after 60m |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-10-29 14:15:00 | 411.20 | 2025-10-31 09:15:00 | 415.40 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-10-31 14:15:00 | 410.90 | 2025-11-03 14:15:00 | 417.25 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-11-04 10:15:00 | 409.60 | 2025-11-14 14:15:00 | 389.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 10:15:00 | 409.60 | 2025-11-17 09:15:00 | 368.64 | TARGET_HIT | 0.50 | 10.00% |
