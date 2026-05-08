# COALINDIA (COALINDIA)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 456.55
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 4 |
| ALERT3 | 5 |
| PENDING | 12 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 1 |
| ENTRY2 | 8 |
| PARTIAL | 0 |
| TARGET_HIT | 3 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / Stop hits / Partials:** 3 / 6 / 0
- **Avg / median % per leg:** 1.67% / -1.13%
- **Sum % (uncompounded):** 15.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 3 | 2 | 0 | 5.35% | 26.7% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.45% | -2.4% |
| BUY @ 3rd Alert (retest2) | 4 | 3 | 75.0% | 3 | 1 | 0 | 7.30% | 29.2% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.92% | -11.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.92% | -11.7% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.45% | -2.4% |
| retest2 (combined) | 8 | 3 | 37.5% | 3 | 5 | 0 | 2.19% | 17.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 14:15:00 | 393.35 | 387.94 | 387.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 10:15:00 | 394.10 | 388.10 | 388.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 389.40 | 389.82 | 388.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 13:15:00 | 389.40 | 389.82 | 388.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 389.40 | 389.82 | 388.98 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-09-29 10:15:00 | 391.35 | 389.83 | 389.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-29 11:15:00 | 389.65 | 389.83 | 389.00 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-30 10:15:00 | 391.20 | 389.79 | 389.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-30 11:15:00 | 389.05 | 389.78 | 389.01 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-01 09:15:00 | 392.80 | 389.78 | 389.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 10:15:00 | 391.60 | 389.80 | 389.04 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 388.40 | 389.81 | 389.06 | SL hit (close<static) qty=1.00 sl=388.80 alert=retest2 |

### Cycle 2 — SELL (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 13:15:00 | 382.60 | 388.37 | 388.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 381.85 | 388.30 | 388.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 12:15:00 | 387.40 | 386.76 | 387.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 12:15:00 | 387.40 | 386.76 | 387.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 387.40 | 386.76 | 387.49 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-10-17 13:15:00 | 385.90 | 386.79 | 387.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-17 14:15:00 | 388.60 | 386.81 | 387.48 | ENTRY2 sustain failed after 60m |

### Cycle 3 — BUY (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 10:15:00 | 397.20 | 388.14 | 388.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 398.35 | 389.03 | 388.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 13:15:00 | 382.60 | 389.12 | 388.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 13:15:00 | 382.60 | 389.12 | 388.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 382.60 | 389.12 | 388.62 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 15:15:00 | 378.00 | 388.19 | 388.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 374.70 | 388.05 | 388.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 12:15:00 | 386.80 | 385.73 | 386.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 12:15:00 | 386.80 | 385.73 | 386.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 386.80 | 385.73 | 386.84 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-11-13 12:15:00 | 384.95 | 385.76 | 386.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 13:15:00 | 383.85 | 385.74 | 386.81 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 388.20 | 385.71 | 386.74 | SL hit (close>static) qty=1.00 sl=387.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-18 14:15:00 | 384.10 | 385.83 | 386.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 15:15:00 | 383.65 | 385.81 | 386.73 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-12 10:15:00 | 384.40 | 380.69 | 382.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:15:00 | 383.60 | 380.72 | 382.87 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-17 15:15:00 | 384.40 | 381.24 | 382.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 384.50 | 381.27 | 382.90 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-12-23 09:15:00 | 397.40 | 382.15 | 383.20 | SL hit (close>static) qty=1.00 sl=387.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 09:15:00 | 397.40 | 382.15 | 383.20 | SL hit (close>static) qty=1.00 sl=387.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 09:15:00 | 397.40 | 382.15 | 383.20 | SL hit (close>static) qty=1.00 sl=387.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 13:15:00 | 402.50 | 384.24 | 384.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 405.35 | 384.80 | 384.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 419.30 | 419.57 | 408.06 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-02-02 14:15:00 | 423.85 | 419.44 | 408.34 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 15:15:00 | 422.80 | 419.47 | 408.41 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 412.45 | 422.91 | 413.19 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 412.45 | 422.91 | 413.19 | SL hit (close<ema400) qty=1.00 sl=413.19 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-16 09:15:00 | 419.60 | 422.11 | 413.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 10:15:00 | 420.15 | 422.09 | 413.15 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-18 14:15:00 | 418.20 | 421.66 | 413.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 15:15:00 | 418.20 | 421.62 | 413.72 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-20 09:15:00 | 424.00 | 421.30 | 413.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:15:00 | 420.80 | 421.30 | 413.90 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Target hit | 2026-03-12 10:15:00 | 462.17 | 431.12 | 422.25 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-12 10:15:00 | 460.02 | 431.12 | 422.25 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-12 10:15:00 | 462.88 | 431.12 | 422.25 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-10-01 10:15:00 | 391.60 | 2025-10-01 14:15:00 | 388.40 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-11-13 13:15:00 | 383.85 | 2025-11-17 09:15:00 | 388.20 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-11-18 15:15:00 | 383.65 | 2025-12-23 09:15:00 | 397.40 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2025-12-12 11:15:00 | 383.60 | 2025-12-23 09:15:00 | 397.40 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2025-12-18 09:15:00 | 384.50 | 2025-12-23 09:15:00 | 397.40 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest1 | 2026-02-02 15:15:00 | 422.80 | 2026-02-13 09:15:00 | 412.45 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2026-02-16 10:15:00 | 420.15 | 2026-03-12 10:15:00 | 462.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-18 15:15:00 | 418.20 | 2026-03-12 10:15:00 | 460.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-20 10:15:00 | 420.80 | 2026-03-12 10:15:00 | 462.88 | TARGET_HIT | 1.00 | 10.00% |
