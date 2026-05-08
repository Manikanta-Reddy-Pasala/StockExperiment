# KOTAKBANK (KOTAKBANK)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 381.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 4 |
| PENDING | 9 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 1
- **Avg / median % per leg:** -0.23% / -2.47%
- **Sum % (uncompounded):** -1.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.54% | -14.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.54% | -14.2% |
| SELL (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.18% | 12.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.18% | 12.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.23% | -1.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 12:15:00 | 398.62 | 423.59 | 423.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 390.22 | 414.58 | 418.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 399.62 | 397.77 | 404.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 11:15:00 | 404.56 | 397.88 | 404.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 404.56 | 397.88 | 404.80 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-09-25 14:15:00 | 402.20 | 401.56 | 405.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-25 15:15:00 | 402.96 | 401.57 | 405.29 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-26 09:15:00 | 400.00 | 401.55 | 405.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:15:00 | 397.88 | 401.52 | 405.23 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-01 10:15:00 | 407.70 | 401.12 | 404.65 | SL hit (close>static) qty=1.00 sl=406.10 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 426.64 | 407.62 | 407.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 11:15:00 | 428.36 | 407.83 | 407.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 423.80 | 424.02 | 417.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 419.94 | 423.84 | 417.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 419.94 | 423.84 | 417.94 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-11-03 12:15:00 | 422.00 | 423.74 | 417.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 13:15:00 | 423.52 | 423.74 | 418.01 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 417.36 | 423.29 | 418.11 | SL hit (close<static) qty=1.00 sl=417.58 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-21 13:15:00 | 421.86 | 420.64 | 418.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-21 14:15:00 | 417.54 | 420.61 | 418.17 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-27 10:15:00 | 423.68 | 419.96 | 418.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 11:15:00 | 425.02 | 420.01 | 418.14 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 09:15:00 | 424.20 | 429.57 | 427.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 10:15:00 | 423.90 | 429.51 | 427.01 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-19 09:15:00 | 422.30 | 428.97 | 426.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 10:15:00 | 422.60 | 428.91 | 426.79 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 427.60 | 428.87 | 426.80 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-20 09:15:00 | 429.70 | 428.85 | 426.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-20 10:15:00 | 428.60 | 428.85 | 426.83 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 405.90 | 427.49 | 426.37 | SL hit (close<static) qty=1.00 sl=417.58 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 405.90 | 427.49 | 426.37 | SL hit (close<static) qty=1.00 sl=417.58 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 405.90 | 427.49 | 426.37 | SL hit (close<static) qty=1.00 sl=417.58 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 410.40 | 425.22 | 425.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 406.50 | 423.79 | 424.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 422.70 | 419.66 | 422.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 11:15:00 | 422.70 | 419.66 | 422.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 422.70 | 419.66 | 422.10 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-27 14:15:00 | 414.65 | 422.89 | 423.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 15:15:00 | 416.05 | 422.82 | 423.17 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 395.25 | 418.57 | 420.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-12 15:15:00 | 374.44 | 410.64 | 416.36 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-26 10:15:00 | 397.88 | 2025-10-01 10:15:00 | 407.70 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-11-03 13:15:00 | 423.52 | 2025-11-06 11:15:00 | 417.36 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-11-27 11:15:00 | 425.02 | 2026-01-27 09:15:00 | 405.90 | STOP_HIT | 1.00 | -4.50% |
| BUY | retest2 | 2026-01-16 10:15:00 | 423.90 | 2026-01-27 09:15:00 | 405.90 | STOP_HIT | 1.00 | -4.25% |
| BUY | retest2 | 2026-01-19 10:15:00 | 422.60 | 2026-01-27 09:15:00 | 405.90 | STOP_HIT | 1.00 | -3.95% |
| SELL | retest2 | 2026-02-27 15:15:00 | 416.05 | 2026-03-09 09:15:00 | 395.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 15:15:00 | 416.05 | 2026-03-12 15:15:00 | 374.44 | TARGET_HIT | 0.50 | 10.00% |
