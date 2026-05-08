# BEL (BEL)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:30:00 (1238 bars)
- **Last close:** 439.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty @ 15% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 15%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 2 |
| ALERT3 | 3 |
| PENDING | 10 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 5
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -3.08% / -1.54%
- **Sum % (uncompounded):** -9.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -6.67% | -6.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -6.67% | -6.7% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.28% | -2.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.28% | -2.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.08% | -9.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 12:15:00 | 390.45 | 405.49 | 405.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 13:15:00 | 389.85 | 405.34 | 405.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 400.50 | 400.34 | 402.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 11:15:00 | 402.25 | 400.39 | 402.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 402.25 | 400.39 | 402.56 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-12-24 13:15:00 | 400.55 | 400.41 | 402.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 14:15:00 | 400.25 | 400.41 | 402.54 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 404.35 | 400.44 | 402.53 | SL hit (close>static) qty=1.00 sl=403.65 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-26 12:15:00 | 400.85 | 400.53 | 402.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 13:15:00 | 398.70 | 400.51 | 402.53 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 404.85 | 399.50 | 401.69 | SL hit (close>static) qty=1.00 sl=403.65 alert=retest2 |

### Cycle 2 — BUY (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 09:15:00 | 420.65 | 403.61 | 403.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 430.30 | 408.89 | 406.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 440.90 | 443.83 | 432.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 432.80 | 443.42 | 433.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 432.80 | 443.42 | 433.01 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-17 13:15:00 | 440.05 | 442.07 | 432.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 439.50 | 442.05 | 432.90 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 410.20 | 440.41 | 433.06 | SL hit (close<static) qty=1.00 sl=425.80 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 438.55 | 429.77 | 428.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 10:15:00 | 440.35 | 429.88 | 428.82 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-28 09:15:00 | 439.95 | 439.32 | 434.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-28 10:15:00 | 437.20 | 439.29 | 434.87 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-29 09:15:00 | 438.70 | 439.12 | 434.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:15:00 | 438.85 | 439.12 | 434.93 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-29 14:15:00 | 437.65 | 439.08 | 435.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 15:15:00 | 437.50 | 439.07 | 435.01 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 426.60 | 438.94 | 434.97 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-05-04 09:15:00 | 436.95 | 438.40 | 434.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-05-04 10:15:00 | 435.10 | 438.37 | 434.83 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-05-05 15:15:00 | 436.40 | 437.77 | 434.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 438.00 | 437.78 | 434.74 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2026-05-07 10:15:00 | 436.30 | 437.72 | 434.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 11:15:00 | 437.85 | 437.72 | 434.85 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-12-24 14:15:00 | 400.25 | 2025-12-26 09:15:00 | 404.35 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-12-26 13:15:00 | 398.70 | 2026-01-02 09:15:00 | 404.85 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-03-17 14:15:00 | 439.50 | 2026-03-23 09:15:00 | 410.20 | STOP_HIT | 1.00 | -6.67% |
