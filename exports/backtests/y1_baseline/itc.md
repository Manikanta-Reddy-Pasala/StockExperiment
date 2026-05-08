# ITC (ITC)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 307.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 1 |
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
- **Avg / median % per leg:** 4.85% / 5.00%
- **Sum % (uncompounded):** 14.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.85% | 14.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.44% | -0.4% |
| SELL @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.44% | -0.4% |
| retest2 (combined) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 12:15:00 | 420.85 | 410.45 | 410.44 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 14:15:00 | 404.00 | 410.47 | 410.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 10:15:00 | 403.05 | 408.76 | 409.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 13:15:00 | 408.35 | 408.27 | 409.20 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-11-24 10:15:00 | 406.50 | 408.24 | 409.16 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 11:15:00 | 405.30 | 408.21 | 409.14 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 407.10 | 403.73 | 405.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-23 14:15:00 | 407.10 | 403.73 | 405.63 | SL hit (close>ema400) qty=1.00 sl=405.63 alert=retest1 |
| Cross detected — sustain check pending | 2025-12-26 09:15:00 | 405.00 | 404.01 | 405.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 404.70 | 404.01 | 405.68 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 09:15:00 | 384.46 | 403.47 | 405.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-01 11:15:00 | 364.23 | 402.74 | 404.81 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-11-24 11:15:00 | 405.30 | 2025-12-23 14:15:00 | 407.10 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-12-26 10:15:00 | 404.70 | 2026-01-01 09:15:00 | 384.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-26 10:15:00 | 404.70 | 2026-01-01 11:15:00 | 364.23 | TARGET_HIT | 0.50 | 10.00% |
