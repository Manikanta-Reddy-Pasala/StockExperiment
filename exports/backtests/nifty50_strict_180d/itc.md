# ITC (ITC)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:30:00 (1238 bars)
- **Last close:** 307.45
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 2 |
| PENDING | 7 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 2 / 2 / 2
- **Avg / median % per leg:** 4.66% / 5.00%
- **Sum % (uncompounded):** 27.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 4 | 66.7% | 2 | 2 | 2 | 4.66% | 28.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 4 | 66.7% | 2 | 2 | 2 | 4.66% | 28.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 6 | 4 | 66.7% | 2 | 2 | 2 | 4.66% | 28.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 10:15:00 | 403.10 | 406.93 | 406.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 09:15:00 | 401.60 | 406.73 | 406.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 10:15:00 | 405.10 | 404.39 | 405.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 10:15:00 | 405.10 | 404.39 | 405.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 405.10 | 404.39 | 405.41 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-12-16 12:15:00 | 402.00 | 404.35 | 405.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 13:15:00 | 403.00 | 404.34 | 405.37 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-22 12:15:00 | 403.05 | 403.63 | 404.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 13:15:00 | 403.00 | 403.63 | 404.85 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-23 14:15:00 | 407.10 | 403.70 | 404.84 | SL hit (close>static) qty=1.00 sl=405.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 14:15:00 | 407.10 | 403.70 | 404.84 | SL hit (close>static) qty=1.00 sl=405.65 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-29 09:15:00 | 402.55 | 404.00 | 404.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:15:00 | 402.00 | 403.98 | 404.90 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-31 15:15:00 | 402.65 | 403.69 | 404.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 379.20 | 403.45 | 404.53 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 09:15:00 | 381.90 | 403.45 | 404.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-02 09:15:00 | 361.80 | 400.74 | 403.12 | Target hit (10%) qty=0.50 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 09:15:00 | 360.24 | 400.74 | 403.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-06 13:15:00 | 341.28 | 392.16 | 398.44 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 313.55 | 304.49 | 313.91 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-05-05 09:15:00 | 310.90 | 306.24 | 313.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:15:00 | 310.25 | 306.28 | 313.92 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-05-05 15:15:00 | 310.95 | 306.53 | 313.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-05-06 09:15:00 | 311.85 | 306.58 | 313.84 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-05-06 10:15:00 | 310.10 | 306.61 | 313.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:15:00 | 310.00 | 306.65 | 313.81 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-12-16 13:15:00 | 403.00 | 2025-12-23 14:15:00 | 407.10 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-12-22 13:15:00 | 403.00 | 2025-12-23 14:15:00 | 407.10 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-12-29 10:15:00 | 402.00 | 2026-01-01 09:15:00 | 381.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 10:15:00 | 402.00 | 2026-01-02 09:15:00 | 361.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-01 09:15:00 | 379.20 | 2026-01-02 09:15:00 | 360.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 09:15:00 | 379.20 | 2026-01-06 13:15:00 | 341.28 | TARGET_HIT | 0.50 | 10.00% |
