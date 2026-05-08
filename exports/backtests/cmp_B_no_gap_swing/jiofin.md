# JIOFIN (JIOFIN)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 249.01
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
| ALERT2_SKIP | 0 |
| ALERT3 | 1 |
| PENDING | 5 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 2 |
| ENTRY2 | 2 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 2.23% / -1.05%
- **Sum % (uncompounded):** 11.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 2.23% | 11.2% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.39% | -2.8% |
| SELL @ 3rd Alert (retest2) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.65% | 14.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.39% | -2.8% |
| retest2 (combined) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.65% | 14.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 10:15:00 | 295.20 | 310.85 | 310.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 291.90 | 305.13 | 306.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 302.30 | 300.48 | 303.41 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-12-24 13:15:00 | 298.60 | 300.45 | 303.34 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 14:15:00 | 298.60 | 300.44 | 303.32 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-26 10:15:00 | 298.50 | 300.39 | 303.25 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-26 11:15:00 | 299.05 | 300.37 | 303.23 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-26 12:15:00 | 297.20 | 300.34 | 303.20 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 13:15:00 | 297.30 | 300.31 | 303.17 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 301.55 | 298.92 | 301.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-02 12:15:00 | 302.10 | 298.95 | 301.97 | SL hit (close>ema400) qty=1.00 sl=301.97 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-02 12:15:00 | 302.10 | 298.95 | 301.97 | SL hit (close>ema400) qty=1.00 sl=301.97 alert=retest1 |
| Cross detected — sustain check pending | 2026-01-05 09:15:00 | 300.50 | 299.05 | 301.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 10:15:00 | 301.15 | 299.07 | 301.96 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 304.30 | 299.10 | 301.78 | SL hit (close>static) qty=1.00 sl=303.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-08 10:15:00 | 296.50 | 299.32 | 301.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:15:00 | 296.55 | 299.30 | 301.78 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 281.72 | 298.06 | 300.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 14:15:00 | 266.90 | 292.42 | 297.44 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-12-24 14:15:00 | 298.60 | 2026-01-02 12:15:00 | 302.10 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest1 | 2025-12-26 13:15:00 | 297.30 | 2026-01-02 12:15:00 | 302.10 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-01-05 10:15:00 | 301.15 | 2026-01-07 10:15:00 | 304.30 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-01-08 11:15:00 | 296.55 | 2026-01-12 11:15:00 | 281.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:15:00 | 296.55 | 2026-01-20 14:15:00 | 266.90 | TARGET_HIT | 0.50 | 10.00% |
