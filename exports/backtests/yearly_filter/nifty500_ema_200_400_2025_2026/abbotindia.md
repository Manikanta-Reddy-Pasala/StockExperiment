# Abbott India Ltd. (ABBOTINDIA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 26850.00
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
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 3 |
| PARTIAL | 5 |
| TARGET_HIT | 6 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 0
- **Target hits / Stop hits / Partials:** 1 / 5 / 5
- **Avg / median % per leg:** 4.51% / 3.99%
- **Sum % (uncompounded):** 49.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 1 | 100.0% | 1 | 0 | 0 | 9.20% | 9.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 1 | 100.0% | 1 | 0 | 0 | 9.20% | 9.2% |
| SELL (all) | 10 | 10 | 100.0% | 0 | 5 | 5 | 4.04% | 40.4% |
| SELL @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 3.55% | 21.3% |
| SELL @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 0 | 2 | 2 | 4.79% | 19.2% |
| retest1 (combined) | 6 | 6 | 100.0% | 0 | 3 | 3 | 3.55% | 21.3% |
| retest2 (combined) | 5 | 5 | 100.0% | 1 | 2 | 2 | 5.67% | 28.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 31030.00 | 32679.58 | 32686.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 13:15:00 | 30945.00 | 32554.72 | 32622.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 15:15:00 | 29780.00 | 29700.23 | 30225.87 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 09:15:00 | 29515.00 | 29700.23 | 30225.87 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 10:00:00 | 29565.00 | 29698.89 | 30222.58 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 10:45:00 | 29565.00 | 29697.45 | 30219.25 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 30055.00 | 29702.87 | 30209.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:30:00 | 29830.00 | 29725.55 | 30200.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:30:00 | 29865.00 | 29707.86 | 30159.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 28338.50 | 29463.00 | 29942.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 28371.75 | 29463.00 | 29942.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 14:15:00 | 28039.25 | 29400.24 | 29899.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 14:15:00 | 28086.75 | 29400.24 | 29899.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 14:15:00 | 28086.75 | 29400.24 | 29899.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-12-24 15:15:00 | 28750.00 | 28647.72 | 29286.12 | SL hit (close>ema200) qty=0.50 sl=28647.72 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 14:00:00 | 30120.00 | 2025-06-26 10:15:00 | 32890.00 | TARGET_HIT | 1.00 | 9.20% |
| SELL | retest1 | 2025-11-25 09:15:00 | 29515.00 | 2025-12-09 09:15:00 | 28338.50 | PARTIAL | 0.50 | 3.99% |
| SELL | retest1 | 2025-11-25 10:00:00 | 29565.00 | 2025-12-09 09:15:00 | 28371.75 | PARTIAL | 0.50 | 4.04% |
| SELL | retest1 | 2025-11-25 10:45:00 | 29565.00 | 2025-12-09 14:15:00 | 28039.25 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2025-11-27 09:30:00 | 29830.00 | 2025-12-09 14:15:00 | 28086.75 | PARTIAL | 0.50 | 5.84% |
| SELL | retest2 | 2025-12-01 09:30:00 | 29865.00 | 2025-12-09 14:15:00 | 28086.75 | PARTIAL | 0.50 | 5.95% |
| SELL | retest1 | 2025-11-25 09:15:00 | 29515.00 | 2025-12-24 15:15:00 | 28750.00 | STOP_HIT | 0.50 | 2.59% |
| SELL | retest1 | 2025-11-25 10:00:00 | 29565.00 | 2025-12-24 15:15:00 | 28750.00 | STOP_HIT | 0.50 | 2.76% |
| SELL | retest1 | 2025-11-25 10:45:00 | 29565.00 | 2025-12-24 15:15:00 | 28750.00 | STOP_HIT | 0.50 | 2.76% |
| SELL | retest2 | 2025-11-27 09:30:00 | 29830.00 | 2025-12-24 15:15:00 | 28750.00 | STOP_HIT | 0.50 | 3.62% |
| SELL | retest2 | 2025-12-01 09:30:00 | 29865.00 | 2025-12-24 15:15:00 | 28750.00 | STOP_HIT | 0.50 | 3.73% |
