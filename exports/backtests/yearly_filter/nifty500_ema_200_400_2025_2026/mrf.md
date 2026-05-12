# MRF Ltd. (MRF)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 130490.00
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
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 8 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 3 |
| TARGET_HIT | 4 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 8
- **Target hits / Stop hits / Partials:** 4 / 8 / 3
- **Avg / median % per leg:** 3.05% / -1.02%
- **Sum % (uncompounded):** 45.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 1 | 11.1% | 1 | 8 | 0 | 0.08% | 0.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 1 | 11.1% | 1 | 8 | 0 | 0.08% | 0.7% |
| SELL (all) | 6 | 6 | 100.0% | 3 | 0 | 3 | 7.50% | 45.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 6 | 100.0% | 3 | 0 | 3 | 7.50% | 45.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 7 | 46.7% | 4 | 8 | 3 | 3.05% | 45.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 15:15:00 | 148510.00 | 153067.73 | 153080.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 148100.00 | 151991.20 | 152478.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 12:15:00 | 143855.00 | 141379.96 | 145534.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 12:15:00 | 143855.00 | 141379.96 | 145534.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 143855.00 | 141379.96 | 145534.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 143855.00 | 141379.96 | 145534.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 146700.00 | 141432.90 | 145540.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 11:00:00 | 143215.00 | 144873.99 | 146277.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 13:45:00 | 143190.00 | 144824.09 | 146231.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:45:00 | 143080.00 | 144859.01 | 146130.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 136054.25 | 144203.07 | 145713.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 136030.50 | 144203.07 | 145713.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 135926.00 | 144203.07 | 145713.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-20 13:15:00 | 128893.50 | 138565.53 | 141846.72 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-11 10:30:00 | 142995.00 | 2025-08-11 15:15:00 | 141500.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-08-11 11:00:00 | 143060.00 | 2025-08-11 15:15:00 | 141500.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-08-11 11:45:00 | 143005.00 | 2025-08-11 15:15:00 | 141500.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-08-11 12:15:00 | 142955.00 | 2025-08-11 15:15:00 | 141500.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-08-12 09:45:00 | 142400.00 | 2025-08-12 14:15:00 | 140600.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-08-18 10:30:00 | 142400.00 | 2025-08-29 09:15:00 | 140765.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-08-28 10:30:00 | 142900.00 | 2025-08-29 09:15:00 | 140765.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-08-28 14:15:00 | 142410.00 | 2025-08-29 09:15:00 | 140765.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-09-01 14:15:00 | 144650.00 | 2025-10-08 11:15:00 | 159115.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-24 11:00:00 | 143215.00 | 2026-03-04 09:15:00 | 136054.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 13:45:00 | 143190.00 | 2026-03-04 09:15:00 | 136030.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 10:45:00 | 143080.00 | 2026-03-04 09:15:00 | 135926.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 11:00:00 | 143215.00 | 2026-03-20 13:15:00 | 128893.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 13:45:00 | 143190.00 | 2026-03-20 13:15:00 | 128871.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-27 10:45:00 | 143080.00 | 2026-03-20 13:15:00 | 128772.00 | TARGET_HIT | 0.50 | 10.00% |
