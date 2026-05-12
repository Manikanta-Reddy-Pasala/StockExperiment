# Castrol India Ltd. (CASTROLIND)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 185.00
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
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 5 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 5 |
| TARGET_HIT | 5 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 6
- **Target hits / Stop hits / Partials:** 5 / 7 / 5
- **Avg / median % per leg:** 3.26% / 5.00%
- **Sum % (uncompounded):** 55.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 1 | 14.3% | 0 | 7 | 0 | -2.79% | -19.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 1 | 14.3% | 0 | 7 | 0 | -2.79% | -19.5% |
| SELL (all) | 10 | 10 | 100.0% | 5 | 0 | 5 | 7.50% | 75.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 10 | 100.0% | 5 | 0 | 5 | 7.50% | 75.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 17 | 11 | 64.7% | 5 | 7 | 5 | 3.26% | 55.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 218.70 | 207.03 | 206.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 222.25 | 208.66 | 207.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 212.50 | 213.14 | 210.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 212.50 | 213.14 | 210.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 212.50 | 213.14 | 210.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 213.45 | 213.14 | 210.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 14:30:00 | 213.52 | 212.90 | 210.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 09:45:00 | 213.43 | 212.89 | 210.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 15:15:00 | 208.00 | 212.52 | 210.59 | SL hit (close<static) qty=1.00 sl=208.08 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 11:15:00 | 206.55 | 215.56 | 215.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 205.51 | 213.36 | 214.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 09:15:00 | 206.12 | 204.13 | 208.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 09:15:00 | 206.12 | 204.13 | 208.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 206.12 | 204.13 | 208.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 203.70 | 204.21 | 208.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 11:00:00 | 203.70 | 204.20 | 208.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 12:15:00 | 203.75 | 204.06 | 207.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 12:45:00 | 203.57 | 204.06 | 207.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 204.55 | 201.57 | 204.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:30:00 | 203.79 | 201.59 | 204.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 193.51 | 199.98 | 202.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 193.51 | 199.98 | 202.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 193.56 | 199.98 | 202.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 193.39 | 199.98 | 202.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 193.60 | 199.98 | 202.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-15 09:15:00 | 183.33 | 190.91 | 194.69 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-13 10:15:00 | 213.45 | 2025-06-18 15:15:00 | 208.00 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-06-16 14:30:00 | 213.52 | 2025-06-18 15:15:00 | 208.00 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-06-17 09:45:00 | 213.43 | 2025-06-18 15:15:00 | 208.00 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-06-26 10:00:00 | 213.91 | 2025-08-06 11:15:00 | 215.50 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2025-08-01 09:15:00 | 220.56 | 2025-08-06 11:15:00 | 215.50 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-08-01 09:45:00 | 221.24 | 2025-08-06 11:15:00 | 215.50 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-08-05 09:15:00 | 223.15 | 2025-08-14 09:15:00 | 205.95 | STOP_HIT | 1.00 | -7.71% |
| SELL | retest2 | 2025-09-18 09:15:00 | 203.70 | 2025-11-03 09:15:00 | 193.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 11:00:00 | 203.70 | 2025-11-03 09:15:00 | 193.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 12:15:00 | 203.75 | 2025-11-03 09:15:00 | 193.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 12:45:00 | 203.57 | 2025-11-03 09:15:00 | 193.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 09:30:00 | 203.79 | 2025-11-03 09:15:00 | 193.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 09:15:00 | 203.70 | 2025-12-15 09:15:00 | 183.33 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-18 11:00:00 | 203.70 | 2025-12-15 09:15:00 | 183.33 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-22 12:15:00 | 203.75 | 2025-12-15 09:15:00 | 183.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-22 12:45:00 | 203.57 | 2025-12-15 09:15:00 | 183.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-16 09:30:00 | 203.79 | 2025-12-15 09:15:00 | 183.41 | TARGET_HIT | 0.50 | 10.00% |
