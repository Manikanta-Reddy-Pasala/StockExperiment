# National Aluminium Co. Ltd. (NATIONALUM)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 401.75
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
| ALERT3 | 7 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 8
- **Target hits / Stop hits / Partials:** 2 / 8 / 0
- **Avg / median % per leg:** 0.68% / -1.15%
- **Sum % (uncompounded):** 6.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 2 | 20.0% | 2 | 8 | 0 | 0.68% | 6.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 2 | 20.0% | 2 | 8 | 0 | 0.68% | 6.8% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 10 | 2 | 20.0% | 2 | 8 | 0 | 0.68% | 6.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 185.45 | 176.39 | 176.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 188.00 | 176.51 | 176.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 15:15:00 | 181.50 | 181.64 | 179.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:15:00 | 182.42 | 181.64 | 179.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 188.93 | 190.12 | 186.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 14:45:00 | 189.76 | 190.07 | 186.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 185.83 | 189.86 | 186.78 | SL hit (close<static) qty=1.00 sl=186.02 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-29 14:45:00 | 189.76 | 2025-07-31 09:15:00 | 185.83 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-08-08 10:15:00 | 189.84 | 2025-08-26 14:15:00 | 185.87 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-08-13 09:15:00 | 190.31 | 2025-08-26 14:15:00 | 185.87 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-08-13 12:15:00 | 189.63 | 2025-08-26 14:15:00 | 185.87 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-08-14 15:00:00 | 187.93 | 2025-08-26 14:15:00 | 185.87 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-08-18 09:30:00 | 188.03 | 2025-08-26 14:15:00 | 185.87 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-08-28 11:30:00 | 187.62 | 2025-08-28 12:15:00 | 184.81 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-08-29 11:00:00 | 188.00 | 2025-08-29 14:15:00 | 186.17 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-09-01 09:15:00 | 188.57 | 2025-09-03 12:15:00 | 207.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-01 10:30:00 | 188.65 | 2025-09-03 12:15:00 | 207.52 | TARGET_HIT | 1.00 | 10.00% |
