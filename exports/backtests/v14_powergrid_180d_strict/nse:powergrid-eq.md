# NSE:POWERGRID-EQ (NSE:POWERGRID-EQ)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 313.90
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
| ALERT2_SKIP | 0 |
| ALERT3 | 4 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 5
- **Target hits / Stop hits / Partials:** 1 / 6 / 4
- **Avg / median % per leg:** 2.44% / 2.63%
- **Sum % (uncompounded):** 26.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 1 | 5 | 3 | 2.14% | 19.2% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 0 | 4 | 3 | 1.55% | 10.8% |
| BUY @ 3rd Alert (retest2) | 2 | 1 | 50.0% | 1 | 1 | 0 | 4.21% | 8.4% |
| SELL (all) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.81% | 7.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.81% | 7.6% |
| retest1 (combined) | 7 | 3 | 42.9% | 0 | 4 | 3 | 1.55% | 10.8% |
| retest2 (combined) | 4 | 3 | 75.0% | 1 | 2 | 1 | 4.01% | 16.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 11:15:00 | 292.80 | 269.48 | 269.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 13:15:00 | 293.80 | 269.95 | 269.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 289.75 | 290.18 | 282.80 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 11:45:00 | 293.65 | 290.22 | 282.89 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 12:45:00 | 293.10 | 290.24 | 282.94 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 13:15:00 | 293.35 | 290.24 | 282.94 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 308.33 | 292.36 | 284.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 307.76 | 292.36 | 284.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 308.02 | 292.36 | 284.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 291.90 | 292.94 | 285.50 | SL hit (close<ema200) qty=0.50 sl=292.94 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 291.90 | 292.94 | 285.50 | SL hit (close<ema200) qty=0.50 sl=292.94 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 291.90 | 292.94 | 285.50 | SL hit (close<ema200) qty=0.50 sl=292.94 alert=retest1 |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 11:15:00 | 294.50 | 292.94 | 285.50 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | SL hit (close<ema400) qty=1.00 sl=289.21 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-04-02 09:30:00 | 287.25 | 295.14 | 289.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 284.50 | 295.04 | 289.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:45:00 | 285.60 | 295.04 | 289.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 290.95 | 294.71 | 289.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 292.60 | 294.71 | 289.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 10:15:00 | 287.95 | 294.61 | 289.17 | SL hit (close<static) qty=1.00 sl=289.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:30:00 | 292.10 | 294.50 | 289.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-20 10:15:00 | 321.31 | 299.38 | 293.20 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-01-06 11:15:00 | 268.55 | 2026-01-12 09:15:00 | 255.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 11:15:00 | 268.55 | 2026-01-29 15:15:00 | 261.50 | STOP_HIT | 0.50 | 2.63% |
| BUY | retest1 | 2026-03-09 11:45:00 | 293.65 | 2026-03-13 09:15:00 | 308.33 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-03-09 12:45:00 | 293.10 | 2026-03-13 09:15:00 | 307.76 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-03-09 13:15:00 | 293.35 | 2026-03-13 09:15:00 | 308.02 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-03-09 11:45:00 | 293.65 | 2026-03-16 10:15:00 | 291.90 | STOP_HIT | 0.50 | -0.60% |
| BUY | retest1 | 2026-03-09 12:45:00 | 293.10 | 2026-03-16 10:15:00 | 291.90 | STOP_HIT | 0.50 | -0.41% |
| BUY | retest1 | 2026-03-09 13:15:00 | 293.35 | 2026-03-16 10:15:00 | 291.90 | STOP_HIT | 0.50 | -0.49% |
| BUY | retest1 | 2026-03-16 11:15:00 | 294.50 | 2026-04-02 09:15:00 | 286.60 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2026-04-06 09:15:00 | 292.60 | 2026-04-06 10:15:00 | 287.95 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-04-06 13:30:00 | 292.10 | 2026-04-20 10:15:00 | 321.31 | TARGET_HIT | 1.00 | 10.00% |
