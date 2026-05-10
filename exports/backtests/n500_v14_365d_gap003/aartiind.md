# Aarti Industries Ltd. (AARTIIND)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 486.00
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
| ALERT3 | 11 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 0 |
| TARGET_HIT | 3 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 8
- **Target hits / Stop hits / Partials:** 3 / 8 / 0
- **Avg / median % per leg:** 0.42% / -1.99%
- **Sum % (uncompounded):** 4.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 3 | 27.3% | 3 | 8 | 0 | 0.42% | 4.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 3 | 27.3% | 3 | 8 | 0 | 0.42% | 4.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 3 | 27.3% | 3 | 8 | 0 | 0.42% | 4.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 12:15:00 | 447.85 | 374.88 | 374.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 14:15:00 | 454.90 | 376.39 | 375.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 417.15 | 429.09 | 410.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-04 10:00:00 | 417.15 | 429.09 | 410.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 408.15 | 428.33 | 410.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:30:00 | 416.25 | 428.17 | 410.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 14:45:00 | 422.45 | 427.45 | 410.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 13:45:00 | 415.95 | 426.89 | 410.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 09:45:00 | 415.05 | 425.38 | 410.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-12 12:15:00 | 456.56 | 426.31 | 412.42 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 413.90 | 426.94 | 413.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:00:00 | 413.90 | 426.94 | 413.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 416.60 | 426.84 | 413.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 11:15:00 | 419.00 | 426.84 | 413.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 11:45:00 | 424.55 | 426.81 | 413.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 13:00:00 | 417.75 | 426.43 | 413.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 14:15:00 | 411.45 | 426.19 | 414.74 | SL hit (close<static) qty=1.00 sl=412.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-19 14:15:00 | 411.45 | 426.19 | 414.74 | SL hit (close<static) qty=1.00 sl=412.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-19 14:15:00 | 411.45 | 426.19 | 414.74 | SL hit (close<static) qty=1.00 sl=412.65 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 418.15 | 426.07 | 414.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 415.45 | 425.89 | 414.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 415.35 | 425.89 | 414.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 410.65 | 425.66 | 414.97 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 410.65 | 425.66 | 414.97 | SL hit (close<static) qty=1.00 sl=412.65 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 410.65 | 425.66 | 414.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 407.45 | 425.48 | 414.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 11:00:00 | 407.45 | 425.48 | 414.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 409.20 | 424.14 | 414.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:30:00 | 405.15 | 424.14 | 414.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 416.00 | 424.11 | 415.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:30:00 | 416.15 | 424.11 | 415.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 413.75 | 423.89 | 415.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 410.85 | 423.89 | 415.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 407.40 | 423.73 | 415.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:00:00 | 407.40 | 423.73 | 415.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-30 14:15:00 | 399.50 | 422.72 | 415.01 | SL hit (close<static) qty=1.00 sl=400.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 14:15:00 | 399.50 | 422.72 | 415.01 | SL hit (close<static) qty=1.00 sl=400.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 14:15:00 | 399.50 | 422.72 | 415.01 | SL hit (close<static) qty=1.00 sl=400.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 411.65 | 422.29 | 414.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 12:00:00 | 414.65 | 422.22 | 414.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 398.85 | 421.61 | 414.78 | SL hit (close<static) qty=1.00 sl=410.80 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 416.50 | 418.27 | 413.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 09:30:00 | 414.45 | 419.63 | 414.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 09:15:00 | 455.90 | 426.25 | 419.32 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-22 13:15:00 | 458.15 | 427.51 | 420.09 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-03-05 09:30:00 | 416.25 | 2026-03-12 12:15:00 | 456.56 | TARGET_HIT | 1.00 | 9.68% |
| BUY | retest2 | 2026-03-05 14:45:00 | 422.45 | 2026-03-19 14:15:00 | 411.45 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2026-03-06 13:45:00 | 415.95 | 2026-03-19 14:15:00 | 411.45 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-03-10 09:45:00 | 415.05 | 2026-03-19 14:15:00 | 411.45 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-03-16 11:15:00 | 419.00 | 2026-03-23 09:15:00 | 410.65 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2026-03-16 11:45:00 | 424.55 | 2026-03-30 14:15:00 | 399.50 | STOP_HIT | 1.00 | -5.90% |
| BUY | retest2 | 2026-03-17 13:00:00 | 417.75 | 2026-03-30 14:15:00 | 399.50 | STOP_HIT | 1.00 | -4.37% |
| BUY | retest2 | 2026-03-20 09:15:00 | 418.15 | 2026-03-30 14:15:00 | 399.50 | STOP_HIT | 1.00 | -4.46% |
| BUY | retest2 | 2026-04-01 12:00:00 | 414.65 | 2026-04-02 09:15:00 | 398.85 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest2 | 2026-04-08 09:15:00 | 416.50 | 2026-04-22 09:15:00 | 455.90 | TARGET_HIT | 1.00 | 9.46% |
| BUY | retest2 | 2026-04-13 09:30:00 | 414.45 | 2026-04-22 13:15:00 | 458.15 | TARGET_HIT | 1.00 | 10.54% |
