# Power Finance Corporation Ltd. (PFC)

## Backtest Summary

- **Window:** 2025-10-27 09:15:00 → 2026-05-08 15:15:00 (917 bars)
- **Last close:** 461.60
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
| ALERT3 | 4 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / Stop hits / Partials:** 2 / 6 / 0
- **Avg / median % per leg:** -0.70% / -3.16%
- **Sum % (uncompounded):** -5.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 2 | 6 | 0 | -0.70% | -5.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 2 | 25.0% | 2 | 6 | 0 | -0.70% | -5.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 2 | 25.0% | 2 | 6 | 0 | -0.70% | -5.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 13:15:00 | 383.70 | 367.51 | 367.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 386.25 | 368.69 | 368.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 397.55 | 401.95 | 390.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-04 10:00:00 | 397.55 | 401.95 | 390.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 388.35 | 402.67 | 391.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 10:00:00 | 401.50 | 401.81 | 391.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 11:00:00 | 400.85 | 403.70 | 394.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 405.05 | 406.48 | 397.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 400.90 | 406.41 | 397.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 398.25 | 406.33 | 397.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:45:00 | 397.15 | 406.33 | 397.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 397.40 | 406.24 | 397.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 397.40 | 406.24 | 397.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 400.20 | 406.18 | 397.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 09:15:00 | 407.90 | 405.97 | 397.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 12:45:00 | 402.85 | 405.62 | 397.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 388.80 | 405.21 | 397.82 | SL hit (close<static) qty=1.00 sl=395.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 388.80 | 405.21 | 397.82 | SL hit (close<static) qty=1.00 sl=395.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 10:15:00 | 384.55 | 405.01 | 397.75 | SL hit (close<static) qty=1.00 sl=385.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 10:15:00 | 384.55 | 405.01 | 397.75 | SL hit (close<static) qty=1.00 sl=385.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 10:15:00 | 384.55 | 405.01 | 397.75 | SL hit (close<static) qty=1.00 sl=385.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 10:15:00 | 384.55 | 405.01 | 397.75 | SL hit (close<static) qty=1.00 sl=385.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 15:00:00 | 403.35 | 402.91 | 397.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 404.05 | 402.89 | 397.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 443.69 | 409.42 | 401.85 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-15 09:15:00 | 444.46 | 409.42 | 401.85 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-03-10 10:00:00 | 401.50 | 2026-03-30 09:15:00 | 388.80 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2026-03-16 11:00:00 | 400.85 | 2026-03-30 09:15:00 | 388.80 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2026-03-24 09:15:00 | 405.05 | 2026-03-30 10:15:00 | 384.55 | STOP_HIT | 1.00 | -5.06% |
| BUY | retest2 | 2026-03-24 10:15:00 | 400.90 | 2026-03-30 10:15:00 | 384.55 | STOP_HIT | 1.00 | -4.08% |
| BUY | retest2 | 2026-03-25 09:15:00 | 407.90 | 2026-03-30 10:15:00 | 384.55 | STOP_HIT | 1.00 | -5.72% |
| BUY | retest2 | 2026-03-27 12:45:00 | 402.85 | 2026-03-30 10:15:00 | 384.55 | STOP_HIT | 1.00 | -4.54% |
| BUY | retest2 | 2026-04-02 15:00:00 | 403.35 | 2026-04-15 09:15:00 | 443.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 09:15:00 | 404.05 | 2026-04-15 09:15:00 | 444.46 | TARGET_HIT | 1.00 | 10.00% |
