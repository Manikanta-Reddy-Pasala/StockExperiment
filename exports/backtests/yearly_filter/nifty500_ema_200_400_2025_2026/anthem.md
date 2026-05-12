# Anthem Biosciences Ltd. (ANTHEM)

## Backtest Summary

- **Window:** 2025-07-21 09:15:00 → 2026-05-11 15:15:00 (1381 bars)
- **Last close:** 785.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 6 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 6
- **Target hits / Stop hits / Partials:** 0 / 6 / 0
- **Avg / median % per leg:** -2.97% / -3.10%
- **Sum % (uncompounded):** -17.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.90% | -11.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.90% | -11.6% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.10% | -6.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.10% | -6.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.97% | -17.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 13:15:00 | 693.85 | 659.50 | 659.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 14:15:00 | 694.50 | 659.85 | 659.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 668.30 | 669.19 | 664.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 09:15:00 | 668.30 | 669.19 | 664.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 668.30 | 669.19 | 664.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:00:00 | 668.30 | 669.19 | 664.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 662.00 | 669.12 | 664.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 11:00:00 | 662.00 | 669.12 | 664.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 665.75 | 669.08 | 664.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 12:30:00 | 667.95 | 669.09 | 664.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:15:00 | 677.25 | 669.10 | 664.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 09:45:00 | 670.00 | 669.07 | 664.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 15:15:00 | 655.45 | 668.73 | 664.89 | SL hit (close<static) qty=1.00 sl=660.00 alert=retest2 |

### Cycle 2 — SELL (started 2026-03-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 12:15:00 | 639.55 | 661.65 | 661.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 10:15:00 | 636.65 | 660.66 | 661.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 656.60 | 653.33 | 657.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-25 10:00:00 | 656.60 | 653.33 | 657.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 649.00 | 653.29 | 657.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 15:00:00 | 645.00 | 653.05 | 656.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 12:00:00 | 645.00 | 652.64 | 656.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 13:15:00 | 665.00 | 652.71 | 656.60 | SL hit (close>static) qty=1.00 sl=658.00 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 10:15:00 | 719.00 | 659.82 | 659.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 11:15:00 | 728.35 | 660.50 | 659.98 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-03-04 12:30:00 | 667.95 | 2026-03-06 15:15:00 | 655.45 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2026-03-05 09:15:00 | 677.25 | 2026-03-06 15:15:00 | 655.45 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2026-03-06 09:45:00 | 670.00 | 2026-03-06 15:15:00 | 655.45 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2026-03-10 09:15:00 | 674.90 | 2026-03-11 09:15:00 | 645.50 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2026-03-25 15:00:00 | 645.00 | 2026-03-27 13:15:00 | 665.00 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2026-03-27 12:00:00 | 645.00 | 2026-03-27 13:15:00 | 665.00 | STOP_HIT | 1.00 | -3.10% |
