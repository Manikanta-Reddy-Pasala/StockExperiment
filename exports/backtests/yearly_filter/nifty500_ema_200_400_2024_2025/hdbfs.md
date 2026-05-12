# HDB Financial Services Ltd. (HDBFS)

## Backtest Summary

- **Window:** 2025-07-02 09:15:00 → 2026-05-11 15:15:00 (1472 bars)
- **Last close:** 689.00
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
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 3 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -0.71% / -0.73%
- **Sum % (uncompounded):** -2.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.71% | -2.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.71% | -2.1% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.71% | -2.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 759.10 | 753.10 | 753.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 762.50 | 753.26 | 753.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 11:15:00 | 753.00 | 756.89 | 755.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 11:15:00 | 753.00 | 756.89 | 755.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 753.00 | 756.89 | 755.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 753.00 | 756.89 | 755.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 755.25 | 756.87 | 755.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 13:45:00 | 756.45 | 756.86 | 755.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 15:15:00 | 752.40 | 756.78 | 755.20 | SL hit (close<static) qty=1.00 sl=752.50 alert=retest2 |

### Cycle 2 — SELL (started 2026-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 15:15:00 | 712.20 | 754.47 | 754.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 705.50 | 750.55 | 752.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 11:15:00 | 729.50 | 729.09 | 738.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 12:00:00 | 729.50 | 729.09 | 738.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 690.60 | 633.28 | 663.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 15:15:00 | 685.00 | 636.14 | 664.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 13:15:00 | 682.90 | 643.38 | 665.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 15:15:00 | 685.35 | 656.31 | 666.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-01-08 13:45:00 | 756.45 | 2026-01-08 15:15:00 | 752.40 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2026-01-12 10:15:00 | 757.60 | 2026-01-12 10:15:00 | 752.05 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-01-12 12:00:00 | 758.55 | 2026-01-19 09:15:00 | 752.05 | STOP_HIT | 1.00 | -0.86% |
