# The New India Assurance Company Ltd. (NIACL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 163.20
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
| ALERT2_SKIP | 0 |
| ALERT3 | 7 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 1
- **Target hits / Stop hits / Partials:** 2 / 3 / 3
- **Avg / median % per leg:** 5.32% / 5.00%
- **Sum % (uncompounded):** 42.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 1 | 50.0% | 1 | 1 | 0 | 4.33% | 8.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 1 | 50.0% | 1 | 1 | 0 | 4.33% | 8.7% |
| SELL (all) | 6 | 6 | 100.0% | 1 | 2 | 3 | 5.65% | 33.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 6 | 100.0% | 1 | 2 | 3 | 5.65% | 33.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 7 | 87.5% | 2 | 3 | 3 | 5.32% | 42.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 15:15:00 | 174.50 | 166.88 | 166.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 176.35 | 166.98 | 166.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 14:15:00 | 182.82 | 183.34 | 177.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 15:00:00 | 182.82 | 183.34 | 177.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 177.73 | 183.14 | 178.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:30:00 | 177.74 | 183.14 | 178.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 177.79 | 183.08 | 178.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 176.83 | 183.08 | 178.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 177.90 | 183.03 | 178.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 177.90 | 183.03 | 178.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 176.40 | 182.97 | 178.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 09:15:00 | 178.00 | 182.97 | 178.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 13:15:00 | 175.61 | 184.43 | 182.36 | SL hit (close<static) qty=1.00 sl=176.01 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 182.04 | 189.83 | 189.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 178.82 | 188.48 | 189.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 09:15:00 | 152.40 | 151.79 | 159.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-01 10:00:00 | 152.40 | 151.79 | 159.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 157.75 | 150.74 | 157.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:00:00 | 157.75 | 150.74 | 157.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 159.32 | 150.82 | 157.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:00:00 | 159.32 | 150.82 | 157.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 160.13 | 150.91 | 157.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 14:00:00 | 159.14 | 151.00 | 157.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 14:45:00 | 158.80 | 151.07 | 157.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 156.65 | 151.15 | 157.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 09:15:00 | 151.18 | 151.71 | 157.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 10:15:00 | 150.86 | 151.71 | 157.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 11:15:00 | 151.88 | 151.71 | 157.09 | SL hit (close>ema200) qty=0.50 sl=151.71 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 15:15:00 | 163.20 | 146.10 | 146.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 166.89 | 146.31 | 146.18 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-20 09:15:00 | 178.00 | 2025-07-28 13:15:00 | 175.61 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-07-30 09:15:00 | 185.84 | 2025-07-30 14:15:00 | 204.42 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-10 14:00:00 | 159.14 | 2026-02-13 09:15:00 | 151.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 14:45:00 | 158.80 | 2026-02-13 10:15:00 | 150.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 14:00:00 | 159.14 | 2026-02-13 11:15:00 | 151.88 | STOP_HIT | 0.50 | 4.56% |
| SELL | retest2 | 2026-02-10 14:45:00 | 158.80 | 2026-02-13 11:15:00 | 151.88 | STOP_HIT | 0.50 | 4.36% |
| SELL | retest2 | 2026-02-11 09:15:00 | 156.65 | 2026-02-24 12:15:00 | 148.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 09:15:00 | 156.65 | 2026-03-02 09:15:00 | 140.99 | TARGET_HIT | 0.50 | 10.00% |
