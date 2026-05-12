# Force Motors Ltd. (FORCEMOT)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 20851.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 7 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 0
- **Avg / median % per leg:** 0.27% / -1.23%
- **Sum % (uncompounded):** 1.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 1 | 14.3% | 1 | 6 | 0 | 0.27% | 1.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 1 | 14.3% | 1 | 6 | 0 | 0.27% | 1.9% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 1 | 14.3% | 1 | 6 | 0 | 0.27% | 1.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 14:15:00 | 16638.00 | 17198.00 | 17199.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 15:15:00 | 16530.00 | 17191.35 | 17195.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 09:15:00 | 17418.00 | 17193.61 | 17196.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 17418.00 | 17193.61 | 17196.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 17418.00 | 17193.61 | 17196.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:00:00 | 17418.00 | 17193.61 | 17196.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 17759.00 | 17199.23 | 17199.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:45:00 | 17759.00 | 17199.23 | 17199.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 11:15:00 | 17648.00 | 17203.70 | 17201.98 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 16339.00 | 17196.13 | 17199.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 13:15:00 | 16282.00 | 17140.94 | 17171.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 14:15:00 | 17121.00 | 17029.71 | 17109.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 14:15:00 | 17121.00 | 17029.71 | 17109.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 17121.00 | 17029.71 | 17109.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 15:00:00 | 17121.00 | 17029.71 | 17109.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 17269.00 | 17032.09 | 17109.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 17615.00 | 17032.09 | 17109.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 18332.00 | 17186.32 | 17183.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 09:15:00 | 18613.00 | 17249.31 | 17215.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 10:15:00 | 17408.00 | 17437.48 | 17325.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 10:30:00 | 17350.00 | 17437.48 | 17325.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 17307.00 | 17436.18 | 17325.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:30:00 | 17262.00 | 17436.18 | 17325.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 17275.00 | 17434.58 | 17325.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:30:00 | 17240.00 | 17434.58 | 17325.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 17300.00 | 17441.81 | 17338.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 17300.00 | 17441.81 | 17338.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 17427.00 | 17441.67 | 17338.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 12:00:00 | 17461.00 | 17441.86 | 17339.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 15:15:00 | 17452.00 | 17442.07 | 17341.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 17289.00 | 17453.94 | 17355.14 | SL hit (close<static) qty=1.00 sl=17301.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 12:15:00 | 19166.00 | 21237.66 | 21241.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 10:15:00 | 18956.00 | 21139.10 | 21191.04 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-18 12:00:00 | 17461.00 | 2025-11-21 09:15:00 | 17289.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-11-18 15:15:00 | 17452.00 | 2025-11-21 09:15:00 | 17289.00 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-11-28 09:15:00 | 17537.00 | 2025-12-05 09:15:00 | 17168.00 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-12-12 12:00:00 | 17515.00 | 2025-12-15 10:15:00 | 17300.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-15 10:15:00 | 17450.00 | 2025-12-15 12:15:00 | 17199.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-12-16 12:00:00 | 17423.00 | 2025-12-17 10:15:00 | 17181.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-12-19 09:15:00 | 17940.00 | 2025-12-30 14:15:00 | 19734.00 | TARGET_HIT | 1.00 | 10.00% |
