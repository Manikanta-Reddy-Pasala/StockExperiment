# Linde India Ltd. (LINDEINDIA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 7765.00
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
| ALERT2_SKIP | 0 |
| ALERT3 | 10 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 9 |
| TARGET_HIT | 9 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 1
- **Target hits / Stop hits / Partials:** 9 / 1 / 9
- **Avg / median % per leg:** 6.99% / 5.00%
- **Sum % (uncompounded):** 132.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.28% | -2.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.28% | -2.3% |
| SELL (all) | 18 | 18 | 100.0% | 9 | 0 | 9 | 7.50% | 135.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 18 | 100.0% | 9 | 0 | 9 | 7.50% | 135.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 18 | 94.7% | 9 | 1 | 9 | 6.99% | 132.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 6639.00 | 6773.76 | 6773.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 6606.00 | 6769.01 | 6771.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 11:15:00 | 6559.00 | 6466.98 | 6573.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-26 12:00:00 | 6559.00 | 6466.98 | 6573.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 6525.00 | 6467.56 | 6573.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 13:45:00 | 6490.50 | 6468.00 | 6573.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 14:45:00 | 6474.00 | 6467.35 | 6572.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:30:00 | 6480.00 | 6443.34 | 6530.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:00:00 | 6491.50 | 6446.73 | 6526.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 6528.00 | 6447.88 | 6521.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:00:00 | 6528.00 | 6447.88 | 6521.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 6513.50 | 6448.53 | 6521.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 6484.50 | 6455.83 | 6521.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 10:30:00 | 6491.00 | 6456.22 | 6521.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 12:30:00 | 6495.50 | 6457.00 | 6521.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 13:45:00 | 6487.00 | 6445.37 | 6506.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 6479.00 | 6446.25 | 6505.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:45:00 | 6400.00 | 6445.79 | 6505.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:15:00 | 6165.97 | 6382.82 | 6457.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:15:00 | 6150.30 | 6382.82 | 6457.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:15:00 | 6156.00 | 6382.82 | 6457.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:15:00 | 6166.92 | 6382.82 | 6457.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:15:00 | 6160.27 | 6382.82 | 6457.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:15:00 | 6166.45 | 6382.82 | 6457.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:15:00 | 6170.72 | 6382.82 | 6457.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:15:00 | 6162.65 | 6382.82 | 6457.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 09:15:00 | 6080.00 | 6253.85 | 6354.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-11-06 14:15:00 | 5841.45 | 6149.98 | 6269.40 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 12:15:00 | 6606.00 | 6004.07 | 6003.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 13:15:00 | 6909.00 | 6105.42 | 6056.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 10:15:00 | 6597.50 | 6769.09 | 6532.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 11:00:00 | 6597.50 | 6769.09 | 6532.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-08 15:00:00 | 6898.50 | 2025-07-10 09:15:00 | 6741.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-08-26 13:45:00 | 6490.50 | 2025-10-06 10:15:00 | 6165.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 14:45:00 | 6474.00 | 2025-10-06 10:15:00 | 6150.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-09 11:30:00 | 6480.00 | 2025-10-06 10:15:00 | 6156.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-11 11:00:00 | 6491.50 | 2025-10-06 10:15:00 | 6166.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 09:45:00 | 6484.50 | 2025-10-06 10:15:00 | 6160.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 10:30:00 | 6491.00 | 2025-10-06 10:15:00 | 6166.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 12:30:00 | 6495.50 | 2025-10-06 10:15:00 | 6170.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 13:45:00 | 6487.00 | 2025-10-06 10:15:00 | 6162.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 10:45:00 | 6400.00 | 2025-10-24 09:15:00 | 6080.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 13:45:00 | 6490.50 | 2025-11-06 14:15:00 | 5841.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-26 14:45:00 | 6474.00 | 2025-11-06 14:15:00 | 5826.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-09 11:30:00 | 6480.00 | 2025-11-06 14:15:00 | 5832.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-11 11:00:00 | 6491.50 | 2025-11-06 14:15:00 | 5842.35 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-17 09:45:00 | 6484.50 | 2025-11-06 14:15:00 | 5836.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-17 10:30:00 | 6491.00 | 2025-11-06 14:15:00 | 5841.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-17 12:30:00 | 6495.50 | 2025-11-06 14:15:00 | 5845.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-23 13:45:00 | 6487.00 | 2025-11-06 14:15:00 | 5838.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-24 10:45:00 | 6400.00 | 2025-11-11 10:15:00 | 5760.00 | TARGET_HIT | 0.50 | 10.00% |
