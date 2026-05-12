# Union Bank of India (UNIONBANK)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 166.50
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
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 6 |
| TARGET_HIT | 7 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 2
- **Target hits / Stop hits / Partials:** 1 / 8 / 6
- **Avg / median % per leg:** 3.74% / 3.77%
- **Sum % (uncompounded):** 56.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| SELL (all) | 14 | 12 | 85.7% | 0 | 8 | 6 | 3.29% | 46.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 12 | 85.7% | 0 | 8 | 6 | 3.29% | 46.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 13 | 86.7% | 1 | 8 | 6 | 3.74% | 56.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 128.68 | 139.96 | 140.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 13:15:00 | 127.21 | 139.83 | 139.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 14:15:00 | 137.43 | 137.12 | 138.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 14:45:00 | 137.32 | 137.12 | 138.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 137.67 | 137.14 | 138.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 11:30:00 | 137.40 | 137.14 | 138.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 12:00:00 | 137.49 | 137.14 | 138.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 14:15:00 | 136.90 | 137.16 | 138.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 15:00:00 | 136.94 | 137.15 | 138.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 138.08 | 137.17 | 138.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:30:00 | 137.84 | 137.17 | 138.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 137.51 | 137.16 | 138.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:30:00 | 136.20 | 137.14 | 138.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 14:00:00 | 136.16 | 137.13 | 138.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:15:00 | 130.53 | 136.78 | 137.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:15:00 | 130.62 | 136.78 | 137.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 11:15:00 | 130.06 | 136.64 | 137.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 11:15:00 | 130.09 | 136.64 | 137.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 12:15:00 | 129.39 | 136.57 | 137.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 12:15:00 | 129.35 | 136.57 | 137.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 13:15:00 | 132.30 | 132.25 | 134.90 | SL hit (close>ema200) qty=0.50 sl=132.25 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 138.46 | 136.02 | 136.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 139.80 | 136.10 | 136.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 136.41 | 136.91 | 136.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 136.41 | 136.91 | 136.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 136.41 | 136.91 | 136.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 136.41 | 136.91 | 136.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 136.27 | 136.90 | 136.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 136.21 | 136.90 | 136.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 136.72 | 136.90 | 136.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 138.88 | 136.90 | 136.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-07 13:15:00 | 152.77 | 142.80 | 140.07 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 13:15:00 | 166.46 | 177.08 | 177.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 10:15:00 | 165.18 | 176.04 | 176.57 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-20 11:30:00 | 137.40 | 2025-08-26 09:15:00 | 130.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-20 12:00:00 | 137.49 | 2025-08-26 09:15:00 | 130.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-20 14:15:00 | 136.90 | 2025-08-26 11:15:00 | 130.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-20 15:00:00 | 136.94 | 2025-08-26 11:15:00 | 130.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 12:30:00 | 136.20 | 2025-08-26 12:15:00 | 129.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 14:00:00 | 136.16 | 2025-08-26 12:15:00 | 129.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-20 11:30:00 | 137.40 | 2025-09-10 13:15:00 | 132.30 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2025-08-20 12:00:00 | 137.49 | 2025-09-10 13:15:00 | 132.30 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2025-08-20 14:15:00 | 136.90 | 2025-09-10 13:15:00 | 132.30 | STOP_HIT | 0.50 | 3.36% |
| SELL | retest2 | 2025-08-20 15:00:00 | 136.94 | 2025-09-10 13:15:00 | 132.30 | STOP_HIT | 0.50 | 3.39% |
| SELL | retest2 | 2025-08-22 12:30:00 | 136.20 | 2025-09-10 13:15:00 | 132.30 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2025-08-22 14:00:00 | 136.16 | 2025-09-10 13:15:00 | 132.30 | STOP_HIT | 0.50 | 2.83% |
| SELL | retest2 | 2025-09-17 11:30:00 | 135.87 | 2025-09-19 09:15:00 | 138.65 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-09-18 09:15:00 | 136.18 | 2025-09-19 09:15:00 | 138.65 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-10-20 09:15:00 | 138.88 | 2025-11-07 13:15:00 | 152.77 | TARGET_HIT | 1.00 | 10.00% |
