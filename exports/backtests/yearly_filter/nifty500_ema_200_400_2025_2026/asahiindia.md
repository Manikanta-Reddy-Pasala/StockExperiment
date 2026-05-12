# Asahi India Glass Ltd. (ASAHIINDIA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 836.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 11 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 2 |
| TARGET_HIT | 4 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 4
- **Target hits / Stop hits / Partials:** 4 / 4 / 2
- **Avg / median % per leg:** 4.12% / 5.00%
- **Sum % (uncompounded):** 41.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 2 | 4 | 0 | 1.86% | 11.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 2 | 33.3% | 2 | 4 | 0 | 1.86% | 11.2% |
| SELL (all) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 10 | 6 | 60.0% | 4 | 4 | 2 | 4.12% | 41.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 935.30 | 962.19 | 962.32 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 981.90 | 962.53 | 962.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 993.90 | 963.07 | 962.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 961.90 | 970.26 | 966.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 10:15:00 | 961.90 | 970.26 | 966.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 961.90 | 970.26 | 966.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 961.90 | 970.26 | 966.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 964.00 | 970.20 | 966.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:45:00 | 959.90 | 970.20 | 966.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 964.50 | 969.96 | 966.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:15:00 | 965.00 | 969.96 | 966.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 967.90 | 969.94 | 966.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:00:00 | 973.60 | 969.98 | 966.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:30:00 | 975.90 | 970.05 | 966.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 15:15:00 | 974.00 | 970.05 | 966.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 950.90 | 969.90 | 966.73 | SL hit (close<static) qty=1.00 sl=962.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 934.50 | 964.87 | 965.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 09:15:00 | 931.20 | 964.03 | 964.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 914.00 | 894.04 | 921.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-18 10:00:00 | 914.00 | 894.04 | 921.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 946.30 | 894.56 | 921.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 946.30 | 894.56 | 921.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 885.00 | 894.46 | 921.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 14:15:00 | 876.60 | 894.56 | 921.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 14:45:00 | 866.95 | 894.26 | 920.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 15:15:00 | 832.77 | 891.07 | 918.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 09:15:00 | 823.60 | 890.50 | 917.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-23 09:15:00 | 788.94 | 885.30 | 914.30 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-26 11:30:00 | 721.80 | 2025-07-01 12:15:00 | 793.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-26 15:00:00 | 717.50 | 2025-07-01 12:15:00 | 789.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-05 13:00:00 | 973.60 | 2026-02-06 09:15:00 | 950.90 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-02-05 13:30:00 | 975.90 | 2026-02-06 09:15:00 | 950.90 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2026-02-05 15:15:00 | 974.00 | 2026-02-06 09:15:00 | 950.90 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2026-02-06 13:00:00 | 974.80 | 2026-02-13 09:15:00 | 959.70 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-03-18 14:15:00 | 876.60 | 2026-03-19 15:15:00 | 832.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 14:45:00 | 866.95 | 2026-03-20 09:15:00 | 823.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 14:15:00 | 876.60 | 2026-03-23 09:15:00 | 788.94 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-18 14:45:00 | 866.95 | 2026-03-24 10:15:00 | 780.26 | TARGET_HIT | 0.50 | 10.00% |
