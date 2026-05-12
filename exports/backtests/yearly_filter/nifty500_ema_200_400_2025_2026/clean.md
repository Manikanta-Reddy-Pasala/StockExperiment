# Clean Science and Technology Ltd. (CLEAN)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 891.90
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
| ALERT3 | 8 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 0
- **Avg / median % per leg:** -10.46% / -11.48%
- **Sum % (uncompounded):** -52.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -10.46% | -52.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -10.46% | -52.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -10.46% | -52.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 14:15:00 | 1425.20 | 1255.50 | 1254.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 1441.50 | 1271.93 | 1263.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 1407.20 | 1412.68 | 1360.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 15:00:00 | 1407.20 | 1412.68 | 1360.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1350.60 | 1444.01 | 1407.19 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 10:15:00 | 1233.90 | 1379.04 | 1379.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 13:15:00 | 1224.40 | 1357.26 | 1367.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 09:15:00 | 1195.70 | 1192.79 | 1239.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 09:30:00 | 1200.90 | 1192.79 | 1239.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 747.30 | 721.12 | 757.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:30:00 | 751.40 | 721.12 | 757.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 752.00 | 722.94 | 757.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:45:00 | 746.30 | 723.45 | 757.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 14:30:00 | 747.85 | 724.15 | 757.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 09:30:00 | 742.40 | 727.02 | 757.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 13:15:00 | 745.15 | 727.72 | 757.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 755.85 | 728.69 | 757.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:30:00 | 758.25 | 728.69 | 757.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 753.90 | 729.44 | 757.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:30:00 | 757.35 | 729.44 | 757.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 759.00 | 730.58 | 757.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 12:00:00 | 759.00 | 730.58 | 757.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 766.45 | 730.93 | 757.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 12:30:00 | 770.05 | 730.93 | 757.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 764.05 | 731.26 | 757.10 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 831.95 | 732.94 | 757.56 | SL hit (close>static) qty=1.00 sl=770.00 alert=retest2 |

### Cycle 3 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 875.60 | 775.43 | 775.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 15:15:00 | 879.00 | 779.40 | 777.05 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-23 10:00:00 | 1359.00 | 2025-05-23 14:15:00 | 1438.60 | STOP_HIT | 1.00 | -5.86% |
| SELL | retest2 | 2026-04-16 11:45:00 | 746.30 | 2026-04-23 09:15:00 | 831.95 | STOP_HIT | 1.00 | -11.48% |
| SELL | retest2 | 2026-04-16 14:30:00 | 747.85 | 2026-04-23 09:15:00 | 831.95 | STOP_HIT | 1.00 | -11.25% |
| SELL | retest2 | 2026-04-20 09:30:00 | 742.40 | 2026-04-23 09:15:00 | 831.95 | STOP_HIT | 1.00 | -12.06% |
| SELL | retest2 | 2026-04-20 13:15:00 | 745.15 | 2026-04-23 09:15:00 | 831.95 | STOP_HIT | 1.00 | -11.65% |
