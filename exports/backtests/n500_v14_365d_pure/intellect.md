# Intellect Design Arena Ltd. (INTELLECT)

## Backtest Summary

- **Window:** 2025-01-15 09:15:00 → 2026-05-08 15:15:00 (2263 bars)
- **Last close:** 808.00
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
| ALERT2_SKIP | 0 |
| ALERT3 | 10 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 9 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 12
- **Target hits / Stop hits / Partials:** 0 / 12 / 4
- **Avg / median % per leg:** -1.76% / -1.98%
- **Sum % (uncompounded):** -28.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 4 | 25.0% | 0 | 12 | 4 | -1.76% | -28.1% |
| SELL @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 3 | 2 | -2.99% | -15.0% |
| SELL @ 3rd Alert (retest2) | 11 | 3 | 27.3% | 0 | 9 | 2 | -1.20% | -13.1% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 3 | 2 | -2.99% | -15.0% |
| retest2 (combined) | 11 | 3 | 27.3% | 0 | 9 | 2 | -1.20% | -13.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 09:15:00 | 927.20 | 1056.53 | 1057.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 10:15:00 | 921.70 | 1047.16 | 1052.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 993.40 | 980.33 | 1006.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 10:00:00 | 993.40 | 980.33 | 1006.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 1007.35 | 981.02 | 1006.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:00:00 | 1007.35 | 981.02 | 1006.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 1010.25 | 981.31 | 1006.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:30:00 | 1013.05 | 981.31 | 1006.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 1017.80 | 981.67 | 1006.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 15:00:00 | 1017.80 | 981.67 | 1006.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 1018.00 | 982.03 | 1006.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1011.35 | 982.03 | 1006.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:30:00 | 1010.20 | 982.58 | 1006.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 1008.40 | 983.94 | 1006.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1031.35 | 984.41 | 1007.00 | SL hit (close>static) qty=1.00 sl=1020.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1031.35 | 984.41 | 1007.00 | SL hit (close>static) qty=1.00 sl=1020.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1031.35 | 984.41 | 1007.00 | SL hit (close>static) qty=1.00 sl=1020.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 1013.00 | 986.70 | 1007.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 1020.85 | 987.04 | 1007.56 | SL hit (close>static) qty=1.00 sl=1020.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 1009.10 | 998.43 | 1007.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:00:00 | 1009.10 | 998.43 | 1007.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 1008.90 | 998.53 | 1007.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 993.95 | 998.86 | 1007.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 12:15:00 | 944.25 | 996.71 | 1006.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 997.05 | 995.53 | 1005.30 | SL hit (close>ema200) qty=0.50 sl=995.53 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:00:00 | 1000.15 | 995.54 | 1005.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 09:15:00 | 950.14 | 991.44 | 1002.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 13:15:00 | 992.55 | 986.02 | 998.32 | SL hit (close>ema200) qty=0.50 sl=986.02 alert=retest2 |

### Cycle 2 — BUY (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 10:15:00 | 1208.00 | 1009.66 | 1008.72 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 1007.50 | 1050.16 | 1050.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 11:15:00 | 1003.00 | 1049.25 | 1049.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 682.45 | 676.55 | 741.50 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 13:00:00 | 673.75 | 676.96 | 738.55 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 09:15:00 | 644.10 | 677.37 | 735.75 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 09:15:00 | 640.06 | 677.06 | 735.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:45:00 | 673.00 | 675.56 | 731.97 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 13:15:00 | 681.85 | 675.59 | 731.42 | SL hit (close>ema200) qty=0.50 sl=675.59 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 707.05 | 676.17 | 730.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:30:00 | 699.20 | 676.67 | 730.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:15:00 | 664.24 | 678.95 | 725.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-22 13:15:00 | 695.00 | 678.81 | 724.29 | SL hit (close>ema200) qty=0.50 sl=678.81 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:15:00 | 695.30 | 680.92 | 723.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 10:15:00 | 695.00 | 681.11 | 723.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 725.05 | 683.74 | 721.74 | SL hit (close>ema400) qty=1.00 sl=721.74 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 725.05 | 683.74 | 721.74 | SL hit (close>ema400) qty=1.00 sl=721.74 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-28 10:15:00 | 739.55 | 684.29 | 721.83 | SL hit (close>static) qty=1.00 sl=732.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-28 10:15:00 | 739.55 | 684.29 | 721.83 | SL hit (close>static) qty=1.00 sl=732.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-11 09:15:00 | 1011.35 | 2025-09-12 09:15:00 | 1031.35 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-09-11 10:30:00 | 1010.20 | 2025-09-12 09:15:00 | 1031.35 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-09-12 09:15:00 | 1008.40 | 2025-09-12 09:15:00 | 1031.35 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-09-15 09:15:00 | 1013.00 | 2025-09-15 09:15:00 | 1020.85 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-10-13 09:15:00 | 993.95 | 2025-10-14 12:15:00 | 944.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-13 09:15:00 | 993.95 | 2025-10-15 09:15:00 | 997.05 | STOP_HIT | 0.50 | -0.31% |
| SELL | retest2 | 2025-10-15 12:00:00 | 1000.15 | 2025-10-20 09:15:00 | 950.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-15 12:00:00 | 1000.15 | 2025-10-24 13:15:00 | 992.55 | STOP_HIT | 0.50 | 0.76% |
| SELL | retest1 | 2026-04-09 13:00:00 | 673.75 | 2026-04-13 09:15:00 | 640.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-09 13:00:00 | 673.75 | 2026-04-15 13:15:00 | 681.85 | STOP_HIT | 0.50 | -1.20% |
| SELL | retest1 | 2026-04-13 09:15:00 | 644.10 | 2026-04-22 09:15:00 | 664.24 | PARTIAL | 0.50 | -3.13% |
| SELL | retest1 | 2026-04-13 09:15:00 | 644.10 | 2026-04-22 13:15:00 | 695.00 | STOP_HIT | 0.50 | -7.90% |
| SELL | retest1 | 2026-04-15 11:45:00 | 673.00 | 2026-04-28 09:15:00 | 725.05 | STOP_HIT | 1.00 | -7.73% |
| SELL | retest2 | 2026-04-16 11:30:00 | 699.20 | 2026-04-28 09:15:00 | 725.05 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2026-04-24 09:15:00 | 695.30 | 2026-04-28 10:15:00 | 739.55 | STOP_HIT | 1.00 | -6.36% |
| SELL | retest2 | 2026-04-24 10:15:00 | 695.00 | 2026-04-28 10:15:00 | 739.55 | STOP_HIT | 1.00 | -6.41% |
