# Sai Life Sciences Ltd. (SAILIFE)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 1117.90
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
| ENTRY2 | 4 |
| PARTIAL | 1 |
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 2 / 2 / 1
- **Avg / median % per leg:** 3.87% / 5.00%
- **Sum % (uncompounded):** 19.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 2 | 1 | 0 | 6.03% | 18.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 2 | 66.7% | 2 | 1 | 0 | 6.03% | 18.1% |
| SELL (all) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.64% | 1.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.64% | 1.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 5 | 3 | 60.0% | 2 | 2 | 1 | 3.87% | 19.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 818.50 | 886.11 | 886.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 800.10 | 885.25 | 885.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 870.05 | 866.71 | 875.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 870.05 | 866.71 | 875.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 870.05 | 866.71 | 875.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 10:00:00 | 855.60 | 867.19 | 875.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 14:15:00 | 812.82 | 865.45 | 874.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-09 09:15:00 | 887.45 | 860.10 | 870.78 | SL hit (close>ema200) qty=0.50 sl=860.10 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 13:15:00 | 920.00 | 879.61 | 879.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 928.05 | 880.90 | 880.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 11:15:00 | 951.95 | 957.14 | 927.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 12:00:00 | 951.95 | 957.14 | 927.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 949.10 | 973.83 | 947.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 949.10 | 973.83 | 947.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 950.00 | 973.59 | 947.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 956.55 | 973.44 | 947.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 09:15:00 | 938.35 | 972.68 | 948.09 | SL hit (close<static) qty=1.00 sl=944.05 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-02-04 10:00:00 | 855.60 | 2026-02-04 14:15:00 | 812.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 10:00:00 | 855.60 | 2026-02-09 09:15:00 | 887.45 | STOP_HIT | 0.50 | -3.72% |
| BUY | retest2 | 2026-04-02 11:30:00 | 956.55 | 2026-04-06 09:15:00 | 938.35 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-04-08 09:15:00 | 959.40 | 2026-04-27 14:15:00 | 1055.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-21 11:15:00 | 975.10 | 2026-04-27 14:15:00 | 1072.61 | TARGET_HIT | 1.00 | 10.00% |
