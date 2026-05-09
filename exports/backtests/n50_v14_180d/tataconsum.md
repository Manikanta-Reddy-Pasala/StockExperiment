# TATACONSUM (TATACONSUM)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 1176.60
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
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 8 |
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
- **Avg / median % per leg:** -1.81% / -1.22%
- **Sum % (uncompounded):** -10.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.58% | -7.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.58% | -7.9% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.98% | -3.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.98% | -3.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.81% | -10.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 14:15:00 | 1124.50 | 1161.68 | 1161.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 1115.20 | 1156.69 | 1158.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-09 11:15:00 | 1082.30 | 1076.96 | 1104.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-09 11:45:00 | 1081.90 | 1076.96 | 1104.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1102.40 | 1078.71 | 1102.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 15:15:00 | 1092.00 | 1079.78 | 1102.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 1124.50 | 1081.55 | 1102.40 | SL hit (close>static) qty=1.00 sl=1108.30 alert=retest2 |

### Cycle 2 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 1148.60 | 1117.10 | 1116.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 1163.00 | 1118.63 | 1117.75 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-12-05 10:45:00 | 1151.60 | 2025-12-09 09:15:00 | 1137.60 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-12-10 10:00:00 | 1155.00 | 2025-12-10 12:15:00 | 1141.70 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-12-11 11:15:00 | 1151.60 | 2025-12-11 14:15:00 | 1141.10 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-12-12 15:00:00 | 1150.70 | 2026-01-28 09:15:00 | 1129.00 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-12-15 10:30:00 | 1160.60 | 2026-01-28 09:15:00 | 1129.00 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2026-04-15 15:15:00 | 1092.00 | 2026-04-17 09:15:00 | 1124.50 | STOP_HIT | 1.00 | -2.98% |
