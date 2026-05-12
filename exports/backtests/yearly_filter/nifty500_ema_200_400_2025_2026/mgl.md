# Mahanagar Gas Ltd. (MGL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1173.50
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
| ALERT3 | 7 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 3
- **Target hits / Stop hits / Partials:** 0 / 6 / 3
- **Avg / median % per leg:** 1.61% / 2.95%
- **Sum % (uncompounded):** 14.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.10% | -9.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.10% | -9.3% |
| SELL (all) | 6 | 6 | 100.0% | 0 | 3 | 3 | 3.96% | 23.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 6 | 100.0% | 0 | 3 | 3 | 3.96% | 23.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 6 | 66.7% | 0 | 6 | 3 | 1.61% | 14.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 1285.60 | 1407.91 | 1408.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 13:15:00 | 1274.70 | 1357.25 | 1377.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 1331.60 | 1316.51 | 1345.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 10:00:00 | 1331.60 | 1316.51 | 1345.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1350.00 | 1316.84 | 1345.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 1350.00 | 1316.84 | 1345.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 1354.70 | 1317.22 | 1345.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 11:30:00 | 1340.60 | 1322.36 | 1345.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:30:00 | 1346.70 | 1323.35 | 1345.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:30:00 | 1344.00 | 1324.58 | 1345.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 13:15:00 | 1279.37 | 1321.94 | 1342.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 15:15:00 | 1276.80 | 1321.07 | 1341.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 09:15:00 | 1273.57 | 1320.58 | 1341.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 10:15:00 | 1304.40 | 1303.05 | 1326.88 | SL hit (close>ema200) qty=0.50 sl=1303.05 alert=retest2 |

### Cycle 2 — BUY (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 14:15:00 | 1150.90 | 1081.30 | 1080.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 1154.50 | 1092.50 | 1086.92 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-28 11:00:00 | 1363.40 | 2025-05-30 14:15:00 | 1323.80 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-05-29 09:30:00 | 1371.60 | 2025-05-30 14:15:00 | 1323.80 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2025-06-02 09:30:00 | 1366.40 | 2025-06-02 12:15:00 | 1326.60 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-09-18 11:30:00 | 1340.60 | 2025-09-24 13:15:00 | 1279.37 | PARTIAL | 0.50 | 4.57% |
| SELL | retest2 | 2025-09-19 09:30:00 | 1346.70 | 2025-09-24 15:15:00 | 1276.80 | PARTIAL | 0.50 | 5.19% |
| SELL | retest2 | 2025-09-22 09:30:00 | 1344.00 | 2025-09-25 09:15:00 | 1273.57 | PARTIAL | 0.50 | 5.24% |
| SELL | retest2 | 2025-09-18 11:30:00 | 1340.60 | 2025-10-07 10:15:00 | 1304.40 | STOP_HIT | 0.50 | 2.70% |
| SELL | retest2 | 2025-09-19 09:30:00 | 1346.70 | 2025-10-07 10:15:00 | 1304.40 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2025-09-22 09:30:00 | 1344.00 | 2025-10-07 10:15:00 | 1304.40 | STOP_HIT | 0.50 | 2.95% |
