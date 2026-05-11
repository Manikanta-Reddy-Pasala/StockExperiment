# TATACONSUM (TATACONSUM)

## Backtest Summary

- **Window:** 2025-10-28 09:15:00 → 2026-05-08 15:15:00 (910 bars)
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
| ALERT2_SKIP | 1 |
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 15
- **Target hits / Stop hits / Partials:** 1 / 15 / 1
- **Avg / median % per leg:** -0.27% / -1.19%
- **Sum % (uncompounded):** -4.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 17 | 2 | 11.8% | 1 | 15 | 1 | -0.27% | -4.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 17 | 2 | 11.8% | 1 | 15 | 1 | -0.27% | -4.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 17 | 2 | 11.8% | 1 | 15 | 1 | -0.27% | -4.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 1088.00 | 1165.74 | 1166.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 1085.00 | 1164.94 | 1165.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 1167.20 | 1161.12 | 1163.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 1167.20 | 1161.12 | 1163.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1167.20 | 1161.12 | 1163.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:00:00 | 1167.20 | 1161.12 | 1163.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 1168.20 | 1161.19 | 1163.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 1168.20 | 1161.19 | 1163.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1157.50 | 1160.92 | 1163.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 10:45:00 | 1147.00 | 1160.79 | 1163.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 12:00:00 | 1146.70 | 1160.65 | 1163.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 13:00:00 | 1147.90 | 1160.53 | 1163.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:30:00 | 1147.60 | 1160.26 | 1163.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 1160.40 | 1159.98 | 1162.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:30:00 | 1161.00 | 1159.98 | 1162.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 1169.90 | 1160.06 | 1162.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-09 12:15:00 | 1169.90 | 1160.06 | 1162.80 | SL hit (close>static) qty=1.00 sl=1163.60 alert=retest2 |

### Cycle 2 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 1146.00 | 1117.64 | 1117.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 1163.00 | 1118.64 | 1118.09 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-02-05 10:45:00 | 1147.00 | 2026-02-09 12:15:00 | 1169.90 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2026-02-05 12:00:00 | 1146.70 | 2026-02-09 12:15:00 | 1169.90 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2026-02-05 13:00:00 | 1147.90 | 2026-02-09 12:15:00 | 1169.90 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-02-06 09:30:00 | 1147.60 | 2026-02-09 12:15:00 | 1169.90 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-02-10 11:15:00 | 1153.90 | 2026-02-18 14:15:00 | 1169.70 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-02-10 13:00:00 | 1153.90 | 2026-02-18 14:15:00 | 1169.70 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-02-11 11:15:00 | 1154.40 | 2026-02-18 14:15:00 | 1169.70 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-02-18 09:45:00 | 1151.20 | 2026-02-18 14:15:00 | 1169.70 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-02-19 13:45:00 | 1156.00 | 2026-02-20 10:15:00 | 1164.50 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-02-19 15:00:00 | 1159.20 | 2026-02-20 10:15:00 | 1164.50 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2026-02-20 10:00:00 | 1158.00 | 2026-02-20 10:15:00 | 1164.50 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-02-20 13:00:00 | 1156.40 | 2026-02-23 09:15:00 | 1170.20 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-02-27 09:15:00 | 1146.80 | 2026-03-09 09:15:00 | 1089.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 09:15:00 | 1146.80 | 2026-03-23 09:15:00 | 1032.12 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-28 09:15:00 | 1153.40 | 2026-04-29 11:15:00 | 1164.80 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-04-28 11:30:00 | 1152.00 | 2026-04-29 11:15:00 | 1164.80 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2026-04-28 13:15:00 | 1154.20 | 2026-04-29 11:15:00 | 1164.80 | STOP_HIT | 1.00 | -0.92% |
