# Signatureglobal (India) Ltd. (SIGNATURE)

## Backtest Summary

- **Window:** 2025-07-25 09:15:00 → 2026-05-08 15:15:00 (1346 bars)
- **Last close:** 903.00
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
| ALERT3 | 10 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 13 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 10
- **Target hits / Stop hits / Partials:** 1 / 12 / 3
- **Avg / median % per leg:** 0.26% / -0.23%
- **Sum % (uncompounded):** 4.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.53% | -12.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.53% | -12.2% |
| SELL (all) | 8 | 6 | 75.0% | 1 | 4 | 3 | 2.05% | 16.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 6 | 75.0% | 1 | 4 | 3 | 2.05% | 16.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 16 | 6 | 37.5% | 1 | 12 | 3 | 0.26% | 4.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 11:15:00 | 1127.40 | 1098.26 | 1098.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 15:15:00 | 1129.20 | 1099.44 | 1098.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 09:15:00 | 1104.30 | 1104.80 | 1101.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 09:15:00 | 1104.30 | 1104.80 | 1101.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 1104.30 | 1104.80 | 1101.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 10:45:00 | 1113.50 | 1104.88 | 1101.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 15:00:00 | 1111.40 | 1105.92 | 1102.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:30:00 | 1113.40 | 1106.04 | 1102.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 13:30:00 | 1111.30 | 1106.23 | 1102.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1117.40 | 1121.08 | 1112.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 1117.40 | 1121.08 | 1112.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 1113.20 | 1120.85 | 1112.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:00:00 | 1113.20 | 1120.85 | 1112.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 1127.80 | 1120.92 | 1112.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:45:00 | 1130.20 | 1121.01 | 1113.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 12:30:00 | 1130.50 | 1121.22 | 1113.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 15:00:00 | 1129.70 | 1121.40 | 1113.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 14:00:00 | 1130.20 | 1121.78 | 1113.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1112.10 | 1122.43 | 1114.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:30:00 | 1112.00 | 1122.43 | 1114.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 1108.80 | 1122.29 | 1114.89 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 1108.80 | 1122.29 | 1114.89 | SL hit (close<static) qty=1.00 sl=1110.50 alert=retest2 |

### Cycle 2 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 999.00 | 1108.15 | 1108.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 951.50 | 1104.63 | 1106.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 14:15:00 | 952.75 | 937.27 | 993.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-10 15:00:00 | 952.75 | 937.27 | 993.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 989.50 | 940.12 | 992.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 993.55 | 940.12 | 992.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 986.00 | 941.46 | 991.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 13:45:00 | 984.00 | 941.88 | 991.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:30:00 | 982.00 | 942.29 | 991.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 1028.50 | 943.57 | 992.00 | SL hit (close>static) qty=1.00 sl=992.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-12-09 10:45:00 | 1113.50 | 2026-01-07 10:15:00 | 1108.80 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-12-10 15:00:00 | 1111.40 | 2026-01-07 10:15:00 | 1108.80 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-12-11 10:30:00 | 1113.40 | 2026-01-07 10:15:00 | 1108.80 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-12-11 13:30:00 | 1111.30 | 2026-01-07 10:15:00 | 1108.80 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2026-01-01 09:45:00 | 1130.20 | 2026-01-07 13:15:00 | 1099.20 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2026-01-01 12:30:00 | 1130.50 | 2026-01-07 13:15:00 | 1099.20 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2026-01-01 15:00:00 | 1129.70 | 2026-01-07 13:15:00 | 1099.20 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2026-01-02 14:00:00 | 1130.20 | 2026-01-07 13:15:00 | 1099.20 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2026-02-12 13:45:00 | 984.00 | 2026-02-13 09:15:00 | 1028.50 | STOP_HIT | 1.00 | -4.52% |
| SELL | retest2 | 2026-02-12 14:30:00 | 982.00 | 2026-02-13 09:15:00 | 1028.50 | STOP_HIT | 1.00 | -4.74% |
| SELL | retest2 | 2026-02-23 09:45:00 | 973.70 | 2026-02-24 14:15:00 | 930.72 | PARTIAL | 0.50 | 4.41% |
| SELL | retest2 | 2026-02-23 12:00:00 | 979.70 | 2026-02-25 09:15:00 | 925.01 | PARTIAL | 0.50 | 5.58% |
| SELL | retest2 | 2026-02-23 09:45:00 | 973.70 | 2026-02-27 13:15:00 | 973.40 | STOP_HIT | 0.50 | 0.03% |
| SELL | retest2 | 2026-02-23 12:00:00 | 979.70 | 2026-02-27 13:15:00 | 973.40 | STOP_HIT | 0.50 | 0.64% |
| SELL | retest2 | 2026-03-02 09:15:00 | 960.00 | 2026-03-04 11:15:00 | 912.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 960.00 | 2026-03-09 10:15:00 | 864.00 | TARGET_HIT | 0.50 | 10.00% |
