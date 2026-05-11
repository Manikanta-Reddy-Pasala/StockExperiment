# Atul Ltd. (ATUL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 7090.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 1 |
| TARGET_HIT | 3 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 10
- **Target hits / Stop hits / Partials:** 3 / 11 / 1
- **Avg / median % per leg:** 0.71% / -1.53%
- **Sum % (uncompounded):** 10.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 3 | 4 | 0 | 2.71% | 19.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 3 | 42.9% | 3 | 4 | 0 | 2.71% | 19.0% |
| SELL (all) | 8 | 2 | 25.0% | 0 | 7 | 1 | -1.03% | -8.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 2 | 25.0% | 0 | 7 | 1 | -1.03% | -8.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 5 | 33.3% | 3 | 11 | 1 | 0.71% | 10.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 13:15:00 | 6673.00 | 6948.83 | 6949.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 6571.00 | 6939.72 | 6944.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 10:15:00 | 6480.00 | 6478.32 | 6618.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:30:00 | 6477.00 | 6478.32 | 6618.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 6227.00 | 5931.17 | 6099.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 6227.00 | 5931.17 | 6099.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 6220.00 | 5934.04 | 6099.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 6220.50 | 5934.04 | 6099.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 6150.50 | 5948.75 | 6101.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 6150.50 | 5948.75 | 6101.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 6103.00 | 5950.29 | 6101.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 09:30:00 | 6089.50 | 5958.99 | 6102.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 15:15:00 | 5785.02 | 5945.25 | 6061.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 5885.00 | 5883.99 | 6004.72 | SL hit (close>ema200) qty=0.50 sl=5883.99 alert=retest2 |

### Cycle 2 — BUY (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 09:15:00 | 6165.00 | 6022.96 | 6022.34 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 5821.00 | 6021.90 | 6022.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 5755.50 | 6019.25 | 6020.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 11:15:00 | 5974.00 | 5952.79 | 5983.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 11:15:00 | 5974.00 | 5952.79 | 5983.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 5974.00 | 5952.79 | 5983.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:00:00 | 5974.00 | 5952.79 | 5983.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 6023.00 | 5953.49 | 5984.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 13:00:00 | 6023.00 | 5953.49 | 5984.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 13:15:00 | 6120.00 | 5955.14 | 5984.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 13:45:00 | 6131.50 | 5955.14 | 5984.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 5986.50 | 5996.11 | 6003.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:15:00 | 5940.00 | 5996.11 | 6003.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 6403.50 | 5996.20 | 6003.10 | SL hit (close>static) qty=1.00 sl=6032.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 15:15:00 | 6200.00 | 6010.32 | 6010.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 6310.00 | 6021.90 | 6015.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 6383.50 | 6405.66 | 6268.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-04 10:00:00 | 6383.50 | 6405.66 | 6268.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 6350.00 | 6403.10 | 6269.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:30:00 | 6307.50 | 6403.10 | 6269.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 6278.50 | 6412.44 | 6285.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:45:00 | 6275.50 | 6412.44 | 6285.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 6317.50 | 6411.50 | 6285.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 14:45:00 | 6431.00 | 6344.55 | 6268.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 12:15:00 | 6249.00 | 6342.48 | 6269.09 | SL hit (close<static) qty=1.00 sl=6255.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-11-20 09:30:00 | 6089.50 | 2025-12-01 15:15:00 | 5785.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 09:30:00 | 6089.50 | 2025-12-10 09:15:00 | 5885.00 | STOP_HIT | 0.50 | 3.36% |
| SELL | retest2 | 2025-12-19 10:15:00 | 6090.50 | 2025-12-29 13:15:00 | 6160.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-12-23 15:15:00 | 6051.00 | 2025-12-29 13:15:00 | 6160.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-12-30 12:00:00 | 6072.00 | 2025-12-31 15:15:00 | 6151.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-01-12 11:15:00 | 5986.50 | 2026-01-12 12:15:00 | 6078.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-01-12 12:45:00 | 6001.00 | 2026-01-12 13:15:00 | 6184.50 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2026-02-02 10:15:00 | 5940.00 | 2026-02-03 09:15:00 | 6403.50 | STOP_HIT | 1.00 | -7.80% |
| BUY | retest2 | 2026-03-13 14:45:00 | 6431.00 | 2026-03-16 12:15:00 | 6249.00 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-03-18 09:45:00 | 6360.00 | 2026-03-19 13:15:00 | 6247.50 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2026-03-30 14:45:00 | 6362.00 | 2026-04-06 09:15:00 | 6160.50 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2026-04-01 09:45:00 | 6367.50 | 2026-04-06 09:15:00 | 6160.50 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2026-04-09 12:30:00 | 6483.50 | 2026-05-06 13:15:00 | 7071.90 | TARGET_HIT | 1.00 | 9.08% |
| BUY | retest2 | 2026-04-10 09:15:00 | 6501.00 | 2026-05-06 14:15:00 | 7131.85 | TARGET_HIT | 1.00 | 9.70% |
| BUY | retest2 | 2026-04-15 10:00:00 | 6429.00 | 2026-05-07 09:15:00 | 7151.10 | TARGET_HIT | 1.00 | 11.23% |
