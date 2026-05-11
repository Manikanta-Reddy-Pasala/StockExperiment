# Trent Ltd. (TRENT)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 4249.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 34 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 23
- **Target hits / Stop hits / Partials:** 1 / 23 / 1
- **Avg / median % per leg:** -2.25% / -3.51%
- **Sum % (uncompounded):** -56.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.70% | -16.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.70% | -16.2% |
| SELL (all) | 19 | 2 | 10.5% | 1 | 17 | 1 | -2.10% | -39.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 19 | 2 | 10.5% | 1 | 17 | 1 | -2.10% | -39.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 2 | 8.0% | 1 | 23 | 1 | -2.25% | -56.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 12:15:00 | 6450.00 | 6973.32 | 6974.80 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 12:15:00 | 6960.50 | 6935.46 | 6935.34 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 6886.10 | 6934.97 | 6935.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 6802.80 | 6933.66 | 6934.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 09:15:00 | 6949.45 | 6933.28 | 6934.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 09:15:00 | 6949.45 | 6933.28 | 6934.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 6949.45 | 6933.28 | 6934.24 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2024-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 12:15:00 | 6965.30 | 6935.26 | 6935.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 14:15:00 | 7069.35 | 6945.72 | 6940.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 14:15:00 | 6932.05 | 6963.79 | 6950.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 14:15:00 | 6932.05 | 6963.79 | 6950.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 6932.05 | 6963.79 | 6950.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 6932.05 | 6963.79 | 6950.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 6975.00 | 6963.90 | 6950.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:15:00 | 6919.45 | 6963.90 | 6950.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 6907.00 | 6963.33 | 6950.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 10:45:00 | 6982.60 | 6963.44 | 6950.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 11:30:00 | 6975.95 | 6963.46 | 6950.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 12:30:00 | 6980.20 | 6963.66 | 6950.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 13:00:00 | 6983.50 | 6963.66 | 6950.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 6992.65 | 7021.29 | 6983.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:45:00 | 6997.00 | 7021.29 | 6983.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 6977.65 | 7020.86 | 6983.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:30:00 | 6958.80 | 7020.86 | 6983.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 6970.00 | 7020.35 | 6983.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 09:15:00 | 7031.55 | 7020.35 | 6983.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 12:30:00 | 6997.50 | 7020.06 | 6983.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-07 13:15:00 | 6945.95 | 7019.32 | 6983.45 | SL hit (close<static) qty=1.00 sl=6956.90 alert=retest2 |

### Cycle 5 — SELL (started 2025-01-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 13:15:00 | 6505.90 | 6947.76 | 6949.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 6407.40 | 6935.05 | 6943.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 13:15:00 | 6199.95 | 6173.19 | 6471.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-01 14:00:00 | 6199.95 | 6173.19 | 6471.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 5488.50 | 5226.15 | 5509.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:45:00 | 5497.25 | 5226.15 | 5509.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 5564.00 | 5235.34 | 5505.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:00:00 | 5564.00 | 5235.34 | 5505.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 5577.50 | 5238.74 | 5505.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:30:00 | 5592.15 | 5238.74 | 5505.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 12:15:00 | 5588.00 | 5245.52 | 5506.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 12:30:00 | 5586.25 | 5245.52 | 5506.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 14:15:00 | 5362.00 | 5152.89 | 5371.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 14:45:00 | 5362.50 | 5152.89 | 5371.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 15:15:00 | 5352.00 | 5154.87 | 5371.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-22 09:15:00 | 5398.00 | 5154.87 | 5371.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 5351.50 | 5156.83 | 5370.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-22 13:00:00 | 5312.00 | 5162.17 | 5370.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-22 15:15:00 | 5309.00 | 5165.61 | 5370.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-23 09:30:00 | 5295.50 | 5167.71 | 5369.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 09:15:00 | 5287.50 | 5175.49 | 5367.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 5364.00 | 5181.67 | 5366.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 12:30:00 | 5367.00 | 5181.67 | 5366.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 5339.50 | 5183.24 | 5366.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 14:30:00 | 5319.00 | 5184.63 | 5366.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 15:00:00 | 5323.00 | 5184.63 | 5366.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-29 15:15:00 | 5510.00 | 5190.87 | 5350.33 | SL hit (close>static) qty=1.00 sl=5418.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 5661.00 | 5388.31 | 5387.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 11:15:00 | 5796.00 | 5459.52 | 5426.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 5541.50 | 5551.36 | 5481.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 5541.50 | 5551.36 | 5481.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 5541.50 | 5551.36 | 5481.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:45:00 | 5533.00 | 5551.36 | 5481.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 5758.00 | 5843.24 | 5682.86 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 5363.50 | 5581.54 | 5582.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 5345.00 | 5576.95 | 5579.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 09:15:00 | 5400.50 | 5362.00 | 5451.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 10:00:00 | 5400.50 | 5362.00 | 5451.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 5436.00 | 5358.15 | 5440.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:45:00 | 5440.00 | 5358.15 | 5440.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 5415.00 | 5358.72 | 5440.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 10:15:00 | 5396.50 | 5362.62 | 5440.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:00:00 | 5403.50 | 5364.21 | 5439.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 15:00:00 | 5395.50 | 5367.21 | 5438.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 10:30:00 | 5395.00 | 5368.46 | 5437.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 5563.50 | 5371.03 | 5437.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 5563.50 | 5371.03 | 5437.00 | SL hit (close>static) qty=1.00 sl=5445.00 alert=retest2 |

### Cycle 8 — BUY (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 12:15:00 | 4330.60 | 3908.61 | 3907.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 13:15:00 | 4362.00 | 3913.12 | 3909.38 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-12-31 10:45:00 | 6982.60 | 2025-01-07 13:15:00 | 6945.95 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-12-31 11:30:00 | 6975.95 | 2025-01-07 13:15:00 | 6945.95 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2024-12-31 12:30:00 | 6980.20 | 2025-01-08 09:15:00 | 6731.20 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2024-12-31 13:00:00 | 6983.50 | 2025-01-08 09:15:00 | 6731.20 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2025-01-07 09:15:00 | 7031.55 | 2025-01-08 09:15:00 | 6731.20 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest2 | 2025-01-07 12:30:00 | 6997.50 | 2025-01-08 09:15:00 | 6731.20 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2025-04-22 13:00:00 | 5312.00 | 2025-04-29 15:15:00 | 5510.00 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2025-04-22 15:15:00 | 5309.00 | 2025-04-29 15:15:00 | 5510.00 | STOP_HIT | 1.00 | -3.79% |
| SELL | retest2 | 2025-04-23 09:30:00 | 5295.50 | 2025-04-29 15:15:00 | 5510.00 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2025-04-24 09:15:00 | 5287.50 | 2025-04-29 15:15:00 | 5510.00 | STOP_HIT | 1.00 | -4.21% |
| SELL | retest2 | 2025-04-24 14:30:00 | 5319.00 | 2025-04-29 15:15:00 | 5510.00 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2025-04-24 15:00:00 | 5323.00 | 2025-04-29 15:15:00 | 5510.00 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2025-04-30 09:15:00 | 5200.00 | 2025-05-05 15:15:00 | 5380.00 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2025-05-06 09:45:00 | 5317.00 | 2025-05-12 11:15:00 | 5381.50 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-08-12 10:15:00 | 5396.50 | 2025-08-18 09:15:00 | 5563.50 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-08-12 14:00:00 | 5403.50 | 2025-08-18 09:15:00 | 5563.50 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-08-13 15:00:00 | 5395.50 | 2025-08-18 09:15:00 | 5563.50 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-08-14 10:30:00 | 5395.00 | 2025-08-18 09:15:00 | 5563.50 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-08-22 12:45:00 | 5445.50 | 2025-09-03 14:15:00 | 5481.50 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-08-25 14:30:00 | 5444.50 | 2025-09-04 09:15:00 | 5636.00 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2025-09-01 15:00:00 | 5444.50 | 2025-09-04 09:15:00 | 5636.00 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2025-09-02 10:00:00 | 5445.50 | 2025-09-04 09:15:00 | 5636.00 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2025-09-02 13:15:00 | 5424.50 | 2025-09-04 09:15:00 | 5636.00 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2025-09-08 13:30:00 | 5404.00 | 2025-09-12 12:15:00 | 5133.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-08 13:30:00 | 5404.00 | 2025-09-23 10:15:00 | 4863.60 | TARGET_HIT | 0.50 | 10.00% |
