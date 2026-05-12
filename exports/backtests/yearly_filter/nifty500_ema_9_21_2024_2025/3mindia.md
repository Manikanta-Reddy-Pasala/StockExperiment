# 3M India Ltd. (3MINDIA)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 32070.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 184 |
| ALERT1 | 118 |
| ALERT2 | 116 |
| ALERT2_SKIP | 62 |
| ALERT3 | 293 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 138 |
| PARTIAL | 14 |
| TARGET_HIT | 0 |
| STOP_HIT | 148 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 161 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 65 / 96
- **Target hits / Stop hits / Partials:** 0 / 147 / 14
- **Avg / median % per leg:** 0.29% / -0.69%
- **Sum % (uncompounded):** 46.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 76 | 21 | 27.6% | 0 | 76 | 0 | -0.36% | -27.0% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 7 | 0 | -0.67% | -4.7% |
| BUY @ 3rd Alert (retest2) | 69 | 19 | 27.5% | 0 | 69 | 0 | -0.32% | -22.3% |
| SELL (all) | 85 | 44 | 51.8% | 0 | 71 | 14 | 0.86% | 73.1% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.13% | -2.3% |
| SELL @ 3rd Alert (retest2) | 83 | 44 | 53.0% | 0 | 69 | 14 | 0.91% | 75.4% |
| retest1 (combined) | 9 | 2 | 22.2% | 0 | 9 | 0 | -0.78% | -7.0% |
| retest2 (combined) | 152 | 63 | 41.4% | 0 | 138 | 14 | 0.35% | 53.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 29115.25 | 28931.30 | 28917.54 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 15:15:00 | 28780.00 | 28902.77 | 28912.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 13:15:00 | 28557.15 | 28789.38 | 28850.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 09:15:00 | 29098.70 | 28810.91 | 28841.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 09:15:00 | 29098.70 | 28810.91 | 28841.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 29098.70 | 28810.91 | 28841.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 10:00:00 | 29098.70 | 28810.91 | 28841.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 10:15:00 | 29149.95 | 28878.72 | 28869.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 11:15:00 | 30144.35 | 29131.84 | 28985.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 15:15:00 | 30139.95 | 30272.05 | 29878.95 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-18 11:30:00 | 30510.00 | 30420.11 | 30015.00 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 31086.00 | 31215.79 | 31040.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:30:00 | 31059.95 | 31215.79 | 31040.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 31099.90 | 31192.61 | 31046.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 11:30:00 | 31047.35 | 31192.61 | 31046.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 12:15:00 | 31019.50 | 31157.99 | 31043.63 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-23 12:15:00 | 31019.50 | 31157.99 | 31043.63 | SL hit (close<ema400) qty=1.00 sl=31043.63 alert=retest1 |

### Cycle 4 — SELL (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 09:15:00 | 30875.30 | 30984.38 | 30985.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 11:15:00 | 30700.00 | 30896.56 | 30943.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 10:15:00 | 30871.00 | 30795.95 | 30860.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 10:15:00 | 30871.00 | 30795.95 | 30860.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 30871.00 | 30795.95 | 30860.22 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 15:15:00 | 30960.00 | 30904.25 | 30897.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-28 09:15:00 | 31200.00 | 30963.40 | 30925.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 10:15:00 | 30877.80 | 30946.28 | 30920.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 10:15:00 | 30877.80 | 30946.28 | 30920.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 30877.80 | 30946.28 | 30920.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:30:00 | 30831.15 | 30946.28 | 30920.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 11:15:00 | 30678.70 | 30892.76 | 30898.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 12:15:00 | 30534.35 | 30821.08 | 30865.59 | Break + close below crossover candle low |

### Cycle 7 — BUY (started 2024-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 14:15:00 | 34200.05 | 31426.99 | 31128.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 09:15:00 | 34662.60 | 32487.41 | 31688.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 15:15:00 | 33000.00 | 33431.30 | 32635.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-30 10:00:00 | 33074.05 | 33359.85 | 32675.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 33350.05 | 33598.61 | 33302.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 33350.05 | 33598.61 | 33302.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 15:15:00 | 33840.00 | 33798.69 | 33586.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:15:00 | 33285.00 | 33798.69 | 33586.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 33600.00 | 33758.95 | 33587.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 33365.20 | 33758.95 | 33587.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 32690.00 | 33545.16 | 33505.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 32690.00 | 33545.16 | 33505.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 32425.00 | 33321.13 | 33407.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 13:15:00 | 31858.55 | 32906.69 | 33196.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 32964.05 | 32799.73 | 33039.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 10:15:00 | 32964.05 | 32799.73 | 33039.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 32964.05 | 32799.73 | 33039.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 32964.05 | 32799.73 | 33039.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 32855.20 | 32810.82 | 33022.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:30:00 | 33040.30 | 32810.82 | 33022.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 33250.50 | 32913.70 | 33033.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:00:00 | 33250.50 | 32913.70 | 33033.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 33167.65 | 32964.49 | 33045.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:45:00 | 33397.00 | 32964.49 | 33045.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 34249.10 | 33179.09 | 33126.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 09:15:00 | 35475.05 | 34408.08 | 34086.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 10:15:00 | 35917.30 | 36032.21 | 35566.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 11:00:00 | 35917.30 | 36032.21 | 35566.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 36540.00 | 36658.23 | 36371.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 09:15:00 | 36827.60 | 36658.23 | 36371.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 14:15:00 | 37178.30 | 37713.18 | 37784.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 14:15:00 | 37178.30 | 37713.18 | 37784.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 14:15:00 | 36614.35 | 37266.35 | 37443.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 37205.95 | 37163.82 | 37360.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-01 10:15:00 | 37348.65 | 37163.82 | 37360.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 37186.95 | 37168.45 | 37344.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:30:00 | 37310.10 | 37168.45 | 37344.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 11:15:00 | 37616.95 | 37258.15 | 37369.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 11:45:00 | 37629.80 | 37258.15 | 37369.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 12:15:00 | 37763.90 | 37359.30 | 37405.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 12:30:00 | 37749.80 | 37359.30 | 37405.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 13:15:00 | 37746.20 | 37436.68 | 37436.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 14:15:00 | 37801.20 | 37509.58 | 37469.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 14:15:00 | 38730.05 | 38738.47 | 38206.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 15:00:00 | 38730.05 | 38738.47 | 38206.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 38171.05 | 38607.90 | 38279.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 11:00:00 | 38171.05 | 38607.90 | 38279.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 11:15:00 | 39155.00 | 38717.32 | 38359.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 12:45:00 | 39300.00 | 38823.88 | 38440.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:30:00 | 39299.95 | 39444.64 | 39147.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 15:15:00 | 38629.00 | 39004.97 | 39031.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 15:15:00 | 38629.00 | 39004.97 | 39031.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 09:15:00 | 38574.95 | 38918.97 | 38989.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 10:15:00 | 38962.05 | 38927.58 | 38987.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 10:15:00 | 38962.05 | 38927.58 | 38987.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 38962.05 | 38927.58 | 38987.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 38962.05 | 38927.58 | 38987.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 39199.60 | 38981.99 | 39006.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:45:00 | 39150.40 | 38981.99 | 39006.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 12:15:00 | 39211.00 | 39027.79 | 39025.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 09:15:00 | 39456.05 | 39212.39 | 39122.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 10:15:00 | 39353.85 | 39681.52 | 39486.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 10:15:00 | 39353.85 | 39681.52 | 39486.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 39353.85 | 39681.52 | 39486.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:45:00 | 39360.00 | 39681.52 | 39486.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 39465.00 | 39638.21 | 39484.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 09:15:00 | 39581.05 | 39535.20 | 39476.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 10:15:00 | 39600.00 | 39502.16 | 39467.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 12:00:00 | 39557.60 | 39553.70 | 39499.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 13:45:00 | 39560.00 | 39550.41 | 39506.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 39400.35 | 39520.40 | 39497.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 15:00:00 | 39400.35 | 39520.40 | 39497.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 39350.50 | 39486.42 | 39483.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 39544.05 | 39486.42 | 39483.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-12 09:15:00 | 39289.60 | 39447.05 | 39466.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 09:15:00 | 39289.60 | 39447.05 | 39466.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 13:15:00 | 39116.85 | 39297.75 | 39382.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 10:15:00 | 39299.40 | 39109.02 | 39244.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 10:15:00 | 39299.40 | 39109.02 | 39244.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 39299.40 | 39109.02 | 39244.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 11:00:00 | 39299.40 | 39109.02 | 39244.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 39112.40 | 39109.69 | 39232.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 12:15:00 | 38960.85 | 39109.69 | 39232.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 15:15:00 | 38989.95 | 39127.96 | 39212.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 09:30:00 | 38961.50 | 39080.29 | 39174.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 10:00:00 | 39000.00 | 39080.29 | 39174.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 39012.00 | 39066.63 | 39159.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 11:30:00 | 38988.85 | 39059.19 | 39147.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 12:15:00 | 38965.05 | 39059.19 | 39147.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 37040.45 | 37754.65 | 38094.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-22 09:15:00 | 37893.00 | 37754.65 | 38094.39 | SL hit (close>static) qty=0.50 sl=37754.65 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 09:15:00 | 38841.85 | 38299.45 | 38236.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 10:15:00 | 39224.85 | 38484.53 | 38326.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 13:15:00 | 38524.95 | 38626.87 | 38444.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 13:15:00 | 38524.95 | 38626.87 | 38444.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 38524.95 | 38626.87 | 38444.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:45:00 | 38596.60 | 38626.87 | 38444.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 38452.55 | 38592.00 | 38445.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 15:15:00 | 38406.00 | 38592.00 | 38445.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 38406.00 | 38554.80 | 38442.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:45:00 | 38481.05 | 38531.68 | 38441.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 38649.95 | 38555.34 | 38460.71 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 09:15:00 | 38130.00 | 38393.26 | 38415.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 10:15:00 | 37836.00 | 38281.81 | 38362.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 09:15:00 | 38438.05 | 37999.04 | 38146.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 38438.05 | 37999.04 | 38146.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 38438.05 | 37999.04 | 38146.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 38438.05 | 37999.04 | 38146.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 38612.30 | 38121.69 | 38188.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 11:00:00 | 38612.30 | 38121.69 | 38188.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 38736.75 | 38190.49 | 38186.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 09:15:00 | 40131.00 | 38939.27 | 38602.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 15:15:00 | 39480.00 | 39552.27 | 39133.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 09:15:00 | 39343.00 | 39552.27 | 39133.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 39356.75 | 39513.16 | 39153.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 11:15:00 | 39854.95 | 39491.18 | 39306.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 11:15:00 | 39894.10 | 39478.86 | 39387.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 15:15:00 | 38899.80 | 39357.01 | 39372.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 15:15:00 | 38899.80 | 39357.01 | 39372.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 38400.20 | 39165.65 | 39284.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 39083.20 | 38640.64 | 38874.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 39083.20 | 38640.64 | 38874.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 39083.20 | 38640.64 | 38874.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:30:00 | 38400.05 | 38687.02 | 38833.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 13:15:00 | 37802.15 | 37602.28 | 37596.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 13:15:00 | 37802.15 | 37602.28 | 37596.73 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 09:15:00 | 37507.30 | 37597.62 | 37597.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 36997.20 | 37441.13 | 37523.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 09:15:00 | 37523.00 | 37281.57 | 37402.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 09:15:00 | 37523.00 | 37281.57 | 37402.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 37523.00 | 37281.57 | 37402.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 11:15:00 | 37197.35 | 37332.26 | 37414.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 15:15:00 | 35337.48 | 35816.32 | 36175.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-20 10:15:00 | 35941.05 | 35814.45 | 36110.35 | SL hit (close>ema200) qty=0.50 sl=35814.45 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 11:15:00 | 35952.10 | 35902.85 | 35902.13 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 13:15:00 | 35668.10 | 35857.04 | 35881.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 14:15:00 | 35298.00 | 35745.23 | 35828.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 13:15:00 | 35444.55 | 35431.96 | 35612.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 13:15:00 | 35444.55 | 35431.96 | 35612.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 35444.55 | 35431.96 | 35612.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:00:00 | 35444.55 | 35431.96 | 35612.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 35099.95 | 35365.56 | 35565.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 14:45:00 | 34839.30 | 35203.12 | 35381.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 09:15:00 | 35778.75 | 35272.47 | 35379.38 | SL hit (close>static) qty=1.00 sl=35640.10 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 12:15:00 | 35868.00 | 35492.20 | 35462.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 14:15:00 | 35891.85 | 35621.93 | 35529.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 09:15:00 | 35616.00 | 35660.43 | 35565.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 35616.00 | 35660.43 | 35565.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 35616.00 | 35660.43 | 35565.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:30:00 | 35572.40 | 35660.43 | 35565.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 35565.90 | 35641.31 | 35573.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 12:00:00 | 35565.90 | 35641.31 | 35573.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 12:15:00 | 35524.55 | 35617.96 | 35569.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 12:30:00 | 35509.10 | 35617.96 | 35569.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 13:15:00 | 35435.60 | 35581.49 | 35556.89 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 14:15:00 | 35365.40 | 35538.27 | 35539.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 10:15:00 | 35287.50 | 35451.51 | 35495.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 15:15:00 | 35300.00 | 35295.60 | 35391.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 09:15:00 | 35758.90 | 35295.60 | 35391.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 35538.40 | 35344.16 | 35404.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:30:00 | 35587.05 | 35344.16 | 35404.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 35480.10 | 35371.35 | 35411.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:30:00 | 35599.95 | 35371.35 | 35411.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 35498.45 | 35396.77 | 35419.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:30:00 | 35568.65 | 35396.77 | 35419.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 12:15:00 | 35590.00 | 35435.41 | 35434.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 13:15:00 | 35800.00 | 35508.33 | 35467.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 10:15:00 | 35532.10 | 35568.57 | 35515.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 10:15:00 | 35532.10 | 35568.57 | 35515.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 35532.10 | 35568.57 | 35515.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 15:15:00 | 35840.00 | 35528.11 | 35509.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 10:15:00 | 35445.70 | 35711.58 | 35667.27 | SL hit (close<static) qty=1.00 sl=35502.05 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 12:15:00 | 34961.00 | 35524.11 | 35587.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 14:15:00 | 34916.05 | 35309.24 | 35473.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 35911.05 | 35400.92 | 35484.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 09:15:00 | 35911.05 | 35400.92 | 35484.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 35911.05 | 35400.92 | 35484.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:45:00 | 35887.50 | 35400.92 | 35484.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 35784.70 | 35477.68 | 35511.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 11:15:00 | 35651.50 | 35477.68 | 35511.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 15:15:00 | 35288.00 | 35078.52 | 35062.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 15:15:00 | 35288.00 | 35078.52 | 35062.36 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 12:15:00 | 35021.25 | 35098.18 | 35106.58 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 35260.00 | 35116.83 | 35111.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 10:15:00 | 35350.00 | 35190.85 | 35147.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 10:15:00 | 35699.95 | 35713.61 | 35548.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 10:45:00 | 35608.75 | 35713.61 | 35548.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 35700.00 | 35712.17 | 35621.70 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2024-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 13:15:00 | 35343.00 | 35534.81 | 35559.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 35147.85 | 35416.66 | 35495.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 09:15:00 | 35000.00 | 34974.80 | 35128.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 09:15:00 | 35000.00 | 34974.80 | 35128.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 35000.00 | 34974.80 | 35128.63 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2024-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 13:15:00 | 35467.00 | 35188.17 | 35186.46 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 10:15:00 | 35002.00 | 35252.58 | 35270.51 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 09:15:00 | 35528.60 | 35273.54 | 35263.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 09:15:00 | 35749.90 | 35545.86 | 35426.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 10:15:00 | 35172.80 | 35471.24 | 35403.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 10:15:00 | 35172.80 | 35471.24 | 35403.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 35172.80 | 35471.24 | 35403.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:45:00 | 35100.95 | 35471.24 | 35403.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 34990.00 | 35375.00 | 35365.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:30:00 | 35000.00 | 35375.00 | 35365.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 12:15:00 | 34943.90 | 35288.78 | 35327.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 14:15:00 | 34456.95 | 35058.77 | 35212.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 34181.95 | 34135.37 | 34514.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 14:15:00 | 34486.35 | 34278.00 | 34447.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 34486.35 | 34278.00 | 34447.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 34486.35 | 34278.00 | 34447.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 34455.00 | 34313.40 | 34447.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:15:00 | 34792.65 | 34313.40 | 34447.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 34301.70 | 34342.95 | 34440.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 09:15:00 | 34066.80 | 34466.13 | 34468.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 09:15:00 | 34673.80 | 34507.66 | 34487.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 09:15:00 | 34673.80 | 34507.66 | 34487.19 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 13:15:00 | 34348.00 | 34477.72 | 34484.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 15:15:00 | 34181.60 | 34392.53 | 34442.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 14:15:00 | 33850.00 | 33835.84 | 33993.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 14:45:00 | 33850.85 | 33835.84 | 33993.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 34150.10 | 33898.69 | 34007.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 34698.00 | 33898.69 | 34007.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 34799.95 | 34078.94 | 34079.80 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 34845.95 | 34232.34 | 34149.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 15:15:00 | 34980.00 | 34659.42 | 34415.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 34663.00 | 34813.69 | 34580.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 13:00:00 | 34663.00 | 34813.69 | 34580.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 34486.20 | 34748.19 | 34572.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:30:00 | 34500.00 | 34748.19 | 34572.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 34424.65 | 34683.48 | 34558.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:30:00 | 34352.55 | 34683.48 | 34558.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 34380.00 | 34593.43 | 34537.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:00:00 | 34380.00 | 34593.43 | 34537.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 34297.00 | 34534.14 | 34515.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:30:00 | 34309.80 | 34534.14 | 34515.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 11:15:00 | 34286.75 | 34484.67 | 34495.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 13:15:00 | 34000.00 | 34367.29 | 34438.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 09:15:00 | 34364.65 | 34134.30 | 34219.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 34364.65 | 34134.30 | 34219.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 34364.65 | 34134.30 | 34219.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:30:00 | 34490.25 | 34134.30 | 34219.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 34128.20 | 34133.08 | 34211.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 12:45:00 | 33983.75 | 34104.70 | 34184.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 14:30:00 | 33980.00 | 34083.00 | 34160.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 10:15:00 | 33949.90 | 34060.21 | 34135.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 09:15:00 | 33949.50 | 34136.80 | 34151.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 33656.00 | 34040.64 | 34106.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 11:00:00 | 33567.35 | 33945.98 | 34057.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 13:30:00 | 33607.65 | 33774.44 | 33939.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 15:15:00 | 33600.05 | 33765.57 | 33920.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 10:15:00 | 33903.50 | 33889.78 | 33889.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 10:15:00 | 33903.50 | 33889.78 | 33889.50 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-10-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 12:15:00 | 33851.75 | 33884.47 | 33887.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 10:15:00 | 33576.10 | 33791.62 | 33840.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 14:15:00 | 33879.85 | 33690.66 | 33762.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 14:15:00 | 33879.85 | 33690.66 | 33762.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 14:15:00 | 33879.85 | 33690.66 | 33762.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 14:45:00 | 33961.85 | 33690.66 | 33762.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 15:15:00 | 33810.00 | 33714.53 | 33767.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:15:00 | 33530.30 | 33714.53 | 33767.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 33650.00 | 33617.37 | 33694.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:00:00 | 33650.00 | 33617.37 | 33694.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 33491.55 | 33592.20 | 33676.35 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 09:15:00 | 34004.50 | 33748.30 | 33733.29 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 10:15:00 | 33431.90 | 33766.44 | 33782.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 14:15:00 | 33285.10 | 33514.54 | 33628.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 14:15:00 | 33500.00 | 33399.05 | 33495.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 14:15:00 | 33500.00 | 33399.05 | 33495.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 33500.00 | 33399.05 | 33495.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 15:00:00 | 33500.00 | 33399.05 | 33495.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 33400.00 | 33399.24 | 33486.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:15:00 | 34794.60 | 33399.24 | 33486.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 34700.00 | 33659.39 | 33596.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 35319.95 | 34523.20 | 34144.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 35303.00 | 35647.11 | 35103.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 10:00:00 | 35303.00 | 35647.11 | 35103.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 35445.15 | 35897.17 | 35673.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 35445.15 | 35897.17 | 35673.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 35350.00 | 35787.73 | 35643.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 35583.65 | 35787.73 | 35643.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 35423.25 | 35666.29 | 35610.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:00:00 | 35423.25 | 35666.29 | 35610.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 12:15:00 | 35368.10 | 35552.00 | 35564.93 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 09:15:00 | 35950.00 | 35590.74 | 35573.90 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 35553.10 | 35787.09 | 35794.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 12:15:00 | 35118.00 | 35518.21 | 35657.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 32057.00 | 32024.84 | 32689.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 10:00:00 | 32057.00 | 32024.84 | 32689.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 10:15:00 | 31300.00 | 31045.33 | 31343.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 11:00:00 | 31300.00 | 31045.33 | 31343.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 11:15:00 | 31500.00 | 31136.27 | 31358.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 12:00:00 | 31500.00 | 31136.27 | 31358.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 12:15:00 | 31520.30 | 31213.07 | 31372.77 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2024-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 15:15:00 | 31896.70 | 31524.87 | 31490.35 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 12:15:00 | 31405.35 | 31461.98 | 31466.29 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 14:15:00 | 31723.70 | 31516.75 | 31490.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 11:15:00 | 31798.90 | 31615.47 | 31551.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 13:15:00 | 31571.45 | 31609.75 | 31559.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 13:15:00 | 31571.45 | 31609.75 | 31559.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 31571.45 | 31609.75 | 31559.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:45:00 | 31562.05 | 31609.75 | 31559.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 31730.75 | 31633.95 | 31575.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 15:00:00 | 31730.75 | 31633.95 | 31575.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 32030.05 | 31731.74 | 31631.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 11:00:00 | 32400.00 | 31865.39 | 31701.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:45:00 | 32185.00 | 32034.46 | 31873.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 10:45:00 | 32207.90 | 32057.58 | 31899.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 11:30:00 | 32235.80 | 32105.46 | 31935.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 31652.05 | 32047.26 | 31980.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:30:00 | 31789.90 | 32047.26 | 31980.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 31795.35 | 31996.88 | 31963.68 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-02 11:15:00 | 31652.05 | 31927.91 | 31935.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 11:15:00 | 31652.05 | 31927.91 | 31935.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 12:15:00 | 31608.00 | 31863.93 | 31905.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 10:15:00 | 31194.70 | 31182.91 | 31411.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-04 11:00:00 | 31194.70 | 31182.91 | 31411.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 31121.00 | 31170.53 | 31385.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 12:45:00 | 31018.15 | 31141.02 | 31352.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 14:15:00 | 31002.10 | 31127.76 | 31326.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 15:15:00 | 31444.00 | 31263.78 | 31260.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2024-12-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 15:15:00 | 31444.00 | 31263.78 | 31260.02 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 31200.00 | 31251.03 | 31254.56 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 12:15:00 | 31460.60 | 31271.53 | 31260.91 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 14:15:00 | 31199.00 | 31249.21 | 31252.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 15:15:00 | 31090.00 | 31217.37 | 31237.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 31360.50 | 31245.99 | 31248.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 31360.50 | 31245.99 | 31248.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 31360.50 | 31245.99 | 31248.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:15:00 | 31434.00 | 31245.99 | 31248.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 31241.65 | 31245.12 | 31247.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:30:00 | 31456.80 | 31245.12 | 31247.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 11:15:00 | 31234.85 | 31243.07 | 31246.75 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 13:15:00 | 31438.15 | 31269.11 | 31257.15 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 15:15:00 | 31160.60 | 31249.58 | 31250.48 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 09:15:00 | 31327.00 | 31265.06 | 31257.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 12:15:00 | 31607.00 | 31357.29 | 31303.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 13:15:00 | 31470.00 | 31491.37 | 31424.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-12 13:45:00 | 31491.20 | 31491.37 | 31424.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 31318.30 | 31461.34 | 31422.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:15:00 | 31419.95 | 31461.34 | 31422.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 31390.00 | 31447.07 | 31419.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 11:45:00 | 31508.00 | 31455.62 | 31428.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 12:15:00 | 31531.55 | 31455.62 | 31428.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 10:00:00 | 31515.00 | 31535.51 | 31485.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 12:45:00 | 31507.00 | 31512.33 | 31485.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 31501.00 | 31510.06 | 31487.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:45:00 | 31488.75 | 31510.06 | 31487.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 31517.80 | 31511.61 | 31489.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 14:30:00 | 31500.00 | 31511.61 | 31489.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 31518.00 | 31512.89 | 31492.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:15:00 | 31300.00 | 31512.89 | 31492.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-17 09:15:00 | 31233.10 | 31456.93 | 31468.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 31233.10 | 31456.93 | 31468.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 31066.00 | 31257.53 | 31359.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 13:15:00 | 30960.00 | 30938.60 | 31055.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 13:15:00 | 30960.00 | 30938.60 | 31055.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 30960.00 | 30938.60 | 31055.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:00:00 | 30960.00 | 30938.60 | 31055.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 31105.10 | 30971.90 | 31059.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:45:00 | 31063.95 | 30971.90 | 31059.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 30975.60 | 30972.64 | 31052.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 31400.00 | 30972.64 | 31052.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 31036.45 | 30985.40 | 31050.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:30:00 | 31360.90 | 30985.40 | 31050.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 31050.00 | 30998.32 | 31050.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:45:00 | 31043.25 | 30998.32 | 31050.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 31019.15 | 31002.49 | 31047.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:00:00 | 31019.15 | 31002.49 | 31047.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 30746.95 | 30877.77 | 30964.28 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 14:15:00 | 31000.00 | 30914.26 | 30907.69 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 09:15:00 | 30817.15 | 30904.57 | 30905.03 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 10:15:00 | 30972.95 | 30918.25 | 30911.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 11:15:00 | 31000.00 | 30934.60 | 30919.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 13:15:00 | 30896.90 | 30931.57 | 30920.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 13:15:00 | 30896.90 | 30931.57 | 30920.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 30896.90 | 30931.57 | 30920.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 14:00:00 | 30896.90 | 30931.57 | 30920.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 30941.90 | 30933.64 | 30922.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 14:30:00 | 30900.05 | 30933.64 | 30922.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 15:15:00 | 30808.00 | 30908.51 | 30912.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 11:15:00 | 30491.50 | 30824.09 | 30872.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 14:15:00 | 30544.10 | 30346.72 | 30519.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 14:15:00 | 30544.10 | 30346.72 | 30519.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 30544.10 | 30346.72 | 30519.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 30544.10 | 30346.72 | 30519.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 30579.95 | 30393.37 | 30524.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 30255.45 | 30393.37 | 30524.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 11:30:00 | 30415.00 | 30402.84 | 30436.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 09:15:00 | 30800.40 | 30387.20 | 30402.38 | SL hit (close>static) qty=1.00 sl=30669.60 alert=retest2 |

### Cycle 63 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 30894.60 | 30488.68 | 30447.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 11:15:00 | 31518.20 | 30694.58 | 30544.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 09:15:00 | 30758.50 | 30835.83 | 30687.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 09:15:00 | 30758.50 | 30835.83 | 30687.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 30758.50 | 30835.83 | 30687.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:45:00 | 30798.35 | 30835.83 | 30687.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 30845.00 | 30831.45 | 30711.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 11:30:00 | 30710.00 | 30831.45 | 30711.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 30684.05 | 30801.97 | 30708.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:00:00 | 30684.05 | 30801.97 | 30708.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 30610.00 | 30763.57 | 30699.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:45:00 | 30612.60 | 30763.57 | 30699.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 30700.00 | 30750.86 | 30699.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 30700.00 | 30750.86 | 30699.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 30525.00 | 30705.69 | 30683.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 30356.00 | 30705.69 | 30683.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 30219.95 | 30608.54 | 30641.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 30115.90 | 30510.01 | 30593.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 14:15:00 | 30376.75 | 30314.85 | 30458.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 14:15:00 | 30376.75 | 30314.85 | 30458.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 30376.75 | 30314.85 | 30458.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 15:00:00 | 30376.75 | 30314.85 | 30458.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 30011.00 | 30254.08 | 30417.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:15:00 | 30444.35 | 30254.08 | 30417.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 30060.00 | 30215.26 | 30385.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 11:45:00 | 29952.00 | 30160.25 | 30328.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-07 14:15:00 | 30775.00 | 30330.06 | 30368.60 | SL hit (close>static) qty=1.00 sl=30644.75 alert=retest2 |

### Cycle 65 — BUY (started 2025-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 15:15:00 | 31000.00 | 30464.04 | 30426.00 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 30172.20 | 30383.15 | 30394.43 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 11:15:00 | 30488.60 | 30404.24 | 30402.99 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 29990.00 | 30467.03 | 30483.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 29801.10 | 30131.93 | 30301.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 29236.50 | 29131.92 | 29536.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 29236.50 | 29131.92 | 29536.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 29236.50 | 29131.92 | 29536.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:30:00 | 28891.45 | 29057.45 | 29307.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 13:00:00 | 28930.10 | 29026.44 | 29231.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 12:15:00 | 29704.10 | 29217.66 | 29216.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 12:15:00 | 29704.10 | 29217.66 | 29216.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 13:15:00 | 30194.25 | 29412.98 | 29305.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 12:15:00 | 29667.00 | 29795.39 | 29603.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 13:00:00 | 29667.00 | 29795.39 | 29603.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 29571.00 | 29750.51 | 29600.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:45:00 | 29625.15 | 29750.51 | 29600.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 29570.00 | 29714.41 | 29597.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 15:15:00 | 29700.00 | 29714.41 | 29597.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 29700.00 | 29711.53 | 29606.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 09:45:00 | 29709.05 | 29711.74 | 29616.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 11:30:00 | 29737.00 | 29719.48 | 29635.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 14:00:00 | 29711.65 | 29730.80 | 29656.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 15:15:00 | 29502.00 | 29665.44 | 29638.37 | SL hit (close<static) qty=1.00 sl=29535.50 alert=retest2 |

### Cycle 70 — SELL (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 09:15:00 | 29201.00 | 29572.55 | 29598.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 29021.95 | 29462.43 | 29546.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 29050.00 | 28983.95 | 29144.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 29050.00 | 28983.95 | 29144.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 29281.80 | 29018.69 | 29131.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 29281.80 | 29018.69 | 29131.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 29760.40 | 29167.03 | 29188.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 29760.40 | 29167.03 | 29188.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 11:15:00 | 30000.00 | 29333.62 | 29262.11 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 15:15:00 | 29094.65 | 29426.92 | 29439.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 28607.85 | 29263.11 | 29363.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 14:15:00 | 29200.05 | 29074.96 | 29210.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 14:15:00 | 29200.05 | 29074.96 | 29210.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 14:15:00 | 29200.05 | 29074.96 | 29210.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 15:00:00 | 29200.05 | 29074.96 | 29210.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 15:15:00 | 29265.00 | 29112.97 | 29215.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 09:15:00 | 28900.00 | 29112.97 | 29215.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 28761.10 | 29042.59 | 29174.09 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2025-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 14:15:00 | 29409.00 | 29232.76 | 29222.16 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 10:15:00 | 29100.05 | 29220.97 | 29222.03 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 29463.35 | 29266.90 | 29241.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 29834.15 | 29411.56 | 29313.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 15:15:00 | 29565.00 | 29868.17 | 29735.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 15:15:00 | 29565.00 | 29868.17 | 29735.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 15:15:00 | 29565.00 | 29868.17 | 29735.81 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 29211.60 | 29642.69 | 29667.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 13:15:00 | 29138.20 | 29541.80 | 29619.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 14:15:00 | 30000.80 | 29633.60 | 29654.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 14:15:00 | 30000.80 | 29633.60 | 29654.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 30000.80 | 29633.60 | 29654.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 15:00:00 | 30000.80 | 29633.60 | 29654.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 15:15:00 | 29899.90 | 29686.86 | 29676.65 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 14:15:00 | 29401.05 | 29651.42 | 29680.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 09:15:00 | 28852.55 | 29455.42 | 29583.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 14:15:00 | 28500.00 | 28450.24 | 28653.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 14:15:00 | 28500.00 | 28450.24 | 28653.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 28500.00 | 28450.24 | 28653.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 28500.00 | 28450.24 | 28653.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 28275.00 | 28417.74 | 28603.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 14:00:00 | 27965.00 | 28272.65 | 28471.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 27842.15 | 28265.57 | 28432.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 26566.75 | 27404.76 | 27743.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-12 14:15:00 | 27573.20 | 27309.58 | 27546.79 | SL hit (close>ema200) qty=0.50 sl=27309.58 alert=retest2 |

### Cycle 79 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 27137.40 | 26892.56 | 26867.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 14:15:00 | 27463.45 | 27077.93 | 26976.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 26977.40 | 27093.83 | 27003.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 26977.40 | 27093.83 | 27003.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 26977.40 | 27093.83 | 27003.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 26977.40 | 27093.83 | 27003.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 27323.25 | 27139.72 | 27032.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 11:30:00 | 27470.80 | 27189.86 | 27065.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 12:45:00 | 27458.00 | 27230.98 | 27095.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 26616.60 | 27151.22 | 27105.34 | SL hit (close<static) qty=1.00 sl=26897.60 alert=retest2 |

### Cycle 80 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 26436.10 | 27008.19 | 27044.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 26076.30 | 26425.94 | 26577.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 26338.50 | 26142.43 | 26281.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 14:15:00 | 26338.50 | 26142.43 | 26281.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 26338.50 | 26142.43 | 26281.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 15:00:00 | 26338.50 | 26142.43 | 26281.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 26348.00 | 26183.54 | 26287.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 26017.10 | 26183.54 | 26287.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 10:15:00 | 27104.20 | 26233.97 | 26217.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 27104.20 | 26233.97 | 26217.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 27325.80 | 26452.34 | 26317.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 27710.75 | 27895.65 | 27591.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 27710.75 | 27895.65 | 27591.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 27810.00 | 27878.52 | 27611.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 27556.00 | 27878.52 | 27611.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 27612.35 | 27825.29 | 27611.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 27635.00 | 27825.29 | 27611.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 27447.65 | 27749.76 | 27596.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 27447.65 | 27749.76 | 27596.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 27424.60 | 27684.73 | 27581.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 27424.60 | 27684.73 | 27581.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 27463.50 | 27521.48 | 27521.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 27098.40 | 27373.50 | 27450.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 12:15:00 | 27578.40 | 27357.32 | 27418.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 12:15:00 | 27578.40 | 27357.32 | 27418.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 27578.40 | 27357.32 | 27418.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:00:00 | 27578.40 | 27357.32 | 27418.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 27632.40 | 27412.34 | 27438.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:15:00 | 27735.15 | 27412.34 | 27438.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 14:15:00 | 27971.80 | 27524.23 | 27486.87 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 12:15:00 | 27279.80 | 27474.71 | 27490.26 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 14:15:00 | 28027.15 | 27582.73 | 27536.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 15:15:00 | 28120.00 | 27690.19 | 27589.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-17 09:15:00 | 27767.55 | 27815.46 | 27723.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 27767.55 | 27815.46 | 27723.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 27767.55 | 27815.46 | 27723.47 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 13:15:00 | 27434.90 | 27640.67 | 27663.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 14:15:00 | 27260.00 | 27564.53 | 27626.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 10:15:00 | 27560.00 | 27520.63 | 27586.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 10:15:00 | 27560.00 | 27520.63 | 27586.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 27560.00 | 27520.63 | 27586.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:45:00 | 27610.00 | 27520.63 | 27586.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 11:15:00 | 27533.00 | 27523.11 | 27581.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 12:00:00 | 27533.00 | 27523.11 | 27581.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 12:15:00 | 27446.10 | 27507.70 | 27569.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 13:45:00 | 27372.70 | 27490.00 | 27555.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 27900.00 | 27583.21 | 27583.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 27900.00 | 27583.21 | 27583.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 10:15:00 | 28076.30 | 27681.83 | 27627.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 27865.20 | 28016.28 | 27858.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 27865.20 | 28016.28 | 27858.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 27865.20 | 28016.28 | 27858.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 27865.20 | 28016.28 | 27858.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 28010.00 | 28015.02 | 27872.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 14:00:00 | 28060.65 | 28018.03 | 27908.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 12:45:00 | 28080.10 | 28014.57 | 27952.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 14:00:00 | 28062.20 | 28024.10 | 27962.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 15:00:00 | 28500.00 | 28119.28 | 28011.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 28380.90 | 28494.51 | 28375.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 28375.00 | 28494.51 | 28375.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 28335.70 | 28462.74 | 28371.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:30:00 | 28349.05 | 28462.74 | 28371.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 28449.55 | 28460.11 | 28378.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:30:00 | 28363.85 | 28460.11 | 28378.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 28400.10 | 28448.20 | 28387.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 28558.30 | 28448.20 | 28387.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:00:00 | 28562.40 | 28471.04 | 28403.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 10:15:00 | 28322.40 | 28441.31 | 28395.90 | SL hit (close<static) qty=1.00 sl=28325.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 28169.85 | 28356.77 | 28363.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 28083.00 | 28302.02 | 28337.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 15:15:00 | 27995.00 | 27977.77 | 28102.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-28 09:15:00 | 28157.00 | 27977.77 | 28102.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 28117.55 | 28005.73 | 28104.10 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 12:15:00 | 28439.10 | 28189.32 | 28172.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 28999.80 | 28362.72 | 28254.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 28414.10 | 28442.96 | 28314.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-01 09:45:00 | 28370.50 | 28442.96 | 28314.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 28212.00 | 28396.77 | 28305.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:00:00 | 28212.00 | 28396.77 | 28305.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 28399.00 | 28397.22 | 28313.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:45:00 | 28200.00 | 28397.22 | 28313.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 28301.15 | 28376.97 | 28325.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 15:00:00 | 28301.15 | 28376.97 | 28325.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 15:15:00 | 28125.00 | 28326.58 | 28306.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 09:45:00 | 28365.00 | 28341.26 | 28315.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 10:45:00 | 28382.65 | 28675.16 | 28611.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 12:15:00 | 28329.20 | 28540.92 | 28557.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 28329.20 | 28540.92 | 28557.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 28014.25 | 28435.58 | 28508.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 27653.70 | 27582.46 | 27920.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 27706.65 | 27582.46 | 27920.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 27667.65 | 27599.50 | 27897.08 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 15:15:00 | 28156.80 | 27982.74 | 27968.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 28826.00 | 28151.39 | 28046.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 30405.00 | 30464.37 | 29930.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 30405.00 | 30464.37 | 29930.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 14:15:00 | 30020.00 | 30219.18 | 30142.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 15:00:00 | 30020.00 | 30219.18 | 30142.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 15:15:00 | 30090.00 | 30193.34 | 30137.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 09:15:00 | 30195.00 | 30193.34 | 30137.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 10:15:00 | 30000.00 | 30135.74 | 30120.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 11:00:00 | 30000.00 | 30135.74 | 30120.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-22 12:15:00 | 29995.00 | 30101.07 | 30106.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-22 13:15:00 | 29885.00 | 30057.86 | 30086.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-23 14:15:00 | 30000.00 | 29964.97 | 30011.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-23 15:00:00 | 30000.00 | 29964.97 | 30011.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 15:15:00 | 30000.00 | 29971.97 | 30010.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:15:00 | 30095.00 | 29971.97 | 30010.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 29995.00 | 29976.58 | 30009.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 11:30:00 | 29890.00 | 29960.21 | 29996.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 13:45:00 | 29880.00 | 29935.33 | 29978.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 15:15:00 | 30100.00 | 29878.42 | 29876.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 15:15:00 | 30100.00 | 29878.42 | 29876.17 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 10:15:00 | 29770.00 | 29854.59 | 29865.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 13:15:00 | 29530.00 | 29742.27 | 29807.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 29735.00 | 29659.84 | 29746.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 29735.00 | 29659.84 | 29746.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 29735.00 | 29659.84 | 29746.52 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 12:15:00 | 29995.00 | 29797.64 | 29793.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 14:15:00 | 30105.00 | 29890.69 | 29838.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 29975.00 | 30099.30 | 29998.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 15:15:00 | 29975.00 | 30099.30 | 29998.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 29975.00 | 30099.30 | 29998.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:15:00 | 29895.00 | 30099.30 | 29998.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 30015.00 | 30082.44 | 29999.87 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 15:15:00 | 29660.00 | 29926.63 | 29954.23 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 30130.00 | 29970.68 | 29966.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 14:15:00 | 30220.00 | 30067.23 | 30015.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 10:15:00 | 30060.00 | 30072.02 | 30031.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 11:15:00 | 29880.00 | 30033.62 | 30017.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 29880.00 | 30033.62 | 30017.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 12:00:00 | 29880.00 | 30033.62 | 30017.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 29845.00 | 29995.89 | 30001.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 29695.00 | 29935.71 | 29973.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 15:15:00 | 29665.00 | 29605.39 | 29737.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-08 09:15:00 | 29400.00 | 29605.39 | 29737.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 29530.00 | 29590.31 | 29718.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 14:45:00 | 29375.00 | 29500.61 | 29625.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 11:45:00 | 29300.00 | 29286.36 | 29340.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-13 09:15:00 | 29815.00 | 29354.77 | 29347.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 09:15:00 | 29815.00 | 29354.77 | 29347.46 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 15:15:00 | 29100.00 | 29342.14 | 29366.11 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 29505.00 | 29392.77 | 29386.08 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 13:15:00 | 29130.00 | 29340.22 | 29364.28 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 14:15:00 | 29585.00 | 29379.78 | 29360.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 29735.00 | 29457.26 | 29399.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 29565.00 | 29637.09 | 29570.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 13:15:00 | 29565.00 | 29637.09 | 29570.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 29565.00 | 29637.09 | 29570.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 29565.00 | 29637.09 | 29570.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 29650.00 | 29639.67 | 29578.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:30:00 | 29580.00 | 29639.67 | 29578.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 29580.00 | 29621.39 | 29580.04 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 29400.00 | 29567.37 | 29570.88 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 29650.00 | 29568.81 | 29567.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 13:15:00 | 29945.00 | 29665.51 | 29613.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 12:15:00 | 29860.00 | 29914.81 | 29793.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 12:30:00 | 29860.00 | 29914.81 | 29793.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 30055.00 | 30137.05 | 30021.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:45:00 | 30135.00 | 30137.05 | 30021.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 30130.00 | 30197.01 | 30099.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:30:00 | 30075.00 | 30197.01 | 30099.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 30075.00 | 30172.61 | 30097.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:45:00 | 30080.00 | 30172.61 | 30097.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 30215.00 | 30181.09 | 30108.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:45:00 | 30080.00 | 30181.09 | 30108.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 30145.00 | 30173.87 | 30111.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 29980.00 | 30173.87 | 30111.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 30090.00 | 30157.09 | 30109.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 29995.00 | 30157.09 | 30109.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 30135.00 | 30152.68 | 30111.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 30270.00 | 30152.68 | 30111.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 13:15:00 | 29990.00 | 30125.93 | 30111.29 | SL hit (close<static) qty=1.00 sl=30005.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 15:15:00 | 30020.00 | 30089.40 | 30096.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 29960.00 | 30063.52 | 30083.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 10:15:00 | 29000.00 | 28998.53 | 29297.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-30 10:30:00 | 29015.00 | 28998.53 | 29297.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 29335.00 | 29082.06 | 29284.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:30:00 | 29350.00 | 29082.06 | 29284.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 29315.00 | 29128.65 | 29287.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 15:15:00 | 29230.00 | 29161.92 | 29288.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 11:30:00 | 29270.00 | 29200.15 | 29266.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 29265.00 | 29265.63 | 29281.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 29560.00 | 29324.50 | 29306.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 29560.00 | 29324.50 | 29306.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 10:15:00 | 29620.00 | 29383.60 | 29334.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 13:15:00 | 29815.00 | 29833.42 | 29669.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 14:00:00 | 29815.00 | 29833.42 | 29669.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 29675.00 | 29801.73 | 29670.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:45:00 | 29660.00 | 29801.73 | 29670.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 29660.00 | 29773.39 | 29669.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 29735.00 | 29773.39 | 29669.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 10:00:00 | 29755.00 | 29769.71 | 29676.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 10:45:00 | 29765.00 | 29771.77 | 29686.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 13:15:00 | 29605.00 | 29723.02 | 29684.56 | SL hit (close<static) qty=1.00 sl=29640.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 09:15:00 | 29550.00 | 29662.75 | 29664.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 10:15:00 | 29500.00 | 29630.20 | 29649.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 29525.00 | 29456.47 | 29533.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 29525.00 | 29456.47 | 29533.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 29525.00 | 29456.47 | 29533.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 11:00:00 | 29375.00 | 29440.18 | 29518.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 11:00:00 | 29450.00 | 29433.66 | 29469.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 11:45:00 | 29420.00 | 29430.93 | 29464.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 15:15:00 | 29355.00 | 29439.00 | 29459.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 29355.00 | 29422.20 | 29450.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 29515.00 | 29422.20 | 29450.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 29495.00 | 29436.76 | 29454.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 11:00:00 | 29355.00 | 29420.41 | 29445.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 11:30:00 | 29370.00 | 29432.32 | 29448.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 12:15:00 | 29395.00 | 29432.32 | 29448.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 28675.00 | 28526.09 | 28525.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 09:15:00 | 28675.00 | 28526.09 | 28525.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 10:15:00 | 28775.00 | 28663.79 | 28617.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 11:15:00 | 28570.00 | 28645.03 | 28612.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 11:15:00 | 28570.00 | 28645.03 | 28612.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 28570.00 | 28645.03 | 28612.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:00:00 | 28570.00 | 28645.03 | 28612.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 28465.00 | 28609.02 | 28599.38 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 13:15:00 | 28500.00 | 28587.22 | 28590.35 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 10:15:00 | 28645.00 | 28596.21 | 28591.08 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 28445.00 | 28580.07 | 28593.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 13:15:00 | 28400.00 | 28503.40 | 28551.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 28785.00 | 28529.94 | 28548.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 28785.00 | 28529.94 | 28548.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 28785.00 | 28529.94 | 28548.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 28785.00 | 28529.94 | 28548.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 28670.00 | 28557.95 | 28559.83 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 28770.00 | 28600.36 | 28578.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 29305.00 | 28783.17 | 28678.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 15:15:00 | 28900.00 | 28965.43 | 28835.60 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 09:15:00 | 29370.00 | 28965.43 | 28835.60 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 29565.00 | 29536.95 | 29427.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 11:15:00 | 29670.00 | 29555.56 | 29445.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 10:30:00 | 29670.00 | 29702.36 | 29592.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 29600.00 | 29689.58 | 29635.40 | SL hit (close<ema400) qty=1.00 sl=29635.40 alert=retest1 |

### Cycle 114 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 30985.00 | 31192.16 | 31204.82 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 15:15:00 | 31315.00 | 31221.98 | 31216.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 10:15:00 | 31575.00 | 31337.07 | 31272.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 11:15:00 | 31165.00 | 31302.65 | 31262.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 11:15:00 | 31165.00 | 31302.65 | 31262.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 31165.00 | 31302.65 | 31262.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 12:00:00 | 31165.00 | 31302.65 | 31262.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 31235.00 | 31289.12 | 31260.23 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2025-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 13:15:00 | 31030.00 | 31237.30 | 31239.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 14:15:00 | 30875.00 | 31164.84 | 31206.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 15:15:00 | 30900.00 | 30890.37 | 31002.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 15:15:00 | 30900.00 | 30890.37 | 31002.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 30900.00 | 30890.37 | 31002.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 31065.00 | 30890.37 | 31002.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 30960.00 | 30904.29 | 30999.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:30:00 | 31155.00 | 30904.29 | 30999.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 31030.00 | 30929.44 | 31001.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:00:00 | 31030.00 | 30929.44 | 31001.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 31305.00 | 31004.55 | 31029.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:30:00 | 31360.00 | 31004.55 | 31029.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 13:15:00 | 31130.00 | 31049.71 | 31046.86 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 14:15:00 | 30930.00 | 31025.77 | 31036.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 15:15:00 | 30850.00 | 30990.62 | 31019.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 12:15:00 | 30845.00 | 30819.91 | 30914.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-31 13:00:00 | 30845.00 | 30819.91 | 30914.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 31070.00 | 30869.93 | 30928.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:30:00 | 31040.00 | 30869.93 | 30928.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 30805.00 | 30856.94 | 30917.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:15:00 | 31050.00 | 30856.94 | 30917.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 31050.00 | 30895.55 | 30929.17 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 09:15:00 | 31465.00 | 31009.44 | 30977.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 10:15:00 | 31710.00 | 31149.55 | 31044.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 30965.00 | 31275.15 | 31158.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 14:15:00 | 30965.00 | 31275.15 | 31158.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 30965.00 | 31275.15 | 31158.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 30965.00 | 31275.15 | 31158.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 30800.00 | 31180.12 | 31125.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 30610.00 | 31180.12 | 31125.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 30610.00 | 31066.10 | 31078.73 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 15:15:00 | 30950.00 | 30801.34 | 30787.19 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 30460.00 | 30733.07 | 30757.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 30215.00 | 30629.46 | 30708.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 12:15:00 | 30595.00 | 30558.65 | 30658.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 12:15:00 | 30595.00 | 30558.65 | 30658.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 30595.00 | 30558.65 | 30658.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:00:00 | 30595.00 | 30558.65 | 30658.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 30925.00 | 30631.92 | 30682.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:00:00 | 30925.00 | 30631.92 | 30682.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 32000.00 | 30905.54 | 30802.34 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 30770.00 | 31163.13 | 31214.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 11:15:00 | 30610.00 | 31052.50 | 31159.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 13:15:00 | 31020.00 | 30996.80 | 31112.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 13:45:00 | 31000.00 | 30996.80 | 31112.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 30665.00 | 30937.12 | 31057.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 10:45:00 | 30595.00 | 30877.70 | 31019.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 15:15:00 | 30450.00 | 30747.13 | 30902.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 11:45:00 | 30515.00 | 30432.83 | 30568.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 13:15:00 | 30950.00 | 30608.20 | 30571.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 13:15:00 | 30950.00 | 30608.20 | 30571.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 14:15:00 | 31080.00 | 30702.56 | 30618.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 12:15:00 | 31000.00 | 31034.69 | 30847.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 13:00:00 | 31000.00 | 31034.69 | 30847.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 30950.00 | 31010.60 | 30868.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 30950.00 | 31010.60 | 30868.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 30945.00 | 30997.48 | 30875.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 30850.00 | 30997.48 | 30875.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 30905.00 | 30978.98 | 30878.45 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 30625.00 | 30850.36 | 30862.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 10:15:00 | 30425.00 | 30765.29 | 30823.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 15:15:00 | 30655.00 | 30627.10 | 30719.80 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 09:15:00 | 30275.00 | 30627.10 | 30719.80 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 10:30:00 | 30500.00 | 30590.14 | 30685.70 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 30730.00 | 30621.27 | 30676.28 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-25 13:15:00 | 30730.00 | 30621.27 | 30676.28 | SL hit (close>ema400) qty=1.00 sl=30676.28 alert=retest1 |

### Cycle 127 — BUY (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 14:15:00 | 30740.00 | 30568.03 | 30548.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 30940.00 | 30665.14 | 30597.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 31140.00 | 31251.08 | 31060.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 14:15:00 | 31140.00 | 31251.08 | 31060.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 31140.00 | 31251.08 | 31060.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 31140.00 | 31251.08 | 31060.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 31140.00 | 31228.86 | 31068.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 31260.00 | 31228.86 | 31068.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 14:15:00 | 30990.00 | 31226.66 | 31148.35 | SL hit (close<static) qty=1.00 sl=31050.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 09:15:00 | 30755.00 | 31092.86 | 31098.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 15:15:00 | 30380.00 | 30664.40 | 30849.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 30940.00 | 30524.00 | 30637.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 30940.00 | 30524.00 | 30637.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 30940.00 | 30524.00 | 30637.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 30910.00 | 30524.00 | 30637.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 30845.00 | 30588.20 | 30656.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:45:00 | 30845.00 | 30588.20 | 30656.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-09-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 14:15:00 | 30845.00 | 30721.54 | 30705.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 31300.00 | 30865.79 | 30775.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 13:15:00 | 30860.00 | 30997.28 | 30879.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 13:15:00 | 30860.00 | 30997.28 | 30879.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 30860.00 | 30997.28 | 30879.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:00:00 | 30860.00 | 30997.28 | 30879.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 30845.00 | 30966.83 | 30876.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 10:00:00 | 30995.00 | 30938.57 | 30877.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 11:15:00 | 31005.00 | 30906.85 | 30868.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 13:15:00 | 30780.00 | 30847.03 | 30847.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 13:15:00 | 30780.00 | 30847.03 | 30847.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 14:15:00 | 30665.00 | 30810.62 | 30831.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 31015.00 | 30813.80 | 30826.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 31015.00 | 30813.80 | 30826.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 31015.00 | 30813.80 | 30826.78 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 31020.00 | 30855.04 | 30844.35 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 14:15:00 | 30700.00 | 30862.66 | 30878.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 14:15:00 | 30535.00 | 30775.00 | 30829.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 15:15:00 | 30640.00 | 30591.51 | 30681.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 09:15:00 | 30700.00 | 30591.51 | 30681.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 30500.00 | 30573.21 | 30664.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 12:00:00 | 30405.00 | 30523.05 | 30625.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 10:00:00 | 30400.00 | 30379.57 | 30503.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 14:45:00 | 30400.00 | 30327.93 | 30425.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 28884.75 | 29334.12 | 29561.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 28880.00 | 29334.12 | 29561.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 28880.00 | 29334.12 | 29561.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-26 12:15:00 | 29230.00 | 29200.93 | 29397.03 | SL hit (close>ema200) qty=0.50 sl=29200.93 alert=retest2 |

### Cycle 133 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 29360.00 | 29183.16 | 29171.98 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 10:15:00 | 29115.00 | 29158.62 | 29162.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 13:15:00 | 29060.00 | 29138.81 | 29152.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 13:15:00 | 29095.00 | 29069.76 | 29104.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-03 14:00:00 | 29095.00 | 29069.76 | 29104.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 29105.00 | 29076.81 | 29104.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 15:15:00 | 29130.00 | 29076.81 | 29104.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 29130.00 | 29087.45 | 29106.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 29175.00 | 29087.45 | 29106.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 29255.00 | 29120.96 | 29120.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 10:15:00 | 29315.00 | 29159.77 | 29138.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 15:15:00 | 29205.00 | 29221.01 | 29181.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 09:15:00 | 29300.00 | 29221.01 | 29181.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 29180.00 | 29212.81 | 29181.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:00:00 | 29180.00 | 29212.81 | 29181.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 29200.00 | 29210.24 | 29183.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:30:00 | 29150.00 | 29210.24 | 29183.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 29085.00 | 29185.20 | 29174.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:30:00 | 29110.00 | 29185.20 | 29174.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 29100.00 | 29168.16 | 29167.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:30:00 | 29085.00 | 29168.16 | 29167.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 13:15:00 | 29080.00 | 29150.53 | 29159.75 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 12:15:00 | 29200.00 | 29166.05 | 29161.51 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 29030.00 | 29151.07 | 29157.57 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 29215.00 | 29157.41 | 29154.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 29335.00 | 29211.74 | 29181.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 29350.00 | 29418.57 | 29324.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 11:00:00 | 29350.00 | 29418.57 | 29324.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 29350.00 | 29404.85 | 29326.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:45:00 | 29360.00 | 29404.85 | 29326.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 29170.00 | 29357.88 | 29312.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:00:00 | 29170.00 | 29357.88 | 29312.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 28975.00 | 29281.31 | 29282.01 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 29275.00 | 29208.20 | 29202.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 15:15:00 | 29390.00 | 29244.56 | 29219.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 29200.00 | 29235.65 | 29218.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 29200.00 | 29235.65 | 29218.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 29200.00 | 29235.65 | 29218.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:00:00 | 29200.00 | 29235.65 | 29218.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 29365.00 | 29261.52 | 29231.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 12:30:00 | 29380.00 | 29288.17 | 29249.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 13:15:00 | 29380.00 | 29288.17 | 29249.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 14:00:00 | 29380.00 | 29306.54 | 29261.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 09:30:00 | 29455.00 | 29384.79 | 29310.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 29625.00 | 29511.42 | 29418.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:30:00 | 29540.00 | 29511.42 | 29418.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 29620.00 | 29690.24 | 29618.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:00:00 | 29620.00 | 29690.24 | 29618.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 29595.00 | 29671.19 | 29616.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:30:00 | 29605.00 | 29671.19 | 29616.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 29620.00 | 29660.95 | 29616.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:30:00 | 29555.00 | 29660.95 | 29616.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 29600.00 | 29648.76 | 29615.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 29625.00 | 29648.76 | 29615.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 29645.00 | 29648.01 | 29617.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:15:00 | 29520.00 | 29648.01 | 29617.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 29595.00 | 29637.41 | 29615.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:00:00 | 29725.00 | 29643.99 | 29623.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 29520.00 | 29783.95 | 29790.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 29520.00 | 29783.95 | 29790.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 10:15:00 | 29440.00 | 29659.46 | 29727.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 13:15:00 | 29680.00 | 29608.05 | 29681.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 14:00:00 | 29680.00 | 29608.05 | 29681.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 29775.00 | 29641.44 | 29690.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 29775.00 | 29641.44 | 29690.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 29675.00 | 29648.15 | 29688.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 29670.00 | 29648.15 | 29688.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 29735.00 | 29665.52 | 29692.95 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 11:15:00 | 29785.00 | 29714.93 | 29712.23 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 29465.00 | 29670.56 | 29692.88 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 29900.00 | 29716.45 | 29711.71 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 09:15:00 | 29570.00 | 29699.73 | 29705.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 10:15:00 | 29400.00 | 29639.78 | 29677.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 15:15:00 | 29595.00 | 29565.78 | 29618.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 09:15:00 | 29700.00 | 29565.78 | 29618.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 147 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 30550.00 | 29762.63 | 29703.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 10:15:00 | 30675.00 | 29945.10 | 29791.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 35235.00 | 35550.89 | 34452.85 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:45:00 | 36200.00 | 35604.14 | 34989.85 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 12:45:00 | 36025.00 | 35805.72 | 35245.73 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 14:00:00 | 36130.00 | 35870.58 | 35326.12 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 13:15:00 | 36055.00 | 35839.91 | 35563.13 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 35605.00 | 35848.63 | 35641.27 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 35605.00 | 35848.63 | 35641.27 | SL hit (close<ema400) qty=1.00 sl=35641.27 alert=retest1 |

### Cycle 148 — SELL (started 2025-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 15:15:00 | 35600.00 | 35894.03 | 35926.47 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 36355.00 | 35986.22 | 35965.43 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 35920.00 | 36054.59 | 36061.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 11:15:00 | 35890.00 | 36021.67 | 36046.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 15:15:00 | 36000.00 | 35980.49 | 36013.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 09:15:00 | 35930.00 | 35980.49 | 36013.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 35780.00 | 35940.39 | 35992.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 10:45:00 | 35690.00 | 35898.31 | 35968.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 11:15:00 | 35665.00 | 35898.31 | 35968.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 15:15:00 | 35640.00 | 35781.50 | 35882.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 33905.50 | 34346.08 | 34516.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 33881.75 | 34346.08 | 34516.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 33858.00 | 34346.08 | 34516.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 34500.00 | 34376.86 | 34514.98 | SL hit (close>ema200) qty=0.50 sl=34376.86 alert=retest2 |

### Cycle 151 — BUY (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 14:15:00 | 34895.00 | 34634.09 | 34603.95 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 10:15:00 | 34265.00 | 34541.34 | 34570.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 10:15:00 | 34055.00 | 34349.54 | 34430.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 34365.00 | 34352.63 | 34424.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 12:00:00 | 34365.00 | 34352.63 | 34424.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 34365.00 | 34355.10 | 34419.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 34500.00 | 34355.10 | 34419.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 34510.00 | 34386.08 | 34427.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:00:00 | 34510.00 | 34386.08 | 34427.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 34615.00 | 34431.87 | 34444.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 34615.00 | 34431.87 | 34444.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 34405.00 | 34426.49 | 34441.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 34655.00 | 34426.49 | 34441.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 35340.00 | 34609.19 | 34522.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 10:15:00 | 35540.00 | 34795.36 | 34615.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 15:15:00 | 35375.00 | 35484.21 | 35230.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:15:00 | 35265.00 | 35484.21 | 35230.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 35330.00 | 35453.37 | 35239.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 12:00:00 | 35460.00 | 35440.56 | 35269.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 35565.00 | 35408.02 | 35310.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 11:00:00 | 35460.00 | 35439.53 | 35342.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 14:15:00 | 35065.00 | 35308.74 | 35306.78 | SL hit (close<static) qty=1.00 sl=35155.00 alert=retest2 |

### Cycle 154 — SELL (started 2025-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 15:15:00 | 35135.00 | 35274.00 | 35291.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 12:15:00 | 35005.00 | 35140.52 | 35214.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 12:15:00 | 35070.00 | 35008.10 | 35094.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 12:30:00 | 35035.00 | 35008.10 | 35094.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 35205.00 | 35047.48 | 35104.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 35205.00 | 35047.48 | 35104.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 34785.00 | 34994.98 | 35075.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 35110.00 | 34994.98 | 35075.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 35100.00 | 35015.98 | 35077.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 35250.00 | 35015.98 | 35077.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 35290.00 | 35070.79 | 35096.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:30:00 | 35260.00 | 35070.79 | 35096.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 35420.00 | 35140.63 | 35126.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 35535.00 | 35219.50 | 35163.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 35415.00 | 35415.07 | 35312.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 12:00:00 | 35415.00 | 35415.07 | 35312.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 35225.00 | 35377.06 | 35304.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:30:00 | 35210.00 | 35377.06 | 35304.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 35155.00 | 35332.65 | 35290.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:30:00 | 35165.00 | 35332.65 | 35290.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 35090.00 | 35234.09 | 35249.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 12:15:00 | 34825.00 | 35050.87 | 35151.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 15:15:00 | 34395.00 | 34340.85 | 34550.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 09:15:00 | 34405.00 | 34340.85 | 34550.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 34535.00 | 34379.68 | 34549.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:30:00 | 34705.00 | 34379.68 | 34549.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 34740.00 | 34451.74 | 34566.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:30:00 | 34715.00 | 34451.74 | 34566.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 34820.00 | 34525.40 | 34589.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:00:00 | 34820.00 | 34525.40 | 34589.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 35620.00 | 34807.45 | 34710.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 13:15:00 | 35970.00 | 35239.15 | 35044.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 36175.00 | 36316.02 | 36054.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 36175.00 | 36316.02 | 36054.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 36175.00 | 36316.02 | 36054.50 | EMA400 retest candle locked (from upside) |

### Cycle 158 — SELL (started 2026-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 13:15:00 | 35345.00 | 35878.12 | 35912.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 14:15:00 | 35000.00 | 35702.50 | 35829.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 35065.00 | 34936.55 | 35232.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-08 10:00:00 | 35065.00 | 34936.55 | 35232.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 35235.00 | 34996.24 | 35232.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 35235.00 | 34996.24 | 35232.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 34830.00 | 34962.99 | 35196.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 15:00:00 | 34750.00 | 34942.57 | 35130.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 13:00:00 | 34590.00 | 34724.82 | 34933.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:15:00 | 34590.00 | 34308.33 | 34483.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 15:15:00 | 34840.00 | 34546.08 | 34524.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 34840.00 | 34546.08 | 34524.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 13:15:00 | 35215.00 | 34876.31 | 34707.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 10:15:00 | 35065.00 | 35156.03 | 34916.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 10:45:00 | 35020.00 | 35156.03 | 34916.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 34880.00 | 35100.83 | 34913.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:00:00 | 34880.00 | 35100.83 | 34913.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 34855.00 | 35051.66 | 34908.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:30:00 | 34825.00 | 35051.66 | 34908.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 34835.00 | 35008.33 | 34901.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:45:00 | 34840.00 | 35008.33 | 34901.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 35070.00 | 35117.46 | 34986.13 | EMA400 retest candle locked (from upside) |

### Cycle 160 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 34410.00 | 34887.05 | 34917.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 34160.00 | 34665.31 | 34807.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 11:15:00 | 34080.00 | 34017.24 | 34299.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:00:00 | 34080.00 | 34017.24 | 34299.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 33990.00 | 34001.06 | 34184.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 34205.00 | 34001.06 | 34184.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 33730.00 | 33896.24 | 34033.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:30:00 | 33645.00 | 33773.60 | 33952.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 33840.00 | 33476.84 | 33472.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 33840.00 | 33476.84 | 33472.61 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 33345.00 | 33455.22 | 33467.81 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2026-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 15:15:00 | 34710.00 | 33713.74 | 33580.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 14:15:00 | 35670.00 | 35210.81 | 34925.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 35260.00 | 35274.92 | 35007.70 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 14:15:00 | 35775.00 | 35491.55 | 35202.83 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 35180.00 | 35456.99 | 35238.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-05 15:15:00 | 35180.00 | 35456.99 | 35238.95 | SL hit (close<ema400) qty=1.00 sl=35238.95 alert=retest1 |

### Cycle 164 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 35965.00 | 36915.17 | 36987.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 35005.00 | 36179.02 | 36587.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 35805.00 | 35476.21 | 36020.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 13:00:00 | 35805.00 | 35476.21 | 36020.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 36035.00 | 35587.96 | 36022.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 11:15:00 | 35685.00 | 35923.35 | 36064.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 36125.00 | 36023.43 | 36046.88 | SL hit (close>static) qty=1.00 sl=36100.00 alert=retest2 |

### Cycle 165 — BUY (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 14:15:00 | 37000.00 | 36232.00 | 36136.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 37700.00 | 36640.48 | 36346.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 36905.00 | 36970.14 | 36657.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 15:00:00 | 36905.00 | 36970.14 | 36657.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 36945.00 | 36965.11 | 36683.35 | EMA400 retest candle locked (from upside) |

### Cycle 166 — SELL (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 14:15:00 | 36325.00 | 36523.00 | 36546.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 15:15:00 | 35990.00 | 36416.40 | 36496.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 15:15:00 | 36425.00 | 36286.79 | 36372.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 15:15:00 | 36425.00 | 36286.79 | 36372.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 36425.00 | 36286.79 | 36372.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 13:30:00 | 36015.00 | 36176.88 | 36278.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 14:30:00 | 35700.00 | 36109.50 | 36238.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:45:00 | 35865.00 | 36057.12 | 36172.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 15:15:00 | 36675.00 | 36182.62 | 36188.36 | SL hit (close>static) qty=1.00 sl=36425.00 alert=retest2 |

### Cycle 167 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 36710.00 | 36288.10 | 36235.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 11:15:00 | 36945.00 | 36501.42 | 36395.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 36770.00 | 36897.83 | 36669.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 36770.00 | 36897.83 | 36669.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 36770.00 | 36897.83 | 36669.47 | EMA400 retest candle locked (from upside) |

### Cycle 168 — SELL (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 14:15:00 | 35800.00 | 36518.92 | 36562.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 35180.00 | 36168.11 | 36390.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 34585.00 | 34449.49 | 35003.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 34585.00 | 34449.49 | 35003.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 34145.00 | 33763.73 | 33949.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:00:00 | 34145.00 | 33763.73 | 33949.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 34000.00 | 33810.98 | 33954.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:45:00 | 34270.00 | 33810.98 | 33954.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 33950.00 | 33838.78 | 33953.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 34175.00 | 33838.78 | 33953.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 34220.00 | 33915.03 | 33977.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:45:00 | 34420.00 | 33915.03 | 33977.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 34505.00 | 34033.02 | 34025.84 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 33670.00 | 34048.13 | 34051.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 33250.00 | 33836.01 | 33950.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 34205.00 | 33785.92 | 33875.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 13:15:00 | 34205.00 | 33785.92 | 33875.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 34205.00 | 33785.92 | 33875.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:45:00 | 34160.00 | 33785.92 | 33875.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 34295.00 | 33887.74 | 33914.00 | EMA400 retest candle locked (from downside) |

### Cycle 171 — BUY (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 15:15:00 | 34250.00 | 33960.19 | 33944.54 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 33720.00 | 33912.15 | 33924.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 33305.00 | 33790.72 | 33867.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 32910.00 | 32798.06 | 33141.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 15:00:00 | 32910.00 | 32798.06 | 33141.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 33410.00 | 32952.76 | 33154.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 33410.00 | 32952.76 | 33154.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 33545.00 | 33071.21 | 33189.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 13:45:00 | 33230.00 | 33234.70 | 33247.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 09:45:00 | 33250.00 | 33177.41 | 33214.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 31568.50 | 32298.63 | 32604.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 31587.50 | 32298.63 | 32604.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 31610.00 | 31319.31 | 31739.13 | SL hit (close>ema200) qty=0.50 sl=31319.31 alert=retest2 |

### Cycle 173 — BUY (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 15:15:00 | 31900.00 | 31799.61 | 31794.98 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 31245.00 | 31688.69 | 31744.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 31185.00 | 31433.26 | 31591.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 31210.00 | 30572.91 | 30893.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 31210.00 | 30572.91 | 30893.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 31210.00 | 30572.91 | 30893.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 12:30:00 | 30150.00 | 30417.45 | 30741.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:00:00 | 29955.00 | 30417.45 | 30741.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 09:30:00 | 30260.00 | 29584.37 | 29613.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 10:15:00 | 31055.00 | 29878.50 | 29744.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 31055.00 | 29878.50 | 29744.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 14:15:00 | 31555.00 | 30652.18 | 30190.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 10:15:00 | 31385.00 | 31433.82 | 31171.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 14:15:00 | 31300.00 | 31377.21 | 31225.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 31300.00 | 31377.21 | 31225.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 15:00:00 | 31300.00 | 31377.21 | 31225.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 31400.00 | 31381.77 | 31241.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 31630.00 | 31381.77 | 31241.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 31435.00 | 31408.93 | 31387.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 10:00:00 | 31490.00 | 31425.15 | 31396.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 13:15:00 | 31280.00 | 31366.30 | 31376.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — SELL (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 13:15:00 | 31280.00 | 31366.30 | 31376.26 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 15:15:00 | 31440.00 | 31388.03 | 31384.84 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 09:15:00 | 31210.00 | 31352.43 | 31368.95 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2026-04-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 12:15:00 | 31620.00 | 31420.60 | 31397.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 09:15:00 | 31985.00 | 31652.37 | 31553.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 09:15:00 | 32670.00 | 32813.92 | 32463.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 10:00:00 | 32670.00 | 32813.92 | 32463.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 32310.00 | 32713.13 | 32449.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 32310.00 | 32713.13 | 32449.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 32470.00 | 32664.51 | 32451.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:45:00 | 32325.00 | 32664.51 | 32451.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 32650.00 | 32661.61 | 32469.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:30:00 | 32570.00 | 32661.61 | 32469.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 33005.00 | 33310.86 | 33037.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:00:00 | 33005.00 | 33310.86 | 33037.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 33340.00 | 33316.69 | 33065.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 13:30:00 | 33350.00 | 33309.35 | 33084.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 15:00:00 | 33420.00 | 33331.48 | 33115.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 32810.00 | 33228.33 | 33139.82 | SL hit (close<static) qty=1.00 sl=33000.00 alert=retest2 |

### Cycle 180 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 32885.00 | 33098.07 | 33099.07 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 14:15:00 | 33250.00 | 33109.96 | 33095.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 33570.00 | 33242.78 | 33160.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 11:15:00 | 33060.00 | 33256.18 | 33184.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 11:15:00 | 33060.00 | 33256.18 | 33184.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 33060.00 | 33256.18 | 33184.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:00:00 | 33060.00 | 33256.18 | 33184.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 32955.00 | 33195.94 | 33163.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:45:00 | 33030.00 | 33195.94 | 33163.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 33115.00 | 33150.04 | 33147.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:15:00 | 33045.00 | 33150.04 | 33147.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 32755.00 | 33071.03 | 33112.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 10:15:00 | 32470.00 | 32950.83 | 33053.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 14:15:00 | 32560.00 | 32410.80 | 32603.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 15:00:00 | 32560.00 | 32410.80 | 32603.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 32525.00 | 32433.64 | 32595.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 32655.00 | 32433.64 | 32595.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 32580.00 | 32462.91 | 32594.54 | EMA400 retest candle locked (from downside) |

### Cycle 183 — BUY (started 2026-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 15:15:00 | 32760.00 | 32660.39 | 32655.03 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 32230.00 | 32574.32 | 32616.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 12:15:00 | 32120.00 | 32385.09 | 32510.94 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-18 11:30:00 | 30510.00 | 2024-05-23 12:15:00 | 31019.50 | STOP_HIT | 1.00 | 1.67% |
| BUY | retest2 | 2024-06-18 09:15:00 | 36827.60 | 2024-06-26 14:15:00 | 37178.30 | STOP_HIT | 1.00 | 0.95% |
| BUY | retest2 | 2024-07-03 12:45:00 | 39300.00 | 2024-07-05 15:15:00 | 38629.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-07-05 09:30:00 | 39299.95 | 2024-07-05 15:15:00 | 38629.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-07-11 09:15:00 | 39581.05 | 2024-07-12 09:15:00 | 39289.60 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-07-11 10:15:00 | 39600.00 | 2024-07-12 09:15:00 | 39289.60 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-07-11 12:00:00 | 39557.60 | 2024-07-12 09:15:00 | 39289.60 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-07-11 13:45:00 | 39560.00 | 2024-07-12 09:15:00 | 39289.60 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-07-15 12:15:00 | 38960.85 | 2024-07-22 09:15:00 | 37040.45 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2024-07-15 12:15:00 | 38960.85 | 2024-07-22 09:15:00 | 37893.00 | STOP_HIT | 0.50 | 2.74% |
| SELL | retest2 | 2024-07-15 15:15:00 | 38989.95 | 2024-07-22 09:15:00 | 37050.00 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2024-07-15 15:15:00 | 38989.95 | 2024-07-22 09:15:00 | 37893.00 | STOP_HIT | 0.50 | 2.81% |
| SELL | retest2 | 2024-07-16 09:30:00 | 38961.50 | 2024-07-22 09:15:00 | 37039.41 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2024-07-16 09:30:00 | 38961.50 | 2024-07-22 09:15:00 | 37893.00 | STOP_HIT | 0.50 | 2.74% |
| SELL | retest2 | 2024-07-16 10:00:00 | 39000.00 | 2024-07-23 09:15:00 | 38841.85 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2024-07-16 11:30:00 | 38988.85 | 2024-07-23 09:15:00 | 38841.85 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2024-07-16 12:15:00 | 38965.05 | 2024-07-23 09:15:00 | 38841.85 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2024-08-01 11:15:00 | 39854.95 | 2024-08-02 15:15:00 | 38899.80 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2024-08-02 11:15:00 | 39894.10 | 2024-08-02 15:15:00 | 38899.80 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2024-08-06 14:30:00 | 38400.05 | 2024-08-12 13:15:00 | 37802.15 | STOP_HIT | 1.00 | 1.56% |
| SELL | retest2 | 2024-08-14 11:15:00 | 37197.35 | 2024-08-19 15:15:00 | 35337.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-14 11:15:00 | 37197.35 | 2024-08-20 10:15:00 | 35941.05 | STOP_HIT | 0.50 | 3.38% |
| SELL | retest2 | 2024-08-26 14:45:00 | 34839.30 | 2024-08-27 09:15:00 | 35778.75 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2024-09-02 15:15:00 | 35840.00 | 2024-09-04 10:15:00 | 35445.70 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-09-05 11:15:00 | 35651.50 | 2024-09-10 15:15:00 | 35288.00 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2024-10-04 09:15:00 | 34066.80 | 2024-10-04 09:15:00 | 34673.80 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-10-15 12:45:00 | 33983.75 | 2024-10-21 10:15:00 | 33903.50 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2024-10-15 14:30:00 | 33980.00 | 2024-10-21 10:15:00 | 33903.50 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2024-10-16 10:15:00 | 33949.90 | 2024-10-21 10:15:00 | 33903.50 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2024-10-17 09:15:00 | 33949.50 | 2024-10-21 10:15:00 | 33903.50 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2024-10-17 11:00:00 | 33567.35 | 2024-10-21 10:15:00 | 33903.50 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-10-17 13:30:00 | 33607.65 | 2024-10-21 10:15:00 | 33903.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-10-17 15:15:00 | 33600.05 | 2024-10-21 10:15:00 | 33903.50 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-11-28 11:00:00 | 32400.00 | 2024-12-02 11:15:00 | 31652.05 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2024-11-29 09:45:00 | 32185.00 | 2024-12-02 11:15:00 | 31652.05 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-11-29 10:45:00 | 32207.90 | 2024-12-02 11:15:00 | 31652.05 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-11-29 11:30:00 | 32235.80 | 2024-12-02 11:15:00 | 31652.05 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-12-04 12:45:00 | 31018.15 | 2024-12-06 15:15:00 | 31444.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-12-04 14:15:00 | 31002.10 | 2024-12-06 15:15:00 | 31444.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-12-13 11:45:00 | 31508.00 | 2024-12-17 09:15:00 | 31233.10 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-12-13 12:15:00 | 31531.55 | 2024-12-17 09:15:00 | 31233.10 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-12-16 10:00:00 | 31515.00 | 2024-12-17 09:15:00 | 31233.10 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-12-16 12:45:00 | 31507.00 | 2024-12-17 09:15:00 | 31233.10 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-12-31 09:15:00 | 30255.45 | 2025-01-02 09:15:00 | 30800.40 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-01-01 11:30:00 | 30415.00 | 2025-01-02 09:15:00 | 30800.40 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-01-07 11:45:00 | 29952.00 | 2025-01-07 14:15:00 | 30775.00 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-01-15 09:30:00 | 28891.45 | 2025-01-16 12:15:00 | 29704.10 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-01-15 13:00:00 | 28930.10 | 2025-01-16 12:15:00 | 29704.10 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-01-20 09:45:00 | 29709.05 | 2025-01-20 15:15:00 | 29502.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-01-20 11:30:00 | 29737.00 | 2025-01-20 15:15:00 | 29502.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-01-20 14:00:00 | 29711.65 | 2025-01-20 15:15:00 | 29502.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-02-07 14:00:00 | 27965.00 | 2025-02-12 09:15:00 | 26566.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 14:00:00 | 27965.00 | 2025-02-12 14:15:00 | 27573.20 | STOP_HIT | 0.50 | 1.40% |
| SELL | retest2 | 2025-02-10 09:15:00 | 27842.15 | 2025-02-17 09:15:00 | 26450.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 27842.15 | 2025-02-17 13:15:00 | 26785.55 | STOP_HIT | 0.50 | 3.79% |
| BUY | retest2 | 2025-02-21 11:30:00 | 27470.80 | 2025-02-24 09:15:00 | 26616.60 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2025-02-21 12:45:00 | 27458.00 | 2025-02-24 09:15:00 | 26616.60 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2025-03-04 09:15:00 | 26017.10 | 2025-03-05 10:15:00 | 27104.20 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2025-03-18 13:45:00 | 27372.70 | 2025-03-19 09:15:00 | 27900.00 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-03-20 14:00:00 | 28060.65 | 2025-03-26 10:15:00 | 28322.40 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2025-03-21 12:45:00 | 28080.10 | 2025-03-26 10:15:00 | 28322.40 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2025-03-21 14:00:00 | 28062.20 | 2025-03-26 12:15:00 | 28169.85 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2025-03-21 15:00:00 | 28500.00 | 2025-03-26 12:15:00 | 28169.85 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-03-26 09:15:00 | 28558.30 | 2025-03-26 12:15:00 | 28169.85 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-03-26 10:00:00 | 28562.40 | 2025-03-26 12:15:00 | 28169.85 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-04-02 09:45:00 | 28365.00 | 2025-04-04 12:15:00 | 28329.20 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-04-04 10:45:00 | 28382.65 | 2025-04-04 12:15:00 | 28329.20 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-04-24 11:30:00 | 29890.00 | 2025-04-25 15:15:00 | 30100.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-04-24 13:45:00 | 29880.00 | 2025-04-25 15:15:00 | 30100.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-05-08 14:45:00 | 29375.00 | 2025-05-13 09:15:00 | 29815.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-05-12 11:45:00 | 29300.00 | 2025-05-13 09:15:00 | 29815.00 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-05-27 11:15:00 | 30270.00 | 2025-05-27 13:15:00 | 29990.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-05-30 15:15:00 | 29230.00 | 2025-06-03 09:15:00 | 29560.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-06-02 11:30:00 | 29270.00 | 2025-06-03 09:15:00 | 29560.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-06-03 09:15:00 | 29265.00 | 2025-06-03 09:15:00 | 29560.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-06-05 09:15:00 | 29735.00 | 2025-06-05 13:15:00 | 29605.00 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-06-05 10:00:00 | 29755.00 | 2025-06-05 13:15:00 | 29605.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-06-05 10:45:00 | 29765.00 | 2025-06-05 13:15:00 | 29605.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-06-09 11:00:00 | 29375.00 | 2025-06-26 09:15:00 | 28675.00 | STOP_HIT | 1.00 | 2.38% |
| SELL | retest2 | 2025-06-10 11:00:00 | 29450.00 | 2025-06-26 09:15:00 | 28675.00 | STOP_HIT | 1.00 | 2.63% |
| SELL | retest2 | 2025-06-10 11:45:00 | 29420.00 | 2025-06-26 09:15:00 | 28675.00 | STOP_HIT | 1.00 | 2.53% |
| SELL | retest2 | 2025-06-10 15:15:00 | 29355.00 | 2025-06-26 09:15:00 | 28675.00 | STOP_HIT | 1.00 | 2.32% |
| SELL | retest2 | 2025-06-11 11:00:00 | 29355.00 | 2025-06-26 09:15:00 | 28675.00 | STOP_HIT | 1.00 | 2.32% |
| SELL | retest2 | 2025-06-11 11:30:00 | 29370.00 | 2025-06-26 09:15:00 | 28675.00 | STOP_HIT | 1.00 | 2.37% |
| SELL | retest2 | 2025-06-11 12:15:00 | 29395.00 | 2025-06-26 09:15:00 | 28675.00 | STOP_HIT | 1.00 | 2.45% |
| BUY | retest1 | 2025-07-07 09:15:00 | 29370.00 | 2025-07-14 09:15:00 | 29600.00 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2025-07-10 11:15:00 | 29670.00 | 2025-07-16 09:15:00 | 29550.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-07-11 10:30:00 | 29670.00 | 2025-07-16 09:15:00 | 29550.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-07-14 12:30:00 | 29670.00 | 2025-07-16 09:15:00 | 29550.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-07-14 13:15:00 | 29705.00 | 2025-07-25 13:15:00 | 30985.00 | STOP_HIT | 1.00 | 4.31% |
| BUY | retest2 | 2025-07-15 09:15:00 | 30100.00 | 2025-07-25 13:15:00 | 30985.00 | STOP_HIT | 1.00 | 2.94% |
| BUY | retest2 | 2025-07-15 13:15:00 | 29765.00 | 2025-07-25 13:15:00 | 30985.00 | STOP_HIT | 1.00 | 4.10% |
| BUY | retest2 | 2025-07-16 09:15:00 | 29905.00 | 2025-07-25 13:15:00 | 30985.00 | STOP_HIT | 1.00 | 3.61% |
| BUY | retest2 | 2025-07-16 11:15:00 | 29840.00 | 2025-07-25 13:15:00 | 30985.00 | STOP_HIT | 1.00 | 3.84% |
| BUY | retest2 | 2025-07-18 14:15:00 | 29905.00 | 2025-07-25 13:15:00 | 30985.00 | STOP_HIT | 1.00 | 3.61% |
| BUY | retest2 | 2025-07-21 09:30:00 | 29965.00 | 2025-07-25 13:15:00 | 30985.00 | STOP_HIT | 1.00 | 3.40% |
| SELL | retest2 | 2025-08-13 10:45:00 | 30595.00 | 2025-08-19 13:15:00 | 30950.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-08-13 15:15:00 | 30450.00 | 2025-08-19 13:15:00 | 30950.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-08-18 11:45:00 | 30515.00 | 2025-08-19 13:15:00 | 30950.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest1 | 2025-08-25 09:15:00 | 30275.00 | 2025-08-25 13:15:00 | 30730.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest1 | 2025-08-25 10:30:00 | 30500.00 | 2025-08-25 13:15:00 | 30730.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-08-26 09:15:00 | 30265.00 | 2025-08-29 14:15:00 | 30740.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-08-26 10:30:00 | 30455.00 | 2025-08-29 14:15:00 | 30740.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-08-26 11:30:00 | 30400.00 | 2025-08-29 14:15:00 | 30740.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-08-26 15:00:00 | 30400.00 | 2025-08-29 14:15:00 | 30740.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-09-03 09:15:00 | 31260.00 | 2025-09-03 14:15:00 | 30990.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-09-10 10:00:00 | 30995.00 | 2025-09-10 13:15:00 | 30780.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-09-10 11:15:00 | 31005.00 | 2025-09-10 13:15:00 | 30780.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-09-17 12:00:00 | 30405.00 | 2025-09-25 14:15:00 | 28884.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 10:00:00 | 30400.00 | 2025-09-25 14:15:00 | 28880.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 14:45:00 | 30400.00 | 2025-09-25 14:15:00 | 28880.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 12:00:00 | 30405.00 | 2025-09-26 12:15:00 | 29230.00 | STOP_HIT | 0.50 | 3.86% |
| SELL | retest2 | 2025-09-18 10:00:00 | 30400.00 | 2025-09-26 12:15:00 | 29230.00 | STOP_HIT | 0.50 | 3.85% |
| SELL | retest2 | 2025-09-18 14:45:00 | 30400.00 | 2025-09-26 12:15:00 | 29230.00 | STOP_HIT | 0.50 | 3.85% |
| BUY | retest2 | 2025-10-16 12:30:00 | 29380.00 | 2025-10-28 14:15:00 | 29520.00 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-10-16 13:15:00 | 29380.00 | 2025-10-28 14:15:00 | 29520.00 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-10-16 14:00:00 | 29380.00 | 2025-10-28 14:15:00 | 29520.00 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-10-17 09:30:00 | 29455.00 | 2025-10-28 14:15:00 | 29520.00 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-10-24 14:00:00 | 29725.00 | 2025-10-28 14:15:00 | 29520.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest1 | 2025-11-10 09:45:00 | 36200.00 | 2025-11-11 15:15:00 | 35605.00 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest1 | 2025-11-10 12:45:00 | 36025.00 | 2025-11-11 15:15:00 | 35605.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest1 | 2025-11-10 14:00:00 | 36130.00 | 2025-11-11 15:15:00 | 35605.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest1 | 2025-11-11 13:15:00 | 36055.00 | 2025-11-11 15:15:00 | 35605.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-11-12 10:45:00 | 36150.00 | 2025-11-14 15:15:00 | 35600.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-11-12 14:00:00 | 36155.00 | 2025-11-14 15:15:00 | 35600.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-11-13 09:15:00 | 36350.00 | 2025-11-14 15:15:00 | 35600.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-11-13 10:00:00 | 36185.00 | 2025-11-14 15:15:00 | 35600.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-11-20 10:45:00 | 35690.00 | 2025-12-05 15:15:00 | 33905.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 11:15:00 | 35665.00 | 2025-12-05 15:15:00 | 33881.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 15:15:00 | 35640.00 | 2025-12-05 15:15:00 | 33858.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 10:45:00 | 35690.00 | 2025-12-08 09:15:00 | 34500.00 | STOP_HIT | 0.50 | 3.33% |
| SELL | retest2 | 2025-11-20 11:15:00 | 35665.00 | 2025-12-08 09:15:00 | 34500.00 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2025-11-20 15:15:00 | 35640.00 | 2025-12-08 09:15:00 | 34500.00 | STOP_HIT | 0.50 | 3.20% |
| BUY | retest2 | 2025-12-16 12:00:00 | 35460.00 | 2025-12-17 14:15:00 | 35065.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-17 09:15:00 | 35565.00 | 2025-12-17 14:15:00 | 35065.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-12-17 11:00:00 | 35460.00 | 2025-12-17 14:15:00 | 35065.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2026-01-08 15:00:00 | 34750.00 | 2026-01-13 15:15:00 | 34840.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-01-09 13:00:00 | 34590.00 | 2026-01-13 15:15:00 | 34840.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-01-13 09:15:00 | 34590.00 | 2026-01-13 15:15:00 | 34840.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-01-23 11:30:00 | 33645.00 | 2026-01-28 14:15:00 | 33840.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2026-02-05 14:15:00 | 35775.00 | 2026-02-05 15:15:00 | 35180.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2026-02-09 09:15:00 | 35480.00 | 2026-02-13 10:15:00 | 35965.00 | STOP_HIT | 1.00 | 1.37% |
| BUY | retest2 | 2026-02-09 10:00:00 | 35425.00 | 2026-02-13 10:15:00 | 35965.00 | STOP_HIT | 1.00 | 1.52% |
| BUY | retest2 | 2026-02-09 11:00:00 | 35410.00 | 2026-02-13 10:15:00 | 35965.00 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2026-02-09 14:45:00 | 35520.00 | 2026-02-13 10:15:00 | 35965.00 | STOP_HIT | 1.00 | 1.25% |
| SELL | retest2 | 2026-02-17 11:15:00 | 35685.00 | 2026-02-18 11:15:00 | 36125.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-02-24 13:30:00 | 36015.00 | 2026-02-25 15:15:00 | 36675.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-02-24 14:30:00 | 35700.00 | 2026-02-25 15:15:00 | 36675.00 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2026-02-25 11:45:00 | 35865.00 | 2026-02-25 15:15:00 | 36675.00 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2026-03-17 13:45:00 | 33230.00 | 2026-03-23 09:15:00 | 31568.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 09:45:00 | 33250.00 | 2026-03-23 09:15:00 | 31587.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 13:45:00 | 33230.00 | 2026-03-24 12:15:00 | 31610.00 | STOP_HIT | 0.50 | 4.88% |
| SELL | retest2 | 2026-03-18 09:45:00 | 33250.00 | 2026-03-24 12:15:00 | 31610.00 | STOP_HIT | 0.50 | 4.93% |
| SELL | retest2 | 2026-04-01 12:30:00 | 30150.00 | 2026-04-08 10:15:00 | 31055.00 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2026-04-01 13:00:00 | 29955.00 | 2026-04-08 10:15:00 | 31055.00 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2026-04-08 09:30:00 | 30260.00 | 2026-04-08 10:15:00 | 31055.00 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2026-04-15 09:15:00 | 31630.00 | 2026-04-17 13:15:00 | 31280.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-04-17 09:15:00 | 31435.00 | 2026-04-17 13:15:00 | 31280.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2026-04-17 10:00:00 | 31490.00 | 2026-04-17 13:15:00 | 31280.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2026-04-28 13:30:00 | 33350.00 | 2026-04-29 11:15:00 | 32810.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2026-04-28 15:00:00 | 33420.00 | 2026-04-29 11:15:00 | 32810.00 | STOP_HIT | 1.00 | -1.83% |
