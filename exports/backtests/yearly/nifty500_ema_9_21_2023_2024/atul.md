# Atul Ltd. (ATUL)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 7090.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 234 |
| ALERT1 | 150 |
| ALERT2 | 146 |
| ALERT2_SKIP | 69 |
| ALERT3 | 414 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 200 |
| PARTIAL | 23 |
| TARGET_HIT | 15 |
| STOP_HIT | 191 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 229 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 100 / 129
- **Target hits / Stop hits / Partials:** 15 / 191 / 23
- **Avg / median % per leg:** 0.95% / -0.41%
- **Sum % (uncompounded):** 218.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 104 | 40 | 38.5% | 9 | 95 | 0 | 0.66% | 68.8% |
| BUY @ 2nd Alert (retest1) | 7 | 1 | 14.3% | 0 | 7 | 0 | -0.86% | -6.0% |
| BUY @ 3rd Alert (retest2) | 97 | 39 | 40.2% | 9 | 88 | 0 | 0.77% | 74.7% |
| SELL (all) | 125 | 60 | 48.0% | 6 | 96 | 23 | 1.20% | 149.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 125 | 60 | 48.0% | 6 | 96 | 23 | 1.20% | 149.5% |
| retest1 (combined) | 7 | 1 | 14.3% | 0 | 7 | 0 | -0.86% | -6.0% |
| retest2 (combined) | 222 | 99 | 44.6% | 15 | 184 | 23 | 1.01% | 224.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 11:15:00 | 6852.00 | 6884.70 | 6886.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 13:15:00 | 6819.85 | 6864.14 | 6876.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 14:15:00 | 6614.60 | 6607.15 | 6646.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-22 14:45:00 | 6603.00 | 6607.15 | 6646.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 15:15:00 | 6650.00 | 6615.72 | 6646.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-24 15:15:00 | 6605.00 | 6637.87 | 6646.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-25 09:30:00 | 6606.15 | 6634.11 | 6643.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-25 10:15:00 | 6668.00 | 6640.89 | 6645.57 | SL hit (close>static) qty=1.00 sl=6650.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 09:15:00 | 6721.00 | 6656.40 | 6649.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 10:15:00 | 6767.40 | 6678.60 | 6660.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 15:15:00 | 6853.80 | 6862.46 | 6823.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-31 09:15:00 | 6822.00 | 6862.46 | 6823.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 6810.65 | 6852.09 | 6821.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 09:45:00 | 6785.70 | 6852.09 | 6821.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 10:15:00 | 6795.10 | 6840.70 | 6819.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 10:45:00 | 6812.55 | 6840.70 | 6819.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 11:15:00 | 6766.40 | 6825.84 | 6814.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 11:45:00 | 6757.00 | 6825.84 | 6814.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2023-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 13:15:00 | 6760.65 | 6802.66 | 6805.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 14:15:00 | 6715.10 | 6785.15 | 6797.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-01 12:15:00 | 6787.05 | 6763.22 | 6779.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 12:15:00 | 6787.05 | 6763.22 | 6779.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 12:15:00 | 6787.05 | 6763.22 | 6779.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 12:45:00 | 6786.10 | 6763.22 | 6779.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 13:15:00 | 6760.00 | 6762.58 | 6777.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 13:30:00 | 6770.05 | 6762.58 | 6777.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 6802.60 | 6767.02 | 6775.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-02 09:45:00 | 6825.00 | 6767.02 | 6775.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 10:15:00 | 6765.00 | 6766.62 | 6774.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-02 10:30:00 | 6800.50 | 6766.62 | 6774.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 11:15:00 | 6765.30 | 6766.36 | 6773.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-02 12:00:00 | 6765.30 | 6766.36 | 6773.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 09:15:00 | 6804.10 | 6768.99 | 6771.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-05 09:45:00 | 6799.95 | 6768.99 | 6771.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 10:15:00 | 6784.75 | 6772.14 | 6772.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-05 13:00:00 | 6770.00 | 6771.69 | 6772.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-05 13:45:00 | 6768.00 | 6770.35 | 6771.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-05 15:15:00 | 6782.25 | 6773.79 | 6772.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2023-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 15:15:00 | 6782.25 | 6773.79 | 6772.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-06 09:15:00 | 6819.90 | 6783.02 | 6777.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-06 10:15:00 | 6749.90 | 6776.39 | 6774.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 10:15:00 | 6749.90 | 6776.39 | 6774.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 10:15:00 | 6749.90 | 6776.39 | 6774.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 11:00:00 | 6749.90 | 6776.39 | 6774.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2023-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 11:15:00 | 6744.75 | 6770.06 | 6771.97 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 09:15:00 | 6810.55 | 6772.15 | 6771.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 10:15:00 | 6890.00 | 6795.72 | 6782.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 10:15:00 | 6884.85 | 6890.88 | 6848.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-08 10:45:00 | 6879.80 | 6890.88 | 6848.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 6857.00 | 6884.11 | 6849.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:00:00 | 6857.00 | 6884.11 | 6849.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 6824.40 | 6872.17 | 6847.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 13:00:00 | 6824.40 | 6872.17 | 6847.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 13:15:00 | 6821.15 | 6861.96 | 6844.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 14:00:00 | 6821.15 | 6861.96 | 6844.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 14:15:00 | 6796.10 | 6848.79 | 6840.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 15:00:00 | 6796.10 | 6848.79 | 6840.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2023-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 10:15:00 | 6778.80 | 6829.30 | 6833.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 11:15:00 | 6728.90 | 6809.22 | 6823.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 09:15:00 | 6912.00 | 6736.46 | 6750.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 09:15:00 | 6912.00 | 6736.46 | 6750.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 6912.00 | 6736.46 | 6750.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 09:30:00 | 6929.45 | 6736.46 | 6750.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 10:15:00 | 6826.00 | 6754.37 | 6757.36 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 11:15:00 | 6892.85 | 6782.06 | 6769.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 12:15:00 | 6913.00 | 6808.25 | 6782.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 10:15:00 | 6950.00 | 6982.41 | 6927.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-15 10:30:00 | 6950.00 | 6982.41 | 6927.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 14:15:00 | 6993.45 | 6978.08 | 6942.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 14:30:00 | 6943.35 | 6978.08 | 6942.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 10:15:00 | 6989.50 | 7034.53 | 7006.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 10:45:00 | 6985.20 | 7034.53 | 7006.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 11:15:00 | 7025.95 | 7032.82 | 7008.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 15:15:00 | 7026.05 | 7016.08 | 7005.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 09:30:00 | 7031.75 | 7020.24 | 7009.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 10:00:00 | 7028.90 | 7020.24 | 7009.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 10:45:00 | 7028.80 | 7022.99 | 7011.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 13:15:00 | 7046.70 | 7083.30 | 7061.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-21 14:00:00 | 7046.70 | 7083.30 | 7061.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 14:15:00 | 7060.00 | 7078.64 | 7061.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-21 15:00:00 | 7060.00 | 7078.64 | 7061.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 15:15:00 | 7063.50 | 7075.61 | 7061.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 09:15:00 | 7050.50 | 7075.61 | 7061.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 7040.20 | 7068.53 | 7059.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 10:00:00 | 7040.20 | 7068.53 | 7059.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 7039.90 | 7062.81 | 7057.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 11:00:00 | 7039.90 | 7062.81 | 7057.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 7035.00 | 7057.24 | 7055.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 11:30:00 | 7028.60 | 7057.24 | 7055.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 13:15:00 | 7053.50 | 7057.10 | 7056.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-22 15:15:00 | 7085.00 | 7060.30 | 7057.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-23 09:15:00 | 6959.95 | 7044.18 | 7050.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2023-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 09:15:00 | 6959.95 | 7044.18 | 7050.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-28 15:15:00 | 6925.00 | 6961.85 | 6981.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-30 12:15:00 | 6965.00 | 6955.19 | 6971.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 12:15:00 | 6965.00 | 6955.19 | 6971.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 12:15:00 | 6965.00 | 6955.19 | 6971.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-30 12:45:00 | 6970.00 | 6955.19 | 6971.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 13:15:00 | 6965.00 | 6957.15 | 6970.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 10:00:00 | 6958.20 | 6968.65 | 6973.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 10:45:00 | 6961.05 | 6967.29 | 6972.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-11 09:15:00 | 6610.29 | 6701.33 | 6754.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-11 09:15:00 | 6613.00 | 6701.33 | 6754.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-07-12 09:15:00 | 6594.10 | 6585.79 | 6658.34 | SL hit (close>ema200) qty=0.50 sl=6585.79 alert=retest2 |

### Cycle 10 — BUY (started 2023-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 11:15:00 | 6627.75 | 6595.23 | 6592.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 09:15:00 | 6637.25 | 6609.39 | 6602.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-19 10:15:00 | 6598.00 | 6607.11 | 6602.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-19 10:15:00 | 6598.00 | 6607.11 | 6602.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 10:15:00 | 6598.00 | 6607.11 | 6602.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-19 10:45:00 | 6596.05 | 6607.11 | 6602.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 11:15:00 | 6572.40 | 6600.17 | 6599.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-19 11:45:00 | 6568.45 | 6600.17 | 6599.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2023-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-19 12:15:00 | 6589.95 | 6598.13 | 6598.78 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 13:15:00 | 6603.65 | 6599.23 | 6599.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 15:15:00 | 6611.85 | 6601.88 | 6600.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 10:15:00 | 6595.20 | 6600.56 | 6600.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 10:15:00 | 6595.20 | 6600.56 | 6600.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 10:15:00 | 6595.20 | 6600.56 | 6600.09 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-07-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 11:15:00 | 6584.15 | 6597.28 | 6598.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 09:15:00 | 6573.90 | 6586.18 | 6592.23 | Break + close below crossover candle low |

### Cycle 14 — BUY (started 2023-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 13:15:00 | 6892.15 | 6620.79 | 6602.68 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 11:15:00 | 6609.80 | 6670.18 | 6676.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-25 12:15:00 | 6584.10 | 6652.97 | 6667.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-27 10:15:00 | 6584.00 | 6572.63 | 6600.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-27 10:30:00 | 6584.60 | 6572.63 | 6600.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 11:15:00 | 6598.70 | 6577.85 | 6600.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 11:30:00 | 6642.30 | 6577.85 | 6600.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 12:15:00 | 6576.80 | 6577.64 | 6598.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 13:15:00 | 6560.00 | 6577.64 | 6598.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-28 11:45:00 | 6565.05 | 6530.77 | 6561.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-28 12:15:00 | 6631.50 | 6550.92 | 6567.75 | SL hit (close>static) qty=1.00 sl=6599.00 alert=retest2 |

### Cycle 16 — BUY (started 2023-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 15:15:00 | 6625.60 | 6584.58 | 6580.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 09:15:00 | 6683.85 | 6604.43 | 6589.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 11:15:00 | 7089.70 | 7091.83 | 6942.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-02 12:00:00 | 7089.70 | 7091.83 | 6942.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 13:15:00 | 7020.35 | 7064.09 | 7013.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-03 13:45:00 | 7021.95 | 7064.09 | 7013.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 14:15:00 | 7009.60 | 7053.19 | 7013.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-03 14:45:00 | 7017.95 | 7053.19 | 7013.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 15:15:00 | 7037.00 | 7049.95 | 7015.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-04 09:15:00 | 7045.00 | 7049.95 | 7015.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-04 10:15:00 | 7051.00 | 7045.95 | 7016.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-04 14:00:00 | 7047.00 | 7054.21 | 7030.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-07 09:30:00 | 7067.65 | 7057.00 | 7038.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 10:15:00 | 7039.25 | 7053.45 | 7038.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 10:45:00 | 7038.55 | 7053.45 | 7038.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 11:15:00 | 7060.25 | 7054.81 | 7040.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-07 12:45:00 | 7066.60 | 7056.45 | 7042.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-08 09:15:00 | 6993.85 | 7043.04 | 7040.73 | SL hit (close<static) qty=1.00 sl=7036.10 alert=retest2 |

### Cycle 17 — SELL (started 2023-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 10:15:00 | 6955.10 | 7025.46 | 7032.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-08 11:15:00 | 6934.05 | 7007.17 | 7023.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-11 09:15:00 | 6872.35 | 6863.35 | 6899.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 09:15:00 | 6872.35 | 6863.35 | 6899.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 6872.35 | 6863.35 | 6899.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 09:30:00 | 6887.00 | 6863.35 | 6899.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 6754.40 | 6764.20 | 6801.24 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-08-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 10:15:00 | 6820.65 | 6806.65 | 6806.65 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-08-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 11:15:00 | 6785.90 | 6802.50 | 6804.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-17 13:15:00 | 6775.45 | 6793.49 | 6800.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-18 13:15:00 | 6780.00 | 6770.14 | 6781.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 13:15:00 | 6780.00 | 6770.14 | 6781.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 13:15:00 | 6780.00 | 6770.14 | 6781.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 14:00:00 | 6780.00 | 6770.14 | 6781.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 14:15:00 | 6757.10 | 6767.53 | 6779.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 15:00:00 | 6757.10 | 6767.53 | 6779.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 6728.80 | 6759.70 | 6774.04 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 10:15:00 | 6787.60 | 6774.95 | 6774.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 12:15:00 | 6796.95 | 6781.11 | 6777.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 13:15:00 | 6770.00 | 6778.89 | 6776.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 13:15:00 | 6770.00 | 6778.89 | 6776.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 13:15:00 | 6770.00 | 6778.89 | 6776.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-22 14:00:00 | 6770.00 | 6778.89 | 6776.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 14:15:00 | 6816.80 | 6786.47 | 6780.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-22 15:15:00 | 6835.00 | 6786.47 | 6780.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-23 10:00:00 | 6863.25 | 6809.59 | 6792.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-23 13:45:00 | 6836.65 | 6832.12 | 6810.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 09:15:00 | 6836.10 | 6818.52 | 6807.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 10:15:00 | 6828.30 | 6821.70 | 6811.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 11:00:00 | 6828.30 | 6821.70 | 6811.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 11:15:00 | 6857.90 | 6828.94 | 6815.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 12:45:00 | 6880.00 | 6843.40 | 6823.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-25 11:15:00 | 6872.25 | 6868.50 | 6846.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-08 09:15:00 | 7518.50 | 7449.66 | 7426.34 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 7356.80 | 7458.24 | 7468.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 7324.85 | 7431.56 | 7455.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 10:15:00 | 7386.00 | 7375.90 | 7413.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 11:00:00 | 7386.00 | 7375.90 | 7413.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 11:15:00 | 7410.00 | 7382.72 | 7413.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 12:00:00 | 7410.00 | 7382.72 | 7413.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 12:15:00 | 7425.00 | 7391.18 | 7414.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 12:45:00 | 7431.85 | 7391.18 | 7414.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 13:15:00 | 7408.40 | 7394.62 | 7413.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 14:15:00 | 7409.05 | 7394.62 | 7413.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 14:15:00 | 7397.30 | 7395.16 | 7412.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 11:00:00 | 7381.85 | 7404.39 | 7412.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 15:15:00 | 7391.10 | 7377.72 | 7394.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 09:30:00 | 7381.05 | 7388.32 | 7396.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 10:15:00 | 7376.15 | 7388.32 | 7396.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 10:15:00 | 7382.00 | 7387.05 | 7395.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 11:15:00 | 7354.30 | 7387.05 | 7395.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 11:45:00 | 7352.10 | 7382.21 | 7392.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 12:15:00 | 7344.00 | 7382.21 | 7392.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-25 09:15:00 | 7021.55 | 7094.50 | 7134.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-09-25 12:15:00 | 7087.55 | 7086.19 | 7119.71 | SL hit (close>ema200) qty=0.50 sl=7086.19 alert=retest2 |

### Cycle 22 — BUY (started 2023-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 13:15:00 | 7055.65 | 7020.72 | 7018.60 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 10:15:00 | 7006.70 | 7019.26 | 7019.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-03 13:15:00 | 6969.50 | 7002.94 | 7011.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-04 14:15:00 | 6957.00 | 6926.45 | 6960.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-04 15:00:00 | 6957.00 | 6926.45 | 6960.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 15:15:00 | 6994.00 | 6939.96 | 6963.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 09:15:00 | 6996.00 | 6939.96 | 6963.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 6969.10 | 6945.79 | 6963.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 09:30:00 | 7007.00 | 6945.79 | 6963.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 10:15:00 | 6981.65 | 6952.96 | 6965.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 11:00:00 | 6981.65 | 6952.96 | 6965.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 11:15:00 | 6976.45 | 6957.66 | 6966.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 12:00:00 | 6976.45 | 6957.66 | 6966.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2023-10-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 14:15:00 | 7031.40 | 6977.91 | 6974.03 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 12:15:00 | 6936.55 | 6978.16 | 6980.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 14:15:00 | 6904.80 | 6955.31 | 6968.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-11 09:15:00 | 6981.20 | 6917.51 | 6931.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 09:15:00 | 6981.20 | 6917.51 | 6931.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 6981.20 | 6917.51 | 6931.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 09:30:00 | 6990.10 | 6917.51 | 6931.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 10:15:00 | 6945.50 | 6923.11 | 6932.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 11:15:00 | 6930.00 | 6923.11 | 6932.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 12:00:00 | 6935.15 | 6925.52 | 6932.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-12 14:30:00 | 6935.75 | 6919.72 | 6922.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-12 15:00:00 | 6942.65 | 6919.72 | 6922.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-12 15:15:00 | 6940.00 | 6923.77 | 6923.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2023-10-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 15:15:00 | 6940.00 | 6923.77 | 6923.64 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 09:15:00 | 6908.15 | 6920.65 | 6922.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 10:15:00 | 6892.00 | 6914.92 | 6919.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-13 11:15:00 | 6943.45 | 6920.63 | 6921.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 11:15:00 | 6943.45 | 6920.63 | 6921.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 11:15:00 | 6943.45 | 6920.63 | 6921.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-13 12:00:00 | 6943.45 | 6920.63 | 6921.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2023-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-13 12:15:00 | 6946.00 | 6925.70 | 6923.88 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 15:15:00 | 6894.80 | 6923.90 | 6923.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-16 14:15:00 | 6882.00 | 6904.46 | 6913.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 10:15:00 | 6950.15 | 6906.52 | 6911.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 10:15:00 | 6950.15 | 6906.52 | 6911.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 10:15:00 | 6950.15 | 6906.52 | 6911.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 10:45:00 | 6955.00 | 6906.52 | 6911.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 11:15:00 | 6920.10 | 6909.24 | 6912.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 13:30:00 | 6902.25 | 6906.54 | 6910.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 14:00:00 | 6890.90 | 6906.54 | 6910.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 09:45:00 | 6897.50 | 6900.62 | 6906.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 09:15:00 | 6557.14 | 6688.06 | 6753.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 09:15:00 | 6552.62 | 6688.06 | 6753.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 14:15:00 | 6546.35 | 6570.64 | 6667.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2023-10-27 09:15:00 | 6212.03 | 6256.89 | 6355.37 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 30 — BUY (started 2023-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 09:15:00 | 6367.90 | 6218.10 | 6214.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 11:15:00 | 6426.90 | 6283.59 | 6246.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 10:15:00 | 6520.10 | 6612.59 | 6508.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 10:15:00 | 6520.10 | 6612.59 | 6508.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 10:15:00 | 6520.10 | 6612.59 | 6508.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 11:00:00 | 6520.10 | 6612.59 | 6508.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 11:15:00 | 6544.50 | 6598.97 | 6511.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 12:00:00 | 6544.50 | 6598.97 | 6511.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 12:15:00 | 6542.00 | 6587.58 | 6514.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 12:30:00 | 6512.20 | 6587.58 | 6514.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 13:15:00 | 6525.00 | 6575.06 | 6515.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 09:15:00 | 6560.00 | 6557.08 | 6516.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 11:30:00 | 6561.00 | 6554.84 | 6525.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-20 09:15:00 | 6635.85 | 6686.57 | 6687.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2023-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 09:15:00 | 6635.85 | 6686.57 | 6687.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-20 10:15:00 | 6610.10 | 6671.28 | 6680.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 09:15:00 | 6616.60 | 6606.07 | 6637.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-21 09:45:00 | 6614.50 | 6606.07 | 6637.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 10:15:00 | 6621.30 | 6609.11 | 6636.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 11:00:00 | 6621.30 | 6609.11 | 6636.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 6577.00 | 6533.65 | 6562.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 10:00:00 | 6577.00 | 6533.65 | 6562.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 10:15:00 | 6576.80 | 6542.28 | 6563.88 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-11-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 14:15:00 | 6619.95 | 6581.38 | 6577.63 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 14:15:00 | 6550.00 | 6577.22 | 6578.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 15:15:00 | 6535.70 | 6568.92 | 6574.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 09:15:00 | 6556.15 | 6545.67 | 6557.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 09:15:00 | 6556.15 | 6545.67 | 6557.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 6556.15 | 6545.67 | 6557.18 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 12:15:00 | 6615.10 | 6567.52 | 6564.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 13:15:00 | 6633.20 | 6580.66 | 6571.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 09:15:00 | 6587.85 | 6598.10 | 6582.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 09:15:00 | 6587.85 | 6598.10 | 6582.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 6587.85 | 6598.10 | 6582.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 10:00:00 | 6587.85 | 6598.10 | 6582.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 10:15:00 | 6583.35 | 6595.15 | 6582.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 11:15:00 | 6593.60 | 6595.15 | 6582.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 11:15:00 | 6610.40 | 6598.20 | 6585.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-01 09:15:00 | 6664.75 | 6593.74 | 6587.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-01 11:00:00 | 6619.00 | 6603.39 | 6592.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-08 13:15:00 | 6741.90 | 6814.33 | 6816.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 13:15:00 | 6741.90 | 6814.33 | 6816.66 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 10:15:00 | 6859.90 | 6814.98 | 6809.87 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 10:15:00 | 6790.00 | 6812.87 | 6814.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 11:15:00 | 6779.80 | 6806.25 | 6811.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 12:15:00 | 6819.40 | 6808.88 | 6812.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 12:15:00 | 6819.40 | 6808.88 | 6812.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 12:15:00 | 6819.40 | 6808.88 | 6812.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 12:45:00 | 6823.35 | 6808.88 | 6812.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 13:15:00 | 6829.75 | 6813.06 | 6813.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 13:45:00 | 6820.15 | 6813.06 | 6813.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2023-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 14:15:00 | 6840.55 | 6818.56 | 6816.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 09:15:00 | 6879.60 | 6834.84 | 6824.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 15:15:00 | 7022.20 | 7024.09 | 6985.65 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-19 09:15:00 | 7074.00 | 7024.09 | 6985.65 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-19 10:45:00 | 7064.50 | 7039.19 | 6999.24 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 11:15:00 | 7044.80 | 7077.06 | 7046.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-12-20 11:15:00 | 7044.80 | 7077.06 | 7046.90 | SL hit (close<ema400) qty=1.00 sl=7046.90 alert=retest1 |

### Cycle 39 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 6872.10 | 7024.22 | 7027.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 6791.60 | 6977.69 | 7005.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 6869.15 | 6863.23 | 6919.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 15:00:00 | 6869.15 | 6863.23 | 6919.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 6914.60 | 6871.91 | 6913.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:00:00 | 6914.60 | 6871.91 | 6913.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 6932.60 | 6884.05 | 6915.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:45:00 | 6912.20 | 6884.05 | 6915.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 11:15:00 | 6929.45 | 6893.13 | 6916.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 11:45:00 | 6935.30 | 6893.13 | 6916.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 12:15:00 | 6940.85 | 6902.67 | 6918.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 13:15:00 | 6956.05 | 6902.67 | 6918.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 13:15:00 | 6938.70 | 6909.88 | 6920.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 14:15:00 | 6945.05 | 6909.88 | 6920.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 14:15:00 | 6991.90 | 6926.28 | 6927.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 15:00:00 | 6991.90 | 6926.28 | 6927.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2023-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 15:15:00 | 6951.05 | 6931.24 | 6929.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 10:15:00 | 7028.15 | 6956.16 | 6941.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 11:15:00 | 7011.15 | 7019.00 | 6990.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 12:00:00 | 7011.15 | 7019.00 | 6990.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 7007.00 | 7016.60 | 6991.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 12:30:00 | 7007.00 | 7016.60 | 6991.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 13:15:00 | 7013.00 | 7015.88 | 6993.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 13:30:00 | 6988.10 | 7015.88 | 6993.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 09:15:00 | 7002.10 | 7012.56 | 6997.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 09:30:00 | 7010.00 | 7012.56 | 6997.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 10:15:00 | 7042.05 | 7018.46 | 7001.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 10:30:00 | 7007.50 | 7018.46 | 7001.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 15:15:00 | 7190.00 | 7157.22 | 7121.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 09:30:00 | 7207.55 | 7155.77 | 7123.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 11:30:00 | 7220.00 | 7169.65 | 7148.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 12:30:00 | 7208.20 | 7172.39 | 7151.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 13:15:00 | 7200.75 | 7172.39 | 7151.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 14:15:00 | 7152.50 | 7172.83 | 7155.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 15:00:00 | 7152.50 | 7172.83 | 7155.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 15:15:00 | 7180.00 | 7174.26 | 7157.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-04 09:45:00 | 7165.00 | 7177.01 | 7160.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 10:15:00 | 7133.15 | 7168.24 | 7157.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-04 10:30:00 | 7136.40 | 7168.24 | 7157.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 11:15:00 | 7119.00 | 7158.39 | 7154.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-04 11:45:00 | 7118.45 | 7158.39 | 7154.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-01-04 12:15:00 | 7116.90 | 7150.09 | 7151.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-01-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 12:15:00 | 7116.90 | 7150.09 | 7151.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-04 14:15:00 | 7072.40 | 7128.14 | 7140.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 09:15:00 | 6885.60 | 6824.27 | 6858.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 09:15:00 | 6885.60 | 6824.27 | 6858.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 6885.60 | 6824.27 | 6858.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 09:30:00 | 6895.00 | 6824.27 | 6858.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 10:15:00 | 6872.80 | 6833.98 | 6860.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 12:15:00 | 6860.00 | 6841.18 | 6860.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 14:00:00 | 6840.60 | 6843.06 | 6858.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 15:00:00 | 6854.05 | 6845.26 | 6858.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-12 10:15:00 | 6899.00 | 6863.48 | 6863.83 | SL hit (close>static) qty=1.00 sl=6890.00 alert=retest2 |

### Cycle 42 — BUY (started 2024-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 11:15:00 | 6886.85 | 6868.15 | 6865.93 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 09:15:00 | 6822.25 | 6858.28 | 6862.16 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2024-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 14:15:00 | 6875.25 | 6861.86 | 6861.56 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 11:15:00 | 6835.50 | 6859.28 | 6860.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 12:15:00 | 6783.05 | 6844.03 | 6853.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 15:15:00 | 6640.00 | 6622.12 | 6677.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-19 09:15:00 | 6715.15 | 6622.12 | 6677.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 6710.95 | 6639.89 | 6680.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 12:15:00 | 6617.65 | 6644.97 | 6676.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 13:15:00 | 6625.85 | 6645.98 | 6673.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 15:00:00 | 6619.65 | 6637.99 | 6665.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 11:15:00 | 6286.77 | 6413.22 | 6512.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 11:15:00 | 6294.56 | 6413.22 | 6512.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 11:15:00 | 6288.67 | 6413.22 | 6512.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-24 09:15:00 | 6319.50 | 6293.71 | 6404.98 | SL hit (close>ema200) qty=0.50 sl=6293.71 alert=retest2 |

### Cycle 46 — BUY (started 2024-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 11:15:00 | 6464.40 | 6310.50 | 6308.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 12:15:00 | 6478.00 | 6344.00 | 6323.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 09:15:00 | 6369.90 | 6387.32 | 6355.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-01 10:00:00 | 6369.90 | 6387.32 | 6355.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 10:15:00 | 6330.00 | 6375.85 | 6352.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 10:45:00 | 6344.80 | 6375.85 | 6352.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 6307.40 | 6362.16 | 6348.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:00:00 | 6307.40 | 6362.16 | 6348.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2024-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 14:15:00 | 6294.00 | 6331.89 | 6336.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 15:15:00 | 6289.00 | 6323.31 | 6332.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-02 09:15:00 | 6336.80 | 6326.01 | 6332.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 09:15:00 | 6336.80 | 6326.01 | 6332.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 6336.80 | 6326.01 | 6332.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 10:00:00 | 6336.80 | 6326.01 | 6332.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 10:15:00 | 6362.00 | 6333.21 | 6335.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 10:30:00 | 6360.30 | 6333.21 | 6335.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2024-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 11:15:00 | 6359.60 | 6338.49 | 6337.73 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 14:15:00 | 6309.80 | 6347.02 | 6347.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 15:15:00 | 6265.05 | 6330.62 | 6340.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 09:15:00 | 6379.95 | 6340.49 | 6343.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 09:15:00 | 6379.95 | 6340.49 | 6343.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 6379.95 | 6340.49 | 6343.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 09:30:00 | 6377.00 | 6340.49 | 6343.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2024-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 10:15:00 | 6369.20 | 6346.23 | 6345.99 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-02-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 09:15:00 | 6293.75 | 6341.10 | 6347.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 10:15:00 | 6256.20 | 6324.12 | 6338.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 12:15:00 | 6279.00 | 6251.05 | 6282.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-09 13:00:00 | 6279.00 | 6251.05 | 6282.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 13:15:00 | 6313.90 | 6263.62 | 6285.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-09 14:00:00 | 6313.90 | 6263.62 | 6285.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 14:15:00 | 6393.00 | 6289.50 | 6295.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-09 15:00:00 | 6393.00 | 6289.50 | 6295.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2024-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 15:15:00 | 6372.00 | 6306.00 | 6302.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-13 14:15:00 | 6400.00 | 6376.43 | 6355.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-14 10:15:00 | 6378.05 | 6380.02 | 6362.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 10:15:00 | 6378.05 | 6380.02 | 6362.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 6378.05 | 6380.02 | 6362.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-14 10:30:00 | 6366.55 | 6380.02 | 6362.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 12:15:00 | 6349.25 | 6372.11 | 6361.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-14 12:30:00 | 6347.75 | 6372.11 | 6361.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 13:15:00 | 6361.30 | 6369.94 | 6361.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-14 13:30:00 | 6350.70 | 6369.94 | 6361.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 6382.00 | 6372.36 | 6363.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-14 14:30:00 | 6367.00 | 6372.36 | 6363.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 10:15:00 | 6380.30 | 6378.76 | 6369.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 10:30:00 | 6392.35 | 6378.76 | 6369.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 11:15:00 | 6364.95 | 6376.00 | 6368.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 12:00:00 | 6364.95 | 6376.00 | 6368.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 12:15:00 | 6376.95 | 6376.19 | 6369.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 12:30:00 | 6371.00 | 6376.19 | 6369.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 13:15:00 | 6356.35 | 6372.22 | 6368.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 14:00:00 | 6356.35 | 6372.22 | 6368.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 14:15:00 | 6360.00 | 6369.78 | 6367.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 15:00:00 | 6360.00 | 6369.78 | 6367.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 15:15:00 | 6373.00 | 6370.42 | 6368.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-16 09:15:00 | 6402.00 | 6370.42 | 6368.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-16 14:00:00 | 6378.70 | 6378.06 | 6373.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-19 09:15:00 | 6357.45 | 6369.72 | 6370.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2024-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 09:15:00 | 6357.45 | 6369.72 | 6370.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 10:15:00 | 6285.30 | 6352.84 | 6362.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-19 15:15:00 | 6339.00 | 6337.52 | 6350.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-20 09:15:00 | 6389.00 | 6337.52 | 6350.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 6310.90 | 6332.20 | 6346.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-20 10:30:00 | 6300.00 | 6323.17 | 6341.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-20 11:00:00 | 6287.05 | 6323.17 | 6341.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 10:15:00 | 6379.00 | 6343.96 | 6342.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2024-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-21 10:15:00 | 6379.00 | 6343.96 | 6342.76 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 14:15:00 | 6311.25 | 6340.37 | 6342.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 09:15:00 | 6256.10 | 6317.54 | 6331.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 13:15:00 | 6312.25 | 6305.09 | 6319.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-22 13:45:00 | 6316.75 | 6305.09 | 6319.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 6322.00 | 6308.48 | 6319.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 15:00:00 | 6322.00 | 6308.48 | 6319.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 6339.00 | 6314.58 | 6321.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:15:00 | 6331.90 | 6314.58 | 6321.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 6330.55 | 6317.77 | 6322.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 10:30:00 | 6305.60 | 6317.50 | 6321.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 14:45:00 | 6304.75 | 6313.75 | 6318.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 11:15:00 | 5990.32 | 6053.90 | 6087.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 11:15:00 | 5989.51 | 6053.90 | 6087.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-06 15:15:00 | 6049.00 | 6039.00 | 6068.36 | SL hit (close>ema200) qty=0.50 sl=6039.00 alert=retest2 |

### Cycle 56 — BUY (started 2024-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 11:15:00 | 6155.35 | 6091.53 | 6087.46 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 10:15:00 | 6035.50 | 6086.49 | 6090.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 14:15:00 | 6015.95 | 6058.83 | 6074.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-12 13:15:00 | 6050.20 | 6032.33 | 6051.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 13:15:00 | 6050.20 | 6032.33 | 6051.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 13:15:00 | 6050.20 | 6032.33 | 6051.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-12 14:00:00 | 6050.20 | 6032.33 | 6051.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 14:15:00 | 6046.00 | 6035.06 | 6050.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 10:00:00 | 6044.15 | 6038.63 | 6049.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-14 11:00:00 | 6035.40 | 6013.92 | 6026.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 15:00:00 | 5988.85 | 5999.87 | 6006.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-28 11:15:00 | 5741.94 | 5787.58 | 5820.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-28 12:15:00 | 5733.63 | 5779.01 | 5813.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-01 09:15:00 | 5806.35 | 5772.34 | 5798.20 | SL hit (close>ema200) qty=0.50 sl=5772.34 alert=retest2 |

### Cycle 58 — BUY (started 2024-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-02 09:15:00 | 5905.85 | 5807.35 | 5803.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 10:15:00 | 5950.00 | 5835.88 | 5817.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 09:15:00 | 5981.50 | 5984.45 | 5952.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 09:15:00 | 5981.50 | 5984.45 | 5952.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 5981.50 | 5984.45 | 5952.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 09:45:00 | 5969.90 | 5984.45 | 5952.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 11:15:00 | 5957.00 | 5982.03 | 5956.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 12:00:00 | 5957.00 | 5982.03 | 5956.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 12:15:00 | 5955.00 | 5976.63 | 5956.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-05 14:15:00 | 5971.45 | 5974.56 | 5957.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-05 15:00:00 | 5975.95 | 5974.84 | 5959.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 09:45:00 | 5979.95 | 5976.72 | 5962.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-08 13:15:00 | 5919.90 | 5960.54 | 5959.62 | SL hit (close<static) qty=1.00 sl=5947.30 alert=retest2 |

### Cycle 59 — SELL (started 2024-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 14:15:00 | 5929.00 | 5954.23 | 5956.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 13:15:00 | 5877.65 | 5929.47 | 5943.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 09:15:00 | 5959.80 | 5921.94 | 5935.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 09:15:00 | 5959.80 | 5921.94 | 5935.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 5959.80 | 5921.94 | 5935.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 10:00:00 | 5959.80 | 5921.94 | 5935.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 10:15:00 | 5997.50 | 5937.05 | 5940.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 10:30:00 | 6023.15 | 5937.05 | 5940.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2024-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 11:15:00 | 6158.90 | 5981.42 | 5960.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 12:15:00 | 6208.15 | 6026.77 | 5983.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 14:15:00 | 6158.75 | 6168.74 | 6104.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-12 15:00:00 | 6158.75 | 6168.74 | 6104.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 6004.45 | 6133.77 | 6099.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 09:30:00 | 5980.00 | 6133.77 | 6099.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2024-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 12:15:00 | 5984.05 | 6063.40 | 6072.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 14:15:00 | 5895.00 | 6013.98 | 6047.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 13:15:00 | 5997.45 | 5967.53 | 6003.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 14:00:00 | 5997.45 | 5967.53 | 6003.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 14:15:00 | 5986.60 | 5971.34 | 6001.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:30:00 | 5949.45 | 5975.47 | 5991.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 14:30:00 | 5920.40 | 5954.64 | 5980.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-23 10:15:00 | 5938.55 | 5914.23 | 5920.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-23 10:15:00 | 5974.00 | 5926.19 | 5925.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2024-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 10:15:00 | 5974.00 | 5926.19 | 5925.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 11:15:00 | 5981.45 | 5937.24 | 5930.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 14:15:00 | 5940.55 | 5950.35 | 5939.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 14:15:00 | 5940.55 | 5950.35 | 5939.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 14:15:00 | 5940.55 | 5950.35 | 5939.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 15:00:00 | 5940.55 | 5950.35 | 5939.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 15:15:00 | 5924.55 | 5945.19 | 5938.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 09:15:00 | 5960.55 | 5945.19 | 5938.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-24 14:15:00 | 5900.00 | 5932.76 | 5935.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 14:15:00 | 5900.00 | 5932.76 | 5935.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-25 09:15:00 | 5871.85 | 5915.34 | 5926.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 13:15:00 | 5889.80 | 5889.10 | 5908.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-25 14:00:00 | 5889.80 | 5889.10 | 5908.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 5916.85 | 5891.26 | 5904.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-26 09:30:00 | 5940.00 | 5891.26 | 5904.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 10:15:00 | 5942.70 | 5901.55 | 5907.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-26 10:30:00 | 5950.25 | 5901.55 | 5907.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2024-04-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 12:15:00 | 5969.00 | 5922.71 | 5916.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 12:15:00 | 6040.90 | 5975.96 | 5958.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 14:15:00 | 5989.05 | 5990.42 | 5968.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-30 15:00:00 | 5989.05 | 5990.42 | 5968.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 5954.10 | 5983.16 | 5967.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:45:00 | 5944.85 | 5975.53 | 5965.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 10:15:00 | 5966.40 | 5973.70 | 5965.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 13:45:00 | 5984.65 | 5974.43 | 5967.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-09 09:15:00 | 6040.50 | 6112.53 | 6118.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 09:15:00 | 6040.50 | 6112.53 | 6118.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 10:15:00 | 5945.10 | 6079.04 | 6102.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 11:15:00 | 5925.40 | 5916.23 | 5983.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-10 12:00:00 | 5925.40 | 5916.23 | 5983.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 5960.80 | 5931.00 | 5973.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 15:00:00 | 5960.80 | 5931.00 | 5973.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 5978.00 | 5940.40 | 5974.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:15:00 | 5959.15 | 5940.40 | 5974.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 5896.20 | 5931.56 | 5967.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 10:30:00 | 5863.85 | 5924.83 | 5960.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 10:00:00 | 5871.00 | 5919.75 | 5943.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-15 14:15:00 | 5873.00 | 5901.89 | 5917.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-15 15:00:00 | 5875.55 | 5896.62 | 5913.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 5886.35 | 5890.95 | 5907.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 13:15:00 | 5867.55 | 5890.67 | 5903.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 14:00:00 | 5864.30 | 5885.39 | 5900.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-17 12:15:00 | 5964.80 | 5916.23 | 5909.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2024-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 12:15:00 | 5964.80 | 5916.23 | 5909.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 13:15:00 | 5970.55 | 5927.10 | 5915.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-18 12:15:00 | 5965.00 | 5968.82 | 5944.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 09:15:00 | 5950.70 | 5968.82 | 5944.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 5924.90 | 5960.04 | 5942.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:45:00 | 5915.45 | 5960.04 | 5942.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 5930.00 | 5954.03 | 5941.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:30:00 | 5910.70 | 5954.03 | 5941.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 5960.00 | 5963.16 | 5953.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:00:00 | 5960.00 | 5963.16 | 5953.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 5959.95 | 5962.52 | 5954.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:45:00 | 5941.20 | 5962.52 | 5954.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 5963.00 | 5962.62 | 5955.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:45:00 | 5941.00 | 5956.29 | 5952.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 10:15:00 | 5925.25 | 5950.08 | 5950.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 14:15:00 | 5881.65 | 5914.57 | 5928.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 09:15:00 | 5910.10 | 5907.99 | 5922.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 09:15:00 | 5910.10 | 5907.99 | 5922.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 5910.10 | 5907.99 | 5922.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 5910.10 | 5907.99 | 5922.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 5905.45 | 5907.48 | 5921.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 11:45:00 | 5898.55 | 5905.97 | 5919.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 14:30:00 | 5900.90 | 5902.75 | 5914.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 09:45:00 | 5901.40 | 5896.52 | 5909.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 5603.62 | 5718.16 | 5727.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 5605.85 | 5718.16 | 5727.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 5606.33 | 5718.16 | 5727.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 12:15:00 | 5308.70 | 5608.32 | 5669.62 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 68 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 5792.80 | 5675.43 | 5667.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 14:15:00 | 5816.95 | 5721.13 | 5690.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 09:15:00 | 6224.05 | 6224.29 | 6183.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-18 10:00:00 | 6224.05 | 6224.29 | 6183.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 6200.00 | 6243.02 | 6217.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:00:00 | 6200.00 | 6243.02 | 6217.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 6246.75 | 6243.76 | 6219.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 11:30:00 | 6285.30 | 6256.39 | 6227.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 09:15:00 | 6263.45 | 6251.37 | 6235.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 12:15:00 | 6348.05 | 6406.83 | 6413.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 12:15:00 | 6348.05 | 6406.83 | 6413.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 6304.80 | 6362.41 | 6381.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 6440.15 | 6370.75 | 6380.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 09:15:00 | 6440.15 | 6370.75 | 6380.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 6440.15 | 6370.75 | 6380.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:45:00 | 6428.30 | 6370.75 | 6380.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 10:15:00 | 6524.70 | 6401.54 | 6393.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 6573.60 | 6470.52 | 6433.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 6577.95 | 6590.57 | 6532.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 12:00:00 | 6577.95 | 6590.57 | 6532.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 6558.30 | 6584.11 | 6534.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:45:00 | 6545.95 | 6584.11 | 6534.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 6528.70 | 6578.60 | 6565.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:15:00 | 6612.95 | 6572.72 | 6563.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 13:15:00 | 6993.30 | 7027.84 | 7029.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2024-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 13:15:00 | 6993.30 | 7027.84 | 7029.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 6953.40 | 7012.95 | 7022.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 7032.45 | 6994.75 | 7011.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 7032.45 | 6994.75 | 7011.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 7032.45 | 6994.75 | 7011.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 7032.45 | 6994.75 | 7011.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 7043.55 | 7004.51 | 7014.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 7039.95 | 7004.51 | 7014.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2024-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 11:15:00 | 7185.65 | 7040.74 | 7029.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 12:15:00 | 7399.20 | 7112.43 | 7063.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 09:15:00 | 7177.70 | 7188.59 | 7121.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-23 10:00:00 | 7177.70 | 7188.59 | 7121.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 7226.65 | 7202.03 | 7139.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:15:00 | 7042.30 | 7202.03 | 7139.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 7238.15 | 7209.25 | 7148.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 7205.00 | 7209.25 | 7148.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 7283.95 | 7279.71 | 7237.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 14:15:00 | 7309.90 | 7270.16 | 7244.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 14:45:00 | 7309.95 | 7280.16 | 7251.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 7308.80 | 7278.53 | 7253.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 15:15:00 | 7793.85 | 7828.56 | 7832.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 15:15:00 | 7793.85 | 7828.56 | 7832.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 7684.65 | 7794.45 | 7816.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 7896.25 | 7772.12 | 7788.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 7896.25 | 7772.12 | 7788.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 7896.25 | 7772.12 | 7788.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 7896.25 | 7772.12 | 7788.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 7814.75 | 7780.64 | 7790.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:30:00 | 7796.85 | 7784.87 | 7790.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 10:15:00 | 7871.35 | 7808.63 | 7800.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 10:15:00 | 7871.35 | 7808.63 | 7800.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 12:15:00 | 7947.05 | 7844.71 | 7818.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 8000.00 | 8013.17 | 7946.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 8000.00 | 8013.17 | 7946.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 12:15:00 | 7959.95 | 8006.26 | 7969.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 13:00:00 | 7959.95 | 8006.26 | 7969.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 7965.00 | 7998.01 | 7969.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:00:00 | 7965.00 | 7998.01 | 7969.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 8006.10 | 7998.83 | 7974.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 8052.95 | 7998.83 | 7974.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 8024.50 | 8003.97 | 7978.96 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 12:15:00 | 7907.90 | 7962.99 | 7964.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 14:15:00 | 7855.50 | 7932.69 | 7950.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 11:15:00 | 7710.20 | 7700.04 | 7775.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 12:00:00 | 7710.20 | 7700.04 | 7775.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 7774.55 | 7714.94 | 7775.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 13:00:00 | 7774.55 | 7714.94 | 7775.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 13:15:00 | 7744.55 | 7720.86 | 7772.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 14:15:00 | 7730.50 | 7720.86 | 7772.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 09:30:00 | 7716.10 | 7725.25 | 7761.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 10:00:00 | 7729.95 | 7725.25 | 7761.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 12:15:00 | 7854.60 | 7762.41 | 7770.63 | SL hit (close>static) qty=1.00 sl=7782.40 alert=retest2 |

### Cycle 76 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 7888.55 | 7787.64 | 7781.35 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 10:15:00 | 7772.95 | 7800.70 | 7801.02 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 12:15:00 | 7830.10 | 7805.84 | 7803.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 13:15:00 | 7860.55 | 7816.78 | 7808.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 11:15:00 | 7945.00 | 7971.51 | 7921.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 11:15:00 | 7945.00 | 7971.51 | 7921.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 11:15:00 | 7945.00 | 7971.51 | 7921.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 11:45:00 | 7930.00 | 7971.51 | 7921.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 7937.80 | 7958.12 | 7923.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:45:00 | 7936.35 | 7958.12 | 7923.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 7923.55 | 7951.21 | 7923.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 15:00:00 | 7923.55 | 7951.21 | 7923.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 7902.85 | 7941.54 | 7921.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 7855.00 | 7941.54 | 7921.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 7781.05 | 7909.44 | 7908.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 7768.60 | 7909.44 | 7908.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 10:15:00 | 7762.15 | 7879.98 | 7895.45 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 15:15:00 | 7925.00 | 7875.00 | 7869.82 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 09:15:00 | 7838.20 | 7885.95 | 7886.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 10:15:00 | 7815.25 | 7871.81 | 7879.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 15:15:00 | 7848.00 | 7835.22 | 7854.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 15:15:00 | 7848.00 | 7835.22 | 7854.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 7848.00 | 7835.22 | 7854.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 7890.65 | 7835.22 | 7854.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 7968.70 | 7861.92 | 7865.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:00:00 | 7968.70 | 7861.92 | 7865.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 10:15:00 | 7937.60 | 7877.06 | 7871.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 11:15:00 | 7970.90 | 7895.82 | 7880.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 10:15:00 | 7930.00 | 7938.91 | 7913.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 11:15:00 | 7935.00 | 7938.91 | 7913.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 7883.00 | 7927.73 | 7910.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:30:00 | 7883.00 | 7927.73 | 7910.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 7861.45 | 7914.47 | 7905.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:00:00 | 7861.45 | 7914.47 | 7905.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 15:15:00 | 7895.00 | 7900.33 | 7900.63 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 09:15:00 | 7965.80 | 7913.42 | 7906.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 10:15:00 | 7994.20 | 7929.58 | 7914.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 13:15:00 | 7943.75 | 7945.80 | 7927.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 13:15:00 | 7943.75 | 7945.80 | 7927.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 7943.75 | 7945.80 | 7927.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 14:00:00 | 7943.75 | 7945.80 | 7927.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 7909.00 | 7938.44 | 7925.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 7909.00 | 7938.44 | 7925.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 7935.00 | 7937.75 | 7926.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 7964.25 | 7937.75 | 7926.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 7983.70 | 7946.94 | 7931.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 09:30:00 | 8028.05 | 7944.69 | 7937.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 14:45:00 | 8017.80 | 7949.40 | 7941.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 7817.85 | 7925.68 | 7932.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 7817.85 | 7925.68 | 7932.42 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 11:15:00 | 7997.30 | 7922.12 | 7919.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 8070.00 | 7973.31 | 7946.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 13:15:00 | 7999.95 | 8004.33 | 7972.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-10 14:15:00 | 7992.10 | 8004.33 | 7972.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 7999.20 | 8003.31 | 7975.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 15:00:00 | 7999.20 | 8003.31 | 7975.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 8019.80 | 8006.08 | 7981.42 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 7866.10 | 7957.79 | 7967.72 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 7992.85 | 7966.90 | 7964.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 11:15:00 | 8016.00 | 7979.89 | 7971.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 14:15:00 | 7965.00 | 7980.35 | 7974.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 14:15:00 | 7965.00 | 7980.35 | 7974.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 7965.00 | 7980.35 | 7974.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 15:00:00 | 7965.00 | 7980.35 | 7974.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 7967.95 | 7977.87 | 7973.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:15:00 | 7910.05 | 7977.87 | 7973.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 09:15:00 | 7896.00 | 7961.50 | 7966.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 11:15:00 | 7893.45 | 7938.70 | 7954.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 7742.10 | 7710.33 | 7776.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 7742.10 | 7710.33 | 7776.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 7742.10 | 7710.33 | 7776.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:30:00 | 7844.00 | 7710.33 | 7776.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 7608.00 | 7659.01 | 7712.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:45:00 | 7597.85 | 7653.78 | 7693.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:30:00 | 7599.65 | 7658.01 | 7691.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 11:15:00 | 7598.35 | 7642.27 | 7663.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 11:15:00 | 7587.45 | 7549.75 | 7547.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 11:15:00 | 7587.45 | 7549.75 | 7547.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 13:15:00 | 7621.45 | 7569.70 | 7557.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 15:15:00 | 7650.00 | 7674.81 | 7636.91 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:15:00 | 7709.45 | 7674.81 | 7636.91 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 7847.50 | 7906.21 | 7838.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:00:00 | 7847.50 | 7906.21 | 7838.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 7839.00 | 7892.77 | 7838.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:45:00 | 7820.05 | 7892.77 | 7838.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 7822.95 | 7878.80 | 7836.76 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-03 14:15:00 | 7822.95 | 7878.80 | 7836.76 | SL hit (close<ema400) qty=1.00 sl=7836.76 alert=retest1 |

### Cycle 91 — SELL (started 2024-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 14:15:00 | 7683.45 | 7811.83 | 7824.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 7634.05 | 7755.46 | 7794.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 7593.55 | 7581.54 | 7666.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:45:00 | 7612.00 | 7581.54 | 7666.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 7607.05 | 7596.64 | 7659.13 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 7815.00 | 7710.39 | 7698.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 7963.65 | 7761.04 | 7722.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 10:15:00 | 7907.70 | 7913.13 | 7838.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 11:00:00 | 7907.70 | 7913.13 | 7838.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 7910.55 | 7935.40 | 7904.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 15:00:00 | 7910.55 | 7935.40 | 7904.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 7866.10 | 7921.54 | 7900.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 7867.50 | 7921.54 | 7900.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 7882.15 | 7913.66 | 7899.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:15:00 | 7960.45 | 7914.74 | 7903.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 11:15:00 | 7706.50 | 7884.98 | 7894.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 11:15:00 | 7706.50 | 7884.98 | 7894.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 7585.00 | 7703.73 | 7753.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 7542.95 | 7516.63 | 7597.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 09:45:00 | 7498.80 | 7516.63 | 7597.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 7621.35 | 7537.57 | 7599.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:00:00 | 7621.35 | 7537.57 | 7599.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 7659.25 | 7561.91 | 7604.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:00:00 | 7659.25 | 7561.91 | 7604.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 7646.10 | 7606.51 | 7614.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:00:00 | 7646.10 | 7606.51 | 7614.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 7652.20 | 7615.65 | 7617.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:30:00 | 7720.00 | 7615.65 | 7617.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 12:15:00 | 7626.70 | 7620.03 | 7619.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 15:15:00 | 7680.00 | 7636.18 | 7627.20 | Break + close above crossover candle high |

### Cycle 95 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 7525.00 | 7613.94 | 7617.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 7498.00 | 7590.75 | 7607.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 09:15:00 | 7556.70 | 7489.30 | 7537.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-28 09:15:00 | 7556.70 | 7489.30 | 7537.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 7556.70 | 7489.30 | 7537.59 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 7637.95 | 7567.33 | 7563.31 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 7493.00 | 7554.86 | 7562.41 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 7649.90 | 7562.72 | 7560.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 7679.95 | 7609.10 | 7584.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 7760.75 | 7797.59 | 7734.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 10:00:00 | 7760.75 | 7797.59 | 7734.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 7717.15 | 7781.50 | 7732.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 7717.15 | 7781.50 | 7732.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 7734.90 | 7772.18 | 7732.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 15:15:00 | 7786.15 | 7768.91 | 7740.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 09:15:00 | 7725.90 | 7906.80 | 7926.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 7725.90 | 7906.80 | 7926.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 11:15:00 | 7674.85 | 7843.32 | 7893.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 7366.95 | 7330.19 | 7455.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 7366.95 | 7330.19 | 7455.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 7214.60 | 7243.99 | 7313.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 7361.05 | 7267.40 | 7317.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 7348.95 | 7283.71 | 7320.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 7320.40 | 7283.71 | 7320.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 7340.60 | 7295.09 | 7322.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:45:00 | 7346.20 | 7295.09 | 7322.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 12:15:00 | 7359.45 | 7307.96 | 7325.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 13:00:00 | 7359.45 | 7307.96 | 7325.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 7330.40 | 7312.45 | 7326.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:15:00 | 7324.15 | 7312.45 | 7326.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:45:00 | 7282.85 | 7305.41 | 7321.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 10:15:00 | 7411.30 | 7303.76 | 7292.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 7411.30 | 7303.76 | 7292.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 14:15:00 | 7463.25 | 7379.95 | 7336.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 12:15:00 | 7396.55 | 7414.17 | 7373.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 13:00:00 | 7396.55 | 7414.17 | 7373.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 13:15:00 | 7384.95 | 7408.32 | 7374.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 13:45:00 | 7394.05 | 7408.32 | 7374.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 7377.15 | 7402.09 | 7374.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 14:30:00 | 7390.00 | 7402.09 | 7374.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 7432.00 | 7408.07 | 7379.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:15:00 | 7348.70 | 7408.07 | 7379.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 7395.95 | 7405.65 | 7381.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 7375.00 | 7405.65 | 7381.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 7362.00 | 7396.92 | 7379.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:45:00 | 7373.00 | 7396.92 | 7379.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 7355.60 | 7388.65 | 7377.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:45:00 | 7341.95 | 7388.65 | 7377.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 7395.00 | 7376.93 | 7373.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:30:00 | 7402.35 | 7377.44 | 7374.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 10:15:00 | 7303.70 | 7362.69 | 7368.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 7303.70 | 7362.69 | 7368.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 11:15:00 | 7300.00 | 7350.16 | 7361.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 13:15:00 | 7290.65 | 7280.40 | 7306.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 14:00:00 | 7290.65 | 7280.40 | 7306.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 7291.60 | 7282.64 | 7304.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 15:00:00 | 7291.60 | 7282.64 | 7304.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 7315.05 | 7289.12 | 7305.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 7257.55 | 7289.12 | 7305.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 7308.00 | 7292.90 | 7305.89 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 11:15:00 | 7378.00 | 7314.48 | 7313.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 10:15:00 | 7400.00 | 7355.20 | 7337.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 11:15:00 | 7378.55 | 7388.07 | 7368.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 11:15:00 | 7378.55 | 7388.07 | 7368.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 7378.55 | 7388.07 | 7368.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 7350.40 | 7388.07 | 7368.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 7406.50 | 7391.76 | 7371.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 09:15:00 | 7415.70 | 7393.23 | 7377.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 09:15:00 | 7345.95 | 7383.77 | 7374.50 | SL hit (close<static) qty=1.00 sl=7368.80 alert=retest2 |

### Cycle 103 — SELL (started 2024-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 14:15:00 | 7355.75 | 7374.16 | 7375.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 15:15:00 | 7335.10 | 7366.34 | 7371.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 7398.60 | 7320.73 | 7333.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 7398.60 | 7320.73 | 7333.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 7398.60 | 7320.73 | 7333.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:45:00 | 7416.95 | 7320.73 | 7333.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 10:15:00 | 7431.30 | 7342.84 | 7342.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 11:15:00 | 7457.95 | 7365.86 | 7352.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 14:15:00 | 7485.05 | 7494.24 | 7449.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-11 14:30:00 | 7485.95 | 7494.24 | 7449.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 7307.60 | 7452.44 | 7438.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 7307.60 | 7452.44 | 7438.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 7245.00 | 7410.95 | 7420.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 7065.00 | 7261.94 | 7333.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 15:15:00 | 7230.00 | 7210.89 | 7269.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 09:15:00 | 7261.20 | 7210.89 | 7269.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 7316.85 | 7232.08 | 7273.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 7303.80 | 7232.08 | 7273.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 7305.85 | 7246.83 | 7276.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:00:00 | 7305.85 | 7246.83 | 7276.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 14:15:00 | 7335.85 | 7292.94 | 7291.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 10:15:00 | 7365.90 | 7325.05 | 7307.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 14:15:00 | 7326.35 | 7329.95 | 7316.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 15:00:00 | 7326.35 | 7329.95 | 7316.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 7358.45 | 7335.65 | 7319.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 09:30:00 | 7363.55 | 7340.11 | 7323.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 10:15:00 | 7360.00 | 7340.11 | 7323.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 09:30:00 | 7387.65 | 7371.74 | 7351.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 10:15:00 | 7377.70 | 7371.74 | 7351.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 10:15:00 | 7374.90 | 7372.37 | 7354.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 11:30:00 | 7417.05 | 7378.96 | 7358.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 15:00:00 | 7412.05 | 7387.02 | 7367.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 09:15:00 | 7297.00 | 7372.38 | 7364.56 | SL hit (close<static) qty=1.00 sl=7346.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 11:15:00 | 7277.30 | 7343.54 | 7352.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 12:15:00 | 7182.75 | 7311.38 | 7336.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 11:15:00 | 7073.60 | 7073.02 | 7136.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 12:00:00 | 7073.60 | 7073.02 | 7136.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 7100.00 | 7090.83 | 7126.24 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2025-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 15:15:00 | 6990.00 | 6944.28 | 6942.21 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 12:15:00 | 6903.60 | 6937.53 | 6940.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 13:15:00 | 6890.80 | 6928.18 | 6935.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 09:15:00 | 6917.45 | 6915.41 | 6927.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 6917.45 | 6915.41 | 6927.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 6917.45 | 6915.41 | 6927.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 10:30:00 | 6869.45 | 6902.48 | 6920.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-07 14:15:00 | 6920.00 | 6901.06 | 6898.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 14:15:00 | 6920.00 | 6901.06 | 6898.86 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 6869.00 | 6893.22 | 6895.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 11:15:00 | 6826.05 | 6879.79 | 6889.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 15:15:00 | 6874.00 | 6852.77 | 6870.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 15:15:00 | 6874.00 | 6852.77 | 6870.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 6874.00 | 6852.77 | 6870.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 6962.40 | 6852.77 | 6870.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 6950.10 | 6872.23 | 6877.71 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2025-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 11:15:00 | 6920.00 | 6881.69 | 6881.06 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 6787.20 | 6881.62 | 6884.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 6711.10 | 6806.56 | 6840.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 11:15:00 | 6701.40 | 6684.65 | 6738.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:30:00 | 6691.40 | 6684.65 | 6738.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 6754.90 | 6699.73 | 6736.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:00:00 | 6754.90 | 6699.73 | 6736.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 6737.60 | 6707.31 | 6736.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:15:00 | 6756.00 | 6707.31 | 6736.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 6756.00 | 6717.05 | 6738.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 6711.80 | 6717.05 | 6738.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 6703.20 | 6714.28 | 6735.08 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 6783.75 | 6742.81 | 6741.74 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 10:15:00 | 6708.15 | 6744.90 | 6747.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 12:15:00 | 6680.35 | 6725.95 | 6737.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 10:15:00 | 6699.00 | 6696.58 | 6716.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-20 11:00:00 | 6699.00 | 6696.58 | 6716.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 6707.90 | 6698.84 | 6716.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:45:00 | 6713.00 | 6698.84 | 6716.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 12:15:00 | 6731.85 | 6705.44 | 6717.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:00:00 | 6731.85 | 6705.44 | 6717.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 6713.90 | 6707.13 | 6717.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:45:00 | 6713.45 | 6707.13 | 6717.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 6730.60 | 6711.83 | 6718.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:45:00 | 6749.95 | 6711.83 | 6718.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 6732.50 | 6715.96 | 6719.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:15:00 | 6829.65 | 6715.96 | 6719.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 09:15:00 | 6784.70 | 6729.71 | 6725.63 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 6671.25 | 6726.94 | 6729.53 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2025-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 14:15:00 | 6793.60 | 6734.75 | 6730.69 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 09:15:00 | 6694.70 | 6733.48 | 6734.65 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 10:15:00 | 6769.90 | 6740.77 | 6737.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-24 12:15:00 | 6783.30 | 6747.07 | 6741.09 | Break + close above crossover candle high |

### Cycle 121 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 6519.05 | 6708.66 | 6725.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 6274.00 | 6597.94 | 6670.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 12:15:00 | 6250.50 | 6250.32 | 6334.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 12:30:00 | 6255.65 | 6250.32 | 6334.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 6368.20 | 6278.99 | 6321.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 10:00:00 | 6368.20 | 6278.99 | 6321.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 6317.25 | 6286.65 | 6321.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 11:15:00 | 6301.45 | 6286.65 | 6321.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 09:45:00 | 6272.00 | 6304.18 | 6318.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 10:00:00 | 6305.30 | 6291.29 | 6301.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 11:45:00 | 6197.05 | 6284.04 | 6297.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 6289.95 | 6251.10 | 6271.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:30:00 | 6295.40 | 6251.10 | 6271.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 6268.10 | 6254.50 | 6271.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:45:00 | 6290.90 | 6254.50 | 6271.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 11:15:00 | 6231.00 | 6249.80 | 6267.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 12:15:00 | 6201.25 | 6249.80 | 6267.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 09:15:00 | 6189.00 | 6202.58 | 6216.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 12:15:00 | 6240.25 | 6224.25 | 6223.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2025-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 12:15:00 | 6240.25 | 6224.25 | 6223.30 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 13:15:00 | 6215.50 | 6222.50 | 6222.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 10:15:00 | 6181.10 | 6207.30 | 6214.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 14:15:00 | 6185.20 | 6183.94 | 6199.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 14:15:00 | 6185.20 | 6183.94 | 6199.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 6185.20 | 6183.94 | 6199.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 6185.20 | 6183.94 | 6199.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 6179.80 | 6172.37 | 6189.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:00:00 | 6179.80 | 6172.37 | 6189.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 6166.35 | 6171.16 | 6187.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:45:00 | 6185.30 | 6171.16 | 6187.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 5515.75 | 5407.59 | 5439.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 5515.75 | 5407.59 | 5439.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 5541.60 | 5434.39 | 5448.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 5562.75 | 5434.39 | 5448.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 5583.00 | 5464.11 | 5460.98 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 15:15:00 | 5500.00 | 5515.06 | 5515.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 5310.00 | 5474.05 | 5496.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 14:15:00 | 5450.00 | 5443.52 | 5469.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-24 14:45:00 | 5448.25 | 5443.52 | 5469.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 5347.00 | 5422.05 | 5455.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 11:30:00 | 5293.80 | 5378.99 | 5429.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 10:00:00 | 5298.50 | 5308.20 | 5369.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 15:15:00 | 5270.00 | 5324.46 | 5330.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:45:00 | 5296.80 | 5308.35 | 5321.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 10:15:00 | 5300.00 | 5306.68 | 5319.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 10:30:00 | 5307.70 | 5306.68 | 5319.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-03 12:15:00 | 5597.95 | 5364.51 | 5343.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2025-03-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 12:15:00 | 5597.95 | 5364.51 | 5343.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 5610.00 | 5541.00 | 5487.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 14:15:00 | 5637.00 | 5639.89 | 5580.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 15:00:00 | 5637.00 | 5639.89 | 5580.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 5617.15 | 5635.35 | 5583.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:15:00 | 5635.00 | 5635.35 | 5583.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 5706.45 | 5649.57 | 5595.03 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 10:15:00 | 5554.85 | 5619.51 | 5627.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 11:15:00 | 5510.05 | 5569.12 | 5594.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 15:15:00 | 5548.00 | 5542.19 | 5571.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 09:15:00 | 5502.00 | 5542.19 | 5571.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 5546.00 | 5542.95 | 5568.95 | EMA400 retest candle locked (from downside) |

### Cycle 128 — BUY (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 10:15:00 | 5670.45 | 5563.83 | 5561.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 11:15:00 | 5777.50 | 5606.57 | 5581.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 09:15:00 | 5665.70 | 5710.09 | 5651.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-18 09:30:00 | 5664.30 | 5710.09 | 5651.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 5677.70 | 5703.61 | 5653.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:30:00 | 5662.65 | 5703.61 | 5653.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 5703.75 | 5692.37 | 5667.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:15:00 | 5744.55 | 5672.64 | 5668.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:45:00 | 5793.20 | 5697.33 | 5679.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 10:15:00 | 5770.75 | 5819.94 | 5826.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 10:15:00 | 5770.75 | 5819.94 | 5826.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 5760.00 | 5792.12 | 5809.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 5790.00 | 5787.73 | 5803.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 10:30:00 | 5787.30 | 5787.73 | 5803.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 5854.80 | 5801.14 | 5808.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:45:00 | 5829.70 | 5801.14 | 5808.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 5860.00 | 5812.91 | 5813.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 5860.00 | 5812.91 | 5813.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 5852.00 | 5820.73 | 5816.74 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 15:15:00 | 5755.50 | 5811.49 | 5813.47 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 5844.00 | 5817.99 | 5816.24 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 11:15:00 | 5776.25 | 5810.76 | 5813.33 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 13:15:00 | 5950.00 | 5834.79 | 5823.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 6051.00 | 5878.03 | 5844.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 5911.45 | 5917.03 | 5869.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-01 10:00:00 | 5911.45 | 5917.03 | 5869.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 5880.40 | 5907.80 | 5873.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:30:00 | 5871.65 | 5907.80 | 5873.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 12:15:00 | 5889.85 | 5904.21 | 5875.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 12:45:00 | 5875.00 | 5904.21 | 5875.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 5805.05 | 5884.38 | 5868.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 13:45:00 | 5810.90 | 5884.38 | 5868.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 5817.65 | 5871.03 | 5864.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 14:45:00 | 5809.00 | 5871.03 | 5864.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 15:15:00 | 5800.00 | 5856.83 | 5858.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 09:15:00 | 5731.00 | 5812.06 | 5831.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 5218.95 | 5209.36 | 5348.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 5218.95 | 5209.36 | 5348.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 5218.95 | 5209.36 | 5348.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 5193.90 | 5206.98 | 5334.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:00:00 | 5197.45 | 5206.98 | 5334.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 13:15:00 | 5185.75 | 5210.10 | 5314.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 11:00:00 | 5195.75 | 5206.70 | 5271.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 5359.20 | 5237.61 | 5257.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 10:00:00 | 5359.20 | 5237.61 | 5257.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 5365.90 | 5263.27 | 5267.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 11:00:00 | 5365.90 | 5263.27 | 5267.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-11 11:15:00 | 5425.40 | 5295.70 | 5282.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 5425.40 | 5295.70 | 5282.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 5660.30 | 5368.62 | 5316.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 14:15:00 | 5621.00 | 5658.83 | 5546.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 15:00:00 | 5621.00 | 5658.83 | 5546.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 5681.50 | 5697.76 | 5643.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 5685.00 | 5697.76 | 5643.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 14:30:00 | 5688.00 | 5684.84 | 5653.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 5740.00 | 5684.48 | 5656.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-23 09:15:00 | 6253.50 | 6025.43 | 5925.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 6814.50 | 6863.92 | 6863.93 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 10:15:00 | 6868.00 | 6864.74 | 6864.30 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 11:15:00 | 6840.50 | 6859.89 | 6862.14 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 6892.00 | 6866.70 | 6864.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 6908.00 | 6874.96 | 6868.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 10:15:00 | 6845.50 | 6869.07 | 6866.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 10:15:00 | 6845.50 | 6869.07 | 6866.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 6845.50 | 6869.07 | 6866.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:45:00 | 6835.00 | 6869.07 | 6866.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 6845.00 | 6864.25 | 6864.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:30:00 | 6822.00 | 6864.25 | 6864.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 12:15:00 | 6813.00 | 6854.00 | 6859.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 6772.00 | 6837.60 | 6851.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 14:15:00 | 6845.50 | 6839.18 | 6851.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 14:15:00 | 6845.50 | 6839.18 | 6851.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 6845.50 | 6839.18 | 6851.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 6845.50 | 6839.18 | 6851.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 6761.00 | 6823.55 | 6842.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 09:15:00 | 6685.50 | 6823.55 | 6842.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 10:15:00 | 6759.00 | 6817.24 | 6838.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 12:15:00 | 6760.00 | 6800.29 | 6826.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 13:00:00 | 6760.00 | 6792.23 | 6820.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 6810.00 | 6785.33 | 6806.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 11:15:00 | 6786.00 | 6789.37 | 6806.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-13 09:15:00 | 6771.50 | 6800.86 | 6807.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-13 14:15:00 | 6875.00 | 6818.23 | 6812.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 14:15:00 | 6875.00 | 6818.23 | 6812.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 15:15:00 | 6885.00 | 6831.58 | 6819.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 11:15:00 | 6832.00 | 6840.57 | 6827.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 11:15:00 | 6832.00 | 6840.57 | 6827.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 6832.00 | 6840.57 | 6827.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:45:00 | 6834.00 | 6840.57 | 6827.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 6814.00 | 6835.26 | 6826.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 6807.50 | 6835.26 | 6826.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 6808.50 | 6829.91 | 6824.53 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2025-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 15:15:00 | 6781.00 | 6815.34 | 6818.55 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 11:15:00 | 6835.00 | 6820.12 | 6820.00 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-05-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 15:15:00 | 6801.00 | 6818.49 | 6819.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 09:15:00 | 6776.00 | 6809.99 | 6815.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 12:15:00 | 6805.00 | 6804.02 | 6811.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 12:15:00 | 6805.00 | 6804.02 | 6811.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 12:15:00 | 6805.00 | 6804.02 | 6811.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 13:00:00 | 6805.00 | 6804.02 | 6811.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 13:15:00 | 6809.00 | 6805.02 | 6810.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 14:00:00 | 6809.00 | 6805.02 | 6810.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 6782.50 | 6800.51 | 6808.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 09:15:00 | 6760.00 | 6798.41 | 6806.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 09:15:00 | 6831.00 | 6804.93 | 6808.81 | SL hit (close>static) qty=1.00 sl=6812.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 6851.00 | 6814.14 | 6812.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 6915.50 | 6845.73 | 6830.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 13:15:00 | 6940.00 | 6941.71 | 6904.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-21 14:00:00 | 6940.00 | 6941.71 | 6904.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 6947.50 | 6944.74 | 6915.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 11:45:00 | 6980.00 | 6957.44 | 6926.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 12:15:00 | 7074.50 | 7104.18 | 7105.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 7074.50 | 7104.18 | 7105.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 13:15:00 | 6983.00 | 7079.94 | 7094.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 09:15:00 | 7100.00 | 7065.95 | 7082.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 7100.00 | 7065.95 | 7082.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 7100.00 | 7065.95 | 7082.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:30:00 | 7104.00 | 7065.95 | 7082.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 7095.00 | 7071.76 | 7083.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:30:00 | 7114.00 | 7071.76 | 7083.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 7112.50 | 7079.91 | 7086.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:45:00 | 7123.00 | 7079.91 | 7086.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 7100.50 | 7084.03 | 7087.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 13:30:00 | 7078.50 | 7088.32 | 7089.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 14:15:00 | 7137.50 | 7098.16 | 7093.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 14:15:00 | 7137.50 | 7098.16 | 7093.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 7147.00 | 7112.70 | 7101.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 14:15:00 | 7089.50 | 7124.23 | 7113.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 14:15:00 | 7089.50 | 7124.23 | 7113.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 7089.50 | 7124.23 | 7113.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 7089.50 | 7124.23 | 7113.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 7071.00 | 7113.58 | 7109.84 | EMA400 retest candle locked (from upside) |

### Cycle 149 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 7061.50 | 7103.17 | 7105.45 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 13:15:00 | 7135.00 | 7107.54 | 7106.05 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 10:15:00 | 7099.50 | 7105.71 | 7106.18 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 11:15:00 | 7114.50 | 7107.47 | 7106.94 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 13:15:00 | 7091.00 | 7106.74 | 7106.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 14:15:00 | 7070.00 | 7099.39 | 7103.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 09:15:00 | 7099.00 | 7092.21 | 7099.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 09:15:00 | 7099.00 | 7092.21 | 7099.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 7099.00 | 7092.21 | 7099.09 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 09:15:00 | 7140.50 | 7100.34 | 7098.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 7194.00 | 7119.07 | 7107.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 7227.00 | 7238.47 | 7205.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 11:00:00 | 7227.00 | 7238.47 | 7205.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 7205.00 | 7231.78 | 7205.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:00:00 | 7205.00 | 7231.78 | 7205.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 7211.50 | 7227.72 | 7205.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:30:00 | 7203.50 | 7227.72 | 7205.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 7332.50 | 7248.68 | 7217.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 14:30:00 | 7361.50 | 7270.44 | 7229.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 7364.00 | 7278.75 | 7237.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 10:15:00 | 7352.50 | 7291.30 | 7246.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 12:15:00 | 7380.00 | 7302.87 | 7259.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 7347.00 | 7388.80 | 7342.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:45:00 | 7359.50 | 7388.80 | 7342.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 7349.00 | 7380.84 | 7343.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:30:00 | 7345.50 | 7380.84 | 7343.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 7333.00 | 7371.27 | 7342.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 7333.00 | 7371.27 | 7342.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 7326.00 | 7362.22 | 7340.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 7314.00 | 7347.37 | 7336.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-11 10:15:00 | 7233.50 | 7324.60 | 7326.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 10:15:00 | 7233.50 | 7324.60 | 7326.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 11:15:00 | 7204.00 | 7300.48 | 7315.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 7033.50 | 7032.86 | 7093.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 7033.50 | 7032.86 | 7093.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 7033.50 | 7032.86 | 7093.15 | EMA400 retest candle locked (from downside) |

### Cycle 156 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 7263.00 | 7128.75 | 7125.51 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 09:15:00 | 7106.00 | 7164.34 | 7164.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 7063.00 | 7144.07 | 7155.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 15:15:00 | 7028.00 | 7022.22 | 7060.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:15:00 | 6991.50 | 7022.22 | 7060.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 7014.00 | 7020.58 | 7056.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 11:30:00 | 6969.50 | 7009.85 | 7045.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 13:15:00 | 6977.00 | 6943.86 | 6957.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 09:15:00 | 7182.00 | 7001.89 | 6980.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 7182.00 | 7001.89 | 6980.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 10:15:00 | 7226.00 | 7046.71 | 7002.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 10:15:00 | 7443.00 | 7446.81 | 7391.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 11:00:00 | 7443.00 | 7446.81 | 7391.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 7378.00 | 7433.05 | 7389.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 7374.00 | 7433.05 | 7389.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 7409.00 | 7428.24 | 7391.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 7410.00 | 7428.24 | 7391.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 14:00:00 | 7411.50 | 7424.89 | 7393.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 10:00:00 | 7411.00 | 7417.94 | 7397.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 10:45:00 | 7443.50 | 7424.85 | 7402.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 7664.00 | 7690.85 | 7640.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 7710.00 | 7690.85 | 7640.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:45:00 | 7736.50 | 7702.57 | 7658.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 7553.00 | 7647.92 | 7647.90 | SL hit (close<static) qty=1.00 sl=7602.50 alert=retest2 |

### Cycle 159 — SELL (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 10:15:00 | 7517.50 | 7621.83 | 7636.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 10:15:00 | 7479.50 | 7547.96 | 7586.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 13:15:00 | 7530.00 | 7527.31 | 7565.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 14:00:00 | 7530.00 | 7527.31 | 7565.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 7543.00 | 7520.18 | 7552.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 11:15:00 | 7410.50 | 7507.55 | 7543.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 15:15:00 | 7575.00 | 7510.70 | 7503.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 7575.00 | 7510.70 | 7503.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 7589.50 | 7526.46 | 7511.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 14:15:00 | 7473.00 | 7552.13 | 7534.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 14:15:00 | 7473.00 | 7552.13 | 7534.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 7473.00 | 7552.13 | 7534.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 7473.00 | 7552.13 | 7534.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 7423.50 | 7526.40 | 7524.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 7552.50 | 7526.40 | 7524.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 10:15:00 | 7473.50 | 7513.52 | 7518.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 10:15:00 | 7473.50 | 7513.52 | 7518.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 12:15:00 | 7458.00 | 7494.65 | 7508.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 11:15:00 | 7423.50 | 7341.95 | 7387.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 11:15:00 | 7423.50 | 7341.95 | 7387.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 7423.50 | 7341.95 | 7387.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:00:00 | 7423.50 | 7341.95 | 7387.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 7339.00 | 7341.36 | 7383.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:15:00 | 7171.00 | 7341.36 | 7383.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 6986.50 | 7270.39 | 7347.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 15:15:00 | 6914.00 | 7206.11 | 7310.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 15:15:00 | 6568.30 | 6646.27 | 6729.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 12:15:00 | 6585.00 | 6583.84 | 6668.76 | SL hit (close>ema200) qty=0.50 sl=6583.84 alert=retest2 |

### Cycle 162 — BUY (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 09:15:00 | 6716.50 | 6667.18 | 6666.77 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 12:15:00 | 6685.00 | 6696.72 | 6697.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 6638.50 | 6679.68 | 6689.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 10:15:00 | 6555.00 | 6533.06 | 6583.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 11:00:00 | 6555.00 | 6533.06 | 6583.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 6570.50 | 6540.55 | 6582.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:30:00 | 6571.50 | 6540.55 | 6582.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 6584.50 | 6552.68 | 6577.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 6584.50 | 6552.68 | 6577.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 6571.50 | 6556.45 | 6576.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 6574.00 | 6556.45 | 6576.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 6528.50 | 6550.86 | 6572.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:15:00 | 6513.50 | 6550.86 | 6572.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 12:15:00 | 6634.50 | 6574.73 | 6578.41 | SL hit (close>static) qty=1.00 sl=6614.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 13:15:00 | 6673.00 | 6594.38 | 6587.01 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 13:15:00 | 6607.00 | 6610.56 | 6610.99 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 6634.50 | 6615.35 | 6613.12 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 6579.00 | 6619.51 | 6621.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 6505.00 | 6596.61 | 6611.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 6509.50 | 6494.70 | 6540.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-13 10:00:00 | 6509.50 | 6494.70 | 6540.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 6350.00 | 6344.79 | 6387.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 6416.00 | 6344.79 | 6387.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 6468.00 | 6369.43 | 6394.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 6468.00 | 6369.43 | 6394.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 6401.00 | 6375.74 | 6395.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:30:00 | 6448.00 | 6375.74 | 6395.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 6415.00 | 6388.36 | 6397.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:45:00 | 6420.00 | 6388.36 | 6397.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 6421.50 | 6394.37 | 6398.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 6421.50 | 6394.37 | 6398.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 6419.00 | 6399.29 | 6400.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 6395.00 | 6399.29 | 6400.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 6384.00 | 6393.63 | 6397.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 11:30:00 | 6372.50 | 6388.90 | 6395.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 14:15:00 | 6416.50 | 6388.71 | 6393.04 | SL hit (close>static) qty=1.00 sl=6402.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 6425.50 | 6398.99 | 6397.15 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 11:15:00 | 6385.00 | 6396.19 | 6396.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 13:15:00 | 6375.00 | 6389.76 | 6393.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 6447.50 | 6389.03 | 6390.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 6447.50 | 6389.03 | 6390.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 6447.50 | 6389.03 | 6390.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 6447.50 | 6389.03 | 6390.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — BUY (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 10:15:00 | 6493.00 | 6409.82 | 6400.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 12:15:00 | 6538.50 | 6448.15 | 6420.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 11:15:00 | 6481.50 | 6488.41 | 6456.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 11:45:00 | 6477.50 | 6488.41 | 6456.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 6483.00 | 6487.33 | 6458.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 13:45:00 | 6506.50 | 6490.97 | 6462.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 14:30:00 | 6506.50 | 6493.77 | 6466.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 6345.00 | 6465.01 | 6458.43 | SL hit (close<static) qty=1.00 sl=6450.50 alert=retest2 |

### Cycle 171 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 6324.50 | 6436.91 | 6446.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 6260.00 | 6350.57 | 6397.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 13:15:00 | 6331.00 | 6323.03 | 6359.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 13:30:00 | 6326.50 | 6323.03 | 6359.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 6278.50 | 6288.22 | 6319.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 15:00:00 | 6278.50 | 6288.22 | 6319.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 6315.50 | 6299.63 | 6317.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:45:00 | 6315.00 | 6299.63 | 6317.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 6323.00 | 6304.30 | 6317.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:45:00 | 6319.00 | 6304.30 | 6317.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 6325.50 | 6308.54 | 6318.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:45:00 | 6325.50 | 6308.54 | 6318.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 6327.50 | 6312.33 | 6319.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:45:00 | 6330.50 | 6312.33 | 6319.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 6285.50 | 6310.59 | 6317.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 6284.00 | 6310.59 | 6317.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 6304.50 | 6309.37 | 6316.44 | EMA400 retest candle locked (from downside) |

### Cycle 172 — BUY (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 12:15:00 | 6344.00 | 6324.48 | 6322.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 14:15:00 | 6377.00 | 6338.27 | 6329.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 6374.00 | 6387.24 | 6367.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 6374.00 | 6387.24 | 6367.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 6374.00 | 6387.24 | 6367.83 | EMA400 retest candle locked (from upside) |

### Cycle 173 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 6297.00 | 6364.86 | 6369.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 6272.00 | 6346.29 | 6360.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 11:15:00 | 6310.00 | 6300.74 | 6325.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 12:00:00 | 6310.00 | 6300.74 | 6325.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 6326.50 | 6305.90 | 6325.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:00:00 | 6326.50 | 6305.90 | 6325.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 6323.00 | 6309.32 | 6325.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:45:00 | 6323.50 | 6309.32 | 6325.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 6333.00 | 6314.05 | 6326.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:15:00 | 6331.00 | 6314.05 | 6326.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 6331.00 | 6317.44 | 6326.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 6296.50 | 6317.44 | 6326.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 6335.00 | 6323.76 | 6328.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:15:00 | 6337.50 | 6323.76 | 6328.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 6342.50 | 6327.51 | 6329.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:30:00 | 6337.00 | 6327.51 | 6329.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 6326.50 | 6328.83 | 6329.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:15:00 | 6335.50 | 6328.83 | 6329.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 6353.50 | 6333.76 | 6331.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 15:15:00 | 6360.00 | 6339.01 | 6334.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 15:15:00 | 6410.00 | 6417.22 | 6398.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 09:15:00 | 6391.50 | 6417.22 | 6398.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 6377.50 | 6409.27 | 6396.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 6374.50 | 6409.27 | 6396.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 6326.00 | 6392.62 | 6389.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:00:00 | 6326.00 | 6392.62 | 6389.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 11:15:00 | 6361.00 | 6386.30 | 6387.19 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 6394.00 | 6387.84 | 6387.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 6468.00 | 6405.30 | 6395.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 12:15:00 | 6420.00 | 6431.35 | 6412.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 13:00:00 | 6420.00 | 6431.35 | 6412.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 6435.00 | 6436.60 | 6421.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 10:15:00 | 6463.00 | 6436.60 | 6421.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 10:45:00 | 6455.00 | 6439.88 | 6424.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 15:00:00 | 6448.50 | 6445.72 | 6432.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 12:15:00 | 6448.00 | 6489.81 | 6495.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 6448.00 | 6489.81 | 6495.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 13:15:00 | 6423.50 | 6476.55 | 6488.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 11:15:00 | 6283.50 | 6269.49 | 6306.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 12:00:00 | 6283.50 | 6269.49 | 6306.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 6049.00 | 6019.97 | 6052.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 6049.00 | 6019.97 | 6052.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 6050.00 | 6025.97 | 6052.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 6015.50 | 6025.97 | 6052.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 6100.00 | 6045.12 | 6051.60 | SL hit (close>static) qty=1.00 sl=6080.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 6194.00 | 6074.89 | 6064.54 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 6050.00 | 6081.19 | 6083.69 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 12:15:00 | 6100.50 | 6081.72 | 6080.88 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 6049.50 | 6077.60 | 6079.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 6034.50 | 6068.98 | 6075.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 6061.00 | 6008.81 | 6024.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 6061.00 | 6008.81 | 6024.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 6061.00 | 6008.81 | 6024.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 6061.00 | 6008.81 | 6024.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 6000.50 | 6007.15 | 6022.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:15:00 | 5995.00 | 6007.15 | 6022.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:45:00 | 5992.50 | 6013.63 | 6020.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 11:15:00 | 5991.00 | 6010.90 | 6018.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 11:45:00 | 5987.00 | 6006.82 | 6015.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 5995.00 | 6000.71 | 6009.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 5987.50 | 6000.71 | 6009.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 5943.00 | 5989.16 | 6003.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:30:00 | 5917.00 | 5974.53 | 5995.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:15:00 | 5859.00 | 5842.01 | 5876.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 5939.50 | 5878.79 | 5872.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 11:15:00 | 5939.50 | 5878.79 | 5872.32 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 12:15:00 | 5853.50 | 5881.32 | 5882.83 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 13:15:00 | 5913.00 | 5887.65 | 5885.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 5922.50 | 5898.84 | 5891.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 11:15:00 | 5917.00 | 5922.95 | 5908.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 11:45:00 | 5916.00 | 5922.95 | 5908.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 5905.00 | 5917.77 | 5908.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 5905.00 | 5917.77 | 5908.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 5940.00 | 5922.21 | 5911.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 10:15:00 | 5967.00 | 5925.92 | 5915.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 13:15:00 | 5874.00 | 5910.95 | 5911.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 5874.00 | 5910.95 | 5911.36 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 5925.00 | 5904.02 | 5903.13 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 5887.00 | 5904.19 | 5904.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 5835.50 | 5880.91 | 5891.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 11:15:00 | 5831.50 | 5799.37 | 5819.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 11:15:00 | 5831.50 | 5799.37 | 5819.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 5831.50 | 5799.37 | 5819.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:00:00 | 5831.50 | 5799.37 | 5819.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 5865.00 | 5812.49 | 5823.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:45:00 | 5868.00 | 5812.49 | 5823.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 5894.00 | 5828.79 | 5829.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:00:00 | 5894.00 | 5828.79 | 5829.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 5911.00 | 5845.24 | 5837.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 5940.00 | 5872.95 | 5851.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 5829.50 | 5878.17 | 5867.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 5829.50 | 5878.17 | 5867.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 5829.50 | 5878.17 | 5867.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 5829.50 | 5878.17 | 5867.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 5825.00 | 5867.54 | 5863.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 5820.00 | 5867.54 | 5863.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 5802.00 | 5854.43 | 5858.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 13:15:00 | 5764.00 | 5825.72 | 5843.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 10:15:00 | 5720.00 | 5715.48 | 5756.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 10:30:00 | 5725.50 | 5715.48 | 5756.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 5758.50 | 5720.78 | 5740.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:15:00 | 5788.00 | 5720.78 | 5740.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 5777.50 | 5732.13 | 5743.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 13:45:00 | 5750.00 | 5745.65 | 5747.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 5744.50 | 5748.72 | 5749.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 5785.50 | 5755.40 | 5752.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 5785.50 | 5755.40 | 5752.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 10:15:00 | 5827.00 | 5769.72 | 5758.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 6100.00 | 6120.48 | 6051.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 09:30:00 | 6061.00 | 6120.48 | 6051.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 6080.00 | 6121.61 | 6087.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:45:00 | 6087.00 | 6121.61 | 6087.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 6069.50 | 6111.19 | 6086.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 6069.50 | 6111.19 | 6086.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 6094.00 | 6107.75 | 6086.91 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 5976.50 | 6061.35 | 6071.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 5976.00 | 6044.28 | 6062.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 10:15:00 | 5943.50 | 5934.31 | 5968.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 14:15:00 | 5970.00 | 5941.60 | 5961.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 5970.00 | 5941.60 | 5961.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 5970.00 | 5941.60 | 5961.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 5980.00 | 5949.28 | 5962.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 5979.00 | 5949.28 | 5962.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 5927.00 | 5945.42 | 5958.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 5893.50 | 5945.82 | 5954.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 5882.50 | 5926.27 | 5936.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 15:15:00 | 5856.00 | 5785.45 | 5782.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 15:15:00 | 5856.00 | 5785.45 | 5782.48 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 5759.00 | 5779.26 | 5780.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 5710.00 | 5765.41 | 5774.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 5740.00 | 5732.19 | 5749.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:00:00 | 5740.00 | 5732.19 | 5749.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 5745.00 | 5734.75 | 5749.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 5752.00 | 5734.75 | 5749.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 5855.00 | 5758.80 | 5759.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 5855.00 | 5758.80 | 5759.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 5858.00 | 5778.64 | 5768.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 5895.50 | 5858.52 | 5839.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 5868.00 | 5890.18 | 5865.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 5868.00 | 5890.18 | 5865.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 5868.00 | 5890.18 | 5865.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 5868.00 | 5890.18 | 5865.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 5845.00 | 5881.14 | 5863.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 5840.50 | 5881.14 | 5863.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 5851.00 | 5875.11 | 5862.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 15:15:00 | 5855.00 | 5861.23 | 5858.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 5803.50 | 5848.69 | 5853.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 5803.50 | 5848.69 | 5853.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 10:15:00 | 5785.00 | 5835.95 | 5847.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 5878.00 | 5764.11 | 5785.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 14:15:00 | 5878.00 | 5764.11 | 5785.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 5878.00 | 5764.11 | 5785.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 5878.00 | 5764.11 | 5785.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 5850.00 | 5781.29 | 5791.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:30:00 | 5764.00 | 5781.93 | 5790.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 5835.00 | 5799.50 | 5796.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — BUY (started 2025-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 15:15:00 | 5835.00 | 5799.50 | 5796.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 6104.00 | 5860.40 | 5824.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 5965.00 | 5990.31 | 5947.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 5967.00 | 5990.31 | 5947.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 6040.00 | 6000.24 | 5956.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 5975.00 | 6000.24 | 5956.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 5963.50 | 5997.66 | 5962.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 5961.00 | 5997.66 | 5962.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 5924.00 | 5982.93 | 5959.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 5924.00 | 5982.93 | 5959.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 5923.50 | 5971.04 | 5956.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:30:00 | 5918.00 | 5971.04 | 5956.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 6047.50 | 6043.62 | 6014.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:30:00 | 6066.00 | 6045.49 | 6017.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 6066.00 | 6045.49 | 6017.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:00:00 | 6078.00 | 6051.99 | 6023.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 12:30:00 | 6065.00 | 6111.33 | 6093.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 6039.00 | 6096.65 | 6089.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 6039.00 | 6096.65 | 6089.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 6052.00 | 6087.72 | 6086.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 09:15:00 | 6084.50 | 6087.72 | 6086.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 6071.50 | 6084.48 | 6084.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — SELL (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 09:15:00 | 6071.50 | 6084.48 | 6084.83 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 6111.00 | 6089.78 | 6087.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 6130.50 | 6097.92 | 6091.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 6107.00 | 6123.13 | 6108.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 6107.00 | 6123.13 | 6108.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 6107.00 | 6123.13 | 6108.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:30:00 | 6096.00 | 6123.13 | 6108.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 6094.00 | 6117.30 | 6107.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:00:00 | 6094.00 | 6117.30 | 6107.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 6111.50 | 6115.77 | 6108.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 13:30:00 | 6141.00 | 6123.82 | 6112.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:30:00 | 6125.50 | 6139.55 | 6125.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 14:00:00 | 6126.00 | 6130.44 | 6124.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 6155.50 | 6121.30 | 6120.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 6127.50 | 6153.00 | 6142.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 6115.00 | 6153.00 | 6142.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 6088.00 | 6140.00 | 6137.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-06 10:15:00 | 6088.00 | 6140.00 | 6137.66 | SL hit (close<static) qty=1.00 sl=6094.50 alert=retest2 |

### Cycle 199 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 6092.00 | 6130.40 | 6133.51 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 6154.00 | 6132.97 | 6132.27 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 15:15:00 | 6077.00 | 6126.22 | 6130.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 6072.00 | 6117.50 | 6126.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 13:15:00 | 6179.00 | 6116.60 | 6122.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 13:15:00 | 6179.00 | 6116.60 | 6122.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 6179.00 | 6116.60 | 6122.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:00:00 | 6179.00 | 6116.60 | 6122.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — BUY (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 14:15:00 | 6186.50 | 6130.58 | 6128.13 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 6079.00 | 6123.29 | 6125.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 10:15:00 | 6009.50 | 6062.61 | 6087.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 6078.00 | 6053.51 | 6077.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 6078.00 | 6053.51 | 6077.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 6078.00 | 6053.51 | 6077.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:00:00 | 6078.00 | 6053.51 | 6077.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 6184.50 | 6079.71 | 6087.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 6184.50 | 6079.71 | 6087.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — BUY (started 2026-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 14:15:00 | 6197.50 | 6103.27 | 6097.64 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 12:15:00 | 6056.00 | 6115.10 | 6116.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 09:15:00 | 6000.00 | 6058.04 | 6086.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 15:15:00 | 6015.50 | 6010.46 | 6044.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 09:15:00 | 5955.00 | 6010.46 | 6044.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 5801.00 | 5731.38 | 5795.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 15:00:00 | 5801.00 | 5731.38 | 5795.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 5829.50 | 5751.01 | 5798.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 5841.00 | 5751.01 | 5798.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 5869.50 | 5774.70 | 5805.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:30:00 | 5775.00 | 5793.41 | 5806.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 14:00:00 | 5771.00 | 5793.41 | 5806.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 15:00:00 | 5763.00 | 5787.33 | 5802.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 12:00:00 | 5755.00 | 5792.28 | 5801.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 5809.00 | 5795.62 | 5801.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 5809.00 | 5795.62 | 5801.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 5780.00 | 5792.50 | 5799.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:15:00 | 5865.50 | 5792.50 | 5799.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 5814.00 | 5796.80 | 5801.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-23 15:15:00 | 5856.50 | 5808.74 | 5806.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — BUY (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 15:15:00 | 5856.50 | 5808.74 | 5806.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 09:15:00 | 5872.50 | 5821.49 | 5812.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 13:15:00 | 5828.50 | 5828.80 | 5819.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 13:15:00 | 5828.50 | 5828.80 | 5819.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 5828.50 | 5828.80 | 5819.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 13:30:00 | 5809.50 | 5828.80 | 5819.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 5836.00 | 5830.24 | 5820.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 15:15:00 | 5845.50 | 5830.24 | 5820.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 10:00:00 | 5851.50 | 5836.93 | 5825.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 5990.00 | 6109.42 | 6113.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 5990.00 | 6109.42 | 6113.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 5886.00 | 6031.06 | 6073.86 | Break + close below crossover candle low |

### Cycle 208 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 6403.50 | 6058.92 | 6057.39 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 12:15:00 | 6555.00 | 6581.38 | 6582.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 15:15:00 | 6539.50 | 6567.49 | 6575.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 6580.50 | 6570.09 | 6576.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 6580.50 | 6570.09 | 6576.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 6580.50 | 6570.09 | 6576.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 6599.50 | 6570.09 | 6576.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 6594.50 | 6574.97 | 6577.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 6597.00 | 6574.97 | 6577.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 6686.50 | 6597.28 | 6587.78 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 6552.00 | 6605.10 | 6611.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 15:15:00 | 6505.00 | 6564.85 | 6582.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 6514.00 | 6498.87 | 6530.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:00:00 | 6514.00 | 6498.87 | 6530.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 6515.00 | 6502.09 | 6529.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:45:00 | 6524.00 | 6502.09 | 6529.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 6509.50 | 6504.76 | 6523.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:45:00 | 6520.00 | 6504.76 | 6523.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 6546.50 | 6513.11 | 6525.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 15:00:00 | 6546.50 | 6513.11 | 6525.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 6544.00 | 6519.29 | 6527.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:15:00 | 6559.50 | 6519.29 | 6527.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 6574.00 | 6530.23 | 6531.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:30:00 | 6582.00 | 6530.23 | 6531.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 6561.00 | 6536.38 | 6534.45 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 13:15:00 | 6522.00 | 6531.41 | 6532.58 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 15:15:00 | 6553.50 | 6533.36 | 6533.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 09:15:00 | 6587.50 | 6544.19 | 6538.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 13:15:00 | 6562.00 | 6562.14 | 6549.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 13:15:00 | 6562.00 | 6562.14 | 6549.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 6562.00 | 6562.14 | 6549.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:45:00 | 6572.50 | 6562.14 | 6549.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 6699.00 | 6589.51 | 6563.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:30:00 | 6567.50 | 6589.51 | 6563.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 6575.00 | 6593.41 | 6570.06 | EMA400 retest candle locked (from upside) |

### Cycle 215 — SELL (started 2026-03-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 15:15:00 | 6526.00 | 6554.58 | 6557.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 6383.50 | 6520.36 | 6541.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 15:15:00 | 6700.00 | 6461.20 | 6486.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 15:15:00 | 6700.00 | 6461.20 | 6486.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 6700.00 | 6461.20 | 6486.53 | EMA400 retest candle locked (from downside) |

### Cycle 216 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 6578.00 | 6500.21 | 6490.58 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 12:15:00 | 6467.50 | 6483.61 | 6484.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 14:15:00 | 6423.00 | 6469.55 | 6477.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-11 10:15:00 | 6124.00 | 6107.83 | 6196.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-11 11:00:00 | 6124.00 | 6107.83 | 6196.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 6179.00 | 6135.89 | 6188.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:30:00 | 6180.00 | 6135.89 | 6188.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 6207.50 | 6150.21 | 6190.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:30:00 | 6185.00 | 6150.21 | 6190.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 6200.00 | 6160.17 | 6191.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 09:15:00 | 6101.00 | 6160.17 | 6191.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 6290.00 | 6178.17 | 6190.79 | SL hit (close>static) qty=1.00 sl=6227.50 alert=retest2 |

### Cycle 218 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 6303.00 | 6203.13 | 6200.99 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 6108.00 | 6205.26 | 6208.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 11:15:00 | 6085.00 | 6181.21 | 6197.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 14:15:00 | 6428.00 | 6188.20 | 6192.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 14:15:00 | 6428.00 | 6188.20 | 6192.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 6428.00 | 6188.20 | 6192.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 15:00:00 | 6428.00 | 6188.20 | 6192.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 220 — BUY (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 15:15:00 | 6448.00 | 6240.16 | 6215.64 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 12:15:00 | 6173.50 | 6218.21 | 6224.31 | EMA200 below EMA400 |

### Cycle 222 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 6342.50 | 6247.93 | 6236.54 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 6228.00 | 6261.53 | 6262.32 | EMA200 below EMA400 |

### Cycle 224 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 6291.50 | 6267.52 | 6264.98 | EMA200 above EMA400 |

### Cycle 225 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 6186.50 | 6253.91 | 6262.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 6068.00 | 6189.25 | 6228.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 6143.00 | 6134.30 | 6177.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 6150.00 | 6134.30 | 6177.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 6155.00 | 6138.44 | 6175.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 6196.00 | 6138.44 | 6175.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 6182.50 | 6147.25 | 6176.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 6182.50 | 6147.25 | 6176.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 6131.50 | 6144.10 | 6172.26 | EMA400 retest candle locked (from downside) |

### Cycle 226 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 6245.00 | 6190.02 | 6186.28 | EMA200 above EMA400 |

### Cycle 227 — SELL (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 11:15:00 | 6175.00 | 6198.91 | 6200.41 | EMA200 below EMA400 |

### Cycle 228 — BUY (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 13:15:00 | 6286.50 | 6218.04 | 6208.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 14:15:00 | 6370.50 | 6248.53 | 6223.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-01 15:15:00 | 6366.00 | 6394.50 | 6330.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-02 09:15:00 | 6316.00 | 6394.50 | 6330.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 6318.50 | 6379.30 | 6329.64 | EMA400 retest candle locked (from upside) |

### Cycle 229 — SELL (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 09:15:00 | 6160.50 | 6292.18 | 6305.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 10:15:00 | 6139.00 | 6261.55 | 6290.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 09:15:00 | 6209.50 | 6193.44 | 6236.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 6209.50 | 6193.44 | 6236.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 6209.50 | 6193.44 | 6236.21 | EMA400 retest candle locked (from downside) |

### Cycle 230 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 6374.00 | 6265.61 | 6252.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 15:15:00 | 6400.00 | 6340.31 | 6298.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 6377.00 | 6398.46 | 6353.18 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 6501.00 | 6398.46 | 6353.18 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:15:00 | 6479.50 | 6411.57 | 6363.25 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 12:30:00 | 6485.00 | 6438.90 | 6389.09 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 14:00:00 | 6500.00 | 6451.12 | 6399.17 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 6381.50 | 6440.14 | 6407.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 6381.50 | 6440.14 | 6407.28 | SL hit (close<ema400) qty=1.00 sl=6407.28 alert=retest1 |

### Cycle 231 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 6333.00 | 6392.65 | 6393.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 15:15:00 | 6299.00 | 6363.50 | 6379.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 6429.00 | 6376.60 | 6383.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 6429.00 | 6376.60 | 6383.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 6429.00 | 6376.60 | 6383.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:00:00 | 6429.00 | 6376.60 | 6383.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 232 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 6486.00 | 6398.48 | 6393.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 12:15:00 | 6565.00 | 6444.43 | 6415.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 09:15:00 | 6566.50 | 6573.45 | 6524.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 10:00:00 | 6566.50 | 6573.45 | 6524.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 6575.00 | 6581.28 | 6552.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:30:00 | 6550.50 | 6581.28 | 6552.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 6543.50 | 6574.96 | 6558.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 13:30:00 | 6540.00 | 6574.96 | 6558.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 6549.00 | 6569.77 | 6558.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 15:00:00 | 6549.00 | 6569.77 | 6558.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 6532.00 | 6562.21 | 6555.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 6562.00 | 6562.21 | 6555.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:30:00 | 6567.00 | 6558.30 | 6554.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 12:30:00 | 6567.00 | 6568.61 | 6560.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 6596.50 | 6568.50 | 6562.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 6591.50 | 6573.10 | 6565.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 15:00:00 | 6659.50 | 6610.55 | 6587.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 10:00:00 | 6650.00 | 6624.75 | 6598.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 10:15:00 | 6668.50 | 6700.84 | 6692.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 12:00:00 | 6716.50 | 6694.64 | 6690.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 6700.00 | 6703.34 | 6696.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-27 15:00:00 | 6700.00 | 6703.34 | 6696.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 6813.00 | 6724.73 | 6707.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 10:15:00 | 6879.00 | 6724.73 | 6707.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:30:00 | 6849.50 | 6857.39 | 6801.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 15:00:00 | 6830.00 | 6820.47 | 6800.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 10:15:00 | 6719.00 | 6783.34 | 6787.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 233 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 6719.00 | 6783.34 | 6787.39 | EMA200 below EMA400 |

### Cycle 234 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 6825.00 | 6788.10 | 6786.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 6964.50 | 6823.38 | 6802.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 11:15:00 | 6911.50 | 6934.57 | 6888.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 11:15:00 | 6911.50 | 6934.57 | 6888.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 6911.50 | 6934.57 | 6888.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 6888.00 | 6934.57 | 6888.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 6893.00 | 6920.96 | 6890.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:00:00 | 6893.00 | 6920.96 | 6890.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 6895.50 | 6915.87 | 6890.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 15:00:00 | 6895.50 | 6915.87 | 6890.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 7005.00 | 7053.17 | 7016.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 7093.50 | 7053.17 | 7016.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-24 15:15:00 | 6605.00 | 2023-05-25 10:15:00 | 6668.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2023-05-25 09:30:00 | 6606.15 | 2023-05-25 10:15:00 | 6668.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2023-06-05 13:00:00 | 6770.00 | 2023-06-05 15:15:00 | 6782.25 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2023-06-05 13:45:00 | 6768.00 | 2023-06-05 15:15:00 | 6782.25 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2023-06-19 15:15:00 | 7026.05 | 2023-06-23 09:15:00 | 6959.95 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2023-06-20 09:30:00 | 7031.75 | 2023-06-23 09:15:00 | 6959.95 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2023-06-20 10:00:00 | 7028.90 | 2023-06-23 09:15:00 | 6959.95 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2023-06-20 10:45:00 | 7028.80 | 2023-06-23 09:15:00 | 6959.95 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2023-06-22 15:15:00 | 7085.00 | 2023-06-23 09:15:00 | 6959.95 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2023-07-03 10:00:00 | 6958.20 | 2023-07-11 09:15:00 | 6610.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-03 10:45:00 | 6961.05 | 2023-07-11 09:15:00 | 6613.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-03 10:00:00 | 6958.20 | 2023-07-12 09:15:00 | 6594.10 | STOP_HIT | 0.50 | 5.23% |
| SELL | retest2 | 2023-07-03 10:45:00 | 6961.05 | 2023-07-12 09:15:00 | 6594.10 | STOP_HIT | 0.50 | 5.27% |
| SELL | retest2 | 2023-07-27 13:15:00 | 6560.00 | 2023-07-28 12:15:00 | 6631.50 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2023-07-28 11:45:00 | 6565.05 | 2023-07-28 12:15:00 | 6631.50 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2023-08-04 09:15:00 | 7045.00 | 2023-08-08 09:15:00 | 6993.85 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2023-08-04 10:15:00 | 7051.00 | 2023-08-08 10:15:00 | 6955.10 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2023-08-04 14:00:00 | 7047.00 | 2023-08-08 10:15:00 | 6955.10 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2023-08-07 09:30:00 | 7067.65 | 2023-08-08 10:15:00 | 6955.10 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2023-08-07 12:45:00 | 7066.60 | 2023-08-08 10:15:00 | 6955.10 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2023-08-22 15:15:00 | 6835.00 | 2023-09-08 09:15:00 | 7518.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-23 10:00:00 | 6863.25 | 2023-09-08 09:15:00 | 7520.32 | TARGET_HIT | 1.00 | 9.57% |
| BUY | retest2 | 2023-08-23 13:45:00 | 6836.65 | 2023-09-08 09:15:00 | 7519.71 | TARGET_HIT | 1.00 | 9.99% |
| BUY | retest2 | 2023-08-24 09:15:00 | 6836.10 | 2023-09-11 09:15:00 | 7549.58 | TARGET_HIT | 1.00 | 10.44% |
| BUY | retest2 | 2023-08-24 12:45:00 | 6880.00 | 2023-09-11 11:15:00 | 7568.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-25 11:15:00 | 6872.25 | 2023-09-11 11:15:00 | 7559.48 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-09-14 11:00:00 | 7381.85 | 2023-09-25 09:15:00 | 7021.55 | PARTIAL | 0.50 | 4.88% |
| SELL | retest2 | 2023-09-14 11:00:00 | 7381.85 | 2023-09-25 12:15:00 | 7087.55 | STOP_HIT | 0.50 | 3.99% |
| SELL | retest2 | 2023-09-14 15:15:00 | 7391.10 | 2023-09-25 15:15:00 | 7012.76 | PARTIAL | 0.50 | 5.12% |
| SELL | retest2 | 2023-09-15 09:30:00 | 7381.05 | 2023-09-26 09:15:00 | 7012.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-15 10:15:00 | 7376.15 | 2023-09-26 09:15:00 | 7007.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-15 11:15:00 | 7354.30 | 2023-09-26 11:15:00 | 6986.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-15 11:45:00 | 7352.10 | 2023-09-26 11:15:00 | 6984.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-15 12:15:00 | 7344.00 | 2023-09-26 12:15:00 | 6976.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-14 15:15:00 | 7391.10 | 2023-09-26 15:15:00 | 7004.70 | STOP_HIT | 0.50 | 5.23% |
| SELL | retest2 | 2023-09-15 09:30:00 | 7381.05 | 2023-09-26 15:15:00 | 7004.70 | STOP_HIT | 0.50 | 5.10% |
| SELL | retest2 | 2023-09-15 10:15:00 | 7376.15 | 2023-09-26 15:15:00 | 7004.70 | STOP_HIT | 0.50 | 5.04% |
| SELL | retest2 | 2023-09-15 11:15:00 | 7354.30 | 2023-09-26 15:15:00 | 7004.70 | STOP_HIT | 0.50 | 4.75% |
| SELL | retest2 | 2023-09-15 11:45:00 | 7352.10 | 2023-09-26 15:15:00 | 7004.70 | STOP_HIT | 0.50 | 4.73% |
| SELL | retest2 | 2023-09-15 12:15:00 | 7344.00 | 2023-09-26 15:15:00 | 7004.70 | STOP_HIT | 0.50 | 4.62% |
| SELL | retest2 | 2023-10-11 11:15:00 | 6930.00 | 2023-10-12 15:15:00 | 6940.00 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2023-10-11 12:00:00 | 6935.15 | 2023-10-12 15:15:00 | 6940.00 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2023-10-12 14:30:00 | 6935.75 | 2023-10-12 15:15:00 | 6940.00 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2023-10-12 15:00:00 | 6942.65 | 2023-10-12 15:15:00 | 6940.00 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2023-10-17 13:30:00 | 6902.25 | 2023-10-23 09:15:00 | 6557.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-17 14:00:00 | 6890.90 | 2023-10-23 09:15:00 | 6552.62 | PARTIAL | 0.50 | 4.91% |
| SELL | retest2 | 2023-10-18 09:45:00 | 6897.50 | 2023-10-23 14:15:00 | 6546.35 | PARTIAL | 0.50 | 5.09% |
| SELL | retest2 | 2023-10-17 13:30:00 | 6902.25 | 2023-10-27 09:15:00 | 6212.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-17 14:00:00 | 6890.90 | 2023-10-27 09:15:00 | 6201.81 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-18 09:45:00 | 6897.50 | 2023-10-27 09:15:00 | 6207.75 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2023-11-08 09:15:00 | 6560.00 | 2023-11-20 09:15:00 | 6635.85 | STOP_HIT | 1.00 | 1.16% |
| BUY | retest2 | 2023-11-08 11:30:00 | 6561.00 | 2023-11-20 09:15:00 | 6635.85 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2023-12-01 09:15:00 | 6664.75 | 2023-12-08 13:15:00 | 6741.90 | STOP_HIT | 1.00 | 1.16% |
| BUY | retest2 | 2023-12-01 11:00:00 | 6619.00 | 2023-12-08 13:15:00 | 6741.90 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest1 | 2023-12-19 09:15:00 | 7074.00 | 2023-12-20 11:15:00 | 7044.80 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2023-12-19 10:45:00 | 7064.50 | 2023-12-20 11:15:00 | 7044.80 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2024-01-02 09:30:00 | 7207.55 | 2024-01-04 12:15:00 | 7116.90 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-01-03 11:30:00 | 7220.00 | 2024-01-04 12:15:00 | 7116.90 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-01-03 12:30:00 | 7208.20 | 2024-01-04 12:15:00 | 7116.90 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-01-03 13:15:00 | 7200.75 | 2024-01-04 12:15:00 | 7116.90 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-01-11 12:15:00 | 6860.00 | 2024-01-12 10:15:00 | 6899.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2024-01-11 14:00:00 | 6840.60 | 2024-01-12 10:15:00 | 6899.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-01-11 15:00:00 | 6854.05 | 2024-01-12 10:15:00 | 6899.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-01-19 12:15:00 | 6617.65 | 2024-01-23 11:15:00 | 6286.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-19 13:15:00 | 6625.85 | 2024-01-23 11:15:00 | 6294.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-19 15:00:00 | 6619.65 | 2024-01-23 11:15:00 | 6288.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-19 12:15:00 | 6617.65 | 2024-01-24 09:15:00 | 6319.50 | STOP_HIT | 0.50 | 4.51% |
| SELL | retest2 | 2024-01-19 13:15:00 | 6625.85 | 2024-01-24 09:15:00 | 6319.50 | STOP_HIT | 0.50 | 4.62% |
| SELL | retest2 | 2024-01-19 15:00:00 | 6619.65 | 2024-01-24 09:15:00 | 6319.50 | STOP_HIT | 0.50 | 4.53% |
| BUY | retest2 | 2024-02-16 09:15:00 | 6402.00 | 2024-02-19 09:15:00 | 6357.45 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-02-16 14:00:00 | 6378.70 | 2024-02-19 09:15:00 | 6357.45 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-02-20 10:30:00 | 6300.00 | 2024-02-21 10:15:00 | 6379.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-02-20 11:00:00 | 6287.05 | 2024-02-21 10:15:00 | 6379.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-02-23 10:30:00 | 6305.60 | 2024-03-06 11:15:00 | 5990.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-23 14:45:00 | 6304.75 | 2024-03-06 11:15:00 | 5989.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-23 10:30:00 | 6305.60 | 2024-03-06 15:15:00 | 6049.00 | STOP_HIT | 0.50 | 4.07% |
| SELL | retest2 | 2024-02-23 14:45:00 | 6304.75 | 2024-03-06 15:15:00 | 6049.00 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2024-03-13 10:00:00 | 6044.15 | 2024-03-28 11:15:00 | 5741.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-14 11:00:00 | 6035.40 | 2024-03-28 12:15:00 | 5733.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-13 10:00:00 | 6044.15 | 2024-04-01 09:15:00 | 5806.35 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2024-03-14 11:00:00 | 6035.40 | 2024-04-01 09:15:00 | 5806.35 | STOP_HIT | 0.50 | 3.80% |
| SELL | retest2 | 2024-03-15 15:00:00 | 5988.85 | 2024-04-02 09:15:00 | 5905.85 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2024-04-05 14:15:00 | 5971.45 | 2024-04-08 13:15:00 | 5919.90 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-04-05 15:00:00 | 5975.95 | 2024-04-08 13:15:00 | 5919.90 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-04-08 09:45:00 | 5979.95 | 2024-04-08 13:15:00 | 5919.90 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-04-18 13:30:00 | 5949.45 | 2024-04-23 10:15:00 | 5974.00 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-04-18 14:30:00 | 5920.40 | 2024-04-23 10:15:00 | 5974.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-04-23 10:15:00 | 5938.55 | 2024-04-23 10:15:00 | 5974.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-04-24 09:15:00 | 5960.55 | 2024-04-24 14:15:00 | 5900.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-05-02 13:45:00 | 5984.65 | 2024-05-09 09:15:00 | 6040.50 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2024-05-13 10:30:00 | 5863.85 | 2024-05-17 12:15:00 | 5964.80 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-05-14 10:00:00 | 5871.00 | 2024-05-17 12:15:00 | 5964.80 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-05-15 14:15:00 | 5873.00 | 2024-05-17 12:15:00 | 5964.80 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-05-15 15:00:00 | 5875.55 | 2024-05-17 12:15:00 | 5964.80 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-05-16 13:15:00 | 5867.55 | 2024-05-17 12:15:00 | 5964.80 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-05-16 14:00:00 | 5864.30 | 2024-05-17 12:15:00 | 5964.80 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-05-27 11:45:00 | 5898.55 | 2024-06-04 09:15:00 | 5603.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 14:30:00 | 5900.90 | 2024-06-04 09:15:00 | 5605.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 09:45:00 | 5901.40 | 2024-06-04 09:15:00 | 5606.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 11:45:00 | 5898.55 | 2024-06-04 12:15:00 | 5308.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-27 14:30:00 | 5900.90 | 2024-06-04 12:15:00 | 5310.81 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-28 09:45:00 | 5901.40 | 2024-06-04 12:15:00 | 5311.26 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-06-19 11:30:00 | 6285.30 | 2024-06-25 12:15:00 | 6348.05 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2024-06-20 09:15:00 | 6263.45 | 2024-06-25 12:15:00 | 6348.05 | STOP_HIT | 1.00 | 1.35% |
| BUY | retest2 | 2024-07-04 11:15:00 | 6612.95 | 2024-07-19 13:15:00 | 6993.30 | STOP_HIT | 1.00 | 5.75% |
| BUY | retest2 | 2024-07-25 14:15:00 | 7309.90 | 2024-08-02 15:15:00 | 7793.85 | STOP_HIT | 1.00 | 6.62% |
| BUY | retest2 | 2024-07-25 14:45:00 | 7309.95 | 2024-08-02 15:15:00 | 7793.85 | STOP_HIT | 1.00 | 6.62% |
| BUY | retest2 | 2024-07-26 09:15:00 | 7308.80 | 2024-08-02 15:15:00 | 7793.85 | STOP_HIT | 1.00 | 6.64% |
| SELL | retest2 | 2024-08-06 14:30:00 | 7796.85 | 2024-08-07 10:15:00 | 7871.35 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-08-14 14:15:00 | 7730.50 | 2024-08-16 12:15:00 | 7854.60 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-08-16 09:30:00 | 7716.10 | 2024-08-16 12:15:00 | 7854.60 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-08-16 10:00:00 | 7729.95 | 2024-08-16 12:15:00 | 7854.60 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-09-05 09:30:00 | 8028.05 | 2024-09-06 09:15:00 | 7817.85 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2024-09-05 14:45:00 | 8017.80 | 2024-09-06 09:15:00 | 7817.85 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2024-09-20 13:45:00 | 7597.85 | 2024-09-27 11:15:00 | 7587.45 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2024-09-20 14:30:00 | 7599.65 | 2024-09-27 11:15:00 | 7587.45 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2024-09-24 11:15:00 | 7598.35 | 2024-09-27 11:15:00 | 7587.45 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest1 | 2024-10-01 09:15:00 | 7709.45 | 2024-10-03 14:15:00 | 7822.95 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2024-10-15 09:15:00 | 7960.45 | 2024-10-15 11:15:00 | 7706.50 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2024-11-04 15:15:00 | 7786.15 | 2024-11-11 09:15:00 | 7725.90 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-11-19 14:15:00 | 7324.15 | 2024-11-25 10:15:00 | 7411.30 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-11-19 14:45:00 | 7282.85 | 2024-11-25 10:15:00 | 7411.30 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-11-28 09:30:00 | 7402.35 | 2024-11-28 10:15:00 | 7303.70 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-12-05 09:15:00 | 7415.70 | 2024-12-05 09:15:00 | 7345.95 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-12-06 09:30:00 | 7421.80 | 2024-12-06 13:15:00 | 7366.05 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-12-06 10:15:00 | 7415.00 | 2024-12-06 13:15:00 | 7366.05 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-12-18 09:30:00 | 7363.55 | 2024-12-20 09:15:00 | 7297.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-12-18 10:15:00 | 7360.00 | 2024-12-20 09:15:00 | 7297.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-12-19 09:30:00 | 7387.65 | 2024-12-20 11:15:00 | 7277.30 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-12-19 10:15:00 | 7377.70 | 2024-12-20 11:15:00 | 7277.30 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-12-19 11:30:00 | 7417.05 | 2024-12-20 11:15:00 | 7277.30 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-12-19 15:00:00 | 7412.05 | 2024-12-20 11:15:00 | 7277.30 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-01-06 10:30:00 | 6869.45 | 2025-01-07 14:15:00 | 6920.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-01-30 11:15:00 | 6301.45 | 2025-02-05 12:15:00 | 6240.25 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2025-01-31 09:45:00 | 6272.00 | 2025-02-05 12:15:00 | 6240.25 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2025-02-01 10:00:00 | 6305.30 | 2025-02-05 12:15:00 | 6240.25 | STOP_HIT | 1.00 | 1.03% |
| SELL | retest2 | 2025-02-01 11:45:00 | 6197.05 | 2025-02-05 12:15:00 | 6240.25 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-02-03 12:15:00 | 6201.25 | 2025-02-05 12:15:00 | 6240.25 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-02-05 09:15:00 | 6189.00 | 2025-02-05 12:15:00 | 6240.25 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-02-25 11:30:00 | 5293.80 | 2025-03-03 12:15:00 | 5597.95 | STOP_HIT | 1.00 | -5.75% |
| SELL | retest2 | 2025-02-27 10:00:00 | 5298.50 | 2025-03-03 12:15:00 | 5597.95 | STOP_HIT | 1.00 | -5.65% |
| SELL | retest2 | 2025-02-28 15:15:00 | 5270.00 | 2025-03-03 12:15:00 | 5597.95 | STOP_HIT | 1.00 | -6.22% |
| SELL | retest2 | 2025-03-03 09:45:00 | 5296.80 | 2025-03-03 12:15:00 | 5597.95 | STOP_HIT | 1.00 | -5.69% |
| BUY | retest2 | 2025-03-21 10:15:00 | 5744.55 | 2025-03-26 10:15:00 | 5770.75 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2025-03-21 10:45:00 | 5793.20 | 2025-03-26 10:15:00 | 5770.75 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-04-08 10:30:00 | 5193.90 | 2025-04-11 11:15:00 | 5425.40 | STOP_HIT | 1.00 | -4.46% |
| SELL | retest2 | 2025-04-08 11:00:00 | 5197.45 | 2025-04-11 11:15:00 | 5425.40 | STOP_HIT | 1.00 | -4.39% |
| SELL | retest2 | 2025-04-08 13:15:00 | 5185.75 | 2025-04-11 11:15:00 | 5425.40 | STOP_HIT | 1.00 | -4.62% |
| SELL | retest2 | 2025-04-09 11:00:00 | 5195.75 | 2025-04-11 11:15:00 | 5425.40 | STOP_HIT | 1.00 | -4.42% |
| BUY | retest2 | 2025-04-17 11:15:00 | 5685.00 | 2025-04-23 09:15:00 | 6253.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-17 14:30:00 | 5688.00 | 2025-04-23 09:15:00 | 6256.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-21 09:15:00 | 5740.00 | 2025-04-25 09:15:00 | 6314.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-09 09:15:00 | 6685.50 | 2025-05-13 14:15:00 | 6875.00 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-05-09 10:15:00 | 6759.00 | 2025-05-13 14:15:00 | 6875.00 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-05-09 12:15:00 | 6760.00 | 2025-05-13 14:15:00 | 6875.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-05-09 13:00:00 | 6760.00 | 2025-05-13 14:15:00 | 6875.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-05-12 11:15:00 | 6786.00 | 2025-05-13 14:15:00 | 6875.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-05-13 09:15:00 | 6771.50 | 2025-05-13 14:15:00 | 6875.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-05-19 09:15:00 | 6760.00 | 2025-05-19 09:15:00 | 6831.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-05-22 11:45:00 | 6980.00 | 2025-05-27 12:15:00 | 7074.50 | STOP_HIT | 1.00 | 1.35% |
| SELL | retest2 | 2025-05-28 13:30:00 | 7078.50 | 2025-05-28 14:15:00 | 7137.50 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-06-06 14:30:00 | 7361.50 | 2025-06-11 10:15:00 | 7233.50 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-06-09 09:15:00 | 7364.00 | 2025-06-11 10:15:00 | 7233.50 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-06-09 10:15:00 | 7352.50 | 2025-06-11 10:15:00 | 7233.50 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-06-09 12:15:00 | 7380.00 | 2025-06-11 10:15:00 | 7233.50 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-06-20 11:30:00 | 6969.50 | 2025-06-25 09:15:00 | 7182.00 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-06-24 13:15:00 | 6977.00 | 2025-06-25 09:15:00 | 7182.00 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2025-07-01 13:15:00 | 7410.00 | 2025-07-09 09:15:00 | 7553.00 | STOP_HIT | 1.00 | 1.93% |
| BUY | retest2 | 2025-07-01 14:00:00 | 7411.50 | 2025-07-09 09:15:00 | 7553.00 | STOP_HIT | 1.00 | 1.91% |
| BUY | retest2 | 2025-07-02 10:00:00 | 7411.00 | 2025-07-09 10:15:00 | 7517.50 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest2 | 2025-07-02 10:45:00 | 7443.50 | 2025-07-09 10:15:00 | 7517.50 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest2 | 2025-07-08 09:15:00 | 7710.00 | 2025-07-09 10:15:00 | 7517.50 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-07-08 11:45:00 | 7736.50 | 2025-07-09 10:15:00 | 7517.50 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-07-11 11:15:00 | 7410.50 | 2025-07-14 15:15:00 | 7575.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-07-16 09:15:00 | 7552.50 | 2025-07-16 10:15:00 | 7473.50 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-07-18 15:15:00 | 6914.00 | 2025-07-24 15:15:00 | 6568.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 15:15:00 | 6914.00 | 2025-07-25 12:15:00 | 6585.00 | STOP_HIT | 0.50 | 4.76% |
| SELL | retest2 | 2025-08-05 10:15:00 | 6513.50 | 2025-08-05 12:15:00 | 6634.50 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-08-20 11:30:00 | 6372.50 | 2025-08-20 14:15:00 | 6416.50 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-08-25 13:45:00 | 6506.50 | 2025-08-26 09:15:00 | 6345.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-08-25 14:30:00 | 6506.50 | 2025-08-26 09:15:00 | 6345.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-09-16 10:15:00 | 6463.00 | 2025-09-19 12:15:00 | 6448.00 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-09-16 10:45:00 | 6455.00 | 2025-09-19 12:15:00 | 6448.00 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-09-16 15:00:00 | 6448.50 | 2025-09-19 12:15:00 | 6448.00 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-10-01 09:15:00 | 6015.50 | 2025-10-01 13:15:00 | 6100.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-10-10 11:15:00 | 5995.00 | 2025-10-17 11:15:00 | 5939.50 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2025-10-13 09:45:00 | 5992.50 | 2025-10-17 11:15:00 | 5939.50 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-10-13 11:15:00 | 5991.00 | 2025-10-17 11:15:00 | 5939.50 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2025-10-13 11:45:00 | 5987.00 | 2025-10-17 11:15:00 | 5939.50 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2025-10-14 10:30:00 | 5917.00 | 2025-10-17 11:15:00 | 5939.50 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-10-16 10:15:00 | 5859.00 | 2025-10-17 11:15:00 | 5939.50 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-10-24 10:15:00 | 5967.00 | 2025-10-24 13:15:00 | 5874.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-11-11 13:45:00 | 5750.00 | 2025-11-12 09:15:00 | 5785.50 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-11-11 15:15:00 | 5744.50 | 2025-11-12 09:15:00 | 5785.50 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-11-27 09:15:00 | 5893.50 | 2025-12-05 15:15:00 | 5856.00 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2025-11-28 09:15:00 | 5882.50 | 2025-12-05 15:15:00 | 5856.00 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2025-12-15 15:15:00 | 5855.00 | 2025-12-16 09:15:00 | 5803.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-12-18 09:30:00 | 5764.00 | 2025-12-18 15:15:00 | 5835.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-26 09:30:00 | 6066.00 | 2025-12-31 09:15:00 | 6071.50 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-12-26 10:15:00 | 6066.00 | 2025-12-31 09:15:00 | 6071.50 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-12-26 11:00:00 | 6078.00 | 2025-12-31 09:15:00 | 6071.50 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-12-30 12:30:00 | 6065.00 | 2025-12-31 09:15:00 | 6071.50 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2025-12-31 09:15:00 | 6084.50 | 2025-12-31 09:15:00 | 6071.50 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2026-01-01 13:30:00 | 6141.00 | 2026-01-06 10:15:00 | 6088.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-01-02 10:30:00 | 6125.50 | 2026-01-06 10:15:00 | 6088.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2026-01-02 14:00:00 | 6126.00 | 2026-01-06 10:15:00 | 6088.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-01-05 09:15:00 | 6155.50 | 2026-01-06 10:15:00 | 6088.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-01-22 13:30:00 | 5775.00 | 2026-01-23 15:15:00 | 5856.50 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2026-01-22 14:00:00 | 5771.00 | 2026-01-23 15:15:00 | 5856.50 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-01-22 15:00:00 | 5763.00 | 2026-01-23 15:15:00 | 5856.50 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-01-23 12:00:00 | 5755.00 | 2026-01-23 15:15:00 | 5856.50 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-01-27 15:15:00 | 5845.50 | 2026-02-01 14:15:00 | 5990.00 | STOP_HIT | 1.00 | 2.47% |
| BUY | retest2 | 2026-01-28 10:00:00 | 5851.50 | 2026-02-01 14:15:00 | 5990.00 | STOP_HIT | 1.00 | 2.37% |
| SELL | retest2 | 2026-03-12 09:15:00 | 6101.00 | 2026-03-12 11:15:00 | 6290.00 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest1 | 2026-04-10 09:15:00 | 6501.00 | 2026-04-13 09:15:00 | 6381.50 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest1 | 2026-04-10 10:15:00 | 6479.50 | 2026-04-13 09:15:00 | 6381.50 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest1 | 2026-04-10 12:30:00 | 6485.00 | 2026-04-13 09:15:00 | 6381.50 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest1 | 2026-04-10 14:00:00 | 6500.00 | 2026-04-13 09:15:00 | 6381.50 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-04-21 09:15:00 | 6562.00 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 2.39% |
| BUY | retest2 | 2026-04-21 10:30:00 | 6567.00 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 2.31% |
| BUY | retest2 | 2026-04-21 12:30:00 | 6567.00 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 2.31% |
| BUY | retest2 | 2026-04-22 09:15:00 | 6596.50 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2026-04-22 15:00:00 | 6659.50 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest2 | 2026-04-23 10:00:00 | 6650.00 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2026-04-27 10:15:00 | 6668.50 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2026-04-27 12:00:00 | 6716.50 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2026-04-28 10:15:00 | 6879.00 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-04-29 09:30:00 | 6849.50 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-04-29 15:00:00 | 6830.00 | 2026-04-30 10:15:00 | 6719.00 | STOP_HIT | 1.00 | -1.63% |
