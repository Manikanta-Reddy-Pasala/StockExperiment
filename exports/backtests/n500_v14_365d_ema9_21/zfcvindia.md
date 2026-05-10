# ZF Commercial Vehicle Control Systems India Ltd. (ZFCVINDIA)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 14532.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 87 |
| ALERT1 | 47 |
| ALERT2 | 46 |
| ALERT2_SKIP | 25 |
| ALERT3 | 137 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 65 |
| PARTIAL | 4 |
| TARGET_HIT | 5 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 67 (incl. partial bookings)
- **Trades open at end:** 6
- **Winners / losers:** 18 / 49
- **Target hits / Stop hits / Partials:** 5 / 58 / 4
- **Avg / median % per leg:** 0.51% / -0.60%
- **Sum % (uncompounded):** 33.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 10 | 29.4% | 5 | 29 | 0 | 0.77% | 26.1% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.52% | -6.1% |
| BUY @ 3rd Alert (retest2) | 30 | 10 | 33.3% | 5 | 25 | 0 | 1.07% | 32.2% |
| SELL (all) | 33 | 8 | 24.2% | 0 | 29 | 4 | 0.24% | 7.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 8 | 24.2% | 0 | 29 | 4 | 0.24% | 7.9% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.52% | -6.1% |
| retest2 (combined) | 63 | 18 | 28.6% | 5 | 54 | 4 | 0.64% | 40.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 13205.00 | 13293.27 | 13300.02 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 13359.00 | 13306.42 | 13305.39 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 15:15:00 | 13199.00 | 13288.72 | 13299.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 13156.00 | 13262.17 | 13286.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 13000.00 | 12985.49 | 13086.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 13000.00 | 12985.49 | 13086.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 13000.00 | 12985.49 | 13086.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 12:00:00 | 12875.00 | 12988.14 | 13024.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 13:30:00 | 12843.00 | 12947.13 | 12998.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 12:15:00 | 13252.00 | 12946.31 | 12919.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 12:15:00 | 13252.00 | 12946.31 | 12919.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 12:15:00 | 13252.00 | 12946.31 | 12919.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 14:15:00 | 13391.00 | 13083.36 | 12989.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 15:15:00 | 13412.00 | 13527.40 | 13394.37 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 09:15:00 | 13594.00 | 13527.40 | 13394.37 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 10:45:00 | 13620.00 | 13545.01 | 13425.33 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 13442.00 | 13524.41 | 13426.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:45:00 | 13427.00 | 13524.41 | 13426.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 13375.00 | 13494.53 | 13422.13 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 13375.00 | 13494.53 | 13422.13 | SL hit (close<ema400) qty=1.00 sl=13422.13 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 13375.00 | 13494.53 | 13422.13 | SL hit (close<ema400) qty=1.00 sl=13422.13 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-06-04 13:00:00 | 13375.00 | 13494.53 | 13422.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 13340.00 | 13463.62 | 13414.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:30:00 | 13367.00 | 13463.62 | 13414.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 13508.00 | 13472.50 | 13423.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:45:00 | 13523.00 | 13469.56 | 13429.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 14:15:00 | 13515.00 | 13494.05 | 13456.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 13854.00 | 13439.59 | 13436.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 14:15:00 | 13536.00 | 13733.36 | 13754.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-10 14:15:00 | 13536.00 | 13733.36 | 13754.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-10 14:15:00 | 13536.00 | 13733.36 | 13754.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 14:15:00 | 13536.00 | 13733.36 | 13754.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 09:15:00 | 13507.00 | 13655.39 | 13713.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 13455.00 | 13348.00 | 13413.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 14:15:00 | 13455.00 | 13348.00 | 13413.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 13455.00 | 13348.00 | 13413.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 13455.00 | 13348.00 | 13413.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 13302.00 | 13338.80 | 13403.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 13176.00 | 13338.80 | 13403.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 13135.00 | 13298.04 | 13379.01 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 14:15:00 | 13341.00 | 13297.53 | 13293.24 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 13137.00 | 13267.00 | 13280.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 11:15:00 | 13120.00 | 13237.60 | 13266.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 13310.00 | 13220.73 | 13240.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 13310.00 | 13220.73 | 13240.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 13310.00 | 13220.73 | 13240.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:15:00 | 13110.00 | 13210.51 | 13228.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 13:30:00 | 13126.00 | 13132.36 | 13143.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 09:15:00 | 13268.00 | 13154.23 | 13150.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 09:15:00 | 13268.00 | 13154.23 | 13150.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 13268.00 | 13154.23 | 13150.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 11:15:00 | 13359.00 | 13274.44 | 13228.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 15:15:00 | 13401.00 | 13434.24 | 13369.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 10:00:00 | 13424.00 | 13432.19 | 13374.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 13301.00 | 13405.95 | 13367.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:00:00 | 13301.00 | 13405.95 | 13367.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 13347.00 | 13394.16 | 13365.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 14:30:00 | 13369.00 | 13369.13 | 13358.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 13280.00 | 13349.84 | 13351.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 13280.00 | 13349.84 | 13351.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 10:15:00 | 13232.00 | 13326.28 | 13340.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 12:15:00 | 13379.00 | 13331.98 | 13340.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 12:15:00 | 13379.00 | 13331.98 | 13340.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 13379.00 | 13331.98 | 13340.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 13379.00 | 13331.98 | 13340.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 13366.00 | 13338.78 | 13342.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:30:00 | 13395.00 | 13338.78 | 13342.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 13337.00 | 13340.22 | 13342.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 13292.00 | 13340.22 | 13342.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 13229.00 | 13317.98 | 13332.47 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 11:15:00 | 13465.00 | 13350.26 | 13344.81 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 15:15:00 | 13238.00 | 13327.66 | 13338.37 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 13418.00 | 13354.37 | 13348.41 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 13321.00 | 13352.47 | 13352.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 14:15:00 | 13287.00 | 13339.38 | 13346.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 10:15:00 | 13172.00 | 13141.76 | 13200.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 10:15:00 | 13172.00 | 13141.76 | 13200.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 13172.00 | 13141.76 | 13200.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 13172.00 | 13141.76 | 13200.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 13200.00 | 13153.41 | 13200.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 13200.00 | 13153.41 | 13200.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 13178.00 | 13158.33 | 13198.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:30:00 | 13223.00 | 13158.33 | 13198.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 13240.00 | 13179.89 | 13201.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:45:00 | 13249.00 | 13179.89 | 13201.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 13179.00 | 13179.71 | 13199.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:45:00 | 13100.00 | 13164.37 | 13190.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:45:00 | 13112.00 | 13158.71 | 13176.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:45:00 | 13112.00 | 13133.77 | 13163.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 13143.00 | 13048.45 | 13044.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 13143.00 | 13048.45 | 13044.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 13143.00 | 13048.45 | 13044.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 13143.00 | 13048.45 | 13044.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 13171.00 | 13072.96 | 13055.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 13082.00 | 13119.07 | 13090.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 13082.00 | 13119.07 | 13090.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 13082.00 | 13119.07 | 13090.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:30:00 | 13093.00 | 13119.07 | 13090.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 13057.00 | 13106.66 | 13087.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 13057.00 | 13106.66 | 13087.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 13123.00 | 13109.93 | 13090.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:30:00 | 13053.00 | 13109.93 | 13090.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 13068.00 | 13101.54 | 13088.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:00:00 | 13068.00 | 13101.54 | 13088.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 13070.00 | 13095.23 | 13087.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:30:00 | 13043.00 | 13095.23 | 13087.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 13105.00 | 13089.95 | 13085.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 13070.00 | 13089.95 | 13085.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 13130.00 | 13097.96 | 13089.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:30:00 | 13242.00 | 13121.37 | 13101.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 15:15:00 | 13189.00 | 13200.79 | 13179.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:30:00 | 13192.00 | 13186.90 | 13177.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 12:00:00 | 13250.00 | 13186.90 | 13177.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 13154.00 | 13180.32 | 13175.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:00:00 | 13154.00 | 13180.32 | 13175.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 13218.00 | 13187.85 | 13179.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:45:00 | 13146.00 | 13187.85 | 13179.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 13220.00 | 13199.27 | 13186.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 10:30:00 | 13261.00 | 13188.45 | 13184.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 11:15:00 | 13114.00 | 13173.56 | 13177.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-22 11:15:00 | 13114.00 | 13173.56 | 13177.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-22 11:15:00 | 13114.00 | 13173.56 | 13177.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-22 11:15:00 | 13114.00 | 13173.56 | 13177.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-22 11:15:00 | 13114.00 | 13173.56 | 13177.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 11:15:00 | 13114.00 | 13173.56 | 13177.67 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 13:15:00 | 13229.00 | 13186.96 | 13183.19 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 15:15:00 | 13115.00 | 13167.93 | 13174.91 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 11:15:00 | 13292.00 | 13191.49 | 13183.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 13:15:00 | 13391.00 | 13250.83 | 13212.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 13:15:00 | 13463.00 | 13506.02 | 13393.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 13:45:00 | 13441.00 | 13506.02 | 13393.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 13565.00 | 13517.82 | 13409.44 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 13220.00 | 13409.57 | 13426.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 13093.00 | 13346.25 | 13396.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 14:15:00 | 13256.00 | 13214.93 | 13282.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 15:00:00 | 13256.00 | 13214.93 | 13282.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 13240.00 | 13219.94 | 13279.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 13295.00 | 13219.94 | 13279.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 13246.00 | 13225.15 | 13276.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 13192.00 | 13225.15 | 13276.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 13:15:00 | 13420.00 | 13290.99 | 13292.53 | SL hit (close>static) qty=1.00 sl=13364.00 alert=retest2 |

### Cycle 20 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 13393.00 | 13311.39 | 13301.67 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 13199.00 | 13289.74 | 13298.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 13166.00 | 13249.99 | 13277.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 13234.00 | 13077.80 | 13135.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 12:15:00 | 13234.00 | 13077.80 | 13135.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 13234.00 | 13077.80 | 13135.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 13234.00 | 13077.80 | 13135.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 13326.00 | 13127.44 | 13152.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:45:00 | 13377.00 | 13127.44 | 13152.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 13350.00 | 13171.95 | 13170.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 09:15:00 | 13473.00 | 13256.65 | 13211.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 13534.00 | 13634.12 | 13525.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 13534.00 | 13634.12 | 13525.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 13534.00 | 13634.12 | 13525.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 13534.00 | 13634.12 | 13525.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 13452.00 | 13597.69 | 13518.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:00:00 | 13452.00 | 13597.69 | 13518.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 13448.00 | 13567.76 | 13512.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:30:00 | 13418.00 | 13567.76 | 13512.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 14:15:00 | 13256.00 | 13446.98 | 13466.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 13155.00 | 13369.87 | 13426.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 14:15:00 | 13320.00 | 13276.39 | 13347.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 15:00:00 | 13320.00 | 13276.39 | 13347.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 13175.00 | 13263.25 | 13329.64 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 13611.00 | 13367.77 | 13362.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 13702.00 | 13434.61 | 13393.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 13362.00 | 13660.64 | 13594.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 13362.00 | 13660.64 | 13594.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 13362.00 | 13660.64 | 13594.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:00:00 | 13362.00 | 13660.64 | 13594.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 13402.00 | 13608.91 | 13576.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:30:00 | 13331.00 | 13608.91 | 13576.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 13500.00 | 13587.30 | 13572.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:45:00 | 13512.00 | 13587.30 | 13572.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 13612.00 | 13592.24 | 13576.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 15:15:00 | 13650.00 | 13593.79 | 13578.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 10:15:00 | 13662.00 | 13610.23 | 13589.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 11:00:00 | 14084.00 | 13704.98 | 13634.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-25 11:15:00 | 15015.00 | 14752.22 | 14598.37 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-25 11:15:00 | 15028.20 | 14752.22 | 14598.37 | Target hit (10%) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 11:15:00 | 14322.00 | 14606.18 | 14611.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 14322.00 | 14606.18 | 14611.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 12:15:00 | 14241.00 | 14533.14 | 14578.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 14147.00 | 14134.71 | 14265.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 14147.00 | 14134.71 | 14265.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 14207.00 | 14149.17 | 14259.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 14308.00 | 14149.17 | 14259.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 14100.00 | 14122.08 | 14203.68 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 12:15:00 | 14239.00 | 14204.23 | 14200.89 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 15:15:00 | 14111.00 | 14187.47 | 14194.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 09:15:00 | 13841.00 | 14118.18 | 14162.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 11:15:00 | 13647.00 | 13544.39 | 13646.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 11:15:00 | 13647.00 | 13544.39 | 13646.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 13647.00 | 13544.39 | 13646.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:00:00 | 13647.00 | 13544.39 | 13646.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 13696.00 | 13574.71 | 13650.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:00:00 | 13696.00 | 13574.71 | 13650.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 13724.00 | 13604.57 | 13657.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:45:00 | 13722.00 | 13604.57 | 13657.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 13672.00 | 13649.94 | 13665.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 13672.00 | 13649.94 | 13665.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 13695.00 | 13658.95 | 13668.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:30:00 | 13714.00 | 13658.95 | 13668.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 13665.00 | 13660.16 | 13668.01 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 13702.00 | 13674.58 | 13673.58 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 13607.00 | 13677.87 | 13678.32 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 13:15:00 | 13721.00 | 13686.50 | 13682.20 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 09:15:00 | 13611.00 | 13677.75 | 13679.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 14:15:00 | 13545.00 | 13626.36 | 13652.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 13658.00 | 13544.76 | 13582.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 13658.00 | 13544.76 | 13582.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 13658.00 | 13544.76 | 13582.36 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 13:15:00 | 13638.00 | 13604.84 | 13602.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 13703.00 | 13632.87 | 13617.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 11:15:00 | 13716.00 | 13725.53 | 13689.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 11:45:00 | 13711.00 | 13725.53 | 13689.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 13658.00 | 13712.02 | 13686.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 13:00:00 | 13658.00 | 13712.02 | 13686.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 13623.00 | 13694.22 | 13680.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 13623.00 | 13694.22 | 13680.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 13550.00 | 13665.38 | 13669.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 11:15:00 | 13465.00 | 13600.67 | 13635.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 13:15:00 | 13341.00 | 13277.75 | 13364.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 13:15:00 | 13341.00 | 13277.75 | 13364.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 13341.00 | 13277.75 | 13364.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:00:00 | 13341.00 | 13277.75 | 13364.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 13322.00 | 13286.60 | 13360.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 15:00:00 | 13322.00 | 13286.60 | 13360.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 13052.00 | 13246.62 | 13329.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 10:30:00 | 13040.00 | 13185.90 | 13294.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 14:45:00 | 13010.00 | 13062.49 | 13193.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:15:00 | 13039.00 | 13049.43 | 13153.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:15:00 | 12388.00 | 12589.30 | 12730.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:15:00 | 12387.05 | 12589.30 | 12730.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 15:15:00 | 12359.50 | 12534.04 | 12657.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 12554.00 | 12538.03 | 12648.06 | SL hit (close>ema200) qty=0.50 sl=12538.03 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 12554.00 | 12538.03 | 12648.06 | SL hit (close>ema200) qty=0.50 sl=12538.03 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 12554.00 | 12538.03 | 12648.06 | SL hit (close>ema200) qty=0.50 sl=12538.03 alert=retest2 |

### Cycle 34 — BUY (started 2025-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 14:15:00 | 12906.00 | 12729.51 | 12710.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 09:15:00 | 13008.00 | 12817.29 | 12755.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 11:15:00 | 13415.00 | 13426.25 | 13238.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 12:00:00 | 13415.00 | 13426.25 | 13238.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 13412.00 | 13495.82 | 13421.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 13412.00 | 13495.82 | 13421.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 13250.00 | 13446.66 | 13405.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 13250.00 | 13446.66 | 13405.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 13238.00 | 13404.93 | 13390.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:30:00 | 13231.00 | 13404.93 | 13390.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 12:15:00 | 13209.00 | 13365.74 | 13374.02 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 13374.00 | 13359.10 | 13357.10 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 10:15:00 | 13281.00 | 13343.48 | 13350.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 11:15:00 | 13225.00 | 13319.79 | 13338.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 14:15:00 | 13296.00 | 13293.62 | 13320.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 14:15:00 | 13296.00 | 13293.62 | 13320.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 13296.00 | 13293.62 | 13320.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 13296.00 | 13293.62 | 13320.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 13299.00 | 13294.69 | 13318.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 13181.00 | 13294.69 | 13318.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 13139.00 | 13263.56 | 13302.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 10:45:00 | 13096.00 | 13230.24 | 13283.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:45:00 | 13049.00 | 13055.38 | 13102.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:30:00 | 13091.00 | 13048.65 | 13083.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 14:00:00 | 13114.00 | 13077.27 | 13087.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 13089.00 | 13079.62 | 13087.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 15:15:00 | 12976.00 | 13079.62 | 13087.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:30:00 | 13011.00 | 13052.12 | 13072.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 12:45:00 | 13042.00 | 13050.48 | 13067.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 15:00:00 | 13025.00 | 13006.83 | 13025.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 13060.00 | 13017.46 | 13028.68 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 13135.00 | 13040.97 | 13038.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 13135.00 | 13040.97 | 13038.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 13135.00 | 13040.97 | 13038.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 13135.00 | 13040.97 | 13038.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 13135.00 | 13040.97 | 13038.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 13135.00 | 13040.97 | 13038.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 13135.00 | 13040.97 | 13038.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 13135.00 | 13040.97 | 13038.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 13135.00 | 13040.97 | 13038.34 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 14:15:00 | 13010.00 | 13034.78 | 13035.77 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 13083.00 | 13044.42 | 13040.06 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 11:15:00 | 13019.00 | 13033.99 | 13035.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 12926.00 | 12996.79 | 13015.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 12911.00 | 12909.58 | 12950.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 12911.00 | 12909.58 | 12950.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 12911.00 | 12909.58 | 12950.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 12911.00 | 12909.58 | 12950.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 12902.00 | 12898.20 | 12928.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 15:00:00 | 12902.00 | 12898.20 | 12928.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 12900.00 | 12898.56 | 12925.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 13040.00 | 12898.56 | 12925.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 13006.00 | 12920.05 | 12933.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 13040.00 | 12920.05 | 12933.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 13090.00 | 12954.04 | 12947.39 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 15:15:00 | 12886.00 | 12943.24 | 12946.84 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 13088.00 | 12975.48 | 12960.71 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 12:15:00 | 12896.00 | 12964.80 | 12968.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 13:15:00 | 12875.00 | 12946.84 | 12960.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 10:15:00 | 12537.00 | 12536.94 | 12678.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 11:15:00 | 12570.00 | 12536.94 | 12678.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 12670.00 | 12594.35 | 12672.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:00:00 | 12670.00 | 12594.35 | 12672.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 12641.00 | 12603.68 | 12669.42 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 11:15:00 | 12796.00 | 12697.83 | 12697.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 13:15:00 | 12864.00 | 12748.05 | 12721.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 10:15:00 | 12750.00 | 12789.45 | 12753.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 10:15:00 | 12750.00 | 12789.45 | 12753.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 12750.00 | 12789.45 | 12753.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 12705.00 | 12789.45 | 12753.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 12730.00 | 12777.56 | 12751.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:30:00 | 12723.00 | 12777.56 | 12751.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 12814.00 | 12784.85 | 12757.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:15:00 | 12702.00 | 12784.85 | 12757.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 12502.00 | 12728.28 | 12733.91 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2025-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 14:15:00 | 12816.00 | 12745.82 | 12741.37 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 12684.00 | 12738.77 | 12739.26 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 12800.00 | 12749.86 | 12743.96 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 14:15:00 | 12600.00 | 12716.87 | 12729.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 12584.00 | 12679.76 | 12710.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 13:15:00 | 12660.00 | 12653.80 | 12685.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 14:00:00 | 12660.00 | 12653.80 | 12685.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 12649.00 | 12652.84 | 12682.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:45:00 | 12696.00 | 12652.84 | 12682.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 12698.00 | 12661.87 | 12683.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 12629.00 | 12661.87 | 12683.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 10:15:00 | 12616.00 | 12613.43 | 12638.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 14:15:00 | 12867.00 | 12668.27 | 12654.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 14:15:00 | 12867.00 | 12668.27 | 12654.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 12867.00 | 12668.27 | 12654.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 11:15:00 | 12887.00 | 12788.93 | 12722.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 12789.00 | 12822.31 | 12757.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 14:15:00 | 12789.00 | 12822.31 | 12757.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 12789.00 | 12822.31 | 12757.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 12789.00 | 12822.31 | 12757.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 12884.00 | 12834.65 | 12768.67 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2025-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 13:15:00 | 12599.00 | 12738.87 | 12743.76 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 09:15:00 | 13064.00 | 12727.62 | 12708.11 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 09:15:00 | 12864.00 | 12878.01 | 12878.75 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 10:15:00 | 12918.00 | 12886.01 | 12882.32 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 12:15:00 | 12861.00 | 12880.04 | 12880.19 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 15:15:00 | 12900.00 | 12883.38 | 12881.36 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 09:15:00 | 12818.00 | 12870.31 | 12875.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 10:15:00 | 12749.00 | 12846.05 | 12864.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 12839.00 | 12815.36 | 12836.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 12839.00 | 12815.36 | 12836.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 12839.00 | 12815.36 | 12836.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 12852.00 | 12815.36 | 12836.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 12924.00 | 12837.08 | 12844.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:30:00 | 12924.00 | 12837.08 | 12844.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 12904.00 | 12850.47 | 12850.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 12969.00 | 12885.46 | 12866.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 15:15:00 | 12801.00 | 12873.93 | 12865.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 15:15:00 | 12801.00 | 12873.93 | 12865.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 12801.00 | 12873.93 | 12865.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 12977.00 | 12873.93 | 12865.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 12886.00 | 12876.35 | 12866.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 11:00:00 | 13050.00 | 12911.08 | 12883.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 12:45:00 | 13063.00 | 12964.41 | 12914.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 12:00:00 | 13052.00 | 13009.68 | 12964.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-01 13:15:00 | 14355.00 | 13730.33 | 13379.15 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-12-01 13:15:00 | 14369.30 | 13730.33 | 13379.15 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-12-01 13:15:00 | 14357.20 | 13730.33 | 13379.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 14416.00 | 14706.18 | 14709.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 15:15:00 | 14270.00 | 14451.31 | 14522.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 09:15:00 | 14246.00 | 14211.15 | 14285.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-15 10:00:00 | 14246.00 | 14211.15 | 14285.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 14266.00 | 14222.12 | 14283.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 14266.00 | 14222.12 | 14283.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 14335.00 | 14244.70 | 14288.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:45:00 | 14302.00 | 14244.70 | 14288.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 14490.00 | 14293.76 | 14306.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:00:00 | 14490.00 | 14293.76 | 14306.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 13:15:00 | 14565.00 | 14348.01 | 14329.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 14:15:00 | 14622.00 | 14402.81 | 14356.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 14:15:00 | 14750.00 | 14972.51 | 14801.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 14:15:00 | 14750.00 | 14972.51 | 14801.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 14750.00 | 14972.51 | 14801.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 14750.00 | 14972.51 | 14801.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 14703.00 | 14918.61 | 14792.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 10:15:00 | 14991.00 | 14880.09 | 14786.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 12:15:00 | 14850.00 | 14869.90 | 14799.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 13:15:00 | 14982.00 | 15138.42 | 15159.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 13:15:00 | 14982.00 | 15138.42 | 15159.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 13:15:00 | 14982.00 | 15138.42 | 15159.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 09:15:00 | 14802.00 | 15030.47 | 15101.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 14900.00 | 14856.41 | 14954.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 14900.00 | 14856.41 | 14954.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 14900.00 | 14856.41 | 14954.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 14877.00 | 14856.41 | 14954.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 14717.00 | 14776.16 | 14858.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:15:00 | 14670.00 | 14747.37 | 14800.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 14953.00 | 14802.06 | 14789.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 14953.00 | 14802.06 | 14789.30 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 10:15:00 | 14713.00 | 14798.08 | 14803.88 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 14966.00 | 14793.69 | 14790.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 13:15:00 | 15029.00 | 14888.21 | 14840.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 15:15:00 | 15060.00 | 15109.50 | 15018.51 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 09:15:00 | 15345.00 | 15109.50 | 15018.51 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 14937.00 | 15221.47 | 15160.81 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 14937.00 | 15221.47 | 15160.81 | SL hit (close<ema400) qty=1.00 sl=15160.81 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 14937.00 | 15221.47 | 15160.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 14934.00 | 15163.98 | 15140.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:30:00 | 14881.00 | 15163.98 | 15140.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 11:15:00 | 14722.00 | 15075.58 | 15102.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 14598.00 | 14766.56 | 14892.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 14150.00 | 14117.35 | 14353.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 14131.00 | 14117.35 | 14353.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 13908.00 | 14057.51 | 14155.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 13896.00 | 14030.83 | 14051.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:00:00 | 13855.00 | 13995.67 | 14033.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 13900.00 | 13783.71 | 13826.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:45:00 | 13885.00 | 13820.38 | 13835.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 12:15:00 | 13959.00 | 13848.10 | 13847.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 12:15:00 | 13959.00 | 13848.10 | 13847.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 12:15:00 | 13959.00 | 13848.10 | 13847.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 12:15:00 | 13959.00 | 13848.10 | 13847.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 12:15:00 | 13959.00 | 13848.10 | 13847.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 13:15:00 | 14033.00 | 13885.08 | 13863.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-22 14:15:00 | 13853.00 | 13878.66 | 13862.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 14:15:00 | 13853.00 | 13878.66 | 13862.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 13853.00 | 13878.66 | 13862.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 15:00:00 | 13853.00 | 13878.66 | 13862.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 13798.00 | 13862.53 | 13857.08 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 13800.00 | 13850.03 | 13851.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 10:15:00 | 13726.00 | 13825.22 | 13840.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 10:15:00 | 13643.00 | 13641.35 | 13722.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 10:30:00 | 13650.00 | 13641.35 | 13722.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 13725.00 | 13658.08 | 13722.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 11:30:00 | 13780.00 | 13658.08 | 13722.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 12:15:00 | 13757.00 | 13677.86 | 13726.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 13:00:00 | 13757.00 | 13677.86 | 13726.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 13736.00 | 13689.49 | 13726.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 13:45:00 | 13706.00 | 13689.49 | 13726.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 14:15:00 | 14065.00 | 13764.59 | 13757.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 14289.00 | 14044.05 | 13923.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 10:15:00 | 14226.00 | 14226.60 | 14111.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 10:30:00 | 14239.00 | 14226.60 | 14111.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 15093.00 | 14399.88 | 14200.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:30:00 | 15200.00 | 14921.01 | 14626.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:00:00 | 15102.00 | 15043.31 | 14866.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 15399.00 | 14995.44 | 14874.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 15157.00 | 15273.23 | 15186.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 15172.00 | 15252.99 | 15185.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 15168.00 | 15252.99 | 15185.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 15075.00 | 15217.39 | 15175.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 15075.00 | 15217.39 | 15175.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 15189.00 | 15211.71 | 15176.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-05 15:15:00 | 15011.00 | 15154.45 | 15159.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-05 15:15:00 | 15011.00 | 15154.45 | 15159.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-05 15:15:00 | 15011.00 | 15154.45 | 15159.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-05 15:15:00 | 15011.00 | 15154.45 | 15159.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 15011.00 | 15154.45 | 15159.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 13:15:00 | 14860.00 | 15059.67 | 15110.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 15001.00 | 14980.29 | 15056.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 09:45:00 | 14961.00 | 14980.29 | 15056.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 15016.00 | 14987.43 | 15053.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:30:00 | 15030.00 | 14987.43 | 15053.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 14962.00 | 14982.35 | 15044.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:30:00 | 15028.00 | 14982.35 | 15044.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 15027.00 | 14991.28 | 15043.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:00:00 | 15027.00 | 14991.28 | 15043.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 15111.00 | 15015.22 | 15049.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:45:00 | 15120.00 | 15015.22 | 15049.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 15111.00 | 15034.38 | 15054.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 15:00:00 | 15111.00 | 15034.38 | 15054.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 15164.00 | 15060.30 | 15064.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:15:00 | 15322.00 | 15060.30 | 15064.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 15360.00 | 15120.24 | 15091.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 09:15:00 | 15600.00 | 15269.29 | 15185.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 14:15:00 | 15801.00 | 15819.87 | 15654.40 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 09:15:00 | 15963.00 | 15816.50 | 15667.91 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 15922.00 | 15837.60 | 15691.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 09:30:00 | 16359.00 | 15983.52 | 15842.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 14:15:00 | 15960.00 | 16127.75 | 15985.53 | SL hit (close<ema400) qty=1.00 sl=15985.53 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 15789.00 | 15916.59 | 15927.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 13:15:00 | 15789.00 | 15916.59 | 15927.45 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 15:15:00 | 16045.00 | 15937.69 | 15934.87 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 10:15:00 | 15776.00 | 15905.56 | 15920.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 11:15:00 | 15728.00 | 15870.05 | 15903.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 12:15:00 | 15388.00 | 15361.72 | 15506.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 13:00:00 | 15388.00 | 15361.72 | 15506.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 15368.00 | 15362.98 | 15493.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:30:00 | 15499.00 | 15362.98 | 15493.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 15638.00 | 15393.60 | 15472.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 15552.00 | 15393.60 | 15472.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 15684.00 | 15451.68 | 15491.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:30:00 | 15553.00 | 15451.68 | 15491.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 15599.00 | 15532.44 | 15523.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 09:15:00 | 15843.00 | 15612.98 | 15564.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 09:15:00 | 15732.00 | 15745.80 | 15671.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:00:00 | 15732.00 | 15745.80 | 15671.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 15651.00 | 15726.84 | 15669.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:00:00 | 15651.00 | 15726.84 | 15669.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 15615.00 | 15704.47 | 15664.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:00:00 | 15615.00 | 15704.47 | 15664.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 15580.00 | 15679.58 | 15656.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 15580.00 | 15679.58 | 15656.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 13:15:00 | 15456.00 | 15634.86 | 15638.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 09:15:00 | 15412.00 | 15565.72 | 15603.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 12:15:00 | 15558.00 | 15542.88 | 15582.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-26 13:00:00 | 15558.00 | 15542.88 | 15582.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 15500.00 | 15534.31 | 15574.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 13:45:00 | 15567.00 | 15534.31 | 15574.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 15608.00 | 15539.24 | 15569.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 15340.00 | 15539.24 | 15569.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 14573.00 | 14717.28 | 14989.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 12:15:00 | 14349.00 | 14328.00 | 14544.58 | SL hit (close>ema200) qty=0.50 sl=14328.00 alert=retest2 |

### Cycle 78 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 13825.00 | 13746.96 | 13744.64 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 13545.00 | 13708.85 | 13727.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 13440.00 | 13655.08 | 13701.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-19 15:15:00 | 13572.00 | 13550.08 | 13625.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 09:15:00 | 13636.00 | 13550.08 | 13625.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 13474.00 | 13534.87 | 13611.68 | EMA400 retest candle locked (from downside) |

### Cycle 80 — BUY (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 14:15:00 | 13812.00 | 13663.27 | 13653.22 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 13421.00 | 13616.54 | 13633.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 13086.00 | 13510.43 | 13584.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 14:15:00 | 13440.00 | 13300.82 | 13441.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 14:15:00 | 13440.00 | 13300.82 | 13441.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 13440.00 | 13300.82 | 13441.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-23 15:00:00 | 13440.00 | 13300.82 | 13441.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 13061.00 | 13252.85 | 13407.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:15:00 | 13398.00 | 13252.85 | 13407.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 13387.00 | 13279.68 | 13405.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 13190.00 | 13268.15 | 13388.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 12:45:00 | 13219.00 | 13271.21 | 13369.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 11:15:00 | 13564.00 | 13390.54 | 13381.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 11:15:00 | 13564.00 | 13390.54 | 13381.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 13564.00 | 13390.54 | 13381.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 15:15:00 | 13802.00 | 13596.22 | 13524.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 13591.00 | 13595.18 | 13530.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 13591.00 | 13595.18 | 13530.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 13591.00 | 13595.18 | 13530.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 15:00:00 | 13914.00 | 13613.52 | 13557.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:30:00 | 13726.00 | 14056.58 | 14022.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:00:00 | 13800.00 | 14056.58 | 14022.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 10:15:00 | 13722.00 | 13989.66 | 13995.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 10:15:00 | 13722.00 | 13989.66 | 13995.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 10:15:00 | 13722.00 | 13989.66 | 13995.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 10:15:00 | 13722.00 | 13989.66 | 13995.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 09:15:00 | 13571.00 | 13790.92 | 13883.57 | Break + close below crossover candle low |

### Cycle 84 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 14542.00 | 13847.43 | 13843.43 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 14127.00 | 14235.34 | 14246.48 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2026-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 13:15:00 | 14365.00 | 14178.61 | 14174.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 14506.00 | 14293.13 | 14232.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 15:15:00 | 14443.00 | 14465.97 | 14361.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 15:15:00 | 14443.00 | 14465.97 | 14361.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 15:15:00 | 14443.00 | 14465.97 | 14361.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 09:45:00 | 14699.00 | 14531.38 | 14400.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 14977.00 | 15086.40 | 15093.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 14977.00 | 15086.40 | 15093.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 14858.00 | 15040.72 | 15072.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 12:15:00 | 15102.00 | 15024.38 | 15057.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 12:15:00 | 15102.00 | 15024.38 | 15057.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 15102.00 | 15024.38 | 15057.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:45:00 | 15034.00 | 15024.38 | 15057.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 14872.00 | 14993.90 | 15040.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 13:45:00 | 15158.00 | 14993.90 | 15040.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 14952.00 | 14985.52 | 15032.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 14952.00 | 14985.52 | 15032.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 14870.00 | 14962.42 | 15017.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 14985.00 | 14962.42 | 15017.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 15101.00 | 14990.14 | 15025.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 15101.00 | 14990.14 | 15025.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 15071.00 | 15006.31 | 15029.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 14985.00 | 15006.31 | 15029.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 12:00:00 | 15019.00 | 15008.85 | 15028.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:30:00 | 14977.00 | 14999.66 | 15021.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:45:00 | 15000.00 | 14938.16 | 14976.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 15000.00 | 14950.53 | 14978.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:30:00 | 15000.00 | 14950.53 | 14978.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 14961.00 | 14919.62 | 14952.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:30:00 | 14902.00 | 14919.62 | 14952.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 14919.00 | 14919.49 | 14949.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:45:00 | 14840.00 | 14895.02 | 14931.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:15:00 | 14855.00 | 14709.66 | 14769.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-19 09:15:00 | 13400.00 | 2025-05-20 15:15:00 | 13205.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-05-19 10:15:00 | 13315.00 | 2025-05-20 15:15:00 | 13205.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-05-19 14:15:00 | 13410.00 | 2025-05-20 15:15:00 | 13205.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-05-20 14:00:00 | 13291.00 | 2025-05-20 15:15:00 | 13205.00 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-05-28 12:00:00 | 12875.00 | 2025-05-30 12:15:00 | 13252.00 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2025-05-28 13:30:00 | 12843.00 | 2025-05-30 12:15:00 | 13252.00 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest1 | 2025-06-04 09:15:00 | 13594.00 | 2025-06-04 12:15:00 | 13375.00 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest1 | 2025-06-04 10:45:00 | 13620.00 | 2025-06-04 12:15:00 | 13375.00 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-06-05 09:45:00 | 13523.00 | 2025-06-10 14:15:00 | 13536.00 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2025-06-05 14:15:00 | 13515.00 | 2025-06-10 14:15:00 | 13536.00 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2025-06-06 09:15:00 | 13854.00 | 2025-06-10 14:15:00 | 13536.00 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-06-20 15:15:00 | 13110.00 | 2025-06-25 09:15:00 | 13268.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-06-24 13:30:00 | 13126.00 | 2025-06-25 09:15:00 | 13268.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-06-30 14:30:00 | 13369.00 | 2025-07-01 09:15:00 | 13280.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-09 09:45:00 | 13100.00 | 2025-07-15 10:15:00 | 13143.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-07-10 09:45:00 | 13112.00 | 2025-07-15 10:15:00 | 13143.00 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-07-10 10:45:00 | 13112.00 | 2025-07-15 10:15:00 | 13143.00 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-07-17 10:30:00 | 13242.00 | 2025-07-22 11:15:00 | 13114.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-07-18 15:15:00 | 13189.00 | 2025-07-22 11:15:00 | 13114.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-07-21 11:30:00 | 13192.00 | 2025-07-22 11:15:00 | 13114.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-07-21 12:00:00 | 13250.00 | 2025-07-22 11:15:00 | 13114.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-07-22 10:30:00 | 13261.00 | 2025-07-22 11:15:00 | 13114.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-07-30 10:15:00 | 13192.00 | 2025-07-30 13:15:00 | 13420.00 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-08-13 15:15:00 | 13650.00 | 2025-08-25 11:15:00 | 15015.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-14 10:15:00 | 13662.00 | 2025-08-25 11:15:00 | 15028.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-14 11:00:00 | 14084.00 | 2025-08-26 11:15:00 | 14322.00 | STOP_HIT | 1.00 | 1.69% |
| SELL | retest2 | 2025-09-23 10:30:00 | 13040.00 | 2025-09-29 11:15:00 | 12388.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 14:45:00 | 13010.00 | 2025-09-29 11:15:00 | 12387.05 | PARTIAL | 0.50 | 4.79% |
| SELL | retest2 | 2025-09-24 11:15:00 | 13039.00 | 2025-09-29 15:15:00 | 12359.50 | PARTIAL | 0.50 | 5.21% |
| SELL | retest2 | 2025-09-23 10:30:00 | 13040.00 | 2025-09-30 09:15:00 | 12554.00 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2025-09-23 14:45:00 | 13010.00 | 2025-09-30 09:15:00 | 12554.00 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2025-09-24 11:15:00 | 13039.00 | 2025-09-30 09:15:00 | 12554.00 | STOP_HIT | 0.50 | 3.72% |
| SELL | retest2 | 2025-10-13 10:45:00 | 13096.00 | 2025-10-21 13:15:00 | 13135.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-10-15 12:45:00 | 13049.00 | 2025-10-21 13:15:00 | 13135.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-10-16 09:30:00 | 13091.00 | 2025-10-21 13:15:00 | 13135.00 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-10-16 14:00:00 | 13114.00 | 2025-10-21 13:15:00 | 13135.00 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-10-16 15:15:00 | 12976.00 | 2025-10-21 13:15:00 | 13135.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-10-17 09:30:00 | 13011.00 | 2025-10-21 13:15:00 | 13135.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-10-17 12:45:00 | 13042.00 | 2025-10-21 13:15:00 | 13135.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-10-20 15:00:00 | 13025.00 | 2025-10-21 13:15:00 | 13135.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-11-11 09:15:00 | 12629.00 | 2025-11-12 14:15:00 | 12867.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-11-12 10:15:00 | 12616.00 | 2025-11-12 14:15:00 | 12867.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-11-27 11:00:00 | 13050.00 | 2025-12-01 13:15:00 | 14355.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-27 12:45:00 | 13063.00 | 2025-12-01 13:15:00 | 14369.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-28 12:00:00 | 13052.00 | 2025-12-01 13:15:00 | 14357.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-18 10:15:00 | 14991.00 | 2025-12-23 13:15:00 | 14982.00 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-12-18 12:15:00 | 14850.00 | 2025-12-23 13:15:00 | 14982.00 | STOP_HIT | 1.00 | 0.89% |
| SELL | retest2 | 2025-12-30 10:15:00 | 14670.00 | 2025-12-31 09:15:00 | 14953.00 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest1 | 2026-01-06 09:15:00 | 15345.00 | 2026-01-07 09:15:00 | 14937.00 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2026-01-20 09:15:00 | 13896.00 | 2026-01-22 12:15:00 | 13959.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-01-20 10:00:00 | 13855.00 | 2026-01-22 12:15:00 | 13959.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2026-01-22 10:15:00 | 13900.00 | 2026-01-22 12:15:00 | 13959.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-01-22 11:45:00 | 13885.00 | 2026-01-22 12:15:00 | 13959.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2026-02-01 12:30:00 | 15200.00 | 2026-02-05 15:15:00 | 15011.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-02-02 14:00:00 | 15102.00 | 2026-02-05 15:15:00 | 15011.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2026-02-03 09:15:00 | 15399.00 | 2026-02-05 15:15:00 | 15011.00 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2026-02-05 09:15:00 | 15157.00 | 2026-02-05 15:15:00 | 15011.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest1 | 2026-02-13 09:15:00 | 15963.00 | 2026-02-16 14:15:00 | 15960.00 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2026-02-16 09:30:00 | 16359.00 | 2026-02-17 13:15:00 | 15789.00 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2026-02-27 09:15:00 | 15340.00 | 2026-03-04 09:15:00 | 14573.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 09:15:00 | 15340.00 | 2026-03-05 12:15:00 | 14349.00 | STOP_HIT | 0.50 | 6.46% |
| SELL | retest2 | 2026-03-24 10:30:00 | 13190.00 | 2026-03-25 11:15:00 | 13564.00 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2026-03-24 12:45:00 | 13219.00 | 2026-03-25 11:15:00 | 13564.00 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2026-03-30 15:00:00 | 13914.00 | 2026-04-06 10:15:00 | 13722.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-04-06 09:30:00 | 13726.00 | 2026-04-06 10:15:00 | 13722.00 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2026-04-06 10:00:00 | 13800.00 | 2026-04-06 10:15:00 | 13722.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-04-20 09:45:00 | 14699.00 | 2026-04-24 09:15:00 | 14977.00 | STOP_HIT | 1.00 | 1.89% |
