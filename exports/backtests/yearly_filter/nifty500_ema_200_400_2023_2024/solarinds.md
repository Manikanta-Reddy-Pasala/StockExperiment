# Solar Industries India Ltd. (SOLARINDS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 16101.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 25 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 22 |
| PARTIAL | 8 |
| TARGET_HIT | 8 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 13
- **Target hits / Stop hits / Partials:** 8 / 14 / 8
- **Avg / median % per leg:** 3.16% / 5.00%
- **Sum % (uncompounded):** 94.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| SELL (all) | 29 | 16 | 55.2% | 7 | 14 | 8 | 2.93% | 84.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 16 | 55.2% | 7 | 14 | 8 | 2.93% | 84.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 30 | 17 | 56.7% | 8 | 14 | 8 | 3.16% | 94.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 15:15:00 | 3727.00 | 3800.27 | 3800.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-05 12:15:00 | 3707.00 | 3797.39 | 3799.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-20 09:15:00 | 3730.40 | 3725.98 | 3755.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-20 10:00:00 | 3730.40 | 3725.98 | 3755.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 3736.05 | 3726.96 | 3755.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-24 11:45:00 | 3720.85 | 3729.17 | 3755.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-24 13:15:00 | 3763.00 | 3729.75 | 3755.15 | SL hit (close>static) qty=1.00 sl=3759.95 alert=retest2 |

### Cycle 2 — BUY (started 2023-08-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 13:15:00 | 3876.85 | 3773.66 | 3773.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-08 09:15:00 | 3900.10 | 3785.20 | 3779.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-13 09:15:00 | 4430.15 | 4436.36 | 4218.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-13 10:00:00 | 4430.15 | 4436.36 | 4218.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 6465.45 | 6771.09 | 6466.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 8719.25 | 8273.84 | 7723.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-22 09:15:00 | 9591.18 | 8691.14 | 8335.51 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 09:15:00 | 9724.60 | 10784.47 | 10785.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 9659.95 | 10525.12 | 10640.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 10774.35 | 10321.99 | 10492.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 09:15:00 | 10774.35 | 10321.99 | 10492.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 10774.35 | 10321.99 | 10492.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:00:00 | 10774.35 | 10321.99 | 10492.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 10692.20 | 10325.67 | 10493.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 11:30:00 | 10607.50 | 10328.98 | 10494.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 13:15:00 | 10859.95 | 10338.18 | 10497.26 | SL hit (close>static) qty=1.00 sl=10848.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 09:15:00 | 10717.95 | 9615.68 | 9611.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 10:15:00 | 10763.00 | 9627.09 | 9616.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 10:15:00 | 16437.00 | 16508.38 | 15298.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 11:00:00 | 16437.00 | 16508.38 | 15298.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 15407.00 | 16425.78 | 15398.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:00:00 | 15407.00 | 16425.78 | 15398.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 15401.00 | 16415.59 | 15398.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 15294.00 | 16415.59 | 15398.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 15270.00 | 16404.19 | 15397.56 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 13:15:00 | 14535.00 | 15004.74 | 15006.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 14470.00 | 14999.42 | 15004.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 11:15:00 | 14575.00 | 14409.16 | 14639.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 12:00:00 | 14575.00 | 14409.16 | 14639.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 14503.00 | 14410.09 | 14638.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 13:30:00 | 14440.00 | 14410.36 | 14637.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 12:30:00 | 14481.00 | 14410.77 | 14631.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 13:30:00 | 14474.00 | 14411.44 | 14630.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 14:45:00 | 14493.00 | 14412.15 | 14629.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 14637.00 | 14416.47 | 14628.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:30:00 | 14635.00 | 14416.47 | 14628.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 14617.00 | 14418.47 | 14628.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:15:00 | 14632.00 | 14418.47 | 14628.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 14588.00 | 14420.16 | 14628.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:15:00 | 14571.00 | 14420.16 | 14628.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:45:00 | 14540.00 | 14421.37 | 14628.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 14862.00 | 14429.30 | 14629.02 | SL hit (close>static) qty=1.00 sl=14669.00 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 15:15:00 | 13241.00 | 13128.09 | 13127.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 13288.00 | 13132.72 | 13130.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 10:15:00 | 13825.00 | 13894.43 | 13596.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:30:00 | 13793.00 | 13894.43 | 13596.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 13329.00 | 13885.90 | 13610.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 13329.00 | 13885.90 | 13610.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 13296.00 | 13880.03 | 13609.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 13296.00 | 13880.03 | 13609.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 10:15:00 | 12529.00 | 13399.38 | 13402.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 12456.00 | 13365.29 | 13384.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 13330.00 | 13329.19 | 13364.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 15:00:00 | 13330.00 | 13329.19 | 13364.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 13290.00 | 13328.92 | 13364.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:30:00 | 13270.00 | 13328.92 | 13364.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 13340.00 | 13329.03 | 13364.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:00:00 | 13340.00 | 13329.03 | 13364.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 13382.00 | 13329.55 | 13364.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:45:00 | 13395.00 | 13329.55 | 13364.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 13408.00 | 13330.34 | 13364.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:45:00 | 13395.00 | 13330.34 | 13364.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 09:15:00 | 14114.00 | 13397.06 | 13395.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 13:15:00 | 14510.00 | 13475.13 | 13435.99 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-07-24 11:45:00 | 3720.85 | 2023-07-24 13:15:00 | 3763.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-04-12 09:15:00 | 8719.25 | 2024-05-22 09:15:00 | 9591.18 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-28 11:30:00 | 10607.50 | 2024-11-28 13:15:00 | 10859.95 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2024-12-02 12:15:00 | 10636.00 | 2024-12-03 09:15:00 | 10849.95 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2024-12-02 13:15:00 | 10636.05 | 2024-12-03 09:15:00 | 10849.95 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2024-12-02 13:45:00 | 10634.55 | 2024-12-03 09:15:00 | 10849.95 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-12-13 11:30:00 | 10384.10 | 2024-12-20 14:15:00 | 9864.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 10:45:00 | 10414.45 | 2024-12-20 14:15:00 | 9893.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 11:45:00 | 10420.00 | 2024-12-20 14:15:00 | 9899.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 11:45:00 | 10417.05 | 2024-12-20 14:15:00 | 9896.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 10:00:00 | 10418.00 | 2024-12-20 14:15:00 | 9897.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 11:30:00 | 10384.10 | 2025-01-13 09:15:00 | 9345.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-16 10:45:00 | 10414.45 | 2025-01-13 09:15:00 | 9373.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-16 11:45:00 | 10420.00 | 2025-01-13 09:15:00 | 9378.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-17 11:45:00 | 10417.05 | 2025-01-13 09:15:00 | 9375.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-18 10:00:00 | 10418.00 | 2025-01-13 09:15:00 | 9376.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-21 13:30:00 | 10449.75 | 2025-03-21 14:15:00 | 10687.20 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-09-12 13:30:00 | 14440.00 | 2025-09-17 09:15:00 | 14862.00 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-09-15 12:30:00 | 14481.00 | 2025-09-17 09:15:00 | 14862.00 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2025-09-15 13:30:00 | 14474.00 | 2025-09-17 09:15:00 | 14862.00 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-09-15 14:45:00 | 14493.00 | 2025-09-17 09:15:00 | 14862.00 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-09-16 13:15:00 | 14571.00 | 2025-09-17 09:15:00 | 14862.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-09-16 13:45:00 | 14540.00 | 2025-09-17 09:15:00 | 14862.00 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-09-18 10:15:00 | 14571.00 | 2025-09-18 10:15:00 | 14683.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-09-22 11:00:00 | 14537.00 | 2025-09-26 13:15:00 | 13810.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 11:00:00 | 14537.00 | 2025-10-07 10:15:00 | 14180.00 | STOP_HIT | 0.50 | 2.46% |
| SELL | retest2 | 2025-11-12 14:15:00 | 14078.00 | 2025-11-24 12:15:00 | 13374.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 15:00:00 | 14080.00 | 2025-11-24 12:15:00 | 13376.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 14:15:00 | 14078.00 | 2025-12-08 09:15:00 | 12670.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-12 15:00:00 | 14080.00 | 2025-12-08 09:15:00 | 12672.00 | TARGET_HIT | 0.50 | 10.00% |
