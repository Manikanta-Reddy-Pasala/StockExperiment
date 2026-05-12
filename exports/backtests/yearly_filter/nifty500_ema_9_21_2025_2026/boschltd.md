# Bosch Ltd. (BOSCHLTD)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 38050.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 73 |
| ALERT1 | 52 |
| ALERT2 | 51 |
| ALERT2_SKIP | 28 |
| ALERT3 | 137 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 74 |
| PARTIAL | 6 |
| TARGET_HIT | 5 |
| STOP_HIT | 71 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 82 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 56
- **Target hits / Stop hits / Partials:** 5 / 71 / 6
- **Avg / median % per leg:** 0.37% / -0.69%
- **Sum % (uncompounded):** 30.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 13 | 41.9% | 5 | 26 | 0 | 1.21% | 37.4% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | 0.06% | 0.1% |
| BUY @ 3rd Alert (retest2) | 29 | 12 | 41.4% | 5 | 24 | 0 | 1.29% | 37.3% |
| SELL (all) | 51 | 13 | 25.5% | 0 | 45 | 6 | -0.13% | -6.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 51 | 13 | 25.5% | 0 | 45 | 6 | -0.13% | -6.8% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 2 | 0 | 0.06% | 0.1% |
| retest2 (combined) | 80 | 25 | 31.2% | 5 | 69 | 6 | 0.38% | 30.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 30885.00 | 30209.99 | 30182.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 30980.00 | 30532.80 | 30352.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 14:15:00 | 31380.00 | 31391.35 | 31121.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 14:45:00 | 31355.00 | 31391.35 | 31121.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 31550.00 | 31681.90 | 31591.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 13:00:00 | 31550.00 | 31681.90 | 31591.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 31505.00 | 31646.52 | 31583.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 31505.00 | 31646.52 | 31583.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 31625.00 | 31629.17 | 31585.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 31620.00 | 31624.34 | 31587.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 31815.00 | 31662.47 | 31607.99 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 31460.00 | 31589.07 | 31593.02 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 31760.00 | 31623.25 | 31608.20 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 31490.00 | 31610.99 | 31614.55 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 14:15:00 | 31740.00 | 31614.76 | 31610.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 15:15:00 | 31840.00 | 31659.80 | 31631.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 12:15:00 | 31975.00 | 32086.66 | 31941.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 12:15:00 | 31975.00 | 32086.66 | 31941.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 31975.00 | 32086.66 | 31941.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:45:00 | 31940.00 | 32086.66 | 31941.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 32490.00 | 32167.33 | 31991.03 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 10:15:00 | 31370.00 | 32067.85 | 32118.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 31190.00 | 31556.53 | 31755.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 14:15:00 | 31410.00 | 31315.89 | 31492.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-30 15:00:00 | 31410.00 | 31315.89 | 31492.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 31385.00 | 31329.71 | 31482.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 10:00:00 | 31175.00 | 31298.77 | 31454.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 11:15:00 | 31255.00 | 31295.01 | 31438.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 12:45:00 | 31260.00 | 31300.61 | 31416.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 15:15:00 | 31210.00 | 31314.39 | 31403.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 31375.00 | 31309.81 | 31384.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:30:00 | 31215.00 | 31289.48 | 31362.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 10:15:00 | 31605.00 | 31375.94 | 31372.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 31605.00 | 31375.94 | 31372.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 11:15:00 | 31780.00 | 31456.75 | 31409.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 14:15:00 | 31440.00 | 31527.30 | 31460.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 14:15:00 | 31440.00 | 31527.30 | 31460.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 31440.00 | 31527.30 | 31460.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 31440.00 | 31527.30 | 31460.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 31330.00 | 31487.84 | 31448.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:15:00 | 31285.00 | 31487.84 | 31448.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 31350.00 | 31460.27 | 31439.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 31745.00 | 31557.76 | 31515.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 11:00:00 | 31700.00 | 31623.36 | 31555.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 12:15:00 | 31700.00 | 31623.69 | 31561.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 13:30:00 | 31685.00 | 31637.16 | 31578.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 31755.00 | 31670.19 | 31609.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:45:00 | 31680.00 | 31670.19 | 31609.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 31655.00 | 31733.82 | 31659.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:30:00 | 31685.00 | 31733.82 | 31659.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 31655.00 | 31718.05 | 31658.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:15:00 | 31595.00 | 31718.05 | 31658.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 31560.00 | 31686.44 | 31649.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 31560.00 | 31686.44 | 31649.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 31490.00 | 31647.15 | 31635.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 31760.00 | 31647.15 | 31635.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 14:15:00 | 31610.00 | 31787.51 | 31754.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 15:15:00 | 31500.00 | 31687.21 | 31712.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 31500.00 | 31687.21 | 31712.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 31260.00 | 31601.76 | 31671.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 31605.00 | 31510.30 | 31587.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 14:15:00 | 31605.00 | 31510.30 | 31587.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 31605.00 | 31510.30 | 31587.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 31605.00 | 31510.30 | 31587.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 31600.00 | 31528.24 | 31588.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:30:00 | 31485.00 | 31548.59 | 31592.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 10:15:00 | 31970.00 | 31632.87 | 31626.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 10:15:00 | 31970.00 | 31632.87 | 31626.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 11:15:00 | 32020.00 | 31710.30 | 31662.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 12:15:00 | 32060.00 | 32138.44 | 31972.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 13:00:00 | 32060.00 | 32138.44 | 31972.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 32270.00 | 32416.97 | 32294.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 32270.00 | 32416.97 | 32294.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 32350.00 | 32403.58 | 32299.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 15:15:00 | 32400.00 | 32387.86 | 32301.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 31600.00 | 32170.92 | 32246.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 09:15:00 | 31600.00 | 32170.92 | 32246.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 15:15:00 | 31540.00 | 31772.94 | 31983.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 09:15:00 | 31750.00 | 31646.00 | 31790.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 09:15:00 | 31750.00 | 31646.00 | 31790.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 31750.00 | 31646.00 | 31790.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:45:00 | 31890.00 | 31646.00 | 31790.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 31785.00 | 31673.80 | 31790.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 11:00:00 | 31785.00 | 31673.80 | 31790.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 31790.00 | 31697.04 | 31790.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:00:00 | 31790.00 | 31697.04 | 31790.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 31750.00 | 31707.63 | 31786.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 14:45:00 | 31650.00 | 31717.88 | 31777.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 09:30:00 | 31600.00 | 31695.25 | 31757.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 12:45:00 | 31645.00 | 31719.69 | 31753.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 13:15:00 | 31840.00 | 31743.75 | 31761.39 | SL hit (close>static) qty=1.00 sl=31830.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 14:15:00 | 32240.00 | 31843.00 | 31804.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 32335.00 | 31982.52 | 31877.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 32475.00 | 32562.18 | 32378.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 10:00:00 | 32475.00 | 32562.18 | 32378.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 32175.00 | 32484.74 | 32359.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 32175.00 | 32484.74 | 32359.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 32280.00 | 32443.80 | 32352.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:15:00 | 32305.00 | 32443.80 | 32352.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 13:30:00 | 32310.00 | 32390.43 | 32342.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 12:15:00 | 32305.00 | 32342.95 | 32334.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 14:00:00 | 32295.00 | 32333.69 | 32331.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 32455.00 | 32357.95 | 32342.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 32615.00 | 32381.36 | 32354.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-04 09:15:00 | 35535.50 | 34138.52 | 33421.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 38035.00 | 38254.65 | 38273.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 10:15:00 | 37765.00 | 38156.72 | 38226.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 37940.00 | 37793.71 | 37910.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 14:15:00 | 37940.00 | 37793.71 | 37910.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 37940.00 | 37793.71 | 37910.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:45:00 | 37975.00 | 37793.71 | 37910.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 37980.00 | 37830.97 | 37917.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 38090.00 | 37830.97 | 37917.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 38005.00 | 37942.93 | 37949.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:15:00 | 37910.00 | 37941.35 | 37948.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 37805.00 | 37881.65 | 37919.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 38170.00 | 37881.89 | 37884.84 | SL hit (close>static) qty=1.00 sl=38095.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 10:15:00 | 38245.00 | 37954.51 | 37917.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 12:15:00 | 39550.00 | 38416.88 | 38187.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 39760.00 | 39801.11 | 39307.00 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 10:30:00 | 40050.00 | 39843.88 | 39371.36 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 40155.00 | 40280.11 | 40048.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 40155.00 | 40280.11 | 40048.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 40600.00 | 40848.13 | 40525.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-05 10:15:00 | 40420.00 | 40762.50 | 40515.92 | SL hit (close<ema400) qty=1.00 sl=40515.92 alert=retest1 |

### Cycle 14 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 38685.00 | 40285.50 | 40395.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 11:15:00 | 38520.00 | 39662.72 | 40076.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 38690.00 | 38579.90 | 39015.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 09:15:00 | 38940.00 | 38579.90 | 39015.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 38850.00 | 38633.92 | 39000.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 11:15:00 | 38505.00 | 38625.14 | 38962.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 11:45:00 | 38540.00 | 38609.11 | 38924.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:30:00 | 38530.00 | 38554.34 | 38823.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 13:15:00 | 38550.00 | 38689.80 | 38727.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 38655.00 | 38682.84 | 38721.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:45:00 | 38760.00 | 38682.84 | 38721.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 38725.00 | 38691.27 | 38721.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 38725.00 | 38691.27 | 38721.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 38700.00 | 38693.02 | 38719.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 38790.00 | 38693.02 | 38719.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 38545.00 | 38663.41 | 38703.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 11:00:00 | 38470.00 | 38624.73 | 38682.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 12:00:00 | 38490.00 | 38597.78 | 38665.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 15:15:00 | 38490.00 | 38618.23 | 38660.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 39580.00 | 38790.06 | 38729.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 39580.00 | 38790.06 | 38729.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 39665.00 | 38965.05 | 38814.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 15:15:00 | 39890.00 | 39901.00 | 39692.84 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:15:00 | 40000.00 | 39901.00 | 39692.84 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 39680.00 | 39845.31 | 39717.22 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-21 11:15:00 | 39680.00 | 39845.31 | 39717.22 | SL hit (close<ema400) qty=1.00 sl=39717.22 alert=retest1 |

### Cycle 16 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 39500.00 | 39659.65 | 39660.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 39230.00 | 39518.98 | 39591.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 39510.00 | 39421.34 | 39505.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 10:15:00 | 39510.00 | 39421.34 | 39505.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 39510.00 | 39421.34 | 39505.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 39510.00 | 39421.34 | 39505.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 39485.00 | 39434.07 | 39503.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:15:00 | 39130.00 | 39435.81 | 39492.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 14:45:00 | 39340.00 | 39288.22 | 39343.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 09:15:00 | 39785.00 | 39407.06 | 39389.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 09:15:00 | 39785.00 | 39407.06 | 39389.15 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 15:15:00 | 39205.00 | 39399.95 | 39409.96 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 10:15:00 | 39825.00 | 39472.97 | 39440.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 11:15:00 | 40145.00 | 39607.37 | 39504.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 41225.00 | 41276.00 | 40904.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 12:45:00 | 41185.00 | 41276.00 | 40904.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 41115.00 | 41206.63 | 41006.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:45:00 | 41160.00 | 41206.63 | 41006.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 40970.00 | 41159.30 | 41002.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:45:00 | 40945.00 | 41159.30 | 41002.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 40785.00 | 41084.44 | 40983.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 40785.00 | 41084.44 | 40983.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 40605.00 | 40988.55 | 40948.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 40605.00 | 40988.55 | 40948.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 40800.00 | 40906.27 | 40915.47 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 10:15:00 | 41010.00 | 40929.22 | 40924.44 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 40895.00 | 40920.90 | 40921.39 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 40950.00 | 40926.72 | 40923.99 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 14:15:00 | 40890.00 | 40919.37 | 40920.90 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 41465.00 | 41025.40 | 40968.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 10:15:00 | 41625.00 | 41145.32 | 41028.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 41215.00 | 41294.92 | 41174.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 10:00:00 | 41215.00 | 41294.92 | 41174.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 41275.00 | 41290.94 | 41183.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 13:15:00 | 41380.00 | 41273.60 | 41192.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 10:45:00 | 41455.00 | 41356.56 | 41277.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 12:15:00 | 41010.00 | 41263.00 | 41247.45 | SL hit (close<static) qty=1.00 sl=41145.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 13:15:00 | 41045.00 | 41219.40 | 41229.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 09:15:00 | 40735.00 | 41068.93 | 41152.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 39805.00 | 39664.51 | 40005.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 10:15:00 | 39955.00 | 39722.61 | 40000.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 39955.00 | 39722.61 | 40000.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 39955.00 | 39722.61 | 40000.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 39860.00 | 39786.47 | 39983.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 14:30:00 | 39765.00 | 39885.78 | 39942.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:45:00 | 39775.00 | 39843.58 | 39886.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 11:15:00 | 39735.00 | 39777.73 | 39810.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 12:00:00 | 39750.00 | 39772.18 | 39804.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 39440.00 | 39610.63 | 39706.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 39160.00 | 39513.05 | 39606.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:15:00 | 37776.75 | 38126.80 | 38471.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:15:00 | 37786.25 | 38126.80 | 38471.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:15:00 | 37748.25 | 38126.80 | 38471.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:15:00 | 37762.50 | 38126.80 | 38471.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 13:15:00 | 38540.00 | 38193.15 | 38440.43 | SL hit (close>ema200) qty=0.50 sl=38193.15 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 13:15:00 | 38325.00 | 38261.99 | 38261.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 38650.00 | 38339.59 | 38296.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 12:15:00 | 38765.00 | 38795.70 | 38629.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 12:45:00 | 38820.00 | 38795.70 | 38629.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 38775.00 | 38801.27 | 38685.69 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 38300.00 | 38571.41 | 38606.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 38245.00 | 38464.30 | 38549.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 38460.00 | 38397.15 | 38477.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 14:15:00 | 38460.00 | 38397.15 | 38477.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 38460.00 | 38397.15 | 38477.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 38460.00 | 38397.15 | 38477.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 38495.00 | 38416.72 | 38479.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 38565.00 | 38416.72 | 38479.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 38790.00 | 38491.38 | 38507.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 38795.00 | 38491.38 | 38507.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 38805.00 | 38554.10 | 38534.67 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 15:15:00 | 38450.00 | 38564.85 | 38576.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 37900.00 | 38431.88 | 38514.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 38325.00 | 38101.95 | 38246.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 38325.00 | 38101.95 | 38246.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 38325.00 | 38101.95 | 38246.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 38265.00 | 38101.95 | 38246.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 38265.00 | 38134.56 | 38248.28 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 38425.00 | 38317.36 | 38309.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 38525.00 | 38367.31 | 38334.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 11:15:00 | 38390.00 | 38401.08 | 38357.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 12:00:00 | 38390.00 | 38401.08 | 38357.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 38550.00 | 38430.86 | 38374.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:30:00 | 38375.00 | 38430.86 | 38374.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 38370.00 | 38443.08 | 38396.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 38445.00 | 38443.08 | 38396.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 38490.00 | 38452.47 | 38405.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 10:45:00 | 38545.00 | 38475.97 | 38420.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 13:45:00 | 38600.00 | 38496.34 | 38443.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 10:15:00 | 38570.00 | 38934.80 | 38953.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 38570.00 | 38934.80 | 38953.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 38475.00 | 38783.67 | 38877.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 10:15:00 | 38650.00 | 38628.43 | 38752.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 11:00:00 | 38650.00 | 38628.43 | 38752.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 38755.00 | 38653.74 | 38752.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:45:00 | 38720.00 | 38653.74 | 38752.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 38790.00 | 38680.99 | 38755.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:45:00 | 38820.00 | 38680.99 | 38755.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 38845.00 | 38713.79 | 38763.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:45:00 | 38865.00 | 38713.79 | 38763.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 39025.00 | 38776.03 | 38787.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 15:00:00 | 39025.00 | 38776.03 | 38787.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 15:15:00 | 39000.00 | 38820.83 | 38806.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 39150.00 | 38886.66 | 38838.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 38750.00 | 38887.86 | 38848.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 11:15:00 | 38750.00 | 38887.86 | 38848.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 38750.00 | 38887.86 | 38848.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 38750.00 | 38887.86 | 38848.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 12:15:00 | 38550.00 | 38820.29 | 38821.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 13:15:00 | 38500.00 | 38756.23 | 38792.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 12:15:00 | 37260.00 | 37197.71 | 37468.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 13:00:00 | 37260.00 | 37197.71 | 37468.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 37315.00 | 37054.12 | 37135.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:00:00 | 37315.00 | 37054.12 | 37135.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 37650.00 | 37173.30 | 37182.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:00:00 | 37650.00 | 37173.30 | 37182.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 14:15:00 | 37865.00 | 37311.64 | 37244.19 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 36900.00 | 37171.65 | 37200.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 36670.00 | 37021.27 | 37113.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 09:15:00 | 36920.00 | 36856.14 | 36959.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 09:15:00 | 36920.00 | 36856.14 | 36959.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 36920.00 | 36856.14 | 36959.36 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 37295.00 | 37033.67 | 37022.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 37470.00 | 37120.93 | 37063.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 10:15:00 | 36645.00 | 37081.54 | 37070.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 10:15:00 | 36645.00 | 37081.54 | 37070.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 36645.00 | 37081.54 | 37070.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 36645.00 | 37081.54 | 37070.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 11:15:00 | 36900.00 | 37045.23 | 37055.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 09:15:00 | 36235.00 | 36701.43 | 36867.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 13:15:00 | 37075.00 | 36648.74 | 36775.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 13:15:00 | 37075.00 | 36648.74 | 36775.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 37075.00 | 36648.74 | 36775.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 37075.00 | 36648.74 | 36775.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 37260.00 | 36770.99 | 36819.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 37260.00 | 36770.99 | 36819.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 37325.00 | 36944.03 | 36893.70 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 36800.00 | 36987.23 | 37001.10 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 37065.00 | 36995.97 | 36995.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 11:15:00 | 37190.00 | 37034.78 | 37013.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 36900.00 | 37176.60 | 37111.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 36900.00 | 37176.60 | 37111.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 36900.00 | 37176.60 | 37111.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 36900.00 | 37176.60 | 37111.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 36840.00 | 37109.28 | 37087.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:45:00 | 36865.00 | 37109.28 | 37087.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 36905.00 | 37068.43 | 37070.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 10:15:00 | 36720.00 | 36959.75 | 37014.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 37090.00 | 36919.44 | 36958.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 37090.00 | 36919.44 | 36958.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 37090.00 | 36919.44 | 36958.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 36965.00 | 36919.44 | 36958.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 37090.00 | 36953.55 | 36970.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 37090.00 | 36953.55 | 36970.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 37060.00 | 36989.02 | 36983.64 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 36790.00 | 36970.78 | 36978.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 12:15:00 | 36660.00 | 36849.84 | 36916.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 36175.00 | 35967.32 | 36185.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 36175.00 | 35967.32 | 36185.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 36175.00 | 35967.32 | 36185.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 36175.00 | 35967.32 | 36185.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 36170.00 | 36007.85 | 36184.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 36170.00 | 36007.85 | 36184.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 36295.00 | 36065.28 | 36194.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:45:00 | 36315.00 | 36065.28 | 36194.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 36325.00 | 36117.23 | 36206.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:30:00 | 36345.00 | 36117.23 | 36206.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 36295.00 | 36264.62 | 36261.33 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 36180.00 | 36250.96 | 36255.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 11:15:00 | 36135.00 | 36227.77 | 36244.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 13:15:00 | 36215.00 | 36197.57 | 36226.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 13:15:00 | 36215.00 | 36197.57 | 36226.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 36215.00 | 36197.57 | 36226.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:00:00 | 36215.00 | 36197.57 | 36226.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 36300.00 | 36218.06 | 36233.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 36300.00 | 36218.06 | 36233.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 36300.00 | 36234.44 | 36239.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 36240.00 | 36234.44 | 36239.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 10:15:00 | 36295.00 | 36251.44 | 36246.45 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 36135.00 | 36228.16 | 36236.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 15:15:00 | 36075.00 | 36177.44 | 36208.82 | Break + close below crossover candle low |

### Cycle 49 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 36445.00 | 36230.96 | 36230.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 36660.00 | 36368.36 | 36301.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 36350.00 | 36543.80 | 36451.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 36350.00 | 36543.80 | 36451.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 36350.00 | 36543.80 | 36451.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 36350.00 | 36543.80 | 36451.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 36215.00 | 36478.04 | 36430.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 36215.00 | 36478.04 | 36430.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 36525.00 | 36463.03 | 36432.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:45:00 | 36490.00 | 36463.03 | 36432.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 36615.00 | 36493.43 | 36448.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:45:00 | 36510.00 | 36493.43 | 36448.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 36880.00 | 36846.06 | 36710.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 10:15:00 | 37000.00 | 36846.06 | 36710.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 10:15:00 | 36565.00 | 36782.09 | 36757.86 | SL hit (close<static) qty=1.00 sl=36595.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 36420.00 | 36685.39 | 36717.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 10:15:00 | 36385.00 | 36560.90 | 36643.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 36705.00 | 36589.72 | 36649.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 36705.00 | 36589.72 | 36649.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 36705.00 | 36589.72 | 36649.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 36705.00 | 36589.72 | 36649.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 36725.00 | 36616.78 | 36656.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:15:00 | 36670.00 | 36650.42 | 36668.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 15:15:00 | 36840.00 | 36709.87 | 36693.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 36840.00 | 36709.87 | 36693.36 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 10:15:00 | 36525.00 | 36680.92 | 36683.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 36405.00 | 36570.30 | 36625.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 36800.00 | 36592.99 | 36624.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 36800.00 | 36592.99 | 36624.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 36800.00 | 36592.99 | 36624.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:00:00 | 36800.00 | 36592.99 | 36624.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 36730.00 | 36620.40 | 36634.34 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 36865.00 | 36669.32 | 36655.31 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 12:15:00 | 36400.00 | 36639.14 | 36670.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 36120.00 | 36409.53 | 36518.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 14:15:00 | 36245.00 | 36237.27 | 36374.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 15:00:00 | 36245.00 | 36237.27 | 36374.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 36270.00 | 36242.65 | 36353.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:00:00 | 36025.00 | 36189.90 | 36309.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:15:00 | 35940.00 | 35881.93 | 35905.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 14:15:00 | 36100.00 | 35935.38 | 35920.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 14:15:00 | 36100.00 | 35935.38 | 35920.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 14:15:00 | 36300.00 | 36104.93 | 36021.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 14:15:00 | 36190.00 | 36251.27 | 36155.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 14:15:00 | 36190.00 | 36251.27 | 36155.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 36190.00 | 36251.27 | 36155.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:45:00 | 36205.00 | 36251.27 | 36155.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 36150.00 | 36231.02 | 36155.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 36215.00 | 36231.02 | 36155.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 36110.00 | 36206.81 | 36151.26 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 35855.00 | 36077.82 | 36104.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 35690.00 | 35917.04 | 36017.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 35570.00 | 35496.00 | 35679.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 15:00:00 | 35570.00 | 35496.00 | 35679.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 35745.00 | 35545.80 | 35685.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 35880.00 | 35545.80 | 35685.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 35825.00 | 35601.64 | 35698.10 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 36000.00 | 35778.65 | 35767.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 37215.00 | 36261.49 | 36050.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 38935.00 | 39078.72 | 38588.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 38935.00 | 39078.72 | 38588.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 38580.00 | 38943.18 | 38645.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:45:00 | 38220.00 | 38943.18 | 38645.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 38460.00 | 38846.55 | 38628.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 38460.00 | 38846.55 | 38628.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 38630.00 | 38803.24 | 38628.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 12:30:00 | 38900.00 | 38869.59 | 38674.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 13:00:00 | 38765.00 | 38787.56 | 38740.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 14:30:00 | 38770.00 | 38758.04 | 38734.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 15:15:00 | 38545.00 | 38715.43 | 38717.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 38545.00 | 38715.43 | 38717.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 38360.00 | 38630.51 | 38675.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 37610.00 | 37571.96 | 37969.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 37745.00 | 37571.96 | 37969.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 37420.00 | 37554.05 | 37893.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:30:00 | 37200.00 | 37614.01 | 37746.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 35340.00 | 35771.47 | 36288.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 35745.00 | 35324.44 | 35736.88 | SL hit (close>ema200) qty=0.50 sl=35324.44 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 35710.00 | 35448.59 | 35417.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 36200.00 | 35598.87 | 35488.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 36230.00 | 36291.61 | 36099.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:00:00 | 36230.00 | 36291.61 | 36099.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 36090.00 | 36251.29 | 36099.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 35990.00 | 36251.29 | 36099.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 36100.00 | 36221.03 | 36099.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 14:15:00 | 36195.00 | 36221.03 | 36099.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 11:30:00 | 36185.00 | 36119.93 | 36088.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 12:30:00 | 36140.00 | 36164.94 | 36112.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 36310.00 | 36911.02 | 36939.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 36310.00 | 36911.02 | 36939.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 15:15:00 | 36100.00 | 36447.26 | 36659.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 36440.00 | 36174.27 | 36389.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 14:15:00 | 36440.00 | 36174.27 | 36389.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 36440.00 | 36174.27 | 36389.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 36440.00 | 36174.27 | 36389.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 36275.00 | 36194.41 | 36379.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 36385.00 | 36194.41 | 36379.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 36250.00 | 36205.53 | 36367.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 10:45:00 | 36105.00 | 36194.42 | 36347.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 11:15:00 | 36075.00 | 36194.42 | 36347.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 11:45:00 | 36085.00 | 36159.54 | 36317.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 13:30:00 | 35950.00 | 36080.31 | 36254.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 35695.00 | 35713.72 | 35901.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-11 13:15:00 | 36470.00 | 36060.61 | 36021.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 13:15:00 | 36470.00 | 36060.61 | 36021.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 14:15:00 | 36635.00 | 36175.49 | 36077.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 35900.00 | 36364.31 | 36284.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 35900.00 | 36364.31 | 36284.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 35900.00 | 36364.31 | 36284.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 36000.00 | 36364.31 | 36284.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 35890.00 | 36269.45 | 36248.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:15:00 | 35850.00 | 36269.45 | 36248.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 11:15:00 | 35990.00 | 36213.56 | 36224.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 35715.00 | 36055.22 | 36145.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 13:15:00 | 35540.00 | 35390.56 | 35513.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 13:15:00 | 35540.00 | 35390.56 | 35513.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 35540.00 | 35390.56 | 35513.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:00:00 | 35540.00 | 35390.56 | 35513.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 35710.00 | 35454.45 | 35531.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 35710.00 | 35454.45 | 35531.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 35760.00 | 35515.56 | 35551.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 35835.00 | 35515.56 | 35551.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 35475.00 | 35532.57 | 35552.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:30:00 | 35615.00 | 35532.57 | 35552.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 35510.00 | 35528.05 | 35548.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:00:00 | 35510.00 | 35528.05 | 35548.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 35385.00 | 35229.78 | 35372.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:00:00 | 35385.00 | 35229.78 | 35372.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 35640.00 | 35311.83 | 35396.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 35640.00 | 35311.83 | 35396.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 35395.00 | 35328.46 | 35396.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 35605.00 | 35328.46 | 35396.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 35260.00 | 35314.77 | 35384.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:45:00 | 35170.00 | 35269.79 | 35338.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:45:00 | 35170.00 | 35252.83 | 35324.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:00:00 | 35170.00 | 35236.26 | 35310.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 35085.00 | 35264.77 | 35306.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 35080.00 | 35227.81 | 35285.81 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 36035.00 | 35363.00 | 35301.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 36035.00 | 35363.00 | 35301.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 10:15:00 | 37250.00 | 35740.40 | 35479.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 36315.00 | 36363.61 | 36003.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 10:45:00 | 36215.00 | 36363.61 | 36003.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 36160.00 | 36459.26 | 36222.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 36160.00 | 36459.26 | 36222.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 36055.00 | 36378.41 | 36207.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:00:00 | 36055.00 | 36378.41 | 36207.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 36125.00 | 36327.73 | 36199.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:30:00 | 36035.00 | 36327.73 | 36199.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 36200.00 | 36302.18 | 36199.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:45:00 | 36330.00 | 36299.75 | 36208.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:15:00 | 36300.00 | 36299.75 | 36208.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:45:00 | 36450.00 | 36328.80 | 36229.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 35840.00 | 36242.43 | 36208.29 | SL hit (close<static) qty=1.00 sl=36125.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 35600.00 | 36113.94 | 36153.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 35295.00 | 35950.16 | 36075.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 11:15:00 | 33265.00 | 33260.88 | 33800.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 11:45:00 | 33315.00 | 33260.88 | 33800.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 32030.00 | 32018.70 | 32338.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 11:00:00 | 31935.00 | 32001.96 | 32302.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 14:15:00 | 30338.25 | 30711.81 | 31166.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 30370.00 | 30305.66 | 30688.98 | SL hit (close>ema200) qty=0.50 sl=30305.66 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 31035.00 | 30736.84 | 30721.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 31125.00 | 30867.78 | 30787.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 30315.00 | 30821.00 | 30800.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 30315.00 | 30821.00 | 30800.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 30315.00 | 30821.00 | 30800.06 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 30315.00 | 30719.80 | 30755.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 30240.00 | 30623.84 | 30709.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 30625.00 | 30422.66 | 30553.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 30625.00 | 30422.66 | 30553.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 30625.00 | 30422.66 | 30553.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 30625.00 | 30422.66 | 30553.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 30670.00 | 30472.12 | 30564.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 30670.00 | 30472.12 | 30564.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 30585.00 | 30494.70 | 30565.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 30680.00 | 30494.70 | 30565.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 30680.00 | 30531.76 | 30576.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:00:00 | 30680.00 | 30531.76 | 30576.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 30620.00 | 30549.41 | 30580.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:30:00 | 30640.00 | 30549.41 | 30580.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 30405.00 | 30520.53 | 30564.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 29790.00 | 30504.42 | 30553.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 30470.00 | 29979.33 | 29976.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 30470.00 | 29979.33 | 29976.65 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 29985.00 | 30105.86 | 30107.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 29600.00 | 29981.40 | 30048.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 30530.00 | 29358.09 | 29552.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 30530.00 | 29358.09 | 29552.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 30530.00 | 29358.09 | 29552.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 30530.00 | 29358.09 | 29552.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 30825.00 | 29651.47 | 29668.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 31030.00 | 29651.47 | 29668.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 30870.00 | 29895.18 | 29777.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 11:15:00 | 31515.00 | 30616.24 | 30259.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 32740.00 | 32849.90 | 32110.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 32740.00 | 32849.90 | 32110.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 37040.00 | 37168.98 | 36475.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 37240.00 | 37106.18 | 36510.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 10:30:00 | 37240.00 | 37056.61 | 36862.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:30:00 | 37265.00 | 37139.56 | 36962.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 37485.00 | 37904.28 | 37923.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 37485.00 | 37904.28 | 37923.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 37405.00 | 37673.44 | 37797.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 37330.00 | 36910.62 | 37195.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 37330.00 | 36910.62 | 37195.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 37330.00 | 36910.62 | 37195.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 37330.00 | 36910.62 | 37195.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 37190.00 | 36966.50 | 37195.32 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 12:15:00 | 37345.00 | 37251.30 | 37249.70 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 37020.00 | 37216.35 | 37235.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 10:15:00 | 36830.00 | 37123.66 | 37189.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 36145.00 | 36115.50 | 36445.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:15:00 | 36440.00 | 36115.50 | 36445.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 36285.00 | 36149.40 | 36430.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:30:00 | 36020.00 | 36150.52 | 36405.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:15:00 | 36085.00 | 36150.52 | 36405.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 11:30:00 | 36110.00 | 36004.04 | 36148.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 12:15:00 | 36125.00 | 36004.04 | 36148.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 36175.00 | 36038.23 | 36150.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:30:00 | 36215.00 | 36038.23 | 36150.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 36035.00 | 36037.58 | 36140.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 14:15:00 | 35890.00 | 36037.58 | 36140.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:30:00 | 35985.00 | 35986.85 | 36079.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 36230.00 | 36035.48 | 36093.12 | SL hit (close>static) qty=1.00 sl=36200.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 36690.00 | 36194.25 | 36153.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 37915.00 | 36610.52 | 36354.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 15:15:00 | 38050.00 | 38096.21 | 37639.57 | EMA200 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-02 10:00:00 | 31175.00 | 2025-06-04 10:15:00 | 31605.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-06-02 11:15:00 | 31255.00 | 2025-06-04 10:15:00 | 31605.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-06-02 12:45:00 | 31260.00 | 2025-06-04 10:15:00 | 31605.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-06-02 15:15:00 | 31210.00 | 2025-06-04 10:15:00 | 31605.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-06-03 11:30:00 | 31215.00 | 2025-06-04 10:15:00 | 31605.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-06-09 09:15:00 | 31745.00 | 2025-06-12 15:15:00 | 31500.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-06-09 11:00:00 | 31700.00 | 2025-06-12 15:15:00 | 31500.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-06-09 12:15:00 | 31700.00 | 2025-06-12 15:15:00 | 31500.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-06-09 13:30:00 | 31685.00 | 2025-06-12 15:15:00 | 31500.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-06-11 09:15:00 | 31760.00 | 2025-06-12 15:15:00 | 31500.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-06-12 14:15:00 | 31610.00 | 2025-06-12 15:15:00 | 31500.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-06-16 09:30:00 | 31485.00 | 2025-06-16 10:15:00 | 31970.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-06-19 15:15:00 | 32400.00 | 2025-06-23 09:15:00 | 31600.00 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-06-25 14:45:00 | 31650.00 | 2025-06-26 13:15:00 | 31840.00 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-06-26 09:30:00 | 31600.00 | 2025-06-26 13:15:00 | 31840.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-06-26 12:45:00 | 31645.00 | 2025-06-26 13:15:00 | 31840.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-07-01 12:15:00 | 32305.00 | 2025-07-04 09:15:00 | 35535.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-01 13:30:00 | 32310.00 | 2025-07-04 09:15:00 | 35541.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-02 12:15:00 | 32305.00 | 2025-07-04 09:15:00 | 35535.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-02 14:00:00 | 32295.00 | 2025-07-04 09:15:00 | 35524.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-03 09:15:00 | 32615.00 | 2025-07-04 14:15:00 | 35876.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-24 14:15:00 | 37910.00 | 2025-07-28 09:15:00 | 38170.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-07-25 09:30:00 | 37805.00 | 2025-07-28 09:15:00 | 38170.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest1 | 2025-07-31 10:30:00 | 40050.00 | 2025-08-05 10:15:00 | 40420.00 | STOP_HIT | 1.00 | 0.92% |
| SELL | retest2 | 2025-08-08 11:15:00 | 38505.00 | 2025-08-18 09:15:00 | 39580.00 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-08-08 11:45:00 | 38540.00 | 2025-08-18 09:15:00 | 39580.00 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-08-08 14:30:00 | 38530.00 | 2025-08-18 09:15:00 | 39580.00 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-08-13 13:15:00 | 38550.00 | 2025-08-18 09:15:00 | 39580.00 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-08-14 11:00:00 | 38470.00 | 2025-08-18 09:15:00 | 39580.00 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2025-08-14 12:00:00 | 38490.00 | 2025-08-18 09:15:00 | 39580.00 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-08-14 15:15:00 | 38490.00 | 2025-08-18 09:15:00 | 39580.00 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest1 | 2025-08-21 09:15:00 | 40000.00 | 2025-08-21 11:15:00 | 39680.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-08-25 14:15:00 | 39130.00 | 2025-08-28 09:15:00 | 39785.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-08-26 14:45:00 | 39340.00 | 2025-08-28 09:15:00 | 39785.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-09-09 13:15:00 | 41380.00 | 2025-09-10 12:15:00 | 41010.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-09-10 10:45:00 | 41455.00 | 2025-09-10 12:15:00 | 41010.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-09-17 14:30:00 | 39765.00 | 2025-09-29 11:15:00 | 37776.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 10:45:00 | 39775.00 | 2025-09-29 11:15:00 | 37786.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 11:15:00 | 39735.00 | 2025-09-29 11:15:00 | 37748.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 12:00:00 | 39750.00 | 2025-09-29 11:15:00 | 37762.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 14:30:00 | 39765.00 | 2025-09-29 13:15:00 | 38540.00 | STOP_HIT | 0.50 | 3.08% |
| SELL | retest2 | 2025-09-19 10:45:00 | 39775.00 | 2025-09-29 13:15:00 | 38540.00 | STOP_HIT | 0.50 | 3.10% |
| SELL | retest2 | 2025-09-22 11:15:00 | 39735.00 | 2025-09-29 13:15:00 | 38540.00 | STOP_HIT | 0.50 | 3.01% |
| SELL | retest2 | 2025-09-22 12:00:00 | 39750.00 | 2025-09-29 13:15:00 | 38540.00 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2025-09-24 09:15:00 | 39160.00 | 2025-10-03 13:15:00 | 38325.00 | STOP_HIT | 1.00 | 2.13% |
| BUY | retest2 | 2025-10-17 10:45:00 | 38545.00 | 2025-10-24 10:15:00 | 38570.00 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-10-17 13:45:00 | 38600.00 | 2025-10-24 10:15:00 | 38570.00 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-12-05 10:15:00 | 37000.00 | 2025-12-08 10:15:00 | 36565.00 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-12-09 14:15:00 | 36670.00 | 2025-12-09 15:15:00 | 36840.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-12-17 12:00:00 | 36025.00 | 2025-12-22 14:15:00 | 36100.00 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-12-22 10:15:00 | 35940.00 | 2025-12-22 14:15:00 | 36100.00 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2026-01-07 12:30:00 | 38900.00 | 2026-01-08 15:15:00 | 38545.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-01-08 13:00:00 | 38765.00 | 2026-01-08 15:15:00 | 38545.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-01-08 14:30:00 | 38770.00 | 2026-01-08 15:15:00 | 38545.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-01-16 09:30:00 | 37200.00 | 2026-01-21 09:15:00 | 35340.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:30:00 | 37200.00 | 2026-01-22 09:15:00 | 35745.00 | STOP_HIT | 0.50 | 3.91% |
| BUY | retest2 | 2026-02-01 14:15:00 | 36195.00 | 2026-02-05 09:15:00 | 36310.00 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2026-02-02 11:30:00 | 36185.00 | 2026-02-05 09:15:00 | 36310.00 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2026-02-02 12:30:00 | 36140.00 | 2026-02-05 09:15:00 | 36310.00 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2026-02-09 10:45:00 | 36105.00 | 2026-02-11 13:15:00 | 36470.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-02-09 11:15:00 | 36075.00 | 2026-02-11 13:15:00 | 36470.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-02-09 11:45:00 | 36085.00 | 2026-02-11 13:15:00 | 36470.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-02-09 13:30:00 | 35950.00 | 2026-02-11 13:15:00 | 36470.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-02-23 10:45:00 | 35170.00 | 2026-02-25 09:15:00 | 36035.00 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-02-23 11:45:00 | 35170.00 | 2026-02-25 09:15:00 | 36035.00 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-02-23 13:00:00 | 35170.00 | 2026-02-25 09:15:00 | 36035.00 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-02-24 09:15:00 | 35085.00 | 2026-02-25 09:15:00 | 36035.00 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2026-02-27 13:45:00 | 36330.00 | 2026-03-02 09:15:00 | 35840.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-02-27 14:15:00 | 36300.00 | 2026-03-02 09:15:00 | 35840.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-02-27 14:45:00 | 36450.00 | 2026-03-02 09:15:00 | 35840.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-03-11 11:00:00 | 31935.00 | 2026-03-13 14:15:00 | 30338.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 11:00:00 | 31935.00 | 2026-03-16 14:15:00 | 30370.00 | STOP_HIT | 0.50 | 4.90% |
| SELL | retest2 | 2026-03-23 09:15:00 | 29790.00 | 2026-03-25 09:15:00 | 30470.00 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2026-04-13 10:45:00 | 37240.00 | 2026-04-23 10:15:00 | 37485.00 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2026-04-16 10:30:00 | 37240.00 | 2026-04-23 10:15:00 | 37485.00 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2026-04-16 14:30:00 | 37265.00 | 2026-04-23 10:15:00 | 37485.00 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2026-05-04 10:30:00 | 36020.00 | 2026-05-06 11:15:00 | 36230.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-05-04 11:15:00 | 36085.00 | 2026-05-06 11:15:00 | 36230.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2026-05-05 11:30:00 | 36110.00 | 2026-05-06 14:15:00 | 36690.00 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-05-05 12:15:00 | 36125.00 | 2026-05-06 14:15:00 | 36690.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-05-05 14:15:00 | 35890.00 | 2026-05-06 14:15:00 | 36690.00 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2026-05-06 10:30:00 | 35985.00 | 2026-05-06 14:15:00 | 36690.00 | STOP_HIT | 1.00 | -1.96% |
