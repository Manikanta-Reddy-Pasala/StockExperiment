# Bosch Ltd. (BOSCHLTD)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 38050.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 1 |
| ALERT3 | 40 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 17 |
| PARTIAL | 2 |
| TARGET_HIT | 3 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 13
- **Target hits / Stop hits / Partials:** 3 / 14 / 2
- **Avg / median % per leg:** 0.95% / -1.30%
- **Sum % (uncompounded):** 18.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 2 | 20.0% | 2 | 8 | 0 | 0.27% | 2.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 2 | 20.0% | 2 | 8 | 0 | 0.27% | 2.7% |
| SELL (all) | 9 | 4 | 44.4% | 1 | 6 | 2 | 1.71% | 15.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 4 | 44.4% | 1 | 6 | 2 | 1.71% | 15.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 6 | 31.6% | 3 | 14 | 2 | 0.95% | 18.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-04 15:15:00 | 18239.90 | 18896.01 | 18896.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-07 10:15:00 | 18200.00 | 18882.81 | 18889.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 13:15:00 | 18565.10 | 18507.37 | 18648.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-28 14:00:00 | 18565.10 | 18507.37 | 18648.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 14:15:00 | 18629.90 | 18510.60 | 18644.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 15:00:00 | 18629.90 | 18510.60 | 18644.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 15:15:00 | 18625.00 | 18511.74 | 18644.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 09:15:00 | 18655.00 | 18511.74 | 18644.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 09:15:00 | 18654.40 | 18513.16 | 18644.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-30 11:45:00 | 18614.90 | 18516.31 | 18645.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-30 12:15:00 | 18706.40 | 18518.20 | 18645.33 | SL hit (close>static) qty=1.00 sl=18695.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 13:15:00 | 19414.40 | 18737.62 | 18736.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 10:15:00 | 19417.30 | 18762.60 | 18749.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-25 10:15:00 | 19010.40 | 19037.37 | 18913.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-25 11:00:00 | 19010.40 | 19037.37 | 18913.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 19141.80 | 19058.15 | 18939.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 14:30:00 | 18893.30 | 19058.15 | 18939.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 18696.10 | 19060.37 | 18945.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 10:00:00 | 18696.10 | 19060.37 | 18945.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 10:15:00 | 18762.60 | 19057.40 | 18944.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 10:30:00 | 18692.10 | 19057.40 | 18944.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 10:15:00 | 18828.00 | 19020.25 | 18933.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-05 10:30:00 | 18803.20 | 19020.25 | 18933.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 13:15:00 | 18920.90 | 19016.44 | 18932.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-05 13:30:00 | 18905.10 | 19016.44 | 18932.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 14:15:00 | 18936.10 | 19015.64 | 18932.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-05 15:00:00 | 18936.10 | 19015.64 | 18932.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 15:15:00 | 18910.00 | 19014.59 | 18932.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-06 09:15:00 | 18969.90 | 19014.59 | 18932.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 19088.70 | 19015.33 | 18933.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-06 10:45:00 | 19150.80 | 19016.67 | 18934.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-09 11:00:00 | 19133.70 | 19030.84 | 18944.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-11-22 09:15:00 | 21065.88 | 19850.56 | 19583.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 12:15:00 | 32089.50 | 32564.39 | 32565.26 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 12:15:00 | 33779.90 | 32573.05 | 32569.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 34163.35 | 32795.76 | 32688.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 14:15:00 | 36489.95 | 36504.76 | 35195.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 15:00:00 | 36489.95 | 36504.76 | 35195.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 35647.60 | 36491.18 | 35417.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 35558.90 | 36491.18 | 35417.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 35460.55 | 36421.59 | 35448.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:00:00 | 35460.55 | 36421.59 | 35448.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 35745.65 | 36414.87 | 35450.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-29 12:45:00 | 36190.75 | 36405.93 | 35455.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-31 11:15:00 | 35161.65 | 36365.39 | 35494.69 | SL hit (close<static) qty=1.00 sl=35325.10 alert=retest2 |

### Cycle 5 — SELL (started 2024-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 12:15:00 | 34657.00 | 35086.92 | 35087.40 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 10:15:00 | 35791.15 | 35092.84 | 35090.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 12:15:00 | 36123.15 | 35110.66 | 35099.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 12:15:00 | 35571.75 | 35599.78 | 35392.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-19 13:00:00 | 35571.75 | 35599.78 | 35392.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 35395.85 | 35596.79 | 35395.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:00:00 | 35395.85 | 35596.79 | 35395.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 35503.15 | 35595.85 | 35395.54 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 11:15:00 | 33691.45 | 35227.67 | 35230.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 33598.30 | 34877.26 | 35037.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 12:15:00 | 27565.10 | 27504.04 | 29028.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-19 12:30:00 | 27605.05 | 27504.04 | 29028.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 27960.00 | 27432.01 | 28221.48 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 30835.00 | 28660.62 | 28655.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 31195.00 | 28794.86 | 28723.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 09:15:00 | 39620.00 | 39808.50 | 38107.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:00:00 | 39620.00 | 39808.50 | 38107.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 38425.00 | 39657.57 | 38475.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 38425.00 | 39657.57 | 38475.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 38375.00 | 39644.81 | 38474.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 38560.00 | 39634.22 | 38475.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 38690.00 | 39624.82 | 38476.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:30:00 | 38535.00 | 39624.82 | 38476.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 38495.00 | 39613.58 | 38476.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 12:00:00 | 38495.00 | 39613.58 | 38476.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 12:15:00 | 38450.00 | 39602.00 | 38476.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:15:00 | 38400.00 | 39602.00 | 38476.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 38250.00 | 39588.55 | 38475.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 38250.00 | 39588.55 | 38475.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 38160.00 | 39574.33 | 38473.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:00:00 | 38160.00 | 39574.33 | 38473.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 38405.00 | 39475.55 | 38461.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 38405.00 | 39475.55 | 38461.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 38250.00 | 39463.35 | 38460.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 38200.00 | 39463.35 | 38460.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 38045.00 | 39449.24 | 38458.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:15:00 | 38010.00 | 39449.24 | 38458.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 38175.00 | 39436.56 | 38456.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:45:00 | 37980.00 | 39436.56 | 38456.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 38270.00 | 39303.24 | 38440.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 38270.00 | 39303.24 | 38440.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 38280.00 | 39284.60 | 38439.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:15:00 | 38400.00 | 39244.82 | 38436.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 10:30:00 | 38440.00 | 39111.69 | 38461.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 13:15:00 | 38400.00 | 39095.18 | 38459.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 15:00:00 | 38460.00 | 39081.22 | 38458.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 38495.00 | 39075.39 | 38458.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 38565.00 | 39075.39 | 38458.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 38790.00 | 39072.55 | 38460.52 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 37900.00 | 39007.87 | 38468.66 | SL hit (close<static) qty=1.00 sl=38010.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 36670.00 | 38258.93 | 38259.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 09:15:00 | 36235.00 | 38000.11 | 38123.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 37035.00 | 37020.23 | 37462.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 09:45:00 | 37130.00 | 37020.23 | 37462.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 37215.00 | 36345.10 | 36816.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 37215.00 | 36345.10 | 36816.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 38515.00 | 36366.69 | 36824.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 38515.00 | 36366.69 | 36824.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 37230.00 | 37180.42 | 37192.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:45:00 | 37275.00 | 37180.42 | 37192.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 37280.00 | 37181.41 | 37192.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 37280.00 | 37181.41 | 37192.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 37610.00 | 37185.67 | 37194.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 37745.00 | 37185.67 | 37194.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2026-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 12:15:00 | 37685.00 | 37207.96 | 37205.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 14:15:00 | 37895.00 | 37218.66 | 37211.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 37210.00 | 37263.14 | 37234.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 37210.00 | 37263.14 | 37234.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 37210.00 | 37263.14 | 37234.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:30:00 | 37200.00 | 37263.14 | 37234.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 37080.00 | 37261.32 | 37233.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:45:00 | 37180.00 | 37261.32 | 37233.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 36335.00 | 37200.60 | 37204.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 35935.00 | 37188.01 | 37197.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 14:15:00 | 36570.00 | 36536.67 | 36821.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 15:00:00 | 36570.00 | 36536.67 | 36821.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 37260.00 | 36507.66 | 36784.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 12:30:00 | 36695.00 | 36613.66 | 36825.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 34860.25 | 36159.62 | 36495.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 36035.00 | 35992.15 | 36373.69 | SL hit (close>ema200) qty=0.50 sl=35992.15 alert=retest2 |

### Cycle 12 — BUY (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 15:15:00 | 38200.00 | 34571.80 | 34557.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 10:15:00 | 38290.00 | 34644.49 | 34593.92 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-08-30 11:45:00 | 18614.90 | 2023-08-30 12:15:00 | 18706.40 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2023-08-30 15:00:00 | 18622.70 | 2023-09-01 09:15:00 | 18709.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2023-08-31 14:45:00 | 18610.10 | 2023-09-01 09:15:00 | 18709.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2023-10-06 10:45:00 | 19150.80 | 2023-11-22 09:15:00 | 21065.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-09 11:00:00 | 19133.70 | 2023-11-22 09:15:00 | 21047.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-29 12:45:00 | 36190.75 | 2024-10-31 11:15:00 | 35161.65 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2024-11-06 09:15:00 | 35932.00 | 2024-11-08 10:15:00 | 35218.65 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-10-03 14:15:00 | 38400.00 | 2025-10-14 09:15:00 | 37900.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-10-09 10:30:00 | 38440.00 | 2025-10-14 09:15:00 | 37900.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-10-09 13:15:00 | 38400.00 | 2025-10-14 09:15:00 | 37900.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-10-09 15:00:00 | 38460.00 | 2025-10-14 09:15:00 | 37900.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-10-20 10:15:00 | 38995.00 | 2025-10-29 09:15:00 | 37600.00 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2025-10-27 14:15:00 | 38945.00 | 2025-10-29 09:15:00 | 37600.00 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2026-02-04 12:30:00 | 36695.00 | 2026-02-20 09:15:00 | 34860.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 12:30:00 | 36695.00 | 2026-02-25 09:15:00 | 36035.00 | STOP_HIT | 0.50 | 1.80% |
| SELL | retest2 | 2026-02-25 11:45:00 | 36910.00 | 2026-03-02 13:15:00 | 35064.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:45:00 | 36910.00 | 2026-03-05 11:15:00 | 33219.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-09 11:15:00 | 37100.00 | 2026-04-20 11:15:00 | 38075.00 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2026-04-09 13:15:00 | 37225.00 | 2026-04-20 11:15:00 | 38075.00 | STOP_HIT | 1.00 | -2.28% |
