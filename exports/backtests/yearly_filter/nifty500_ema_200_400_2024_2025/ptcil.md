# PTC Industries Ltd. (PTCIL)

## Backtest Summary

- **Window:** 2023-06-09 09:15:00 → 2026-05-11 15:15:00 (5037 bars)
- **Last close:** 17080.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 0 |
| ALERT3 | 52 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 54 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 58
- **Target hits / Stop hits / Partials:** 0 / 58 / 0
- **Avg / median % per leg:** -2.95% / -2.45%
- **Sum % (uncompounded):** -170.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 0 | 0.0% | 0 | 50 | 0 | -2.66% | -133.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 50 | 0 | 0.0% | 0 | 50 | 0 | -2.66% | -133.2% |
| SELL (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -4.70% | -37.6% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -6.80% | -27.2% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.61% | -10.4% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -6.80% | -27.2% |
| retest2 (combined) | 54 | 0 | 0.0% | 0 | 54 | 0 | -2.66% | -143.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 10:15:00 | 7220.10 | 7588.40 | 7589.96 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 12:15:00 | 8150.00 | 7586.24 | 7584.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 13:15:00 | 8201.25 | 7592.36 | 7587.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 09:15:00 | 13475.50 | 13530.56 | 12217.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 09:30:00 | 13400.00 | 13530.56 | 12217.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 13764.10 | 14073.79 | 13541.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:30:00 | 13757.05 | 14073.79 | 13541.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 13500.00 | 14059.37 | 13547.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 13500.00 | 14059.37 | 13547.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 15:15:00 | 14000.00 | 14058.78 | 13549.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 13:45:00 | 14100.00 | 13822.60 | 13511.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 14:45:00 | 14195.90 | 13826.31 | 13514.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 13:15:00 | 13200.00 | 13625.76 | 13504.05 | SL hit (close<static) qty=1.00 sl=13201.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 15:15:00 | 12002.00 | 13398.41 | 13400.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 10:15:00 | 11969.95 | 13370.66 | 13386.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 09:15:00 | 12268.95 | 12110.08 | 12530.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-27 10:00:00 | 12268.95 | 12110.08 | 12530.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 12167.00 | 11788.83 | 12132.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 13:00:00 | 12167.00 | 11788.83 | 12132.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 13:15:00 | 12208.00 | 11793.00 | 12132.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 14:00:00 | 12208.00 | 11793.00 | 12132.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 14:15:00 | 12529.90 | 11800.33 | 12134.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 15:00:00 | 12529.90 | 11800.33 | 12134.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 12700.00 | 11814.63 | 12138.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 09:45:00 | 12690.35 | 11814.63 | 12138.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 13:15:00 | 13994.00 | 12373.00 | 12371.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 14:15:00 | 14442.45 | 12393.59 | 12381.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-22 12:15:00 | 14794.30 | 14802.57 | 13863.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-22 12:30:00 | 14794.30 | 14802.57 | 13863.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 14129.40 | 14823.38 | 14015.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 09:15:00 | 14556.00 | 14656.15 | 14004.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:00:00 | 14870.00 | 14635.08 | 14012.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 14:45:00 | 14458.60 | 14613.93 | 14023.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:15:00 | 14429.40 | 14609.93 | 14024.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 14195.10 | 14593.81 | 14092.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:15:00 | 13869.95 | 14593.81 | 14092.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 13650.00 | 14584.42 | 14090.33 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 13650.00 | 14584.42 | 14090.33 | SL hit (close<static) qty=1.00 sl=13800.05 alert=retest2 |

### Cycle 5 — SELL (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 14:15:00 | 10197.95 | 13760.06 | 13763.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 13:15:00 | 10074.15 | 13346.06 | 13549.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 09:15:00 | 12504.55 | 12171.04 | 12810.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-07 10:00:00 | 12504.55 | 12171.04 | 12810.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 12767.65 | 12187.94 | 12797.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:45:00 | 12807.00 | 12187.94 | 12797.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 12647.45 | 12186.53 | 12714.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:45:00 | 12682.40 | 12186.53 | 12714.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 12650.05 | 12206.32 | 12708.80 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 11:15:00 | 14445.00 | 13022.15 | 13020.51 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-05-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 13:15:00 | 12230.00 | 13242.32 | 13245.93 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 11:15:00 | 14786.00 | 13254.50 | 13249.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 11:15:00 | 14972.00 | 13552.97 | 13411.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 09:15:00 | 14439.00 | 14489.55 | 14034.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 10:00:00 | 14439.00 | 14489.55 | 14034.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 14273.00 | 14704.83 | 14280.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 12:00:00 | 14613.00 | 14663.15 | 14280.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 12:00:00 | 14595.00 | 14664.72 | 14294.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 13:45:00 | 14576.00 | 14662.53 | 14296.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 11:15:00 | 14570.00 | 14659.30 | 14302.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 14450.00 | 14780.53 | 14493.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 14450.00 | 14780.53 | 14493.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 14360.00 | 14776.34 | 14492.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 14360.00 | 14776.34 | 14492.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 14124.00 | 14752.78 | 14487.65 | SL hit (close<static) qty=1.00 sl=14125.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 12:15:00 | 13692.00 | 14464.81 | 14465.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 14:15:00 | 13445.00 | 14447.28 | 14456.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 14:15:00 | 14049.00 | 13980.30 | 14161.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 15:00:00 | 14049.00 | 13980.30 | 14161.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 14239.00 | 13979.54 | 14155.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:45:00 | 14211.00 | 13979.54 | 14155.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 14179.00 | 13981.53 | 14155.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 14140.00 | 13981.53 | 14155.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 14411.00 | 13991.89 | 14157.21 | SL hit (close>static) qty=1.00 sl=14290.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 15231.00 | 14290.54 | 14287.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 15566.00 | 14471.75 | 14383.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 17898.00 | 17909.15 | 17153.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 10:45:00 | 17916.00 | 17909.15 | 17153.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 16760.00 | 17828.57 | 17183.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 16760.00 | 17828.57 | 17183.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 16470.00 | 17815.05 | 17180.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:00:00 | 16470.00 | 17815.05 | 17180.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 17586.00 | 17975.59 | 17514.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 15:15:00 | 17700.00 | 17975.59 | 17514.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 10:15:00 | 17741.00 | 17955.46 | 17524.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 14:45:00 | 17640.00 | 17943.13 | 17529.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 09:45:00 | 17768.00 | 17937.74 | 17530.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 11:15:00 | 17482.00 | 17930.29 | 17530.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-12 11:15:00 | 17482.00 | 17930.29 | 17530.84 | SL hit (close<static) qty=1.00 sl=17511.00 alert=retest2 |

### Cycle 11 — SELL (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 13:15:00 | 17311.00 | 17741.88 | 17743.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 14:15:00 | 17173.00 | 17736.22 | 17740.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 16201.00 | 16190.61 | 16674.98 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:45:00 | 16024.00 | 16189.09 | 16657.56 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:00:00 | 16043.00 | 16187.63 | 16654.49 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:30:00 | 16014.00 | 16176.99 | 16635.29 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 13:15:00 | 16021.00 | 16146.61 | 16581.92 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 17116.00 | 16172.19 | 16571.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 17116.00 | 16172.19 | 16571.64 | SL hit (close>ema400) qty=1.00 sl=16571.64 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-09-30 13:45:00 | 14100.00 | 2024-10-17 13:15:00 | 13200.00 | STOP_HIT | 1.00 | -6.38% |
| BUY | retest2 | 2024-09-30 14:45:00 | 14195.90 | 2024-10-17 13:15:00 | 13200.00 | STOP_HIT | 1.00 | -7.02% |
| BUY | retest2 | 2025-02-01 09:15:00 | 14556.00 | 2025-02-10 09:15:00 | 13650.00 | STOP_HIT | 1.00 | -6.22% |
| BUY | retest2 | 2025-02-01 15:00:00 | 14870.00 | 2025-02-10 09:15:00 | 13650.00 | STOP_HIT | 1.00 | -8.20% |
| BUY | retest2 | 2025-02-03 14:45:00 | 14458.60 | 2025-02-10 09:15:00 | 13650.00 | STOP_HIT | 1.00 | -5.59% |
| BUY | retest2 | 2025-02-04 09:15:00 | 14429.40 | 2025-02-10 09:15:00 | 13650.00 | STOP_HIT | 1.00 | -5.40% |
| BUY | retest2 | 2025-06-23 12:00:00 | 14613.00 | 2025-07-11 09:15:00 | 14124.00 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2025-06-24 12:00:00 | 14595.00 | 2025-07-11 09:15:00 | 14124.00 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2025-06-24 13:45:00 | 14576.00 | 2025-07-11 09:15:00 | 14124.00 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2025-06-25 11:15:00 | 14570.00 | 2025-07-11 09:15:00 | 14124.00 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2025-07-14 15:00:00 | 14480.00 | 2025-08-11 09:15:00 | 14380.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-07-15 12:00:00 | 14494.00 | 2025-08-11 09:15:00 | 14380.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-07-16 13:15:00 | 14465.00 | 2025-08-11 09:15:00 | 14380.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-07-16 13:45:00 | 14466.00 | 2025-08-11 09:15:00 | 14380.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-07-25 13:45:00 | 14699.00 | 2025-08-13 09:15:00 | 14071.00 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest2 | 2025-07-25 14:30:00 | 14725.00 | 2025-08-13 09:15:00 | 14071.00 | STOP_HIT | 1.00 | -4.44% |
| BUY | retest2 | 2025-07-28 15:00:00 | 14722.00 | 2025-08-13 09:15:00 | 14071.00 | STOP_HIT | 1.00 | -4.42% |
| BUY | retest2 | 2025-07-29 10:00:00 | 14700.00 | 2025-08-13 09:15:00 | 14071.00 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2025-09-12 09:15:00 | 14140.00 | 2025-09-12 12:15:00 | 14411.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-09-12 15:00:00 | 14148.00 | 2025-09-15 14:15:00 | 14520.00 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2026-01-07 15:15:00 | 17700.00 | 2026-01-12 11:15:00 | 17482.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2026-01-09 10:15:00 | 17741.00 | 2026-01-12 11:15:00 | 17482.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-01-09 14:45:00 | 17640.00 | 2026-01-12 11:15:00 | 17482.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2026-01-12 09:45:00 | 17768.00 | 2026-01-12 11:15:00 | 17482.00 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2026-01-12 13:30:00 | 17677.00 | 2026-01-21 09:15:00 | 17365.00 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2026-01-12 14:00:00 | 17674.00 | 2026-01-21 09:15:00 | 17365.00 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-01-20 09:45:00 | 17698.00 | 2026-01-21 09:15:00 | 17365.00 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2026-01-20 10:30:00 | 17710.00 | 2026-01-21 09:15:00 | 17365.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2026-01-21 14:45:00 | 17650.00 | 2026-01-27 15:15:00 | 17250.00 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2026-01-28 09:15:00 | 17669.00 | 2026-02-05 09:15:00 | 17409.00 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-01-28 09:45:00 | 17633.00 | 2026-02-05 09:15:00 | 17409.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-01-28 10:15:00 | 17660.00 | 2026-02-05 09:15:00 | 17409.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-01-28 12:30:00 | 17864.00 | 2026-02-05 09:15:00 | 17409.00 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2026-01-28 13:00:00 | 17846.00 | 2026-02-05 09:15:00 | 17409.00 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2026-01-28 13:45:00 | 17845.00 | 2026-02-05 12:15:00 | 17270.00 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2026-02-01 12:30:00 | 17850.00 | 2026-02-05 12:15:00 | 17270.00 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2026-02-02 15:15:00 | 18150.00 | 2026-02-05 12:15:00 | 17270.00 | STOP_HIT | 1.00 | -4.85% |
| BUY | retest2 | 2026-02-09 12:00:00 | 18191.00 | 2026-02-19 15:15:00 | 17625.00 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2026-02-09 13:15:00 | 18113.00 | 2026-02-19 15:15:00 | 17625.00 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2026-02-16 14:30:00 | 18139.00 | 2026-02-19 15:15:00 | 17625.00 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-02-20 13:45:00 | 17840.00 | 2026-02-24 09:15:00 | 17781.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2026-02-23 09:15:00 | 17932.00 | 2026-03-02 09:15:00 | 17725.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2026-02-23 10:45:00 | 17800.00 | 2026-03-02 09:15:00 | 17725.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2026-02-23 11:45:00 | 17839.00 | 2026-03-02 09:15:00 | 17725.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-02-23 15:00:00 | 17991.00 | 2026-03-02 10:15:00 | 17601.00 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2026-02-24 15:00:00 | 17953.00 | 2026-03-02 10:15:00 | 17601.00 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2026-02-25 09:30:00 | 17965.00 | 2026-03-02 10:15:00 | 17601.00 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2026-02-25 14:45:00 | 18077.00 | 2026-03-02 10:15:00 | 17601.00 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2026-03-02 14:30:00 | 18030.00 | 2026-03-04 10:15:00 | 17821.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-03-04 13:45:00 | 18022.00 | 2026-03-05 11:15:00 | 17817.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-03-06 09:15:00 | 18200.00 | 2026-03-06 11:15:00 | 17789.00 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2026-03-11 09:15:00 | 18090.00 | 2026-03-11 09:15:00 | 17771.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest1 | 2026-04-28 09:45:00 | 16024.00 | 2026-05-06 09:15:00 | 17116.00 | STOP_HIT | 1.00 | -6.81% |
| SELL | retest1 | 2026-04-28 11:00:00 | 16043.00 | 2026-05-06 09:15:00 | 17116.00 | STOP_HIT | 1.00 | -6.69% |
| SELL | retest1 | 2026-04-29 09:30:00 | 16014.00 | 2026-05-06 09:15:00 | 17116.00 | STOP_HIT | 1.00 | -6.88% |
| SELL | retest1 | 2026-05-04 13:15:00 | 16021.00 | 2026-05-06 09:15:00 | 17116.00 | STOP_HIT | 1.00 | -6.83% |
| SELL | retest2 | 2026-05-11 12:15:00 | 16588.00 | 2026-05-11 14:15:00 | 17068.00 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2026-05-11 13:00:00 | 16574.00 | 2026-05-11 14:15:00 | 17068.00 | STOP_HIT | 1.00 | -2.98% |
