# PTC Industries Ltd. (PTCIL)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 16790.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 93 |
| ALERT1 | 55 |
| ALERT2 | 52 |
| ALERT2_SKIP | 33 |
| ALERT3 | 149 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 64 |
| PARTIAL | 14 |
| TARGET_HIT | 13 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 79 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 42 / 37
- **Target hits / Stop hits / Partials:** 13 / 52 / 14
- **Avg / median % per leg:** 2.04% / 0.53%
- **Sum % (uncompounded):** 161.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 14 | 40.0% | 6 | 28 | 1 | 1.51% | 52.8% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.27% | 13.1% |
| BUY @ 3rd Alert (retest2) | 31 | 12 | 38.7% | 5 | 26 | 0 | 1.28% | 39.7% |
| SELL (all) | 44 | 28 | 63.6% | 7 | 24 | 13 | 2.47% | 108.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 44 | 28 | 63.6% | 7 | 24 | 13 | 2.47% | 108.5% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.27% | 13.1% |
| retest2 (combined) | 75 | 40 | 53.3% | 12 | 50 | 13 | 1.98% | 148.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 12877.00 | 12474.14 | 12435.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 13100.00 | 12735.53 | 12596.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 12789.00 | 12810.19 | 12683.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 13:15:00 | 12789.00 | 12810.19 | 12683.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 12789.00 | 12810.19 | 12683.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:30:00 | 12806.00 | 12810.19 | 12683.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 14067.00 | 14237.84 | 14056.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 14083.00 | 14237.84 | 14056.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 14025.00 | 14195.27 | 14053.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:30:00 | 13925.00 | 14195.27 | 14053.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 14113.00 | 14178.82 | 14058.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 10:45:00 | 14319.00 | 14082.12 | 14047.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 12:00:00 | 14200.00 | 14105.70 | 14061.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 14380.00 | 14181.25 | 14119.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:00:00 | 14166.00 | 14209.21 | 14162.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 14432.00 | 14253.77 | 14187.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 15:15:00 | 14460.00 | 14253.77 | 14187.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:45:00 | 14451.00 | 14329.41 | 14235.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-22 14:15:00 | 15620.00 | 14989.26 | 14617.26 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-05-22 14:15:00 | 15582.60 | 14989.26 | 14617.26 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-05-23 09:15:00 | 15750.90 | 15172.73 | 14771.22 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-05-26 13:15:00 | 15818.00 | 15523.10 | 15251.80 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-05-28 09:15:00 | 15896.10 | 15686.77 | 15531.06 | Target hit (10%) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-29 11:15:00 | 15329.00 | 15507.45 | 15522.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 11:15:00 | 15329.00 | 15507.45 | 15522.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 12:15:00 | 15310.00 | 15467.96 | 15502.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 15536.00 | 15420.62 | 15462.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 15536.00 | 15420.62 | 15462.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 15536.00 | 15420.62 | 15462.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 15536.00 | 15420.62 | 15462.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 15562.00 | 15448.89 | 15471.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 15562.00 | 15448.89 | 15471.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 15571.00 | 15473.31 | 15480.56 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 12:15:00 | 15547.00 | 15488.05 | 15486.60 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 14:15:00 | 15386.00 | 15482.19 | 15485.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 15:15:00 | 14997.00 | 15385.15 | 15440.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 13:15:00 | 15389.00 | 15369.14 | 15409.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 13:30:00 | 15385.00 | 15369.14 | 15409.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 15242.00 | 15343.71 | 15394.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:30:00 | 15383.00 | 15343.71 | 15394.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 15328.00 | 15319.34 | 15373.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 15:15:00 | 15276.00 | 15336.35 | 15362.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 11:00:00 | 15266.00 | 15326.50 | 15351.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 09:15:00 | 14512.20 | 14927.28 | 15108.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 09:15:00 | 14502.70 | 14927.28 | 15108.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-05 15:15:00 | 14858.00 | 14850.37 | 14983.20 | SL hit (close>ema200) qty=0.50 sl=14850.37 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-05 15:15:00 | 14858.00 | 14850.37 | 14983.20 | SL hit (close>ema200) qty=0.50 sl=14850.37 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 09:15:00 | 15670.00 | 14832.16 | 14733.37 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 10:15:00 | 15099.00 | 15219.93 | 15221.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 11:15:00 | 14907.00 | 15157.34 | 15192.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 14877.00 | 14818.54 | 14917.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 14877.00 | 14818.54 | 14917.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 14877.00 | 14818.54 | 14917.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 14882.00 | 14818.54 | 14917.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 14416.00 | 14300.53 | 14451.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:30:00 | 14480.00 | 14300.53 | 14451.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 14613.00 | 14363.02 | 14466.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:00:00 | 14613.00 | 14363.02 | 14466.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 14779.00 | 14446.22 | 14494.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:45:00 | 14742.00 | 14446.22 | 14494.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 14:15:00 | 14674.00 | 14530.62 | 14527.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 14:15:00 | 14749.00 | 14621.62 | 14584.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 09:15:00 | 14495.00 | 14610.60 | 14586.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 09:15:00 | 14495.00 | 14610.60 | 14586.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 14495.00 | 14610.60 | 14586.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:45:00 | 14576.00 | 14610.60 | 14586.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 10:15:00 | 14373.00 | 14563.08 | 14567.34 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 11:15:00 | 14751.00 | 14600.66 | 14584.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 14:15:00 | 14924.00 | 14697.46 | 14634.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 10:15:00 | 15267.00 | 15387.78 | 15210.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 11:00:00 | 15267.00 | 15387.78 | 15210.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 14940.00 | 15298.22 | 15185.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 14958.00 | 15298.22 | 15185.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 14982.00 | 15234.98 | 15167.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 14982.00 | 15234.98 | 15167.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 14930.00 | 15102.28 | 15117.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 14727.00 | 15027.22 | 15082.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 12:15:00 | 15050.00 | 15006.24 | 15056.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 13:00:00 | 15050.00 | 15006.24 | 15056.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 15070.00 | 15019.59 | 15053.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:00:00 | 15070.00 | 15019.59 | 15053.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 14973.00 | 15010.28 | 15046.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:30:00 | 15193.00 | 15070.02 | 15070.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 15136.00 | 15083.22 | 15076.09 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 14864.00 | 15054.70 | 15065.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 14:15:00 | 14766.00 | 14959.25 | 15017.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 13:15:00 | 14858.00 | 14851.06 | 14926.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 13:15:00 | 14858.00 | 14851.06 | 14926.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 14858.00 | 14851.06 | 14926.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:30:00 | 14931.00 | 14851.06 | 14926.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 14898.00 | 14864.28 | 14919.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 14822.00 | 14864.28 | 14919.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:00:00 | 14825.00 | 14765.73 | 14837.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 14799.00 | 14757.67 | 14821.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 14841.00 | 14753.21 | 14782.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 14769.00 | 14757.78 | 14779.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:45:00 | 14771.00 | 14757.78 | 14779.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 14799.00 | 14766.02 | 14781.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:15:00 | 14840.00 | 14766.02 | 14781.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 14795.00 | 14771.82 | 14782.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:30:00 | 14840.00 | 14771.82 | 14782.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 14550.00 | 14727.45 | 14761.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:15:00 | 14536.00 | 14655.39 | 14717.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:15:00 | 14080.90 | 14365.52 | 14519.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:15:00 | 14083.75 | 14365.52 | 14519.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:15:00 | 14098.95 | 14365.52 | 14519.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:15:00 | 14059.05 | 14312.42 | 14480.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 14:15:00 | 14423.00 | 14216.89 | 14370.01 | SL hit (close>ema200) qty=0.50 sl=14216.89 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 14:15:00 | 14423.00 | 14216.89 | 14370.01 | SL hit (close>ema200) qty=0.50 sl=14216.89 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 14:15:00 | 14423.00 | 14216.89 | 14370.01 | SL hit (close>ema200) qty=0.50 sl=14216.89 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 14:15:00 | 14423.00 | 14216.89 | 14370.01 | SL hit (close>ema200) qty=0.50 sl=14216.89 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 14169.00 | 14283.31 | 14386.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 14494.00 | 14372.28 | 14366.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 14494.00 | 14372.28 | 14366.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 14494.00 | 14372.28 | 14366.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 14511.00 | 14400.02 | 14379.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 10:15:00 | 14425.00 | 14441.03 | 14412.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 10:45:00 | 14436.00 | 14441.03 | 14412.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 14452.00 | 14443.22 | 14415.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:00:00 | 14459.00 | 14446.38 | 14419.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:45:00 | 14466.00 | 14451.10 | 14424.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 14510.00 | 14505.81 | 14484.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 11:15:00 | 14422.00 | 14466.10 | 14469.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-18 11:15:00 | 14422.00 | 14466.10 | 14469.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-18 11:15:00 | 14422.00 | 14466.10 | 14469.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 14422.00 | 14466.10 | 14469.69 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 13:15:00 | 14470.00 | 14466.75 | 14466.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 14:15:00 | 14539.00 | 14481.20 | 14473.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 12:15:00 | 14485.00 | 14500.80 | 14488.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 12:15:00 | 14485.00 | 14500.80 | 14488.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 14485.00 | 14500.80 | 14488.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:45:00 | 14481.00 | 14500.80 | 14488.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 14498.00 | 14500.24 | 14489.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:15:00 | 14482.00 | 14500.24 | 14489.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 14541.00 | 14508.39 | 14493.77 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 14311.00 | 14464.90 | 14477.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 12:15:00 | 14230.00 | 14394.41 | 14441.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 14391.00 | 14366.94 | 14419.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 14:15:00 | 14391.00 | 14366.94 | 14419.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 14391.00 | 14366.94 | 14419.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 14391.00 | 14366.94 | 14419.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 14477.00 | 14386.56 | 14419.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 12:00:00 | 14308.00 | 14373.16 | 14407.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 14550.00 | 14403.65 | 14412.27 | SL hit (close>static) qty=1.00 sl=14498.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 15:15:00 | 14560.00 | 14434.92 | 14425.70 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 14398.00 | 14417.93 | 14419.60 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 12:15:00 | 14619.00 | 14458.14 | 14437.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 13:15:00 | 14790.00 | 14685.87 | 14608.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 15:15:00 | 14730.00 | 14753.58 | 14698.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 09:15:00 | 14795.00 | 14753.58 | 14698.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 14727.00 | 14748.26 | 14700.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 14:30:00 | 14960.00 | 14778.67 | 14732.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 13:15:00 | 14845.00 | 14835.30 | 14782.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 14:30:00 | 14848.00 | 14834.59 | 14792.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 11:15:00 | 14683.00 | 14755.09 | 14763.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 11:15:00 | 14683.00 | 14755.09 | 14763.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 11:15:00 | 14683.00 | 14755.09 | 14763.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 11:15:00 | 14683.00 | 14755.09 | 14763.10 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 14832.00 | 14766.64 | 14765.79 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 15:15:00 | 14727.00 | 14758.71 | 14762.27 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 14822.00 | 14771.37 | 14767.70 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 14739.00 | 14791.58 | 14793.95 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 14:15:00 | 15018.00 | 14828.09 | 14808.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 14:15:00 | 15124.00 | 14953.79 | 14899.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 15:15:00 | 14821.00 | 14927.23 | 14892.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 15:15:00 | 14821.00 | 14927.23 | 14892.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 14821.00 | 14927.23 | 14892.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 14642.00 | 14927.23 | 14892.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 14380.00 | 14817.78 | 14845.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 13778.00 | 13988.33 | 14129.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 11:15:00 | 13722.00 | 13718.49 | 13841.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 12:00:00 | 13722.00 | 13718.49 | 13841.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 13647.00 | 13573.55 | 13661.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:45:00 | 13697.00 | 13573.55 | 13661.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 13580.00 | 13574.84 | 13654.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 13593.00 | 13574.84 | 13654.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 13537.00 | 13567.27 | 13643.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 11:15:00 | 13454.00 | 13565.82 | 13635.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 13:30:00 | 13463.00 | 13523.18 | 13596.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 14168.00 | 13600.35 | 13608.75 | SL hit (close>static) qty=1.00 sl=13747.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 14168.00 | 13600.35 | 13608.75 | SL hit (close>static) qty=1.00 sl=13747.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 10:15:00 | 13980.00 | 13676.28 | 13642.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 14:15:00 | 14221.00 | 13894.69 | 13796.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 10:15:00 | 13964.00 | 13968.54 | 13861.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-26 10:45:00 | 13948.00 | 13968.54 | 13861.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 13893.00 | 13950.33 | 13879.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:30:00 | 13851.00 | 13950.33 | 13879.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 13969.00 | 13954.07 | 13887.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 14:30:00 | 13843.00 | 13954.07 | 13887.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 13761.00 | 13918.48 | 13883.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:30:00 | 13757.00 | 13918.48 | 13883.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 13793.00 | 13893.39 | 13875.12 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 13740.00 | 13841.37 | 13853.32 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 15:15:00 | 13965.00 | 13873.41 | 13864.58 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 10:15:00 | 13844.00 | 13856.58 | 13857.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 12:15:00 | 13755.00 | 13832.81 | 13846.56 | Break + close below crossover candle low |

### Cycle 31 — BUY (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 14:15:00 | 14000.00 | 13854.92 | 13853.52 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 10:15:00 | 13845.00 | 13870.65 | 13871.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 12:15:00 | 13819.00 | 13861.02 | 13867.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 13845.00 | 13840.78 | 13854.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 13845.00 | 13840.78 | 13854.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 13845.00 | 13840.78 | 13854.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 10:15:00 | 13812.00 | 13840.78 | 13854.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 13:15:00 | 13710.00 | 13622.86 | 13618.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 13:15:00 | 13710.00 | 13622.86 | 13618.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 14:15:00 | 14049.00 | 13708.09 | 13657.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 13799.00 | 13826.11 | 13748.18 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 14:00:00 | 14001.00 | 13861.09 | 13771.16 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 14520.00 | 14227.49 | 14114.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:30:00 | 14486.00 | 14227.49 | 14114.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 14:15:00 | 14701.05 | 14513.96 | 14342.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-09-17 12:15:00 | 15401.10 | 14897.54 | 14603.60 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 15184.00 | 15312.91 | 15163.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:30:00 | 15157.00 | 15312.91 | 15163.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 14991.00 | 15248.53 | 15148.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:30:00 | 14845.00 | 15248.53 | 15148.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 14800.00 | 15158.82 | 15116.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 15138.00 | 15158.82 | 15116.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 13:15:00 | 15125.00 | 15323.51 | 15333.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 13:15:00 | 15125.00 | 15323.51 | 15333.39 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 15438.00 | 15291.14 | 15284.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 15699.00 | 15470.87 | 15385.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 16210.00 | 16518.73 | 16285.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 16210.00 | 16518.73 | 16285.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 16210.00 | 16518.73 | 16285.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:00:00 | 16210.00 | 16518.73 | 16285.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 16212.00 | 16457.38 | 16278.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:15:00 | 16188.00 | 16457.38 | 16278.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 16236.00 | 16329.38 | 16256.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:45:00 | 16183.00 | 16329.38 | 16256.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 16001.00 | 16263.70 | 16233.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 16001.00 | 16263.70 | 16233.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 16050.00 | 16220.96 | 16216.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 16220.00 | 16220.96 | 16216.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 16118.00 | 16200.37 | 16207.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 16118.00 | 16200.37 | 16207.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 13:15:00 | 15813.00 | 16068.83 | 16138.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 16284.00 | 16070.15 | 16118.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 16284.00 | 16070.15 | 16118.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 16284.00 | 16070.15 | 16118.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 16284.00 | 16070.15 | 16118.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 16538.00 | 16163.72 | 16156.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 14:15:00 | 16699.00 | 16376.86 | 16269.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 09:15:00 | 16415.00 | 16424.99 | 16312.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 10:00:00 | 16415.00 | 16424.99 | 16312.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 16321.00 | 16394.44 | 16317.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:00:00 | 16321.00 | 16394.44 | 16317.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 16263.00 | 16368.15 | 16312.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:45:00 | 16270.00 | 16368.15 | 16312.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 16366.00 | 16367.72 | 16317.28 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 16060.00 | 16275.06 | 16285.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 15959.00 | 16211.84 | 16255.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 16439.00 | 16156.61 | 16203.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 14:15:00 | 16439.00 | 16156.61 | 16203.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 16439.00 | 16156.61 | 16203.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:45:00 | 16368.00 | 16156.61 | 16203.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 16353.00 | 16195.89 | 16217.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 16465.00 | 16195.89 | 16217.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 10:15:00 | 16298.00 | 16244.01 | 16236.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 14:15:00 | 16563.00 | 16331.42 | 16280.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 11:15:00 | 16677.00 | 16703.90 | 16573.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 12:00:00 | 16677.00 | 16703.90 | 16573.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 16794.00 | 16706.31 | 16622.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 17190.00 | 16669.31 | 16634.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 11:45:00 | 16918.00 | 16773.13 | 16695.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 12:30:00 | 16932.00 | 16816.30 | 16722.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 17052.00 | 16887.28 | 16788.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 16774.00 | 16879.46 | 16803.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 17520.00 | 16971.51 | 16903.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 14:45:00 | 17181.00 | 17254.44 | 17189.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 17265.00 | 17225.55 | 17182.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 16868.00 | 17152.99 | 17156.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 16868.00 | 17152.99 | 17156.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 16868.00 | 17152.99 | 17156.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 16868.00 | 17152.99 | 17156.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 16868.00 | 17152.99 | 17156.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 16868.00 | 17152.99 | 17156.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 16868.00 | 17152.99 | 17156.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 10:15:00 | 16868.00 | 17152.99 | 17156.58 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 11:15:00 | 17154.00 | 17112.05 | 17108.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 14:15:00 | 17259.00 | 17139.28 | 17121.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 15:15:00 | 17240.00 | 17317.44 | 17247.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 15:15:00 | 17240.00 | 17317.44 | 17247.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 17240.00 | 17317.44 | 17247.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:45:00 | 17205.00 | 17287.95 | 17240.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 17125.00 | 17255.36 | 17230.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 17125.00 | 17255.36 | 17230.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 17338.00 | 17271.83 | 17241.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 17640.00 | 17241.53 | 17235.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 17172.00 | 17227.62 | 17229.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 17172.00 | 17227.62 | 17229.99 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 10:15:00 | 17392.00 | 17260.50 | 17244.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 12:15:00 | 17515.00 | 17421.62 | 17355.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 15:15:00 | 17440.00 | 17443.55 | 17383.67 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:15:00 | 17658.00 | 17443.55 | 17383.67 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 12:45:00 | 17528.00 | 17535.93 | 17454.87 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 17425.00 | 17513.74 | 17452.15 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-10 13:15:00 | 17425.00 | 17513.74 | 17452.15 | SL hit (close<ema400) qty=1.00 sl=17452.15 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-11-10 13:15:00 | 17425.00 | 17513.74 | 17452.15 | SL hit (close<ema400) qty=1.00 sl=17452.15 alert=retest1 |

### Cycle 44 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 17328.00 | 17412.60 | 17416.24 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 17519.00 | 17411.06 | 17405.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 17647.00 | 17458.25 | 17427.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 15:15:00 | 17512.00 | 17541.41 | 17493.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 15:15:00 | 17512.00 | 17541.41 | 17493.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 17512.00 | 17541.41 | 17493.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 17603.00 | 17541.41 | 17493.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 10:15:00 | 17330.00 | 17501.30 | 17483.48 | SL hit (close<static) qty=1.00 sl=17451.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 11:15:00 | 17344.00 | 17469.84 | 17470.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 12:15:00 | 17290.00 | 17433.87 | 17454.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 17227.00 | 17143.54 | 17250.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 17227.00 | 17143.54 | 17250.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 17227.00 | 17143.54 | 17250.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 17227.00 | 17143.54 | 17250.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 17178.00 | 17150.43 | 17243.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:30:00 | 17208.00 | 17150.43 | 17243.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 17190.00 | 17158.35 | 17238.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:45:00 | 17240.00 | 17158.35 | 17238.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 17261.00 | 17178.88 | 17240.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:00:00 | 17261.00 | 17178.88 | 17240.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 17197.00 | 17182.50 | 17236.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:15:00 | 17268.00 | 17182.50 | 17236.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 17315.00 | 17209.00 | 17243.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:45:00 | 17347.00 | 17209.00 | 17243.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 17345.00 | 17236.20 | 17253.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:15:00 | 17379.00 | 17236.20 | 17253.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 17280.00 | 17242.41 | 17251.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 11:45:00 | 17303.00 | 17242.41 | 17251.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 17279.00 | 17249.73 | 17254.31 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 13:15:00 | 17330.00 | 17265.78 | 17261.19 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 17227.00 | 17258.02 | 17258.08 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 15:15:00 | 17260.00 | 17258.42 | 17258.26 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 09:15:00 | 17226.00 | 17251.94 | 17255.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 10:15:00 | 17180.00 | 17237.55 | 17248.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 12:15:00 | 17256.00 | 17232.51 | 17243.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 12:15:00 | 17256.00 | 17232.51 | 17243.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 17256.00 | 17232.51 | 17243.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:00:00 | 17256.00 | 17232.51 | 17243.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 17280.00 | 17242.01 | 17247.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:00:00 | 17280.00 | 17242.01 | 17247.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 17235.00 | 17240.61 | 17245.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:45:00 | 17250.00 | 17240.61 | 17245.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 17249.00 | 17242.29 | 17246.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 17247.00 | 17242.29 | 17246.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 17172.00 | 17228.23 | 17239.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 15:15:00 | 17100.00 | 17204.69 | 17223.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 10:00:00 | 17088.00 | 17164.60 | 17200.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 17081.00 | 17141.42 | 17160.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:15:00 | 17141.00 | 17152.71 | 17162.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 17140.00 | 17150.17 | 17160.90 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 17653.00 | 17258.53 | 17207.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 17653.00 | 17258.53 | 17207.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 17653.00 | 17258.53 | 17207.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 17653.00 | 17258.53 | 17207.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 17653.00 | 17258.53 | 17207.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 18200.00 | 17850.79 | 17614.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 18003.00 | 18058.33 | 17874.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 10:30:00 | 18027.00 | 18058.33 | 17874.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 18070.00 | 18060.31 | 17954.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:30:00 | 18001.00 | 18060.31 | 17954.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 18201.00 | 18275.34 | 18176.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 18438.00 | 18275.34 | 18176.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 11:30:00 | 18376.00 | 18309.02 | 18217.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 12:45:00 | 18350.00 | 18312.40 | 18268.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 09:45:00 | 18355.00 | 18414.34 | 18384.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 18480.00 | 18884.28 | 18790.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 18424.00 | 18884.28 | 18790.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 18747.00 | 18856.82 | 18786.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:30:00 | 18435.00 | 18856.82 | 18786.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 18906.00 | 18866.66 | 18797.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:45:00 | 18712.00 | 18866.66 | 18797.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 18802.00 | 18853.73 | 18797.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:45:00 | 18755.00 | 18853.73 | 18797.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 18787.00 | 18840.38 | 18796.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:45:00 | 18716.00 | 18840.38 | 18796.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 18946.00 | 18861.50 | 18810.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:30:00 | 18830.00 | 18861.50 | 18810.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 18884.00 | 18866.00 | 18817.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 18930.00 | 18866.00 | 18817.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 18805.00 | 18853.80 | 18816.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-10 13:15:00 | 18723.00 | 18798.64 | 18800.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-10 13:15:00 | 18723.00 | 18798.64 | 18800.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-10 13:15:00 | 18723.00 | 18798.64 | 18800.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-10 13:15:00 | 18723.00 | 18798.64 | 18800.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 18723.00 | 18798.64 | 18800.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 18583.00 | 18755.51 | 18780.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 18739.00 | 18726.53 | 18761.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 18739.00 | 18726.53 | 18761.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 18739.00 | 18726.53 | 18761.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:15:00 | 18448.00 | 18687.32 | 18732.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 09:45:00 | 18475.00 | 18511.24 | 18628.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:30:00 | 18433.00 | 18498.39 | 18603.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 14:15:00 | 17525.60 | 17951.08 | 18213.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 14:15:00 | 17551.25 | 17951.08 | 18213.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 14:15:00 | 17511.35 | 17951.08 | 18213.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-18 09:15:00 | 16603.20 | 17288.35 | 17534.29 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-12-18 09:15:00 | 16627.50 | 17288.35 | 17534.29 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-12-18 10:15:00 | 16589.70 | 17124.68 | 17437.53 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 53 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 17490.00 | 17308.88 | 17296.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 17532.00 | 17382.48 | 17333.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 09:15:00 | 18083.00 | 18201.87 | 18051.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 10:00:00 | 18083.00 | 18201.87 | 18051.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 18082.00 | 18177.90 | 18054.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 14:45:00 | 18180.00 | 18223.82 | 18107.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 14:15:00 | 18087.00 | 18389.46 | 18415.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 14:15:00 | 18087.00 | 18389.46 | 18415.61 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 11:15:00 | 18387.00 | 18352.17 | 18349.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 12:15:00 | 18416.00 | 18364.93 | 18355.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 14:15:00 | 18329.00 | 18366.08 | 18358.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 14:15:00 | 18329.00 | 18366.08 | 18358.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 18329.00 | 18366.08 | 18358.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 18329.00 | 18366.08 | 18358.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 15:15:00 | 18250.00 | 18342.86 | 18348.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 12:15:00 | 18170.00 | 18286.71 | 18319.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 18058.00 | 18045.92 | 18134.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 18058.00 | 18045.92 | 18134.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 18058.00 | 18045.92 | 18134.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 14:15:00 | 17941.00 | 18047.77 | 18108.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 15:00:00 | 17916.00 | 17705.97 | 17723.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 17994.00 | 17763.57 | 17747.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 17994.00 | 17763.57 | 17747.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 17994.00 | 17763.57 | 17747.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 18065.00 | 17823.86 | 17776.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 15:15:00 | 17991.00 | 18007.22 | 17911.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 09:15:00 | 18040.00 | 18007.22 | 17911.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 17866.00 | 17978.98 | 17907.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 17866.00 | 17978.98 | 17907.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 17827.00 | 17948.58 | 17899.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:45:00 | 17840.00 | 17948.58 | 17899.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 17831.00 | 17925.07 | 17893.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:30:00 | 17831.00 | 17925.07 | 17893.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 17863.00 | 17904.48 | 17889.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:30:00 | 17831.00 | 17904.48 | 17889.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 18040.00 | 17988.15 | 17936.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:30:00 | 18000.00 | 17988.15 | 17936.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 17935.00 | 17982.30 | 17942.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:00:00 | 17935.00 | 17982.30 | 17942.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 17752.00 | 17936.24 | 17925.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:30:00 | 17764.00 | 17936.24 | 17925.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 17751.00 | 17899.19 | 17909.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 17636.00 | 17789.34 | 17842.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-20 14:15:00 | 18081.00 | 17780.69 | 17809.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 14:15:00 | 18081.00 | 17780.69 | 17809.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 18081.00 | 17780.69 | 17809.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 15:00:00 | 18081.00 | 17780.69 | 17809.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 17501.00 | 17724.75 | 17781.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 09:15:00 | 17394.00 | 17724.75 | 17781.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 10:15:00 | 17729.00 | 17697.20 | 17696.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 17729.00 | 17697.20 | 17696.15 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 17542.00 | 17701.62 | 17709.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 17485.00 | 17658.30 | 17688.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 10:15:00 | 17547.00 | 17538.66 | 17604.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 11:00:00 | 17547.00 | 17538.66 | 17604.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 17595.00 | 17454.93 | 17524.49 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 17846.00 | 17599.63 | 17579.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 17968.00 | 17710.48 | 17635.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 18030.00 | 18248.57 | 18084.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 12:15:00 | 18030.00 | 18248.57 | 18084.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 18030.00 | 18248.57 | 18084.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 18030.00 | 18248.57 | 18084.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 18076.00 | 18214.05 | 18083.67 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 17887.00 | 18039.44 | 18042.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 17640.00 | 17905.52 | 17975.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 17911.00 | 17827.85 | 17907.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 17911.00 | 17827.85 | 17907.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 17911.00 | 17827.85 | 17907.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:45:00 | 17923.00 | 17827.85 | 17907.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 18150.00 | 17892.28 | 17929.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 18325.00 | 17892.28 | 17929.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 18160.00 | 17945.83 | 17950.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:30:00 | 17990.00 | 17949.06 | 17951.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 11:15:00 | 18191.00 | 17690.18 | 17624.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 18191.00 | 17690.18 | 17624.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 18377.00 | 17891.36 | 17731.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 18251.00 | 18430.90 | 18212.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 10:00:00 | 18251.00 | 18430.90 | 18212.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 18440.00 | 18395.75 | 18296.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 10:45:00 | 18656.00 | 18457.00 | 18333.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 13:15:00 | 18175.00 | 18372.02 | 18392.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 13:15:00 | 18175.00 | 18372.02 | 18392.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 15:15:00 | 18150.00 | 18336.41 | 18373.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 18067.00 | 18004.30 | 18137.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:15:00 | 18019.00 | 18004.30 | 18137.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 18290.00 | 18061.44 | 18151.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 18290.00 | 18061.44 | 18151.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 18320.00 | 18113.15 | 18166.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 18260.00 | 18113.15 | 18166.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 18163.00 | 18174.41 | 18187.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 18163.00 | 18174.41 | 18187.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 18270.00 | 18193.52 | 18195.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:30:00 | 18376.00 | 18193.52 | 18195.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 15:15:00 | 18237.00 | 18202.22 | 18199.13 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 17890.00 | 18149.36 | 18180.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 10:15:00 | 17811.00 | 18081.68 | 18147.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 17921.00 | 17754.02 | 17873.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 13:15:00 | 17921.00 | 17754.02 | 17873.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 17921.00 | 17754.02 | 17873.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 17921.00 | 17754.02 | 17873.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 17853.00 | 17773.81 | 17871.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:30:00 | 18040.00 | 17773.81 | 17871.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 17698.00 | 17758.65 | 17855.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 17932.00 | 17758.65 | 17855.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 17789.00 | 17764.72 | 17849.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 17919.00 | 17764.72 | 17849.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 17797.00 | 17771.18 | 17844.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 17797.00 | 17771.18 | 17844.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 17845.00 | 17785.94 | 17844.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:45:00 | 17839.00 | 17785.94 | 17844.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 17900.00 | 17808.75 | 17849.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:00:00 | 17900.00 | 17808.75 | 17849.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 17934.00 | 17833.80 | 17857.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:00:00 | 17934.00 | 17833.80 | 17857.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 17991.00 | 17865.24 | 17869.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:45:00 | 17901.00 | 17865.24 | 17869.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 17850.00 | 17862.19 | 17867.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 17711.00 | 17862.19 | 17867.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 18010.00 | 17879.81 | 17863.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 18010.00 | 17879.81 | 17863.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 18049.00 | 17935.35 | 17899.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 13:15:00 | 17977.00 | 18050.48 | 17986.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 13:15:00 | 17977.00 | 18050.48 | 17986.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 17977.00 | 18050.48 | 17986.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:00:00 | 17977.00 | 18050.48 | 17986.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 17965.00 | 18033.38 | 17984.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 15:15:00 | 17900.00 | 18033.38 | 17984.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 17900.00 | 18006.71 | 17976.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 17847.00 | 18006.71 | 17976.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 17961.00 | 17989.61 | 17973.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:45:00 | 17920.00 | 17989.61 | 17973.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 17837.00 | 17959.09 | 17961.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 17725.00 | 17880.16 | 17918.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 11:15:00 | 17834.00 | 17826.26 | 17884.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 11:15:00 | 17834.00 | 17826.26 | 17884.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 11:15:00 | 17834.00 | 17826.26 | 17884.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 11:30:00 | 17870.00 | 17826.26 | 17884.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 12:15:00 | 17762.00 | 17813.41 | 17873.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 12:30:00 | 17878.00 | 17813.41 | 17873.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 17993.00 | 17849.33 | 17884.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 13:30:00 | 17980.00 | 17849.33 | 17884.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 17988.00 | 17877.06 | 17893.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 15:00:00 | 17988.00 | 17877.06 | 17893.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 09:15:00 | 17930.00 | 17904.12 | 17904.07 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 17821.00 | 17887.50 | 17896.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 17696.00 | 17849.20 | 17878.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 13:15:00 | 18162.00 | 17884.85 | 17887.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 13:15:00 | 18162.00 | 17884.85 | 17887.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 18162.00 | 17884.85 | 17887.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:00:00 | 18162.00 | 17884.85 | 17887.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 14:15:00 | 17925.00 | 17892.88 | 17891.16 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 15:15:00 | 17840.00 | 17882.30 | 17886.51 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 09:15:00 | 17958.00 | 17897.44 | 17893.01 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 13:15:00 | 17852.00 | 17892.03 | 17892.56 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 18000.00 | 17913.63 | 17902.33 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 11:15:00 | 17789.00 | 17886.04 | 17894.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 17210.00 | 17701.71 | 17800.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 17459.00 | 17371.27 | 17538.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 17459.00 | 17371.27 | 17538.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 17459.00 | 17371.27 | 17538.95 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 17771.00 | 17573.46 | 17567.74 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 17451.00 | 17743.86 | 17770.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 17305.00 | 17656.09 | 17728.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 17118.00 | 16920.24 | 17200.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 17118.00 | 16920.24 | 17200.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 17118.00 | 16920.24 | 17200.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 17118.00 | 16920.24 | 17200.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 16859.00 | 16907.99 | 17169.36 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 17352.00 | 17257.25 | 17248.11 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 17087.00 | 17238.27 | 17246.98 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-19 15:15:00 | 17310.00 | 17225.10 | 17224.44 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 12:15:00 | 17164.00 | 17216.58 | 17221.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 13:15:00 | 17106.00 | 17194.47 | 17211.24 | Break + close below crossover candle low |

### Cycle 83 — BUY (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 14:15:00 | 17480.00 | 17251.57 | 17235.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 15:15:00 | 17490.00 | 17299.26 | 17258.79 | Break + close above crossover candle high |

### Cycle 84 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 16931.00 | 17225.61 | 17228.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 16749.00 | 17130.29 | 17185.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 14:15:00 | 16985.00 | 16920.34 | 17048.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 14:15:00 | 16985.00 | 16920.34 | 17048.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 16985.00 | 16920.34 | 17048.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-23 15:00:00 | 16985.00 | 16920.34 | 17048.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 16708.00 | 16858.62 | 16997.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 11:15:00 | 16525.00 | 16833.29 | 16973.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 13:00:00 | 16604.00 | 16736.99 | 16901.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 13:45:00 | 16584.00 | 16694.19 | 16867.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:00:00 | 16539.00 | 16663.15 | 16837.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 09:15:00 | 15698.75 | 16315.74 | 16556.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 09:15:00 | 15773.80 | 16315.74 | 16556.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 09:15:00 | 15754.80 | 16315.74 | 16556.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 09:15:00 | 15712.05 | 16315.74 | 16556.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-30 10:15:00 | 14872.50 | 15407.22 | 15878.62 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-30 10:15:00 | 14943.60 | 15407.22 | 15878.62 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-30 10:15:00 | 14925.60 | 15407.22 | 15878.62 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-30 10:15:00 | 14885.10 | 15407.22 | 15878.62 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 14998.00 | 15064.44 | 15465.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 14968.00 | 15064.44 | 15465.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 14969.00 | 15000.22 | 15302.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 14579.00 | 14995.74 | 15247.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 15372.00 | 15051.81 | 15050.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 15372.00 | 15051.81 | 15050.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 15372.00 | 15051.81 | 15050.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 15372.00 | 15051.81 | 15050.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 15434.00 | 15128.25 | 15084.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 12:15:00 | 16043.00 | 16140.14 | 15839.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-08 13:00:00 | 16043.00 | 16140.14 | 15839.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 15781.00 | 16061.89 | 15855.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 15:00:00 | 15781.00 | 16061.89 | 15855.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 15:15:00 | 15750.00 | 15999.51 | 15845.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:15:00 | 15562.00 | 15999.51 | 15845.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2026-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 11:15:00 | 15460.00 | 15701.35 | 15730.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 09:15:00 | 15182.00 | 15503.29 | 15595.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 15253.00 | 15154.45 | 15336.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-15 10:00:00 | 15253.00 | 15154.45 | 15336.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 13:15:00 | 15655.00 | 15239.38 | 15316.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:00:00 | 15655.00 | 15239.38 | 15316.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 15699.00 | 15331.31 | 15350.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 15:15:00 | 15898.00 | 15331.31 | 15350.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2026-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 15:15:00 | 15898.00 | 15444.65 | 15400.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 15981.00 | 15551.92 | 15453.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 10:15:00 | 15685.00 | 15821.44 | 15685.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 10:15:00 | 15685.00 | 15821.44 | 15685.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 15685.00 | 15821.44 | 15685.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 10:45:00 | 15692.00 | 15821.44 | 15685.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 11:15:00 | 15744.00 | 15805.95 | 15691.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 09:15:00 | 15828.00 | 15750.71 | 15699.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 09:45:00 | 15805.00 | 15762.77 | 15709.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 14:15:00 | 15889.00 | 15982.37 | 15983.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-22 14:15:00 | 15889.00 | 15982.37 | 15983.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 14:15:00 | 15889.00 | 15982.37 | 15983.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 15660.00 | 15880.80 | 15924.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 16121.00 | 15859.22 | 15892.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 14:15:00 | 16121.00 | 15859.22 | 15892.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 16121.00 | 15859.22 | 15892.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 16121.00 | 15859.22 | 15892.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 16050.00 | 15897.38 | 15906.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 16133.00 | 15897.38 | 15906.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 16201.00 | 15958.10 | 15933.57 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 15952.00 | 16025.29 | 16025.56 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 16047.00 | 16029.63 | 16027.51 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 10:15:00 | 15966.00 | 16016.91 | 16021.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 13:15:00 | 15882.00 | 15974.94 | 16000.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 16060.00 | 15891.44 | 15927.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 14:15:00 | 16060.00 | 15891.44 | 15927.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 16060.00 | 15891.44 | 15927.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:45:00 | 16067.00 | 15891.44 | 15927.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 16140.00 | 15941.16 | 15946.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 16307.00 | 15941.16 | 15946.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 16320.00 | 16016.92 | 15980.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 15:15:00 | 16490.00 | 16318.49 | 16211.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 16784.00 | 16972.30 | 16802.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 14:15:00 | 16784.00 | 16972.30 | 16802.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 16784.00 | 16972.30 | 16802.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 15:00:00 | 16784.00 | 16972.30 | 16802.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 16820.00 | 16941.84 | 16804.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 16950.00 | 16941.84 | 16804.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 14:00:00 | 16867.00 | 16900.00 | 16833.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-20 10:45:00 | 14319.00 | 2025-05-22 14:15:00 | 15620.00 | TARGET_HIT | 1.00 | 9.09% |
| BUY | retest2 | 2025-05-20 12:00:00 | 14200.00 | 2025-05-22 14:15:00 | 15582.60 | TARGET_HIT | 1.00 | 9.74% |
| BUY | retest2 | 2025-05-21 09:15:00 | 14380.00 | 2025-05-23 09:15:00 | 15750.90 | TARGET_HIT | 1.00 | 9.53% |
| BUY | retest2 | 2025-05-21 14:00:00 | 14166.00 | 2025-05-26 13:15:00 | 15818.00 | TARGET_HIT | 1.00 | 11.66% |
| BUY | retest2 | 2025-05-21 15:15:00 | 14460.00 | 2025-05-28 09:15:00 | 15896.10 | TARGET_HIT | 1.00 | 9.93% |
| BUY | retest2 | 2025-05-22 09:45:00 | 14451.00 | 2025-05-29 11:15:00 | 15329.00 | STOP_HIT | 1.00 | 6.08% |
| SELL | retest2 | 2025-06-03 15:15:00 | 15276.00 | 2025-06-05 09:15:00 | 14512.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-04 11:00:00 | 15266.00 | 2025-06-05 09:15:00 | 14502.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-03 15:15:00 | 15276.00 | 2025-06-05 15:15:00 | 14858.00 | STOP_HIT | 0.50 | 2.74% |
| SELL | retest2 | 2025-06-04 11:00:00 | 15266.00 | 2025-06-05 15:15:00 | 14858.00 | STOP_HIT | 0.50 | 2.67% |
| SELL | retest2 | 2025-07-07 09:15:00 | 14822.00 | 2025-07-11 09:15:00 | 14080.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-07 15:00:00 | 14825.00 | 2025-07-11 09:15:00 | 14083.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-08 09:30:00 | 14799.00 | 2025-07-11 09:15:00 | 14098.95 | PARTIAL | 0.50 | 4.73% |
| SELL | retest2 | 2025-07-09 09:15:00 | 14841.00 | 2025-07-11 10:15:00 | 14059.05 | PARTIAL | 0.50 | 5.27% |
| SELL | retest2 | 2025-07-07 09:15:00 | 14822.00 | 2025-07-11 14:15:00 | 14423.00 | STOP_HIT | 0.50 | 2.69% |
| SELL | retest2 | 2025-07-07 15:00:00 | 14825.00 | 2025-07-11 14:15:00 | 14423.00 | STOP_HIT | 0.50 | 2.71% |
| SELL | retest2 | 2025-07-08 09:30:00 | 14799.00 | 2025-07-11 14:15:00 | 14423.00 | STOP_HIT | 0.50 | 2.54% |
| SELL | retest2 | 2025-07-09 09:15:00 | 14841.00 | 2025-07-11 14:15:00 | 14423.00 | STOP_HIT | 0.50 | 2.82% |
| SELL | retest2 | 2025-07-10 10:15:00 | 14536.00 | 2025-07-15 11:15:00 | 14494.00 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2025-07-14 09:15:00 | 14169.00 | 2025-07-15 11:15:00 | 14494.00 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-07-16 13:00:00 | 14459.00 | 2025-07-18 11:15:00 | 14422.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-07-16 13:45:00 | 14466.00 | 2025-07-18 11:15:00 | 14422.00 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-07-18 09:15:00 | 14510.00 | 2025-07-18 11:15:00 | 14422.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-07-24 12:00:00 | 14308.00 | 2025-07-24 14:15:00 | 14550.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-07-31 14:30:00 | 14960.00 | 2025-08-04 11:15:00 | 14683.00 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-08-01 13:15:00 | 14845.00 | 2025-08-04 11:15:00 | 14683.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-08-01 14:30:00 | 14848.00 | 2025-08-04 11:15:00 | 14683.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-08-21 11:15:00 | 13454.00 | 2025-08-22 09:15:00 | 14168.00 | STOP_HIT | 1.00 | -5.31% |
| SELL | retest2 | 2025-08-21 13:30:00 | 13463.00 | 2025-08-22 09:15:00 | 14168.00 | STOP_HIT | 1.00 | -5.24% |
| SELL | retest2 | 2025-09-03 10:15:00 | 13812.00 | 2025-09-10 13:15:00 | 13710.00 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest1 | 2025-09-11 14:00:00 | 14001.00 | 2025-09-16 14:15:00 | 14701.05 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-09-11 14:00:00 | 14001.00 | 2025-09-17 12:15:00 | 15401.10 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-09-22 09:15:00 | 15138.00 | 2025-09-26 13:15:00 | 15125.00 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-10-08 09:15:00 | 16220.00 | 2025-10-08 09:15:00 | 16118.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-10-20 09:15:00 | 17190.00 | 2025-10-29 10:15:00 | 16868.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-10-20 11:45:00 | 16918.00 | 2025-10-29 10:15:00 | 16868.00 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-10-20 12:30:00 | 16932.00 | 2025-10-29 10:15:00 | 16868.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-10-21 13:45:00 | 17052.00 | 2025-10-29 10:15:00 | 16868.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-10-27 09:15:00 | 17520.00 | 2025-10-29 10:15:00 | 16868.00 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2025-10-28 14:45:00 | 17181.00 | 2025-10-29 10:15:00 | 16868.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-10-29 09:15:00 | 17265.00 | 2025-10-29 10:15:00 | 16868.00 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-11-06 09:15:00 | 17640.00 | 2025-11-06 09:15:00 | 17172.00 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest1 | 2025-11-10 09:15:00 | 17658.00 | 2025-11-10 13:15:00 | 17425.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest1 | 2025-11-10 12:45:00 | 17528.00 | 2025-11-10 13:15:00 | 17425.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-11-13 09:15:00 | 17603.00 | 2025-11-13 10:15:00 | 17330.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-11-20 15:15:00 | 17100.00 | 2025-11-24 14:15:00 | 17653.00 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2025-11-21 10:00:00 | 17088.00 | 2025-11-24 14:15:00 | 17653.00 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2025-11-24 09:15:00 | 17081.00 | 2025-11-24 14:15:00 | 17653.00 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2025-11-24 11:15:00 | 17141.00 | 2025-11-24 14:15:00 | 17653.00 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2025-12-02 09:15:00 | 18438.00 | 2025-12-10 13:15:00 | 18723.00 | STOP_HIT | 1.00 | 1.55% |
| BUY | retest2 | 2025-12-02 11:30:00 | 18376.00 | 2025-12-10 13:15:00 | 18723.00 | STOP_HIT | 1.00 | 1.89% |
| BUY | retest2 | 2025-12-03 12:45:00 | 18350.00 | 2025-12-10 13:15:00 | 18723.00 | STOP_HIT | 1.00 | 2.03% |
| BUY | retest2 | 2025-12-05 09:45:00 | 18355.00 | 2025-12-10 13:15:00 | 18723.00 | STOP_HIT | 1.00 | 2.00% |
| SELL | retest2 | 2025-12-11 14:15:00 | 18448.00 | 2025-12-15 14:15:00 | 17525.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 09:45:00 | 18475.00 | 2025-12-15 14:15:00 | 17551.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 11:30:00 | 18433.00 | 2025-12-15 14:15:00 | 17511.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-11 14:15:00 | 18448.00 | 2025-12-18 09:15:00 | 16603.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-12 09:45:00 | 18475.00 | 2025-12-18 09:15:00 | 16627.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-12 11:30:00 | 18433.00 | 2025-12-18 10:15:00 | 16589.70 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-12-26 14:45:00 | 18180.00 | 2025-12-30 14:15:00 | 18087.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2026-01-06 14:15:00 | 17941.00 | 2026-01-12 15:15:00 | 17994.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2026-01-12 15:00:00 | 17916.00 | 2026-01-12 15:15:00 | 17994.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2026-01-21 09:15:00 | 17394.00 | 2026-01-22 10:15:00 | 17729.00 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-02-03 10:30:00 | 17990.00 | 2026-02-09 11:15:00 | 18191.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-02-12 10:45:00 | 18656.00 | 2026-02-13 13:15:00 | 18175.00 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2026-02-24 09:15:00 | 17711.00 | 2026-02-25 09:15:00 | 18010.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2026-03-24 11:15:00 | 16525.00 | 2026-03-27 09:15:00 | 15698.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-24 13:00:00 | 16604.00 | 2026-03-27 09:15:00 | 15773.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-24 13:45:00 | 16584.00 | 2026-03-27 09:15:00 | 15754.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-24 15:00:00 | 16539.00 | 2026-03-27 09:15:00 | 15712.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-24 11:15:00 | 16525.00 | 2026-03-30 10:15:00 | 14872.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-24 13:00:00 | 16604.00 | 2026-03-30 10:15:00 | 14943.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-24 13:45:00 | 16584.00 | 2026-03-30 10:15:00 | 14925.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-24 15:00:00 | 16539.00 | 2026-03-30 10:15:00 | 14885.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 14968.00 | 2026-04-06 12:15:00 | 15372.00 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2026-04-01 13:30:00 | 14969.00 | 2026-04-06 12:15:00 | 15372.00 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2026-04-02 09:15:00 | 14579.00 | 2026-04-06 12:15:00 | 15372.00 | STOP_HIT | 1.00 | -5.44% |
| BUY | retest2 | 2026-04-20 09:15:00 | 15828.00 | 2026-04-22 14:15:00 | 15889.00 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2026-04-20 09:45:00 | 15805.00 | 2026-04-22 14:15:00 | 15889.00 | STOP_HIT | 1.00 | 0.53% |
