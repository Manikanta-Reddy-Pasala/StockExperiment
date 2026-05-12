# Abbott India Ltd. (ABBOTINDIA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 26850.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 61 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 27 |
| PARTIAL | 5 |
| TARGET_HIT | 8 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 35 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 15
- **Target hits / Stop hits / Partials:** 8 / 22 / 5
- **Avg / median % per leg:** 2.14% / 2.59%
- **Sum % (uncompounded):** 74.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 10 | 47.6% | 8 | 13 | 0 | 2.90% | 60.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 10 | 47.6% | 8 | 13 | 0 | 2.90% | 60.8% |
| SELL (all) | 14 | 10 | 71.4% | 0 | 9 | 5 | 1.00% | 14.0% |
| SELL @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 3.55% | 21.3% |
| SELL @ 3rd Alert (retest2) | 8 | 4 | 50.0% | 0 | 6 | 2 | -0.91% | -7.3% |
| retest1 (combined) | 6 | 6 | 100.0% | 0 | 3 | 3 | 3.55% | 21.3% |
| retest2 (combined) | 29 | 14 | 48.3% | 8 | 19 | 2 | 1.85% | 53.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 13:15:00 | 27991.75 | 26523.61 | 26517.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 09:15:00 | 28260.10 | 26568.14 | 26540.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 09:15:00 | 26846.15 | 26867.10 | 26714.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-20 09:30:00 | 26895.95 | 26867.10 | 26714.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 26710.00 | 26879.51 | 26731.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 09:15:00 | 26883.10 | 26875.59 | 26733.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 15:15:00 | 26998.00 | 26866.36 | 26733.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 10:00:00 | 26979.30 | 27914.58 | 27603.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 14:00:00 | 26900.00 | 27876.49 | 27590.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-22 09:15:00 | 29571.41 | 27836.03 | 27619.93 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 12:15:00 | 27367.85 | 28597.69 | 28602.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 13:15:00 | 27311.50 | 28584.89 | 28596.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 11:15:00 | 28157.15 | 28100.20 | 28310.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-02 11:30:00 | 28143.00 | 28100.20 | 28310.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 28327.75 | 28102.04 | 28304.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 11:45:00 | 28297.85 | 28102.04 | 28304.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 12:15:00 | 28332.55 | 28104.33 | 28304.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 13:15:00 | 28340.55 | 28104.33 | 28304.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 13:15:00 | 28240.00 | 28105.68 | 28304.39 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 15:15:00 | 28635.05 | 28456.61 | 28456.58 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 13:15:00 | 28340.85 | 28455.51 | 28456.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 14:15:00 | 28243.75 | 28453.40 | 28454.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 10:15:00 | 28577.70 | 28411.02 | 28432.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 10:15:00 | 28577.70 | 28411.02 | 28432.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 10:15:00 | 28577.70 | 28411.02 | 28432.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:00:00 | 28577.70 | 28411.02 | 28432.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 28745.25 | 28414.34 | 28434.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:45:00 | 28781.75 | 28414.34 | 28434.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 11:15:00 | 29016.95 | 28454.50 | 28453.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 29489.05 | 28494.48 | 28475.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 11:15:00 | 29122.05 | 29140.78 | 28862.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-10 12:00:00 | 29122.05 | 29140.78 | 28862.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 13:15:00 | 28840.05 | 29136.71 | 28862.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 14:00:00 | 28840.05 | 29136.71 | 28862.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 14:15:00 | 28783.95 | 29133.20 | 28862.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 15:00:00 | 28783.95 | 29133.20 | 28862.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 15:15:00 | 28849.10 | 29130.37 | 28862.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 09:15:00 | 28549.95 | 29130.37 | 28862.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 27773.70 | 28655.44 | 28656.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 27659.50 | 28645.53 | 28651.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 13:15:00 | 27530.00 | 27377.24 | 27902.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-05 13:45:00 | 27631.50 | 27377.24 | 27902.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 15:15:00 | 27800.00 | 27385.42 | 27901.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:15:00 | 29241.00 | 27385.42 | 27901.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 29210.15 | 27403.58 | 27908.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 29053.05 | 27403.58 | 27908.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 28223.30 | 27964.78 | 28120.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:45:00 | 28229.55 | 27964.78 | 28120.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 28245.35 | 27967.57 | 28121.37 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 09:15:00 | 29006.75 | 28255.40 | 28254.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-24 14:15:00 | 29822.30 | 28344.48 | 28300.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 14:15:00 | 29600.55 | 29636.08 | 29097.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-13 15:00:00 | 29600.55 | 29636.08 | 29097.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 29560.00 | 30112.77 | 29564.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 29560.00 | 30112.77 | 29564.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 29501.00 | 30106.68 | 29564.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 30279.85 | 30106.68 | 29564.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 29360.00 | 30090.66 | 29577.76 | SL hit (close<static) qty=1.00 sl=29401.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 31030.00 | 32679.58 | 32686.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 13:15:00 | 30945.00 | 32554.72 | 32622.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 15:15:00 | 29780.00 | 29700.23 | 30225.87 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 09:15:00 | 29515.00 | 29700.23 | 30225.87 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 10:00:00 | 29565.00 | 29698.89 | 30222.58 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 10:45:00 | 29565.00 | 29697.45 | 30219.25 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 30055.00 | 29702.87 | 30209.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:30:00 | 29830.00 | 29725.55 | 30200.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:30:00 | 29865.00 | 29707.86 | 30159.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 28338.50 | 29463.00 | 29942.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 28371.75 | 29463.00 | 29942.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 14:15:00 | 28039.25 | 29400.24 | 29899.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 14:15:00 | 28086.75 | 29400.24 | 29899.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 14:15:00 | 28086.75 | 29400.24 | 29899.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-12-24 15:15:00 | 28750.00 | 28647.72 | 29286.12 | SL hit (close>ema200) qty=0.50 sl=28647.72 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-23 09:30:00 | 25950.00 | 2024-06-05 09:15:00 | 27719.95 | STOP_HIT | 1.00 | -6.82% |
| SELL | retest2 | 2024-05-24 09:15:00 | 26046.70 | 2024-06-05 09:15:00 | 27719.95 | STOP_HIT | 1.00 | -6.42% |
| SELL | retest2 | 2024-05-28 11:30:00 | 26001.20 | 2024-06-05 09:15:00 | 27719.95 | STOP_HIT | 1.00 | -6.61% |
| SELL | retest2 | 2024-05-30 09:15:00 | 26011.90 | 2024-06-05 09:15:00 | 27719.95 | STOP_HIT | 1.00 | -6.57% |
| BUY | retest2 | 2024-06-25 09:15:00 | 26883.10 | 2024-08-22 09:15:00 | 29571.41 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-25 15:15:00 | 26998.00 | 2024-08-22 09:15:00 | 29590.00 | TARGET_HIT | 1.00 | 9.60% |
| BUY | retest2 | 2024-08-12 10:00:00 | 26979.30 | 2024-08-22 10:15:00 | 29697.80 | TARGET_HIT | 1.00 | 10.08% |
| BUY | retest2 | 2024-08-12 14:00:00 | 26900.00 | 2024-08-22 10:15:00 | 29677.23 | TARGET_HIT | 1.00 | 10.32% |
| BUY | retest2 | 2024-11-08 09:30:00 | 28721.00 | 2024-11-13 10:15:00 | 28143.85 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-11-08 11:45:00 | 28575.75 | 2024-11-13 10:15:00 | 28143.85 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-11-11 09:30:00 | 28614.40 | 2024-11-13 10:15:00 | 28143.85 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-04-03 09:15:00 | 30279.85 | 2025-04-04 09:15:00 | 29360.00 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-04-11 09:15:00 | 29696.90 | 2025-04-11 09:15:00 | 29325.60 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-04-11 13:00:00 | 29604.85 | 2025-05-07 09:15:00 | 29660.00 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-04-11 14:30:00 | 29630.00 | 2025-05-07 09:15:00 | 29660.00 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2025-04-25 14:45:00 | 29875.00 | 2025-05-07 09:15:00 | 29660.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-04-29 11:15:00 | 29855.00 | 2025-05-07 09:15:00 | 29660.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-04-29 13:30:00 | 29880.00 | 2025-05-07 09:15:00 | 29660.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-04-30 12:45:00 | 29845.00 | 2025-05-07 09:15:00 | 29660.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-05-02 11:30:00 | 30140.00 | 2025-05-07 09:15:00 | 29660.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-05-02 12:30:00 | 30150.00 | 2025-05-09 09:15:00 | 29670.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-05-06 14:30:00 | 30145.00 | 2025-06-26 10:15:00 | 32565.34 | TARGET_HIT | 1.00 | 8.03% |
| BUY | retest2 | 2025-05-07 12:45:00 | 30220.00 | 2025-06-26 10:15:00 | 32593.00 | TARGET_HIT | 1.00 | 7.85% |
| BUY | retest2 | 2025-05-09 14:45:00 | 30210.00 | 2025-06-26 11:15:00 | 33132.00 | TARGET_HIT | 1.00 | 9.67% |
| BUY | retest2 | 2025-05-12 14:00:00 | 30120.00 | 2025-06-26 15:15:00 | 33231.00 | TARGET_HIT | 1.00 | 10.33% |
| SELL | retest1 | 2025-11-25 09:15:00 | 29515.00 | 2025-12-09 09:15:00 | 28338.50 | PARTIAL | 0.50 | 3.99% |
| SELL | retest1 | 2025-11-25 10:00:00 | 29565.00 | 2025-12-09 09:15:00 | 28371.75 | PARTIAL | 0.50 | 4.04% |
| SELL | retest1 | 2025-11-25 10:45:00 | 29565.00 | 2025-12-09 14:15:00 | 28039.25 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2025-11-27 09:30:00 | 29830.00 | 2025-12-09 14:15:00 | 28086.75 | PARTIAL | 0.50 | 5.84% |
| SELL | retest2 | 2025-12-01 09:30:00 | 29865.00 | 2025-12-09 14:15:00 | 28086.75 | PARTIAL | 0.50 | 5.95% |
| SELL | retest1 | 2025-11-25 09:15:00 | 29515.00 | 2025-12-24 15:15:00 | 28750.00 | STOP_HIT | 0.50 | 2.59% |
| SELL | retest1 | 2025-11-25 10:00:00 | 29565.00 | 2025-12-24 15:15:00 | 28750.00 | STOP_HIT | 0.50 | 2.76% |
| SELL | retest1 | 2025-11-25 10:45:00 | 29565.00 | 2025-12-24 15:15:00 | 28750.00 | STOP_HIT | 0.50 | 2.76% |
| SELL | retest2 | 2025-11-27 09:30:00 | 29830.00 | 2025-12-24 15:15:00 | 28750.00 | STOP_HIT | 0.50 | 3.62% |
| SELL | retest2 | 2025-12-01 09:30:00 | 29865.00 | 2025-12-24 15:15:00 | 28750.00 | STOP_HIT | 0.50 | 3.73% |
