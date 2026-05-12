# Abbott India Ltd. (ABBOTINDIA)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 26850.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 162 |
| ALERT1 | 114 |
| ALERT2 | 112 |
| ALERT2_SKIP | 64 |
| ALERT3 | 320 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 136 |
| PARTIAL | 11 |
| TARGET_HIT | 0 |
| STOP_HIT | 140 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 151 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 106
- **Target hits / Stop hits / Partials:** 0 / 140 / 11
- **Avg / median % per leg:** 0.18% / -0.63%
- **Sum % (uncompounded):** 26.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 56 | 10 | 17.9% | 0 | 56 | 0 | -0.55% | -30.9% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.77% | -1.5% |
| BUY @ 3rd Alert (retest2) | 54 | 10 | 18.5% | 0 | 54 | 0 | -0.54% | -29.4% |
| SELL (all) | 95 | 35 | 36.8% | 0 | 84 | 11 | 0.60% | 57.4% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 2 | 0 | 0.31% | 0.6% |
| SELL @ 3rd Alert (retest2) | 93 | 33 | 35.5% | 0 | 82 | 11 | 0.61% | 56.7% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 4 | 0 | -0.23% | -0.9% |
| retest2 (combined) | 147 | 43 | 29.3% | 0 | 136 | 11 | 0.19% | 27.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 14:15:00 | 26438.95 | 26583.80 | 26590.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 15:15:00 | 26360.00 | 26539.04 | 26569.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 26504.00 | 26426.78 | 26489.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 14:15:00 | 26504.00 | 26426.78 | 26489.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 26504.00 | 26426.78 | 26489.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 26504.00 | 26426.78 | 26489.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 26598.75 | 26461.17 | 26499.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 10:00:00 | 26481.00 | 26465.14 | 26498.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 10:30:00 | 26450.00 | 26466.11 | 26495.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 12:45:00 | 26362.90 | 26448.72 | 26482.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-18 11:15:00 | 26689.25 | 26516.19 | 26500.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 11:15:00 | 26689.25 | 26516.19 | 26500.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 12:15:00 | 26748.00 | 26562.56 | 26523.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 26388.30 | 26527.70 | 26510.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 26388.30 | 26527.70 | 26510.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 26388.30 | 26527.70 | 26510.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:30:00 | 26392.65 | 26527.70 | 26510.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 26410.00 | 26504.16 | 26501.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:30:00 | 26375.10 | 26504.16 | 26501.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 11:15:00 | 26409.75 | 26485.28 | 26493.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 12:15:00 | 26290.00 | 26393.78 | 26436.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 14:15:00 | 26145.00 | 26095.95 | 26213.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-23 15:00:00 | 26145.00 | 26095.95 | 26213.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 26158.40 | 26113.09 | 26201.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:45:00 | 26168.90 | 26113.09 | 26201.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 26319.30 | 26154.33 | 26212.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 11:00:00 | 26319.30 | 26154.33 | 26212.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 26170.05 | 26157.47 | 26208.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 15:15:00 | 26169.55 | 26199.91 | 26217.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 10:00:00 | 26100.50 | 26175.17 | 26203.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 12:45:00 | 26131.00 | 26216.14 | 26217.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 13:15:00 | 26230.65 | 26219.04 | 26218.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 13:15:00 | 26230.65 | 26219.04 | 26218.74 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 26162.75 | 26211.62 | 26216.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 11:15:00 | 26021.00 | 26157.24 | 26189.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 26046.90 | 26011.66 | 26091.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 10:00:00 | 26046.90 | 26011.66 | 26091.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 26173.85 | 26044.10 | 26098.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:45:00 | 26169.95 | 26044.10 | 26098.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 26220.00 | 26079.28 | 26109.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 11:45:00 | 26265.70 | 26079.28 | 26109.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 13:15:00 | 26340.00 | 26166.42 | 26146.25 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 25868.75 | 26106.36 | 26124.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 09:15:00 | 25820.80 | 25980.32 | 26045.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 12:15:00 | 25990.00 | 25945.81 | 26010.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 12:15:00 | 25990.00 | 25945.81 | 26010.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 25990.00 | 25945.81 | 26010.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:00:00 | 25990.00 | 25945.81 | 26010.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 25981.70 | 25952.99 | 26007.48 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 26178.20 | 26040.94 | 26024.52 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-03 15:15:00 | 25875.60 | 25995.03 | 26007.56 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 09:15:00 | 26124.00 | 26020.82 | 26018.15 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 25692.50 | 25955.16 | 25988.54 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2024-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 13:15:00 | 26217.70 | 26013.31 | 26007.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 14:15:00 | 26280.25 | 26066.70 | 26032.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 09:15:00 | 27132.80 | 27222.41 | 26823.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 10:00:00 | 27132.80 | 27222.41 | 26823.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 26999.95 | 27100.30 | 26907.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 14:30:00 | 26936.95 | 27100.30 | 26907.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 27725.95 | 27961.27 | 27785.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 10:00:00 | 27725.95 | 27961.27 | 27785.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 27544.70 | 27877.96 | 27763.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 10:45:00 | 27588.00 | 27877.96 | 27763.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2024-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 15:15:00 | 27525.00 | 27672.35 | 27691.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 12:15:00 | 27449.95 | 27566.12 | 27615.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 15:15:00 | 26948.00 | 26913.78 | 27045.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 09:15:00 | 27194.40 | 26913.78 | 27045.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 27331.20 | 26997.27 | 27071.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:30:00 | 27215.95 | 26997.27 | 27071.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 27285.70 | 27054.95 | 27090.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 11:30:00 | 27234.20 | 27090.77 | 27103.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 12:30:00 | 27221.05 | 27082.41 | 27098.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 10:15:00 | 27667.30 | 27032.38 | 26949.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 10:15:00 | 27667.30 | 27032.38 | 26949.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 13:15:00 | 27701.05 | 27329.76 | 27120.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 11:15:00 | 27500.00 | 27501.03 | 27300.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 14:15:00 | 27398.95 | 27489.22 | 27346.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 27398.95 | 27489.22 | 27346.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 27398.95 | 27489.22 | 27346.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 27280.00 | 27447.38 | 27340.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 09:15:00 | 27915.45 | 27447.38 | 27340.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 10:15:00 | 27942.00 | 28054.85 | 28056.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2024-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 10:15:00 | 27942.00 | 28054.85 | 28056.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 13:15:00 | 27833.20 | 27986.36 | 28022.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 09:15:00 | 28029.45 | 27666.28 | 27767.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 09:15:00 | 28029.45 | 27666.28 | 27767.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 28029.45 | 27666.28 | 27767.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:00:00 | 28029.45 | 27666.28 | 27767.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 28087.40 | 27750.51 | 27796.22 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2024-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 12:15:00 | 28369.05 | 27913.82 | 27864.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 13:15:00 | 28434.40 | 28017.94 | 27916.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 28210.75 | 28600.21 | 28489.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 09:15:00 | 28210.75 | 28600.21 | 28489.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 28210.75 | 28600.21 | 28489.08 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2024-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 11:15:00 | 27920.00 | 28401.90 | 28414.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 12:15:00 | 27720.00 | 28265.52 | 28351.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 28000.00 | 27835.28 | 28064.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 11:00:00 | 28000.00 | 27835.28 | 28064.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 27686.75 | 27827.19 | 27967.60 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2024-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 14:15:00 | 28018.65 | 27788.24 | 27787.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 15:15:00 | 28198.00 | 27870.19 | 27824.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 09:15:00 | 28246.15 | 28433.48 | 28231.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 09:15:00 | 28246.15 | 28433.48 | 28231.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 28246.15 | 28433.48 | 28231.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 10:00:00 | 28246.15 | 28433.48 | 28231.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 28152.05 | 28377.19 | 28224.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 11:00:00 | 28152.05 | 28377.19 | 28224.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 11:15:00 | 28080.00 | 28317.76 | 28211.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 12:00:00 | 28080.00 | 28317.76 | 28211.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 12:15:00 | 28014.00 | 28257.00 | 28193.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 12:45:00 | 27900.00 | 28257.00 | 28193.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 09:15:00 | 28024.75 | 28132.63 | 28146.37 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 13:15:00 | 28239.95 | 28163.15 | 28155.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 09:15:00 | 28428.70 | 28217.55 | 28182.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 14:15:00 | 28419.60 | 28440.55 | 28328.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 15:00:00 | 28419.60 | 28440.55 | 28328.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 28400.00 | 28432.44 | 28335.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:30:00 | 28531.00 | 28449.96 | 28352.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 11:30:00 | 28560.00 | 28453.52 | 28370.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 15:00:00 | 28450.00 | 28455.96 | 28393.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 09:15:00 | 28550.35 | 28432.74 | 28388.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 28492.15 | 28444.62 | 28397.74 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-02 14:15:00 | 28204.25 | 28408.45 | 28398.90 | SL hit (close<static) qty=1.00 sl=28332.00 alert=retest2 |

### Cycle 21 — SELL (started 2024-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 15:15:00 | 28244.30 | 28375.62 | 28384.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 27921.00 | 28278.12 | 28338.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 28520.00 | 28188.94 | 28246.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 28520.00 | 28188.94 | 28246.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 28520.00 | 28188.94 | 28246.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 28520.00 | 28188.94 | 28246.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 28370.25 | 28225.20 | 28258.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:00:00 | 28265.90 | 28233.34 | 28258.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 13:15:00 | 28410.00 | 28230.46 | 28210.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 28410.00 | 28230.46 | 28210.39 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 15:15:00 | 28000.00 | 28179.50 | 28190.40 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 28970.65 | 28337.73 | 28261.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 10:15:00 | 29199.95 | 28510.17 | 28346.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 11:15:00 | 28207.15 | 28449.57 | 28333.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 11:15:00 | 28207.15 | 28449.57 | 28333.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 11:15:00 | 28207.15 | 28449.57 | 28333.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 12:00:00 | 28207.15 | 28449.57 | 28333.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 12:15:00 | 27470.00 | 28253.65 | 28255.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 13:15:00 | 27254.00 | 28053.72 | 28164.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 13:15:00 | 27576.35 | 27573.45 | 27805.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-09 14:00:00 | 27576.35 | 27573.45 | 27805.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 27300.00 | 27146.54 | 27334.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:45:00 | 27335.95 | 27146.54 | 27334.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 27400.00 | 27197.23 | 27340.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:00:00 | 27400.00 | 27197.23 | 27340.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 27300.05 | 27217.79 | 27336.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 09:30:00 | 27276.10 | 27247.38 | 27316.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 10:00:00 | 27183.95 | 27247.38 | 27316.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 11:00:00 | 27265.00 | 27250.90 | 27312.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 10:45:00 | 27265.50 | 27149.64 | 27211.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 11:15:00 | 27700.10 | 27259.73 | 27256.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 27700.10 | 27259.73 | 27256.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 12:15:00 | 27921.05 | 27391.99 | 27316.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 10:15:00 | 27809.50 | 27937.14 | 27767.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 11:00:00 | 27809.50 | 27937.14 | 27767.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 27829.70 | 27915.65 | 27773.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 27745.00 | 27915.65 | 27773.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 27731.40 | 27878.80 | 27769.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:45:00 | 27788.80 | 27878.80 | 27769.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 27780.00 | 27859.04 | 27770.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 28031.10 | 27842.58 | 27778.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 14:15:00 | 29666.00 | 29908.58 | 29938.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 14:15:00 | 29666.00 | 29908.58 | 29938.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 15:15:00 | 29625.20 | 29754.41 | 29805.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 09:15:00 | 29872.35 | 29778.00 | 29811.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 29872.35 | 29778.00 | 29811.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 29872.35 | 29778.00 | 29811.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 29872.35 | 29778.00 | 29811.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 29879.90 | 29798.38 | 29817.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:30:00 | 29840.00 | 29798.38 | 29817.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 29908.45 | 29820.39 | 29826.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:45:00 | 29894.15 | 29820.39 | 29826.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2024-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 12:15:00 | 29905.30 | 29837.38 | 29833.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 14:15:00 | 29941.35 | 29872.24 | 29850.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 15:15:00 | 29692.85 | 29836.36 | 29836.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 15:15:00 | 29692.85 | 29836.36 | 29836.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 29692.85 | 29836.36 | 29836.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 29938.55 | 29836.36 | 29836.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 29899.40 | 29848.97 | 29842.08 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 12:15:00 | 29765.70 | 29826.39 | 29833.54 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 14:15:00 | 29909.95 | 29846.88 | 29841.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 15:15:00 | 29927.60 | 29863.03 | 29849.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 09:15:00 | 29789.80 | 29848.38 | 29844.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 29789.80 | 29848.38 | 29844.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 29789.80 | 29848.38 | 29844.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:00:00 | 29789.80 | 29848.38 | 29844.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 29821.90 | 29843.08 | 29842.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:45:00 | 29750.00 | 29843.08 | 29842.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 12:15:00 | 29894.40 | 29870.36 | 29855.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 12:45:00 | 29880.50 | 29870.36 | 29855.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 29881.75 | 29872.64 | 29858.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 13:30:00 | 29904.60 | 29872.64 | 29858.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 29800.00 | 29858.11 | 29852.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 15:00:00 | 29800.00 | 29858.11 | 29852.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2024-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 15:15:00 | 29780.00 | 29842.49 | 29846.23 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 12:15:00 | 29871.00 | 29849.03 | 29848.04 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 13:15:00 | 29750.90 | 29829.40 | 29839.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 14:15:00 | 29529.00 | 29769.32 | 29811.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 11:15:00 | 29766.50 | 29662.24 | 29736.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 11:15:00 | 29766.50 | 29662.24 | 29736.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 29766.50 | 29662.24 | 29736.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:45:00 | 29718.10 | 29662.24 | 29736.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 29656.10 | 29661.01 | 29729.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 13:15:00 | 29550.10 | 29661.01 | 29729.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 09:15:00 | 29586.90 | 29668.05 | 29714.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 11:15:00 | 29576.10 | 29650.53 | 29698.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 12:30:00 | 29585.20 | 29636.97 | 29683.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 29450.00 | 29576.43 | 29639.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 11:15:00 | 29380.10 | 29545.97 | 29619.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:15:00 | 28072.59 | 28724.09 | 29055.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:15:00 | 28107.56 | 28724.09 | 29055.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:15:00 | 28097.29 | 28724.09 | 29055.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:15:00 | 28105.94 | 28724.09 | 29055.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 11:15:00 | 27911.09 | 28556.55 | 28948.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-19 11:15:00 | 28012.40 | 27979.20 | 28393.77 | SL hit (close>ema200) qty=0.50 sl=27979.20 alert=retest2 |

### Cycle 34 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 11:15:00 | 28378.95 | 28228.46 | 28224.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 13:15:00 | 28428.00 | 28293.51 | 28255.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 10:15:00 | 28279.75 | 28309.70 | 28277.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 10:15:00 | 28279.75 | 28309.70 | 28277.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 28279.75 | 28309.70 | 28277.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:00:00 | 28279.75 | 28309.70 | 28277.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 28282.70 | 28304.30 | 28277.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:45:00 | 28273.75 | 28304.30 | 28277.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 28287.95 | 28301.03 | 28278.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 13:00:00 | 28287.95 | 28301.03 | 28278.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 28319.95 | 28304.82 | 28282.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:00:00 | 28319.95 | 28304.82 | 28282.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 28254.05 | 28294.66 | 28279.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:45:00 | 28251.55 | 28294.66 | 28279.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 28274.00 | 28290.53 | 28279.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:15:00 | 28359.95 | 28290.53 | 28279.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 28323.75 | 28297.17 | 28283.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 14:15:00 | 28538.70 | 28330.22 | 28304.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 09:15:00 | 28865.55 | 29042.97 | 29062.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 28865.55 | 29042.97 | 29062.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 28470.05 | 28888.69 | 28986.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 15:15:00 | 28086.00 | 28049.06 | 28253.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 09:15:00 | 28312.00 | 28101.65 | 28259.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 28312.00 | 28101.65 | 28259.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:45:00 | 28314.00 | 28101.65 | 28259.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 28550.00 | 28191.32 | 28285.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 28550.00 | 28191.32 | 28285.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 28538.65 | 28354.54 | 28342.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 15:15:00 | 28600.10 | 28403.65 | 28365.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 11:15:00 | 28556.45 | 28633.36 | 28553.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 11:15:00 | 28556.45 | 28633.36 | 28553.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 28556.45 | 28633.36 | 28553.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:00:00 | 28556.45 | 28633.36 | 28553.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 28447.00 | 28596.09 | 28543.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:00:00 | 28447.00 | 28596.09 | 28543.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 28345.90 | 28546.05 | 28525.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:00:00 | 28345.90 | 28546.05 | 28525.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 28364.20 | 28509.68 | 28510.83 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 10:15:00 | 28728.35 | 28551.76 | 28528.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 12:15:00 | 28816.05 | 28627.34 | 28567.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 14:15:00 | 28831.30 | 28885.54 | 28763.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 15:00:00 | 28831.30 | 28885.54 | 28763.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 15:15:00 | 28761.10 | 28860.65 | 28763.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:15:00 | 29150.00 | 28860.65 | 28763.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 12:30:00 | 28914.00 | 29017.71 | 28940.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 28699.85 | 28940.99 | 28928.54 | SL hit (close<static) qty=1.00 sl=28761.10 alert=retest2 |

### Cycle 39 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 28522.35 | 28857.26 | 28891.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 28272.55 | 28649.98 | 28780.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 09:15:00 | 28912.60 | 28640.40 | 28738.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 09:15:00 | 28912.60 | 28640.40 | 28738.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 28912.60 | 28640.40 | 28738.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:00:00 | 28912.60 | 28640.40 | 28738.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 29181.10 | 28748.54 | 28778.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 11:00:00 | 29181.10 | 28748.54 | 28778.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 11:15:00 | 28998.40 | 28798.51 | 28798.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 14:15:00 | 29199.55 | 28940.03 | 28867.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 14:15:00 | 29139.25 | 29179.51 | 29050.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-21 14:45:00 | 29142.45 | 29179.51 | 29050.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 29308.25 | 29210.52 | 29097.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:45:00 | 29215.35 | 29210.52 | 29097.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 13:15:00 | 29128.10 | 29182.28 | 29111.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 13:30:00 | 29070.00 | 29182.28 | 29111.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 14:15:00 | 28973.20 | 29140.46 | 29099.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 15:00:00 | 28973.20 | 29140.46 | 29099.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 15:15:00 | 29006.20 | 29113.61 | 29090.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:15:00 | 29093.55 | 29113.61 | 29090.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 29045.05 | 29091.46 | 29084.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:45:00 | 29002.75 | 29091.46 | 29084.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 29074.05 | 29087.98 | 29083.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:00:00 | 29074.05 | 29087.98 | 29083.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 29100.65 | 29090.52 | 29084.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:15:00 | 29019.05 | 29090.52 | 29084.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2024-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 13:15:00 | 28962.35 | 29064.88 | 29073.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 14:15:00 | 28441.00 | 28940.11 | 29016.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 14:15:00 | 28662.25 | 28645.10 | 28796.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 15:00:00 | 28662.25 | 28645.10 | 28796.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 28501.05 | 28629.06 | 28763.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 11:00:00 | 28416.55 | 28586.56 | 28732.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 12:00:00 | 28500.00 | 28569.25 | 28711.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 14:00:00 | 28463.95 | 28546.50 | 28676.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 14:45:00 | 28489.70 | 28536.50 | 28659.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 12:15:00 | 28485.65 | 28395.07 | 28530.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 12:30:00 | 28579.95 | 28395.07 | 28530.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 28498.90 | 28415.84 | 28527.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 15:00:00 | 28438.15 | 28420.30 | 28519.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 15:15:00 | 28342.00 | 28199.76 | 28313.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 12:15:00 | 28618.70 | 28407.11 | 28378.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 12:15:00 | 28618.70 | 28407.11 | 28378.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 11:15:00 | 28693.35 | 28467.79 | 28420.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 09:15:00 | 29300.60 | 29329.04 | 29072.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-05 10:00:00 | 29300.60 | 29329.04 | 29072.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 29135.05 | 29290.24 | 29078.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:45:00 | 29158.05 | 29290.24 | 29078.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 29288.70 | 29289.93 | 29097.37 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 09:15:00 | 28915.15 | 29111.29 | 29117.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 13:15:00 | 28240.00 | 28868.43 | 28997.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 09:15:00 | 28824.90 | 28680.79 | 28864.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 09:15:00 | 28824.90 | 28680.79 | 28864.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 28824.90 | 28680.79 | 28864.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:45:00 | 28807.95 | 28680.79 | 28864.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 28837.85 | 28696.92 | 28824.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 13:00:00 | 28837.85 | 28696.92 | 28824.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 28839.00 | 28725.33 | 28825.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 14:15:00 | 28595.85 | 28725.33 | 28825.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 12:00:00 | 28650.60 | 28664.80 | 28750.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 12:30:00 | 28659.90 | 28658.84 | 28739.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 13:00:00 | 28635.00 | 28658.84 | 28739.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 14:15:00 | 28730.75 | 28678.81 | 28735.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 14:30:00 | 28700.00 | 28678.81 | 28735.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 15:15:00 | 28731.00 | 28689.25 | 28734.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:15:00 | 28884.00 | 28689.25 | 28734.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 28883.65 | 28728.13 | 28748.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:45:00 | 28934.05 | 28728.13 | 28748.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-12 10:15:00 | 29002.50 | 28783.00 | 28771.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 10:15:00 | 29002.50 | 28783.00 | 28771.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-12 11:15:00 | 29102.65 | 28846.93 | 28801.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 13:15:00 | 28842.00 | 28859.88 | 28816.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 13:15:00 | 28842.00 | 28859.88 | 28816.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 28842.00 | 28859.88 | 28816.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 14:00:00 | 28842.00 | 28859.88 | 28816.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 28590.00 | 28805.91 | 28795.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 15:00:00 | 28590.00 | 28805.91 | 28795.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2024-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 15:15:00 | 28511.55 | 28747.04 | 28769.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 28208.50 | 28639.33 | 28718.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 27404.95 | 27363.72 | 27598.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 09:30:00 | 27465.55 | 27363.72 | 27598.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 27399.30 | 27172.22 | 27283.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:00:00 | 27399.30 | 27172.22 | 27283.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 27341.85 | 27206.15 | 27288.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:45:00 | 27311.30 | 27206.15 | 27288.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 27524.55 | 27329.64 | 27334.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:30:00 | 27595.95 | 27329.64 | 27334.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 27647.95 | 27393.30 | 27362.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 14:15:00 | 27703.95 | 27455.43 | 27393.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 27463.70 | 27681.02 | 27595.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 09:15:00 | 27463.70 | 27681.02 | 27595.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 27463.70 | 27681.02 | 27595.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 10:00:00 | 27463.70 | 27681.02 | 27595.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 27639.65 | 27672.75 | 27599.11 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2024-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 10:15:00 | 27348.65 | 27531.95 | 27554.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 11:15:00 | 27200.00 | 27465.56 | 27522.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 13:15:00 | 27442.80 | 27437.58 | 27498.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 13:15:00 | 27442.80 | 27437.58 | 27498.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 27442.80 | 27437.58 | 27498.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:45:00 | 27462.35 | 27437.58 | 27498.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 27441.75 | 27438.42 | 27493.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 14:30:00 | 27522.00 | 27438.42 | 27493.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 27561.35 | 27463.00 | 27499.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 10:45:00 | 27337.10 | 27457.52 | 27490.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 13:00:00 | 27337.30 | 27432.52 | 27473.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 15:00:00 | 27367.15 | 27423.03 | 27462.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 10:15:00 | 27784.10 | 27495.14 | 27484.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 10:15:00 | 27784.10 | 27495.14 | 27484.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 27984.45 | 27741.86 | 27628.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 10:15:00 | 28636.45 | 28733.32 | 28486.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 10:45:00 | 28687.95 | 28733.32 | 28486.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 28899.95 | 28766.65 | 28524.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:30:00 | 28535.90 | 28766.65 | 28524.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 28907.20 | 29055.69 | 28912.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 11:30:00 | 29172.00 | 29080.06 | 28948.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 13:45:00 | 29158.45 | 29112.61 | 28987.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 14:45:00 | 29165.60 | 29120.08 | 29002.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 29182.35 | 29125.66 | 29015.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 29122.55 | 29125.04 | 29025.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 10:45:00 | 29265.25 | 29136.40 | 29083.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 12:15:00 | 28880.10 | 29068.45 | 29060.37 | SL hit (close<static) qty=1.00 sl=28888.05 alert=retest2 |

### Cycle 49 — SELL (started 2024-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 13:15:00 | 28826.05 | 29019.97 | 29039.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 28788.50 | 28940.78 | 28995.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 28683.35 | 28628.46 | 28745.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 15:00:00 | 28683.35 | 28628.46 | 28745.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 28635.05 | 28629.78 | 28735.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 28597.20 | 28629.78 | 28735.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 28617.40 | 28627.30 | 28724.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:30:00 | 28480.00 | 28591.14 | 28699.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 11:15:00 | 28745.25 | 28332.70 | 28317.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2024-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 11:15:00 | 28745.25 | 28332.70 | 28317.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 13:15:00 | 29000.00 | 28551.33 | 28424.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 12:15:00 | 28633.40 | 28823.59 | 28653.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 12:15:00 | 28633.40 | 28823.59 | 28653.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 28633.40 | 28823.59 | 28653.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:00:00 | 28633.40 | 28823.59 | 28653.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 28756.55 | 28810.18 | 28662.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:30:00 | 28550.00 | 28810.18 | 28662.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 28666.65 | 28781.47 | 28662.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 28666.65 | 28781.47 | 28662.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 28611.05 | 28747.39 | 28658.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:15:00 | 28475.55 | 28747.39 | 28658.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 28677.15 | 28733.34 | 28659.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:30:00 | 28463.70 | 28733.34 | 28659.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 28820.75 | 28750.82 | 28674.52 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 09:15:00 | 28480.75 | 28615.82 | 28633.32 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 15:15:00 | 28750.00 | 28600.26 | 28587.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 29489.05 | 28778.02 | 28669.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 15:15:00 | 29625.00 | 29685.49 | 29393.60 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 09:30:00 | 29893.95 | 29701.87 | 29427.58 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 10:30:00 | 29924.80 | 29751.50 | 29475.08 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 29795.50 | 29784.55 | 29619.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:30:00 | 29773.85 | 29784.55 | 29619.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 13:15:00 | 29895.45 | 29916.91 | 29811.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 13:30:00 | 29840.30 | 29916.91 | 29811.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 29680.15 | 29870.43 | 29816.73 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-03 09:15:00 | 29680.15 | 29870.43 | 29816.73 | SL hit (close<ema400) qty=1.00 sl=29816.73 alert=retest1 |

### Cycle 53 — SELL (started 2025-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 11:15:00 | 29462.40 | 29733.48 | 29760.37 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-06 13:15:00 | 29893.00 | 29717.89 | 29708.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-06 15:15:00 | 29970.60 | 29805.72 | 29752.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-07 14:15:00 | 29971.65 | 30011.30 | 29906.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 14:15:00 | 29971.65 | 30011.30 | 29906.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 29971.65 | 30011.30 | 29906.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:00:00 | 29971.65 | 30011.30 | 29906.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 30000.00 | 30007.23 | 29922.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:45:00 | 29939.10 | 30007.23 | 29922.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 29912.80 | 29988.35 | 29921.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 11:15:00 | 29759.50 | 29988.35 | 29921.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 11:15:00 | 29784.90 | 29947.66 | 29909.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 11:45:00 | 29782.90 | 29947.66 | 29909.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 12:15:00 | 29620.95 | 29882.32 | 29883.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 15:15:00 | 29500.00 | 29717.29 | 29799.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 10:15:00 | 29750.00 | 29714.47 | 29783.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 10:15:00 | 29750.00 | 29714.47 | 29783.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 29750.00 | 29714.47 | 29783.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 11:00:00 | 29750.00 | 29714.47 | 29783.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 11:15:00 | 29690.00 | 29709.58 | 29775.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 11:45:00 | 29725.15 | 29709.58 | 29775.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 29825.00 | 29732.66 | 29779.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 12:45:00 | 29816.25 | 29732.66 | 29779.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 29778.10 | 29741.75 | 29779.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 15:15:00 | 29600.00 | 29737.15 | 29773.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:45:00 | 29475.15 | 29654.48 | 29728.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 09:15:00 | 28120.00 | 28296.25 | 28581.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 09:15:00 | 28001.39 | 28296.25 | 28581.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-17 09:15:00 | 27620.90 | 27481.28 | 27785.08 | SL hit (close>ema200) qty=0.50 sl=27481.28 alert=retest2 |

### Cycle 56 — BUY (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 12:15:00 | 27871.75 | 27767.94 | 27760.21 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 27659.50 | 27747.18 | 27752.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 27465.05 | 27672.00 | 27715.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 11:15:00 | 27679.90 | 27670.98 | 27707.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 11:45:00 | 27644.10 | 27670.98 | 27707.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 12:15:00 | 27607.95 | 27658.38 | 27698.65 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 27738.60 | 27720.90 | 27719.17 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 10:15:00 | 27645.10 | 27729.15 | 27737.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 27395.00 | 27612.18 | 27677.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 12:15:00 | 25833.35 | 25807.72 | 26220.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 13:00:00 | 25833.35 | 25807.72 | 26220.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 26085.10 | 25758.43 | 26054.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:45:00 | 26152.25 | 25758.43 | 26054.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 26006.15 | 25807.97 | 26049.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 11:30:00 | 25948.90 | 25835.71 | 26040.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 12:00:00 | 25946.65 | 25835.71 | 26040.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 11:15:00 | 26200.00 | 26045.03 | 26060.21 | SL hit (close>static) qty=1.00 sl=26174.90 alert=retest2 |

### Cycle 60 — BUY (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 14:15:00 | 26210.90 | 26090.75 | 26078.49 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 25950.40 | 26065.59 | 26079.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 15:15:00 | 25887.95 | 26030.06 | 26062.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 13:15:00 | 25960.75 | 25948.55 | 26003.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 13:15:00 | 25960.75 | 25948.55 | 26003.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 13:15:00 | 25960.75 | 25948.55 | 26003.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:30:00 | 25891.20 | 25948.55 | 26003.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 14:15:00 | 25999.15 | 25958.67 | 26002.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 15:00:00 | 25999.15 | 25958.67 | 26002.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 25958.00 | 25958.54 | 25998.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:15:00 | 26083.80 | 25958.54 | 25998.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 26001.20 | 25967.07 | 25998.97 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 11:15:00 | 26155.85 | 26039.75 | 26028.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 26518.90 | 26169.13 | 26096.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 09:15:00 | 29383.90 | 29507.98 | 29137.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-11 10:00:00 | 29383.90 | 29507.98 | 29137.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 12:15:00 | 29171.10 | 29388.81 | 29171.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 13:00:00 | 29171.10 | 29388.81 | 29171.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 13:15:00 | 29104.10 | 29331.86 | 29164.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 14:00:00 | 29104.10 | 29331.86 | 29164.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 14:15:00 | 29114.95 | 29288.48 | 29160.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 14:30:00 | 29074.90 | 29288.48 | 29160.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 15:15:00 | 29117.20 | 29254.23 | 29156.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-12 09:15:00 | 28472.05 | 29254.23 | 29156.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2025-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 09:15:00 | 28290.65 | 29061.51 | 29077.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 27920.80 | 28454.14 | 28684.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 09:15:00 | 28019.00 | 27969.42 | 28262.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 10:15:00 | 28055.00 | 27969.42 | 28262.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 28223.30 | 28020.20 | 28258.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:45:00 | 28229.55 | 28020.20 | 28258.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 28245.35 | 28065.23 | 28257.71 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2025-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 09:15:00 | 28607.95 | 28376.61 | 28352.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 11:15:00 | 29015.55 | 28586.24 | 28457.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 11:15:00 | 29505.75 | 29525.84 | 29228.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 12:00:00 | 29505.75 | 29525.84 | 29228.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 29006.75 | 29383.83 | 29269.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 29006.75 | 29383.83 | 29269.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 28880.00 | 29283.06 | 29234.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 28880.00 | 29283.06 | 29234.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 11:15:00 | 28796.70 | 29185.79 | 29194.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 28671.35 | 28978.91 | 29079.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 12:15:00 | 29318.45 | 29005.79 | 29062.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 12:15:00 | 29318.45 | 29005.79 | 29062.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 29318.45 | 29005.79 | 29062.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:45:00 | 29389.90 | 29005.79 | 29062.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 29448.00 | 29094.23 | 29097.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:45:00 | 29510.00 | 29094.23 | 29097.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 14:15:00 | 29822.30 | 29239.84 | 29163.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-25 09:15:00 | 30421.85 | 29565.87 | 29330.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-25 15:15:00 | 29942.05 | 30065.44 | 29743.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-27 09:30:00 | 29872.00 | 30088.35 | 29783.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 13:15:00 | 29580.00 | 29982.87 | 29832.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 14:00:00 | 29580.00 | 29982.87 | 29832.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 30000.00 | 29986.29 | 29848.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-28 15:00:00 | 30369.95 | 29951.48 | 29872.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-03 10:15:00 | 30199.65 | 30033.64 | 29929.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-07 15:15:00 | 30651.95 | 31128.07 | 31187.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 15:15:00 | 30651.95 | 31128.07 | 31187.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 30483.90 | 30863.66 | 31025.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 09:15:00 | 30902.55 | 30794.20 | 30946.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 10:00:00 | 30902.55 | 30794.20 | 30946.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 30781.95 | 30791.75 | 30931.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 11:30:00 | 30610.00 | 30770.61 | 30909.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 12:15:00 | 30674.00 | 30770.61 | 30909.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 12:45:00 | 30673.45 | 30736.49 | 30881.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 10:15:00 | 30450.00 | 29974.13 | 29963.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2025-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 10:15:00 | 30450.00 | 29974.13 | 29963.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 09:15:00 | 30510.00 | 30283.97 | 30149.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 13:15:00 | 30201.00 | 30343.98 | 30230.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 13:15:00 | 30201.00 | 30343.98 | 30230.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 13:15:00 | 30201.00 | 30343.98 | 30230.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 13:30:00 | 30232.85 | 30343.98 | 30230.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 14:15:00 | 30217.85 | 30318.76 | 30229.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 15:15:00 | 30189.20 | 30318.76 | 30229.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 15:15:00 | 30189.20 | 30292.85 | 30225.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:30:00 | 30250.00 | 30244.28 | 30209.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 11:15:00 | 30325.25 | 30228.56 | 30205.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 14:30:00 | 30362.75 | 30285.11 | 30244.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 30621.40 | 30248.09 | 30230.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 30495.95 | 30297.66 | 30255.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:30:00 | 30319.95 | 30297.66 | 30255.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 31172.95 | 31093.32 | 30880.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:30:00 | 30817.85 | 31093.32 | 30880.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 30970.00 | 31042.56 | 30892.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:45:00 | 30872.00 | 31042.56 | 30892.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 30678.50 | 30969.75 | 30873.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:30:00 | 30642.85 | 30969.75 | 30873.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 30923.00 | 30960.40 | 30877.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:45:00 | 30664.70 | 30960.40 | 30877.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 30636.80 | 30895.68 | 30855.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 15:00:00 | 30636.80 | 30895.68 | 30855.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 30600.00 | 30836.54 | 30832.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 30371.35 | 30836.54 | 30832.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-27 09:15:00 | 30519.40 | 30773.12 | 30804.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 09:15:00 | 30519.40 | 30773.12 | 30804.00 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 11:15:00 | 30980.00 | 30736.03 | 30725.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 12:15:00 | 30990.70 | 30786.97 | 30749.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 13:15:00 | 30556.05 | 30740.78 | 30732.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 13:15:00 | 30556.05 | 30740.78 | 30732.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 30556.05 | 30740.78 | 30732.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 13:45:00 | 30540.00 | 30740.78 | 30732.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 14:15:00 | 30349.85 | 30662.60 | 30697.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 15:15:00 | 30245.30 | 30579.14 | 30656.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 09:15:00 | 30021.00 | 29927.94 | 30206.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 30021.00 | 29927.94 | 30206.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 30021.00 | 29927.94 | 30206.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:15:00 | 29704.30 | 29940.82 | 30099.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 28219.08 | 29195.60 | 29584.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-07 15:15:00 | 29107.55 | 28932.60 | 29253.18 | SL hit (close>ema200) qty=0.50 sl=28932.60 alert=retest2 |

### Cycle 72 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 29591.80 | 29326.50 | 29293.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 29604.85 | 29382.17 | 29321.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 14:15:00 | 30755.00 | 30852.65 | 30408.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 15:00:00 | 30755.00 | 30852.65 | 30408.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 30635.00 | 30752.18 | 30589.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:45:00 | 30930.00 | 30769.75 | 30612.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 15:15:00 | 30915.00 | 30966.21 | 30790.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 11:30:00 | 30915.00 | 30973.42 | 30848.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 14:15:00 | 30490.00 | 30831.83 | 30855.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 14:15:00 | 30490.00 | 30831.83 | 30855.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 10:15:00 | 30150.00 | 30576.85 | 30722.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 11:15:00 | 29815.00 | 29812.80 | 30029.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 12:00:00 | 29815.00 | 29812.80 | 30029.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 29735.00 | 29783.05 | 29931.38 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 11:15:00 | 30180.00 | 29949.97 | 29935.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 12:15:00 | 30275.00 | 30014.97 | 29966.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 30465.00 | 30564.97 | 30398.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 10:15:00 | 30405.00 | 30532.97 | 30398.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 30405.00 | 30532.97 | 30398.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:45:00 | 30400.00 | 30532.97 | 30398.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 30355.00 | 30497.38 | 30394.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:45:00 | 30330.00 | 30497.38 | 30394.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 30300.00 | 30457.90 | 30386.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 12:30:00 | 30290.00 | 30457.90 | 30386.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 30250.00 | 30416.32 | 30373.92 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2025-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 15:15:00 | 30025.00 | 30299.45 | 30325.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 29660.00 | 30171.56 | 30265.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 12:15:00 | 30200.00 | 30064.16 | 30182.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 12:15:00 | 30200.00 | 30064.16 | 30182.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 30200.00 | 30064.16 | 30182.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:45:00 | 30220.00 | 30064.16 | 30182.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 30315.00 | 30114.33 | 30194.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 30315.00 | 30114.33 | 30194.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 30315.00 | 30154.46 | 30205.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:45:00 | 30420.00 | 30154.46 | 30205.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 30170.00 | 30157.57 | 30202.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 30230.00 | 30157.57 | 30202.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 30115.00 | 30149.05 | 30194.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 14:45:00 | 29885.00 | 30168.90 | 30193.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 12:00:00 | 29970.00 | 29996.72 | 30092.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 10:15:00 | 30405.00 | 30153.87 | 30126.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 30405.00 | 30153.87 | 30126.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 31000.00 | 30482.18 | 30389.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 13:15:00 | 30500.00 | 30544.48 | 30454.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 13:45:00 | 30520.00 | 30544.48 | 30454.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 30835.00 | 31125.06 | 31028.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:00:00 | 30835.00 | 31125.06 | 31028.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 31050.00 | 31110.05 | 31030.46 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 12:15:00 | 30815.00 | 30987.09 | 30995.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 13:15:00 | 30790.00 | 30947.67 | 30977.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 10:15:00 | 30930.00 | 30873.27 | 30926.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 10:15:00 | 30930.00 | 30873.27 | 30926.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 30930.00 | 30873.27 | 30926.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:00:00 | 30930.00 | 30873.27 | 30926.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 30790.00 | 30856.62 | 30913.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 14:45:00 | 30495.00 | 30696.19 | 30824.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 15:15:00 | 30620.00 | 30380.77 | 30372.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 15:15:00 | 30620.00 | 30380.77 | 30372.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 09:15:00 | 30635.00 | 30431.61 | 30396.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 15:15:00 | 31300.00 | 31674.10 | 31525.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 15:15:00 | 31300.00 | 31674.10 | 31525.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 31300.00 | 31674.10 | 31525.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 11:30:00 | 31920.00 | 31734.66 | 31590.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 13:00:00 | 31945.00 | 31776.73 | 31622.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:15:00 | 31920.00 | 31772.48 | 31660.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 15:15:00 | 31560.00 | 31645.41 | 31655.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 31560.00 | 31645.41 | 31655.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 14:15:00 | 31525.00 | 31613.04 | 31636.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 31600.00 | 31460.84 | 31514.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 31600.00 | 31460.84 | 31514.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 31600.00 | 31460.84 | 31514.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 31615.00 | 31460.84 | 31514.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 31770.00 | 31522.67 | 31537.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 31770.00 | 31522.67 | 31537.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 31675.00 | 31553.14 | 31550.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 14:15:00 | 31845.00 | 31644.25 | 31608.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 09:15:00 | 31655.00 | 31675.32 | 31630.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 31655.00 | 31675.32 | 31630.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 31655.00 | 31675.32 | 31630.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:15:00 | 31640.00 | 31675.32 | 31630.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 31655.00 | 31671.25 | 31632.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 12:30:00 | 31800.00 | 31661.20 | 31634.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 15:15:00 | 31555.00 | 31624.42 | 31623.19 | SL hit (close<static) qty=1.00 sl=31565.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 31465.00 | 31632.57 | 31633.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 13:15:00 | 31435.00 | 31593.05 | 31615.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 31520.00 | 31369.92 | 31469.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 11:15:00 | 31520.00 | 31369.92 | 31469.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 31520.00 | 31369.92 | 31469.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 31520.00 | 31369.92 | 31469.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 31425.00 | 31380.94 | 31465.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:15:00 | 31535.00 | 31380.94 | 31465.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 31540.00 | 31412.75 | 31471.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:00:00 | 31540.00 | 31412.75 | 31471.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 31585.00 | 31447.20 | 31482.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:45:00 | 31625.00 | 31447.20 | 31482.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 31410.00 | 31180.88 | 31282.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 31410.00 | 31180.88 | 31282.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 31445.00 | 31233.70 | 31297.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:45:00 | 31455.00 | 31233.70 | 31297.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 31625.00 | 31351.37 | 31342.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 32525.00 | 31807.09 | 31619.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 34225.00 | 34405.42 | 33701.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:00:00 | 34225.00 | 34405.42 | 33701.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 34670.00 | 34812.09 | 34458.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 34860.00 | 34812.09 | 34458.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:00:00 | 34855.00 | 34772.34 | 34498.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 14:15:00 | 34750.00 | 35036.96 | 35044.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 14:15:00 | 34750.00 | 35036.96 | 35044.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 10:15:00 | 34690.00 | 34891.16 | 34968.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 10:15:00 | 34485.00 | 34302.59 | 34475.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 10:15:00 | 34485.00 | 34302.59 | 34475.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 34485.00 | 34302.59 | 34475.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:45:00 | 34475.00 | 34302.59 | 34475.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 34275.00 | 34297.07 | 34457.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:00:00 | 34090.00 | 34255.66 | 34423.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 34690.00 | 34443.50 | 34416.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 09:15:00 | 34690.00 | 34443.50 | 34416.18 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 34370.00 | 34409.11 | 34414.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 14:15:00 | 34270.00 | 34366.23 | 34392.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 15:15:00 | 34290.00 | 34276.41 | 34321.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 15:15:00 | 34290.00 | 34276.41 | 34321.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 34290.00 | 34276.41 | 34321.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 34435.00 | 34276.41 | 34321.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 34380.00 | 34297.13 | 34326.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 10:30:00 | 34180.00 | 34264.70 | 34309.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 09:30:00 | 34210.00 | 34186.96 | 34241.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 11:45:00 | 34160.00 | 34216.66 | 34246.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 12:15:00 | 34175.00 | 34141.35 | 34139.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 12:15:00 | 34175.00 | 34141.35 | 34139.49 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 12:15:00 | 34100.00 | 34143.66 | 34144.98 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 13:15:00 | 34260.00 | 34166.92 | 34155.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 14:15:00 | 34750.00 | 34283.54 | 34209.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 34700.00 | 34741.95 | 34538.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 14:45:00 | 34560.00 | 34741.95 | 34538.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 34640.00 | 34721.56 | 34548.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 34230.00 | 34721.56 | 34548.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 34135.00 | 34604.25 | 34510.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:45:00 | 34205.00 | 34604.25 | 34510.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 34030.00 | 34418.32 | 34437.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 33945.00 | 34266.32 | 34361.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 33325.00 | 33295.52 | 33622.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 12:15:00 | 33700.00 | 33295.52 | 33622.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 34240.00 | 33484.41 | 33678.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 34240.00 | 33484.41 | 33678.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 34380.00 | 33663.53 | 33742.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:30:00 | 34385.00 | 33663.53 | 33742.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 34720.00 | 33874.82 | 33831.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 12:15:00 | 34995.00 | 34388.74 | 34128.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 34675.00 | 34746.42 | 34406.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 09:45:00 | 34665.00 | 34746.42 | 34406.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 34370.00 | 34668.70 | 34503.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 34370.00 | 34668.70 | 34503.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 34485.00 | 34631.96 | 34501.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:45:00 | 34560.00 | 34580.57 | 34489.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 33980.00 | 34460.45 | 34443.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:45:00 | 33955.00 | 34460.45 | 34443.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 33925.00 | 34353.36 | 34396.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 33745.00 | 34158.35 | 34295.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 34095.00 | 33912.09 | 34097.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 11:15:00 | 34095.00 | 33912.09 | 34097.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 34095.00 | 33912.09 | 34097.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 34095.00 | 33912.09 | 34097.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 34105.00 | 33950.67 | 34097.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 13:15:00 | 34055.00 | 33950.67 | 34097.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:15:00 | 32352.25 | 32757.99 | 33161.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 32725.00 | 32677.06 | 32962.15 | SL hit (close>ema200) qty=0.50 sl=32677.06 alert=retest2 |

### Cycle 92 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 33020.00 | 32908.48 | 32898.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 33320.00 | 32990.79 | 32937.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 14:15:00 | 33005.00 | 33291.97 | 33136.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 14:15:00 | 33005.00 | 33291.97 | 33136.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 33005.00 | 33291.97 | 33136.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 33005.00 | 33291.97 | 33136.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 32250.00 | 33083.58 | 33055.79 | EMA400 retest candle locked (from upside) |

### Cycle 93 — SELL (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 09:15:00 | 32680.00 | 33002.86 | 33021.63 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 11:15:00 | 33650.00 | 33111.03 | 33066.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 12:15:00 | 33720.00 | 33232.83 | 33125.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 32900.00 | 33201.81 | 33132.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 14:15:00 | 32900.00 | 33201.81 | 33132.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 32900.00 | 33201.81 | 33132.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 32900.00 | 33201.81 | 33132.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 32850.00 | 33131.45 | 33106.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 33195.00 | 33131.45 | 33106.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 13:15:00 | 33010.00 | 33100.59 | 33101.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 33010.00 | 33100.59 | 33101.49 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 14:15:00 | 33300.00 | 33140.47 | 33119.53 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 32680.00 | 33040.70 | 33082.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 12:15:00 | 32655.00 | 32963.56 | 33043.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 14:15:00 | 33095.00 | 32780.91 | 32855.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 14:15:00 | 33095.00 | 32780.91 | 32855.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 33095.00 | 32780.91 | 32855.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 33095.00 | 32780.91 | 32855.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 32900.00 | 32804.73 | 32859.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 32920.00 | 32804.73 | 32859.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 32730.00 | 32709.88 | 32776.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 10:15:00 | 32570.00 | 32664.37 | 32717.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 15:15:00 | 32275.00 | 31852.85 | 31803.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 32275.00 | 31852.85 | 31803.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 32395.00 | 31961.28 | 31857.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 13:15:00 | 32410.00 | 32421.20 | 32238.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:30:00 | 32420.00 | 32421.20 | 32238.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 32490.00 | 32447.57 | 32283.12 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 32155.00 | 32271.55 | 32275.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 31500.00 | 32117.24 | 32204.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 31510.00 | 31462.02 | 31766.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:30:00 | 31455.00 | 31462.02 | 31766.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 31145.00 | 30915.43 | 31140.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 31145.00 | 30915.43 | 31140.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 31295.00 | 30991.35 | 31154.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:45:00 | 31300.00 | 30991.35 | 31154.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 31320.00 | 31057.08 | 31169.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:30:00 | 31340.00 | 31057.08 | 31169.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 31240.00 | 31081.33 | 31160.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:30:00 | 31265.00 | 31081.33 | 31160.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 30890.00 | 31043.06 | 31135.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 31280.00 | 31043.06 | 31135.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 31330.00 | 31100.45 | 31153.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 31425.00 | 31100.45 | 31153.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 31145.00 | 31114.09 | 31150.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 14:15:00 | 31000.00 | 31107.02 | 31140.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 12:30:00 | 31025.00 | 31057.61 | 31095.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 10:15:00 | 31290.00 | 31090.49 | 31092.15 | SL hit (close>static) qty=1.00 sl=31245.00 alert=retest2 |

### Cycle 100 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 31340.00 | 31140.39 | 31114.68 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 14:15:00 | 30930.00 | 31094.84 | 31102.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 15:15:00 | 30885.00 | 31052.87 | 31082.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 10:15:00 | 31065.00 | 31046.84 | 31073.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 10:15:00 | 31065.00 | 31046.84 | 31073.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 31065.00 | 31046.84 | 31073.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:30:00 | 31060.00 | 31046.84 | 31073.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 31120.00 | 31061.47 | 31078.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:00:00 | 31120.00 | 31061.47 | 31078.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 31140.00 | 31077.18 | 31083.72 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 13:15:00 | 31145.00 | 31090.74 | 31089.29 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 30960.00 | 31064.59 | 31077.54 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 31145.00 | 31079.67 | 31079.63 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 30910.00 | 31054.47 | 31069.86 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 31220.00 | 31078.98 | 31073.49 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 30945.00 | 31052.19 | 31061.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 10:15:00 | 30855.00 | 30966.69 | 31011.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 11:15:00 | 30640.00 | 30627.83 | 30780.60 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-22 12:15:00 | 30520.00 | 30627.83 | 30780.60 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-22 13:30:00 | 30480.00 | 30579.61 | 30731.69 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 30340.00 | 30324.94 | 30426.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:00:00 | 30175.00 | 30294.95 | 30403.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 10:15:00 | 30405.00 | 30246.62 | 30330.87 | SL hit (close>ema400) qty=1.00 sl=30330.87 alert=retest1 |

### Cycle 108 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 30015.00 | 29687.45 | 29687.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 30095.00 | 29852.03 | 29772.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 14:15:00 | 29985.00 | 30059.29 | 29922.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 14:15:00 | 29985.00 | 30059.29 | 29922.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 29985.00 | 30059.29 | 29922.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 11:30:00 | 30410.00 | 30157.57 | 30010.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 12:00:00 | 30400.00 | 30157.57 | 30010.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 14:15:00 | 29825.00 | 30023.55 | 30036.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 29825.00 | 30023.55 | 30036.10 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 11:15:00 | 30100.00 | 30048.84 | 30042.64 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 29875.00 | 30030.73 | 30037.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 15:15:00 | 29865.00 | 29997.58 | 30021.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 11:15:00 | 29830.00 | 29752.97 | 29845.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 12:00:00 | 29830.00 | 29752.97 | 29845.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 29815.00 | 29765.38 | 29842.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 13:00:00 | 29815.00 | 29765.38 | 29842.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 29905.00 | 29793.30 | 29848.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:00:00 | 29905.00 | 29793.30 | 29848.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 30095.00 | 29853.64 | 29870.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 30095.00 | 29853.64 | 29870.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2025-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 15:15:00 | 30255.00 | 29933.91 | 29905.80 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 12:15:00 | 29915.00 | 29973.06 | 29977.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 09:15:00 | 29755.00 | 29879.29 | 29928.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 10:15:00 | 29900.00 | 29883.44 | 29925.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 10:15:00 | 29900.00 | 29883.44 | 29925.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 29900.00 | 29883.44 | 29925.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:30:00 | 29925.00 | 29883.44 | 29925.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 29810.00 | 29854.80 | 29892.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 11:30:00 | 29650.00 | 29768.85 | 29825.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 13:15:00 | 30125.00 | 29845.06 | 29850.31 | SL hit (close>static) qty=1.00 sl=29995.00 alert=retest2 |

### Cycle 114 — BUY (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 14:15:00 | 29920.00 | 29860.05 | 29856.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 14:15:00 | 30300.00 | 30008.03 | 29929.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 12:15:00 | 29975.00 | 30035.22 | 29971.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 12:15:00 | 29975.00 | 30035.22 | 29971.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 29975.00 | 30035.22 | 29971.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 15:15:00 | 30100.00 | 30032.54 | 29981.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 12:15:00 | 29870.00 | 29955.86 | 29961.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 29870.00 | 29955.86 | 29961.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 09:15:00 | 29775.00 | 29906.59 | 29935.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 29945.00 | 29839.70 | 29876.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 29945.00 | 29839.70 | 29876.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 29945.00 | 29839.70 | 29876.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 29945.00 | 29839.70 | 29876.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 29935.00 | 29858.76 | 29882.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:30:00 | 29965.00 | 29858.76 | 29882.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 29920.00 | 29871.01 | 29885.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 29920.00 | 29871.01 | 29885.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 29865.00 | 29869.81 | 29883.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:30:00 | 29915.00 | 29869.81 | 29883.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 29900.00 | 29875.84 | 29885.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:45:00 | 29950.00 | 29875.84 | 29885.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 29790.00 | 29858.68 | 29876.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:30:00 | 29770.00 | 29858.68 | 29876.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 29790.00 | 29844.94 | 29868.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 29905.00 | 29844.94 | 29868.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 29730.00 | 29821.95 | 29856.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:30:00 | 29865.00 | 29821.95 | 29856.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 29705.00 | 29750.09 | 29806.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:30:00 | 29630.00 | 29708.16 | 29771.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 11:00:00 | 29675.00 | 29701.53 | 29762.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 13:15:00 | 29345.00 | 29302.81 | 29300.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 13:15:00 | 29345.00 | 29302.81 | 29300.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 10:15:00 | 29735.00 | 29407.25 | 29350.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 13:15:00 | 29500.00 | 29501.31 | 29415.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 14:00:00 | 29500.00 | 29501.31 | 29415.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 29560.00 | 29513.05 | 29428.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 29560.00 | 29513.05 | 29428.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 29500.00 | 29510.44 | 29434.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:45:00 | 29340.00 | 29473.35 | 29424.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 29345.00 | 29447.68 | 29417.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:00:00 | 29345.00 | 29447.68 | 29417.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 11:15:00 | 29075.00 | 29373.15 | 29386.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 12:15:00 | 29050.00 | 29308.52 | 29355.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 11:15:00 | 29255.00 | 29087.28 | 29191.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 11:15:00 | 29255.00 | 29087.28 | 29191.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 29255.00 | 29087.28 | 29191.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:00:00 | 29255.00 | 29087.28 | 29191.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 29400.00 | 29149.82 | 29210.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 13:00:00 | 29400.00 | 29149.82 | 29210.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 29520.00 | 29223.86 | 29238.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:00:00 | 29520.00 | 29223.86 | 29238.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 29530.00 | 29285.09 | 29264.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 10:15:00 | 29600.00 | 29416.36 | 29335.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 15:15:00 | 29450.00 | 29518.01 | 29426.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 15:15:00 | 29450.00 | 29518.01 | 29426.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 29450.00 | 29518.01 | 29426.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 29500.00 | 29518.01 | 29426.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 29410.00 | 29496.40 | 29425.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:00:00 | 29410.00 | 29496.40 | 29425.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 29545.00 | 29506.12 | 29436.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:15:00 | 29290.00 | 29506.12 | 29436.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 29295.00 | 29463.90 | 29423.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 15:00:00 | 29605.00 | 29459.80 | 29428.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:30:00 | 29550.00 | 29502.27 | 29454.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:00:00 | 29560.00 | 29502.27 | 29454.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 14:15:00 | 29580.00 | 29501.77 | 29471.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 29530.00 | 29507.42 | 29476.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 29530.00 | 29507.42 | 29476.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 29465.00 | 29513.75 | 29486.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:45:00 | 29510.00 | 29513.75 | 29486.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 29360.00 | 29483.00 | 29474.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:00:00 | 29360.00 | 29483.00 | 29474.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 29365.00 | 29459.40 | 29464.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 29365.00 | 29459.40 | 29464.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 29280.00 | 29423.52 | 29447.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 29410.00 | 29401.05 | 29432.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 14:15:00 | 29410.00 | 29401.05 | 29432.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 29410.00 | 29401.05 | 29432.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 29410.00 | 29401.05 | 29432.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 29210.00 | 29350.67 | 29403.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 29340.00 | 29350.67 | 29403.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 29530.00 | 29378.43 | 29406.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:00:00 | 29530.00 | 29378.43 | 29406.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 12:15:00 | 29635.00 | 29429.74 | 29426.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 10:15:00 | 29670.00 | 29550.94 | 29493.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 29600.00 | 29631.23 | 29567.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 29600.00 | 29631.23 | 29567.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 29600.00 | 29631.23 | 29567.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:30:00 | 29575.00 | 29631.23 | 29567.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 29705.00 | 29645.98 | 29579.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:30:00 | 29615.00 | 29645.98 | 29579.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 29630.00 | 29642.79 | 29584.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:30:00 | 29475.00 | 29642.79 | 29584.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 29550.00 | 29624.23 | 29581.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:30:00 | 29590.00 | 29624.23 | 29581.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 29540.00 | 29607.38 | 29577.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 15:15:00 | 29620.00 | 29595.91 | 29575.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 29485.00 | 29577.58 | 29570.60 | SL hit (close<static) qty=1.00 sl=29500.00 alert=retest2 |

### Cycle 121 — SELL (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 11:15:00 | 29535.00 | 29561.45 | 29564.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 12:15:00 | 29445.00 | 29538.16 | 29553.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 14:15:00 | 29605.00 | 29524.62 | 29542.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 14:15:00 | 29605.00 | 29524.62 | 29542.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 29605.00 | 29524.62 | 29542.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 29605.00 | 29524.62 | 29542.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 29625.00 | 29544.70 | 29550.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 29690.00 | 29544.70 | 29550.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 29305.00 | 29496.81 | 29527.51 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 12:15:00 | 29615.00 | 29528.57 | 29524.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 15:15:00 | 29780.00 | 29603.43 | 29562.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 09:15:00 | 29565.00 | 29595.74 | 29562.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 29565.00 | 29595.74 | 29562.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 29565.00 | 29595.74 | 29562.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:30:00 | 29415.00 | 29595.74 | 29562.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 29555.00 | 29587.59 | 29561.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:00:00 | 29555.00 | 29587.59 | 29561.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 29550.00 | 29580.08 | 29560.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:45:00 | 29575.00 | 29580.08 | 29560.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 29650.00 | 29594.06 | 29568.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 13:30:00 | 29810.00 | 29644.25 | 29593.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 12:15:00 | 29550.00 | 29762.45 | 29772.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 29550.00 | 29762.45 | 29772.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 13:15:00 | 29455.00 | 29700.96 | 29743.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 11:15:00 | 29535.00 | 29507.67 | 29617.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 11:45:00 | 29535.00 | 29507.67 | 29617.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 29670.00 | 29550.91 | 29618.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 29670.00 | 29550.91 | 29618.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 30045.00 | 29649.73 | 29657.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 30045.00 | 29649.73 | 29657.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 15:15:00 | 30010.00 | 29721.78 | 29689.51 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 29360.00 | 29674.21 | 29685.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 29270.00 | 29549.50 | 29623.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 29115.00 | 29057.57 | 29233.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:00:00 | 29115.00 | 29057.57 | 29233.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 29000.00 | 28971.59 | 29082.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 13:15:00 | 28885.00 | 28982.42 | 29068.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 14:00:00 | 28890.00 | 28963.93 | 29052.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:15:00 | 27440.75 | 27729.91 | 27811.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:15:00 | 27445.50 | 27729.91 | 27811.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 10:15:00 | 27850.00 | 27753.93 | 27814.79 | SL hit (close>ema200) qty=0.50 sl=27753.93 alert=retest2 |

### Cycle 126 — BUY (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 14:15:00 | 27965.00 | 27862.13 | 27853.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 10:15:00 | 28050.00 | 27923.89 | 27885.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 09:15:00 | 27960.00 | 28010.73 | 27959.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 27960.00 | 28010.73 | 27959.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 27960.00 | 28010.73 | 27959.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:00:00 | 27960.00 | 28010.73 | 27959.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 28020.00 | 28012.58 | 27965.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 11:30:00 | 28090.00 | 28028.06 | 27976.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 14:00:00 | 28055.00 | 28032.16 | 27987.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 14:45:00 | 28090.00 | 28043.73 | 27996.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 15:15:00 | 27950.00 | 28024.98 | 27992.46 | SL hit (close<static) qty=1.00 sl=27960.00 alert=retest2 |

### Cycle 127 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 28500.00 | 28679.77 | 28701.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 09:15:00 | 28290.00 | 28546.20 | 28630.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 28155.00 | 28142.44 | 28275.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 15:00:00 | 28155.00 | 28142.44 | 28275.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 28300.00 | 28175.96 | 28267.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:00:00 | 28300.00 | 28175.96 | 28267.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 28360.00 | 28212.77 | 28275.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:45:00 | 28405.00 | 28212.77 | 28275.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 14:15:00 | 28420.00 | 28325.55 | 28315.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 09:15:00 | 28600.00 | 28393.15 | 28348.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 10:15:00 | 28365.00 | 28387.52 | 28350.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 11:00:00 | 28365.00 | 28387.52 | 28350.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 28360.00 | 28382.02 | 28351.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 28360.00 | 28382.02 | 28351.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 28230.00 | 28351.61 | 28340.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:45:00 | 28155.00 | 28351.61 | 28340.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 28350.00 | 28351.29 | 28341.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:45:00 | 28470.00 | 28377.03 | 28353.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:30:00 | 28485.00 | 28429.30 | 28383.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 14:15:00 | 28395.00 | 28410.42 | 28389.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 15:15:00 | 28400.00 | 28393.33 | 28383.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 28400.00 | 28394.67 | 28384.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 28155.00 | 28394.67 | 28384.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 28055.00 | 28326.73 | 28354.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 28055.00 | 28326.73 | 28354.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 10:15:00 | 27900.00 | 28241.39 | 28313.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 15:15:00 | 28140.00 | 28125.75 | 28218.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 15:15:00 | 28140.00 | 28125.75 | 28218.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 28140.00 | 28125.75 | 28218.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 09:15:00 | 27785.00 | 28125.75 | 28218.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 11:30:00 | 28035.00 | 28094.63 | 28178.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 14:15:00 | 28310.00 | 28151.25 | 28184.91 | SL hit (close>static) qty=1.00 sl=28250.00 alert=retest2 |

### Cycle 130 — BUY (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 09:15:00 | 28360.00 | 28227.20 | 28215.71 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 14:15:00 | 28005.00 | 28201.41 | 28214.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 13:15:00 | 27900.00 | 28119.38 | 28167.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 28030.00 | 28029.36 | 28107.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-16 10:00:00 | 28030.00 | 28029.36 | 28107.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 28040.00 | 28034.79 | 28097.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:30:00 | 27860.00 | 27983.83 | 28068.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 11:15:00 | 27730.00 | 27598.20 | 27590.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 27730.00 | 27598.20 | 27590.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 10:15:00 | 27815.00 | 27705.02 | 27654.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 27685.00 | 27701.02 | 27656.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 12:00:00 | 27685.00 | 27701.02 | 27656.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 27780.00 | 27749.32 | 27693.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:45:00 | 27705.00 | 27749.32 | 27693.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 27465.00 | 27713.36 | 27687.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:30:00 | 27540.00 | 27713.36 | 27687.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 27465.00 | 27663.69 | 27667.67 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 27715.00 | 27650.49 | 27649.26 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 12:15:00 | 27575.00 | 27636.91 | 27643.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 09:15:00 | 26880.00 | 27476.59 | 27566.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 11:15:00 | 26925.00 | 26874.54 | 27101.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 11:30:00 | 26910.00 | 26874.54 | 27101.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 27365.00 | 26991.11 | 27116.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:00:00 | 27365.00 | 26991.11 | 27116.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 27725.00 | 27137.89 | 27171.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 15:00:00 | 27725.00 | 27137.89 | 27171.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2026-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 15:15:00 | 27595.00 | 27229.31 | 27209.92 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 27180.00 | 27246.35 | 27247.22 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 10:15:00 | 27305.00 | 27258.08 | 27252.47 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 27130.00 | 27232.47 | 27241.34 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 27340.00 | 27253.42 | 27248.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 27500.00 | 27327.67 | 27285.39 | Break + close above crossover candle high |

### Cycle 141 — SELL (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 12:15:00 | 26870.00 | 27244.51 | 27255.45 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 27185.00 | 27115.89 | 27114.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 27375.00 | 27167.71 | 27138.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 10:15:00 | 27375.00 | 27414.99 | 27325.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 11:00:00 | 27375.00 | 27414.99 | 27325.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 27310.00 | 27382.47 | 27332.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:30:00 | 27290.00 | 27382.47 | 27332.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 27290.00 | 27363.98 | 27328.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 27290.00 | 27363.98 | 27328.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 27315.00 | 27354.18 | 27327.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 27235.00 | 27354.18 | 27327.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 27085.00 | 27300.35 | 27305.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 12:15:00 | 26890.00 | 27139.62 | 27223.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 26555.00 | 26536.51 | 26692.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 10:00:00 | 26555.00 | 26536.51 | 26692.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 26455.00 | 26427.35 | 26544.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 26430.00 | 26427.35 | 26544.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 26635.00 | 26479.13 | 26532.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:30:00 | 26565.00 | 26479.13 | 26532.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 26680.00 | 26519.30 | 26545.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 26715.00 | 26519.30 | 26545.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 26585.00 | 26564.16 | 26562.40 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 14:15:00 | 26450.00 | 26544.45 | 26554.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 26415.00 | 26505.05 | 26534.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 26470.00 | 26442.57 | 26477.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 10:15:00 | 26470.00 | 26442.57 | 26477.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 26470.00 | 26442.57 | 26477.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:45:00 | 26535.00 | 26442.57 | 26477.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 26530.00 | 26460.05 | 26481.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 26530.00 | 26460.05 | 26481.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 26390.00 | 26446.04 | 26473.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 26440.00 | 26446.04 | 26473.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 26500.00 | 26418.06 | 26448.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 26490.00 | 26418.06 | 26448.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 26500.00 | 26434.45 | 26452.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 14:15:00 | 26410.00 | 26455.44 | 26459.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 26500.00 | 26464.35 | 26462.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 26500.00 | 26464.35 | 26462.99 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2026-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 15:15:00 | 26425.00 | 26456.48 | 26459.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 26300.00 | 26425.18 | 26445.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 26365.00 | 26354.99 | 26397.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 14:15:00 | 26365.00 | 26354.99 | 26397.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 26365.00 | 26354.99 | 26397.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 14:45:00 | 26395.00 | 26354.99 | 26397.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 26340.00 | 26353.60 | 26389.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:45:00 | 26290.00 | 26355.90 | 26384.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 13:00:00 | 26290.00 | 26342.72 | 26375.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:30:00 | 26305.00 | 26347.80 | 26361.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 26205.00 | 26365.03 | 26366.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 26245.00 | 26341.03 | 26355.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-27 13:15:00 | 26500.00 | 26321.19 | 26333.55 | SL hit (close>static) qty=1.00 sl=26485.00 alert=retest2 |

### Cycle 148 — BUY (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 14:15:00 | 26525.00 | 26361.95 | 26350.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-02 12:15:00 | 26720.00 | 26510.07 | 26434.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-05 13:15:00 | 27560.00 | 27756.77 | 27354.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-05 13:30:00 | 27545.00 | 27756.77 | 27354.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 27395.00 | 27635.43 | 27394.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 27275.00 | 27635.43 | 27394.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 27350.00 | 27578.34 | 27390.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 27355.00 | 27578.34 | 27390.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 27375.00 | 27537.67 | 27389.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 12:15:00 | 27420.00 | 27537.67 | 27389.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 14:15:00 | 27420.00 | 27494.71 | 27394.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 26925.00 | 27324.21 | 27337.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 26925.00 | 27324.21 | 27337.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 13:15:00 | 26625.00 | 26855.11 | 26948.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 11:15:00 | 26750.00 | 26694.33 | 26822.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-13 11:30:00 | 26695.00 | 26694.33 | 26822.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 26965.00 | 26675.86 | 26775.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 14:45:00 | 27055.00 | 26675.86 | 26775.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 26990.00 | 26738.69 | 26794.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:15:00 | 26775.00 | 26738.69 | 26794.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 26870.00 | 26764.95 | 26801.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:30:00 | 26560.00 | 26689.21 | 26747.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 09:30:00 | 26625.00 | 26559.75 | 26657.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 13:00:00 | 26550.00 | 26559.79 | 26632.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 26615.00 | 26427.18 | 26490.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 26485.00 | 26438.74 | 26490.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 26590.00 | 26438.74 | 26490.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 26480.00 | 26446.99 | 26489.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:30:00 | 26415.00 | 26446.99 | 26489.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 26620.00 | 26481.59 | 26501.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:00:00 | 26620.00 | 26481.59 | 26501.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-20 12:15:00 | 26730.00 | 26531.28 | 26521.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 26730.00 | 26531.28 | 26521.87 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 25945.00 | 26478.16 | 26507.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 25640.00 | 26310.53 | 26428.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 26180.00 | 25784.82 | 26000.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 26180.00 | 25784.82 | 26000.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 26180.00 | 25784.82 | 26000.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 26180.00 | 25784.82 | 26000.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 26530.00 | 25933.85 | 26048.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 26530.00 | 25933.85 | 26048.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 26785.00 | 26189.87 | 26150.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 27035.00 | 26466.11 | 26289.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 26395.00 | 26637.97 | 26466.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 14:15:00 | 26395.00 | 26637.97 | 26466.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 26395.00 | 26637.97 | 26466.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 15:00:00 | 26395.00 | 26637.97 | 26466.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 26505.00 | 26611.38 | 26470.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 26195.00 | 26528.10 | 26445.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 26290.00 | 26480.48 | 26430.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 26225.00 | 26480.48 | 26430.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 26265.00 | 26375.81 | 26390.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 26090.00 | 26318.64 | 26362.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 26490.00 | 26087.52 | 26163.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 26490.00 | 26087.52 | 26163.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 26490.00 | 26087.52 | 26163.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 26375.00 | 26087.52 | 26163.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 26310.00 | 26132.02 | 26176.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:15:00 | 26205.00 | 26132.02 | 26176.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 26455.00 | 26232.29 | 26216.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 26455.00 | 26232.29 | 26216.62 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 09:15:00 | 26025.00 | 26248.72 | 26260.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 12:15:00 | 25900.00 | 26121.74 | 26195.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 26140.00 | 25852.87 | 25952.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 26140.00 | 25852.87 | 25952.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 26140.00 | 25852.87 | 25952.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 26200.00 | 25852.87 | 25952.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 25875.00 | 25857.30 | 25945.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 12:45:00 | 25825.00 | 25865.67 | 25934.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:30:00 | 25825.00 | 25863.23 | 25921.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 10:45:00 | 25780.00 | 25871.45 | 25910.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 12:30:00 | 25800.00 | 25852.93 | 25895.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 25790.00 | 25831.88 | 25877.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 25790.00 | 25831.88 | 25877.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 25815.00 | 25817.80 | 25862.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 25575.00 | 25804.70 | 25836.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 25945.00 | 25816.26 | 25814.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 25945.00 | 25816.26 | 25814.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 26090.00 | 25871.01 | 25839.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-15 13:15:00 | 25780.00 | 25876.65 | 25849.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 13:15:00 | 25780.00 | 25876.65 | 25849.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 13:15:00 | 25780.00 | 25876.65 | 25849.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:00:00 | 25780.00 | 25876.65 | 25849.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 25990.00 | 25899.32 | 25862.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:30:00 | 25850.00 | 25899.32 | 25862.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 15:15:00 | 25900.00 | 25899.45 | 25865.55 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 09:15:00 | 25795.00 | 25863.03 | 25868.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 10:15:00 | 25530.00 | 25729.70 | 25791.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 25535.00 | 25486.88 | 25575.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 25535.00 | 25486.88 | 25575.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 25535.00 | 25486.88 | 25575.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:45:00 | 25620.00 | 25486.88 | 25575.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 25500.00 | 25489.50 | 25568.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:45:00 | 25495.00 | 25489.50 | 25568.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 25505.00 | 25492.60 | 25562.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 14:45:00 | 25450.00 | 25480.61 | 25539.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:00:00 | 25465.00 | 25472.59 | 25525.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:30:00 | 25400.00 | 25459.07 | 25514.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 12:45:00 | 25455.00 | 25409.31 | 25449.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 25250.00 | 25377.44 | 25431.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 14:15:00 | 25220.00 | 25377.44 | 25431.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:45:00 | 25235.00 | 25344.49 | 25371.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 15:15:00 | 25550.00 | 25356.33 | 25362.53 | SL hit (close>static) qty=1.00 sl=25485.00 alert=retest2 |

### Cycle 158 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 25515.00 | 25388.06 | 25376.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 25740.00 | 25458.45 | 25409.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 25485.00 | 25500.40 | 25448.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 14:15:00 | 25485.00 | 25500.40 | 25448.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 25485.00 | 25500.40 | 25448.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:30:00 | 25460.00 | 25500.40 | 25448.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 25510.00 | 25502.32 | 25454.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 25345.00 | 25502.32 | 25454.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 25350.00 | 25471.85 | 25445.01 | EMA400 retest candle locked (from upside) |

### Cycle 159 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 25365.00 | 25427.79 | 25428.23 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 25675.00 | 25473.12 | 25446.50 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 25390.00 | 25454.76 | 25461.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 11:15:00 | 25345.00 | 25432.81 | 25450.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 15:15:00 | 25385.00 | 25379.14 | 25415.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 09:15:00 | 25500.00 | 25379.14 | 25415.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 25550.00 | 25413.31 | 25427.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:45:00 | 25630.00 | 25413.31 | 25427.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 25570.00 | 25444.65 | 25440.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 25835.00 | 25562.18 | 25498.38 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-17 10:00:00 | 26481.00 | 2024-05-18 11:15:00 | 26689.25 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-05-17 10:30:00 | 26450.00 | 2024-05-18 11:15:00 | 26689.25 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-05-17 12:45:00 | 26362.90 | 2024-05-18 11:15:00 | 26689.25 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-05-24 15:15:00 | 26169.55 | 2024-05-27 13:15:00 | 26230.65 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2024-05-27 10:00:00 | 26100.50 | 2024-05-27 13:15:00 | 26230.65 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-05-27 12:45:00 | 26131.00 | 2024-05-27 13:15:00 | 26230.65 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2024-06-21 11:30:00 | 27234.20 | 2024-06-26 10:15:00 | 27667.30 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-06-21 12:30:00 | 27221.05 | 2024-06-26 10:15:00 | 27667.30 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-06-28 09:15:00 | 27915.45 | 2024-07-11 10:15:00 | 27942.00 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2024-08-01 09:30:00 | 28531.00 | 2024-08-02 14:15:00 | 28204.25 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-08-01 11:30:00 | 28560.00 | 2024-08-02 14:15:00 | 28204.25 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-08-01 15:00:00 | 28450.00 | 2024-08-02 14:15:00 | 28204.25 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-08-02 09:15:00 | 28550.35 | 2024-08-02 14:15:00 | 28204.25 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-08-06 12:00:00 | 28265.90 | 2024-08-07 13:15:00 | 28410.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-08-14 09:30:00 | 27276.10 | 2024-08-16 11:15:00 | 27700.10 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-08-14 10:00:00 | 27183.95 | 2024-08-16 11:15:00 | 27700.10 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-08-14 11:00:00 | 27265.00 | 2024-08-16 11:15:00 | 27700.10 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-08-16 10:45:00 | 27265.50 | 2024-08-16 11:15:00 | 27700.10 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-08-21 09:15:00 | 28031.10 | 2024-09-03 14:15:00 | 29666.00 | STOP_HIT | 1.00 | 5.83% |
| SELL | retest2 | 2024-09-12 13:15:00 | 29550.10 | 2024-09-18 10:15:00 | 28072.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-13 09:15:00 | 29586.90 | 2024-09-18 10:15:00 | 28107.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-13 11:15:00 | 29576.10 | 2024-09-18 10:15:00 | 28097.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-13 12:30:00 | 29585.20 | 2024-09-18 10:15:00 | 28105.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-16 11:15:00 | 29380.10 | 2024-09-18 11:15:00 | 27911.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-12 13:15:00 | 29550.10 | 2024-09-19 11:15:00 | 28012.40 | STOP_HIT | 0.50 | 5.20% |
| SELL | retest2 | 2024-09-13 09:15:00 | 29586.90 | 2024-09-19 11:15:00 | 28012.40 | STOP_HIT | 0.50 | 5.32% |
| SELL | retest2 | 2024-09-13 11:15:00 | 29576.10 | 2024-09-19 11:15:00 | 28012.40 | STOP_HIT | 0.50 | 5.29% |
| SELL | retest2 | 2024-09-13 12:30:00 | 29585.20 | 2024-09-19 11:15:00 | 28012.40 | STOP_HIT | 0.50 | 5.32% |
| SELL | retest2 | 2024-09-16 11:15:00 | 29380.10 | 2024-09-19 11:15:00 | 28012.40 | STOP_HIT | 0.50 | 4.66% |
| BUY | retest2 | 2024-09-25 14:15:00 | 28538.70 | 2024-10-03 09:15:00 | 28865.55 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2024-10-15 09:15:00 | 29150.00 | 2024-10-17 09:15:00 | 28699.85 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-10-16 12:30:00 | 28914.00 | 2024-10-17 09:15:00 | 28699.85 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-10-25 11:00:00 | 28416.55 | 2024-10-30 12:15:00 | 28618.70 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-10-25 12:00:00 | 28500.00 | 2024-10-30 12:15:00 | 28618.70 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2024-10-25 14:00:00 | 28463.95 | 2024-10-30 12:15:00 | 28618.70 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-10-25 14:45:00 | 28489.70 | 2024-10-30 12:15:00 | 28618.70 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2024-10-28 15:00:00 | 28438.15 | 2024-10-30 12:15:00 | 28618.70 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-10-29 15:15:00 | 28342.00 | 2024-10-30 12:15:00 | 28618.70 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-11-08 14:15:00 | 28595.85 | 2024-11-12 10:15:00 | 29002.50 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-11-11 12:00:00 | 28650.60 | 2024-11-12 10:15:00 | 29002.50 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-11-11 12:30:00 | 28659.90 | 2024-11-12 10:15:00 | 29002.50 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-11-11 13:00:00 | 28635.00 | 2024-11-12 10:15:00 | 29002.50 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-11-28 10:45:00 | 27337.10 | 2024-11-29 10:15:00 | 27784.10 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-11-28 13:00:00 | 27337.30 | 2024-11-29 10:15:00 | 27784.10 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-11-28 15:00:00 | 27367.15 | 2024-11-29 10:15:00 | 27784.10 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-12-09 11:30:00 | 29172.00 | 2024-12-11 12:15:00 | 28880.10 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-12-09 13:45:00 | 29158.45 | 2024-12-11 12:15:00 | 28880.10 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-12-09 14:45:00 | 29165.60 | 2024-12-11 12:15:00 | 28880.10 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-12-10 09:15:00 | 29182.35 | 2024-12-11 12:15:00 | 28880.10 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-12-11 10:45:00 | 29265.25 | 2024-12-11 12:15:00 | 28880.10 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-12-16 10:30:00 | 28480.00 | 2024-12-19 11:15:00 | 28745.25 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest1 | 2024-12-31 09:30:00 | 29893.95 | 2025-01-03 09:15:00 | 29680.15 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest1 | 2024-12-31 10:30:00 | 29924.80 | 2025-01-03 09:15:00 | 29680.15 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-01-09 15:15:00 | 29600.00 | 2025-01-15 09:15:00 | 28120.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 09:45:00 | 29475.15 | 2025-01-15 09:15:00 | 28001.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 15:15:00 | 29600.00 | 2025-01-17 09:15:00 | 27620.90 | STOP_HIT | 0.50 | 6.69% |
| SELL | retest2 | 2025-01-10 09:45:00 | 29475.15 | 2025-01-17 09:15:00 | 27620.90 | STOP_HIT | 0.50 | 6.29% |
| SELL | retest2 | 2025-01-30 11:30:00 | 25948.90 | 2025-01-31 11:15:00 | 26200.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-01-30 12:00:00 | 25946.65 | 2025-01-31 11:15:00 | 26200.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-02-28 15:00:00 | 30369.95 | 2025-03-07 15:15:00 | 30651.95 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2025-03-03 10:15:00 | 30199.65 | 2025-03-07 15:15:00 | 30651.95 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest2 | 2025-03-11 11:30:00 | 30610.00 | 2025-03-19 10:15:00 | 30450.00 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2025-03-11 12:15:00 | 30674.00 | 2025-03-19 10:15:00 | 30450.00 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2025-03-11 12:45:00 | 30673.45 | 2025-03-19 10:15:00 | 30450.00 | STOP_HIT | 1.00 | 0.73% |
| BUY | retest2 | 2025-03-21 09:30:00 | 30250.00 | 2025-03-27 09:15:00 | 30519.40 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest2 | 2025-03-21 11:15:00 | 30325.25 | 2025-03-27 09:15:00 | 30519.40 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2025-03-21 14:30:00 | 30362.75 | 2025-03-27 09:15:00 | 30519.40 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2025-03-24 09:15:00 | 30621.40 | 2025-03-27 09:15:00 | 30519.40 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-04-04 09:15:00 | 29704.30 | 2025-04-07 09:15:00 | 28219.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 09:15:00 | 29704.30 | 2025-04-07 15:15:00 | 29107.55 | STOP_HIT | 0.50 | 2.01% |
| BUY | retest2 | 2025-04-21 09:45:00 | 30930.00 | 2025-04-23 14:15:00 | 30490.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-04-21 15:15:00 | 30915.00 | 2025-04-23 14:15:00 | 30490.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-04-22 11:30:00 | 30915.00 | 2025-04-23 14:15:00 | 30490.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-05-08 14:45:00 | 29885.00 | 2025-05-12 10:15:00 | 30405.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-05-09 12:00:00 | 29970.00 | 2025-05-12 10:15:00 | 30405.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-05-26 14:45:00 | 30495.00 | 2025-05-30 15:15:00 | 30620.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-06-09 11:30:00 | 31920.00 | 2025-06-11 15:15:00 | 31560.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-06-09 13:00:00 | 31945.00 | 2025-06-11 15:15:00 | 31560.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-06-10 09:15:00 | 31920.00 | 2025-06-11 15:15:00 | 31560.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-06-18 12:30:00 | 31800.00 | 2025-06-18 15:15:00 | 31555.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-06-19 09:15:00 | 31775.00 | 2025-06-19 12:15:00 | 31465.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-06-19 10:00:00 | 31770.00 | 2025-06-19 12:15:00 | 31465.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-07-02 09:15:00 | 34860.00 | 2025-07-04 14:15:00 | 34750.00 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-07-02 11:00:00 | 34855.00 | 2025-07-04 14:15:00 | 34750.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-07-09 13:00:00 | 34090.00 | 2025-07-11 09:15:00 | 34690.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-07-16 10:30:00 | 34180.00 | 2025-07-22 12:15:00 | 34175.00 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-07-17 09:30:00 | 34210.00 | 2025-07-22 12:15:00 | 34175.00 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-07-17 11:45:00 | 34160.00 | 2025-07-22 12:15:00 | 34175.00 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-08-04 13:15:00 | 34055.00 | 2025-08-07 09:15:00 | 32352.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-04 13:15:00 | 34055.00 | 2025-08-07 14:15:00 | 32725.00 | STOP_HIT | 0.50 | 3.91% |
| BUY | retest2 | 2025-08-14 09:15:00 | 33195.00 | 2025-08-14 13:15:00 | 33010.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-08-22 10:15:00 | 32570.00 | 2025-09-01 15:15:00 | 32275.00 | STOP_HIT | 1.00 | 0.91% |
| SELL | retest2 | 2025-09-11 14:15:00 | 31000.00 | 2025-09-15 10:15:00 | 31290.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-09-12 12:30:00 | 31025.00 | 2025-09-15 10:15:00 | 31290.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest1 | 2025-09-22 12:15:00 | 30520.00 | 2025-09-25 10:15:00 | 30405.00 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest1 | 2025-09-22 13:30:00 | 30480.00 | 2025-09-25 10:15:00 | 30405.00 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-09-24 13:00:00 | 30175.00 | 2025-10-01 12:15:00 | 30015.00 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest2 | 2025-09-25 13:00:00 | 30195.00 | 2025-10-01 12:15:00 | 30015.00 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2025-10-06 11:30:00 | 30410.00 | 2025-10-07 14:15:00 | 29825.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-10-06 12:00:00 | 30400.00 | 2025-10-07 14:15:00 | 29825.00 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-10-20 11:30:00 | 29650.00 | 2025-10-20 13:15:00 | 30125.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-10-23 15:15:00 | 30100.00 | 2025-10-24 12:15:00 | 29870.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-10-30 09:30:00 | 29630.00 | 2025-11-04 13:15:00 | 29345.00 | STOP_HIT | 1.00 | 0.96% |
| SELL | retest2 | 2025-10-30 11:00:00 | 29675.00 | 2025-11-04 13:15:00 | 29345.00 | STOP_HIT | 1.00 | 1.11% |
| BUY | retest2 | 2025-11-12 15:00:00 | 29605.00 | 2025-11-14 11:15:00 | 29365.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-11-13 09:30:00 | 29550.00 | 2025-11-14 11:15:00 | 29365.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-11-13 10:00:00 | 29560.00 | 2025-11-14 11:15:00 | 29365.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-11-13 14:15:00 | 29580.00 | 2025-11-14 11:15:00 | 29365.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-11-19 15:15:00 | 29620.00 | 2025-11-20 09:15:00 | 29485.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-11-20 10:30:00 | 29580.00 | 2025-11-20 11:15:00 | 29535.00 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2025-11-20 11:15:00 | 29565.00 | 2025-11-20 11:15:00 | 29535.00 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-11-25 13:30:00 | 29810.00 | 2025-11-27 12:15:00 | 29550.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-12-05 13:15:00 | 28885.00 | 2025-12-18 09:15:00 | 27440.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 14:00:00 | 28890.00 | 2025-12-18 09:15:00 | 27445.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 13:15:00 | 28885.00 | 2025-12-18 10:15:00 | 27850.00 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2025-12-05 14:00:00 | 28890.00 | 2025-12-18 10:15:00 | 27850.00 | STOP_HIT | 0.50 | 3.60% |
| BUY | retest2 | 2025-12-22 11:30:00 | 28090.00 | 2025-12-22 15:15:00 | 27950.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-12-22 14:00:00 | 28055.00 | 2025-12-22 15:15:00 | 27950.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-12-22 14:45:00 | 28090.00 | 2025-12-22 15:15:00 | 27950.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-12-23 11:30:00 | 28100.00 | 2026-01-01 13:15:00 | 28500.00 | STOP_HIT | 1.00 | 1.42% |
| BUY | retest2 | 2025-12-24 14:45:00 | 28255.00 | 2026-01-01 13:15:00 | 28500.00 | STOP_HIT | 1.00 | 0.87% |
| BUY | retest2 | 2026-01-07 14:45:00 | 28470.00 | 2026-01-09 09:15:00 | 28055.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-01-08 09:30:00 | 28485.00 | 2026-01-09 09:15:00 | 28055.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2026-01-08 14:15:00 | 28395.00 | 2026-01-09 09:15:00 | 28055.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-01-08 15:15:00 | 28400.00 | 2026-01-09 09:15:00 | 28055.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-01-12 09:15:00 | 27785.00 | 2026-01-12 14:15:00 | 28310.00 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-01-12 11:30:00 | 28035.00 | 2026-01-12 14:15:00 | 28310.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-01-16 12:30:00 | 27860.00 | 2026-01-22 11:15:00 | 27730.00 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2026-02-23 14:15:00 | 26410.00 | 2026-02-23 14:15:00 | 26500.00 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2026-02-25 11:45:00 | 26290.00 | 2026-02-27 13:15:00 | 26500.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-02-25 13:00:00 | 26290.00 | 2026-02-27 13:15:00 | 26500.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-02-26 12:30:00 | 26305.00 | 2026-02-27 13:15:00 | 26500.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-02-27 09:15:00 | 26205.00 | 2026-02-27 13:15:00 | 26500.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2026-03-06 12:15:00 | 27420.00 | 2026-03-09 09:15:00 | 26925.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2026-03-06 14:15:00 | 27420.00 | 2026-03-09 09:15:00 | 26925.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-03-17 11:30:00 | 26560.00 | 2026-03-20 12:15:00 | 26730.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-03-18 09:30:00 | 26625.00 | 2026-03-20 12:15:00 | 26730.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2026-03-18 13:00:00 | 26550.00 | 2026-03-20 12:15:00 | 26730.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2026-03-20 09:15:00 | 26615.00 | 2026-03-20 12:15:00 | 26730.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2026-04-01 11:15:00 | 26205.00 | 2026-04-01 12:15:00 | 26455.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-04-08 12:45:00 | 25825.00 | 2026-04-15 10:15:00 | 25945.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2026-04-08 14:30:00 | 25825.00 | 2026-04-15 10:15:00 | 25945.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2026-04-09 10:45:00 | 25780.00 | 2026-04-15 10:15:00 | 25945.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-04-09 12:30:00 | 25800.00 | 2026-04-15 10:15:00 | 25945.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-04-13 09:15:00 | 25575.00 | 2026-04-15 10:15:00 | 25945.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-04-22 14:45:00 | 25450.00 | 2026-04-28 15:15:00 | 25550.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2026-04-23 10:00:00 | 25465.00 | 2026-04-28 15:15:00 | 25550.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2026-04-23 10:30:00 | 25400.00 | 2026-04-29 09:15:00 | 25515.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-04-24 12:45:00 | 25455.00 | 2026-04-29 09:15:00 | 25515.00 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2026-04-24 14:15:00 | 25220.00 | 2026-04-29 09:15:00 | 25515.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-04-28 10:45:00 | 25235.00 | 2026-04-29 09:15:00 | 25515.00 | STOP_HIT | 1.00 | -1.11% |
