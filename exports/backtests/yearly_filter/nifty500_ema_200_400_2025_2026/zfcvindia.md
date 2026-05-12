# ZF Commercial Vehicle Control Systems India Ltd. (ZFCVINDIA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 14532.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 17 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 2 |
| TARGET_HIT | 6 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 6
- **Target hits / Stop hits / Partials:** 4 / 8 / 2
- **Avg / median % per leg:** 2.69% / 2.98%
- **Sum % (uncompounded):** 37.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 4 | 0 | 0 | 10.00% | 40.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 4 | 0 | 0 | 10.00% | 40.0% |
| SELL (all) | 10 | 4 | 40.0% | 0 | 8 | 2 | -0.23% | -2.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 4 | 40.0% | 0 | 8 | 2 | -0.23% | -2.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 8 | 57.1% | 4 | 8 | 2 | 2.69% | 37.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 12561.00 | 13487.73 | 13492.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 10:15:00 | 12540.00 | 13461.32 | 13478.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 10:15:00 | 13501.00 | 13342.13 | 13413.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 10:15:00 | 13501.00 | 13342.13 | 13413.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 13501.00 | 13342.13 | 13413.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:00:00 | 13501.00 | 13342.13 | 13413.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 13640.00 | 13345.10 | 13414.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:45:00 | 13768.00 | 13345.10 | 13414.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 13447.00 | 13357.19 | 13418.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:15:00 | 13522.00 | 13357.19 | 13418.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 13485.00 | 13358.46 | 13418.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:30:00 | 13511.00 | 13358.46 | 13418.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 13410.00 | 13366.19 | 13417.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:45:00 | 13436.00 | 13366.19 | 13417.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 13380.00 | 13366.33 | 13417.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:45:00 | 13341.00 | 13366.07 | 13416.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 13321.00 | 13366.21 | 13416.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 10:15:00 | 12673.95 | 13134.31 | 13258.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 10:15:00 | 12654.95 | 13134.31 | 13258.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 13:15:00 | 12943.00 | 12936.07 | 13112.33 | SL hit (close>ema200) qty=0.50 sl=12936.07 alert=retest2 |

### Cycle 2 — BUY (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 12:15:00 | 14937.00 | 13153.09 | 13151.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 12:15:00 | 15258.00 | 13878.20 | 13577.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 14:15:00 | 14471.00 | 14576.60 | 14144.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 15:00:00 | 14471.00 | 14576.60 | 14144.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 14024.00 | 14546.66 | 14148.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:45:00 | 13990.00 | 14546.66 | 14148.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 13900.00 | 14540.23 | 14147.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:30:00 | 13914.00 | 14540.23 | 14147.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 14052.00 | 14524.48 | 14145.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 14052.00 | 14524.48 | 14145.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 13907.00 | 14518.34 | 14143.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 13907.00 | 14518.34 | 14143.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 13946.00 | 14502.04 | 14141.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:00:00 | 13946.00 | 14502.04 | 14141.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 14136.00 | 14494.46 | 14141.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 14:45:00 | 14280.00 | 14198.92 | 14069.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 14208.00 | 14199.43 | 14070.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 11:00:00 | 14196.00 | 14199.40 | 14071.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 12:30:00 | 14240.00 | 14200.14 | 14073.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-11 09:15:00 | 15628.80 | 14635.01 | 14359.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 14:15:00 | 13632.00 | 14577.79 | 14579.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 15:15:00 | 13577.00 | 14567.84 | 14574.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 14465.00 | 14120.10 | 14316.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 14465.00 | 14120.10 | 14316.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 14465.00 | 14120.10 | 14316.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 14465.00 | 14120.10 | 14316.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 14197.00 | 14120.86 | 14316.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:45:00 | 14181.00 | 14126.34 | 14316.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 15:15:00 | 14780.00 | 14134.29 | 14318.16 | SL hit (close>static) qty=1.00 sl=14480.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 10:15:00 | 14999.00 | 14382.19 | 14380.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 14:15:00 | 15177.00 | 14410.75 | 14394.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 14:15:00 | 14531.00 | 14561.95 | 14485.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 15:00:00 | 14531.00 | 14561.95 | 14485.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 14523.00 | 14572.12 | 14495.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 14:30:00 | 14523.00 | 14572.12 | 14495.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-10-09 13:45:00 | 13341.00 | 2025-10-31 10:15:00 | 12673.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 10:15:00 | 13321.00 | 2025-10-31 10:15:00 | 12654.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-09 13:45:00 | 13341.00 | 2025-11-13 13:15:00 | 12943.00 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2025-10-10 10:15:00 | 13321.00 | 2025-11-13 13:15:00 | 12943.00 | STOP_HIT | 0.50 | 2.84% |
| BUY | retest2 | 2026-01-28 14:45:00 | 14280.00 | 2026-02-11 09:15:00 | 15628.80 | TARGET_HIT | 1.00 | 9.45% |
| BUY | retest2 | 2026-01-29 10:15:00 | 14208.00 | 2026-02-11 09:15:00 | 15615.60 | TARGET_HIT | 1.00 | 9.91% |
| BUY | retest2 | 2026-01-29 11:00:00 | 14196.00 | 2026-02-11 09:15:00 | 15664.00 | TARGET_HIT | 1.00 | 10.34% |
| BUY | retest2 | 2026-01-29 12:30:00 | 14240.00 | 2026-02-11 10:15:00 | 15708.00 | TARGET_HIT | 1.00 | 10.31% |
| SELL | retest2 | 2026-04-01 13:45:00 | 14181.00 | 2026-04-01 15:15:00 | 14780.00 | STOP_HIT | 1.00 | -4.22% |
| SELL | retest2 | 2026-04-02 13:15:00 | 14160.00 | 2026-04-08 09:15:00 | 14542.00 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2026-04-09 09:45:00 | 14126.00 | 2026-04-17 09:15:00 | 14506.00 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2026-04-13 12:00:00 | 14167.00 | 2026-04-17 09:15:00 | 14506.00 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2026-04-15 11:15:00 | 14050.00 | 2026-04-17 09:15:00 | 14506.00 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2026-04-15 12:30:00 | 14101.00 | 2026-04-17 09:15:00 | 14506.00 | STOP_HIT | 1.00 | -2.87% |
