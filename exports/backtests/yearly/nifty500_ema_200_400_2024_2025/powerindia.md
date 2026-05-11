# Hitachi Energy India Ltd. (POWERINDIA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 33960.00
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
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 28 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 34 |
| PARTIAL | 6 |
| TARGET_HIT | 9 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 40 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 25
- **Target hits / Stop hits / Partials:** 9 / 25 / 6
- **Avg / median % per leg:** 0.33% / -1.54%
- **Sum % (uncompounded):** 13.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 0 | 0.0% | 0 | 11 | 0 | -1.86% | -20.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 0 | 0.0% | 0 | 11 | 0 | -1.86% | -20.5% |
| SELL (all) | 29 | 15 | 51.7% | 9 | 14 | 6 | 1.16% | 33.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 15 | 51.7% | 9 | 14 | 6 | 1.16% | 33.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 40 | 15 | 37.5% | 9 | 25 | 6 | 0.33% | 13.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 15:15:00 | 11511.35 | 13154.60 | 13157.70 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 11:15:00 | 14351.50 | 12946.12 | 12945.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 13:15:00 | 14630.00 | 13074.61 | 13011.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 15:15:00 | 13805.00 | 13845.11 | 13482.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-10 09:15:00 | 13536.30 | 13845.11 | 13482.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 13327.40 | 13839.96 | 13481.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 13327.40 | 13839.96 | 13481.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 13576.15 | 13837.33 | 13481.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 11:15:00 | 13676.15 | 13837.33 | 13481.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-10 14:15:00 | 13269.65 | 13824.79 | 13482.70 | SL hit (close<static) qty=1.00 sl=13328.35 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 11:15:00 | 11487.05 | 13250.56 | 13256.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 11339.95 | 13213.76 | 13238.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 13518.60 | 12607.68 | 12904.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 09:15:00 | 13518.60 | 12607.68 | 12904.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 13518.60 | 12607.68 | 12904.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 12:15:00 | 12465.80 | 12610.09 | 12902.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 12:45:00 | 12468.85 | 12607.76 | 12900.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 14:00:00 | 12466.15 | 12606.35 | 12898.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 12:30:00 | 12370.05 | 12610.43 | 12891.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 13:15:00 | 11842.51 | 12606.74 | 12888.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 13:15:00 | 11845.41 | 12606.74 | 12888.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 13:15:00 | 11842.84 | 12606.74 | 12888.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-03 09:15:00 | 11219.22 | 12584.35 | 12872.83 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 09:15:00 | 13592.00 | 12311.65 | 12306.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 10:15:00 | 13682.00 | 12325.29 | 12313.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 17000.00 | 17071.46 | 15669.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 12:00:00 | 17000.00 | 17071.46 | 15669.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 19260.00 | 19980.46 | 19214.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:00:00 | 19260.00 | 19980.46 | 19214.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 19150.00 | 19972.20 | 19214.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:00:00 | 19150.00 | 19972.20 | 19214.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 19300.00 | 19965.51 | 19214.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:30:00 | 19190.00 | 19965.51 | 19214.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 19085.00 | 19917.12 | 19244.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 15:00:00 | 19085.00 | 19917.12 | 19244.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 19000.00 | 19907.99 | 19243.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 19241.00 | 19907.99 | 19243.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 13:15:00 | 19152.00 | 19877.65 | 19240.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 09:15:00 | 19115.00 | 19851.53 | 19237.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 18900.00 | 19833.08 | 19234.01 | SL hit (close<static) qty=1.00 sl=18920.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 17785.00 | 19138.66 | 19140.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 14:15:00 | 17557.00 | 19101.17 | 19121.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 10:15:00 | 18021.00 | 17971.80 | 18419.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-30 11:00:00 | 18021.00 | 17971.80 | 18419.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 20225.00 | 17966.26 | 18373.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 20225.00 | 17966.26 | 18373.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 12:15:00 | 21865.00 | 18733.88 | 18725.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 22172.00 | 19779.36 | 19310.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 11:15:00 | 20560.00 | 20898.99 | 20110.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 20560.00 | 20898.99 | 20110.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 20100.00 | 20879.19 | 20112.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 20100.00 | 20879.19 | 20112.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 20080.00 | 20871.23 | 20112.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 19980.00 | 20862.37 | 20111.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 19800.00 | 20851.80 | 20110.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:00:00 | 19800.00 | 20851.80 | 20110.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 19355.00 | 20836.90 | 20106.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:00:00 | 19355.00 | 20836.90 | 20106.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 18630.00 | 19671.87 | 19675.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 18415.00 | 19659.37 | 19669.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 19340.00 | 19173.71 | 19383.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 10:00:00 | 19340.00 | 19173.71 | 19383.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 19437.00 | 19176.33 | 19383.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 14:30:00 | 19256.00 | 19207.37 | 19388.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 15:15:00 | 18293.20 | 19199.21 | 19383.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-09 11:15:00 | 17330.40 | 19150.52 | 19356.66 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 8 — BUY (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 12:15:00 | 22483.00 | 18916.79 | 18911.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 09:15:00 | 22565.00 | 19056.13 | 18982.16 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-01-10 11:15:00 | 13676.15 | 2025-01-10 14:15:00 | 13269.65 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2025-01-31 12:15:00 | 12465.80 | 2025-02-01 13:15:00 | 11842.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-31 12:45:00 | 12468.85 | 2025-02-01 13:15:00 | 11845.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-31 14:00:00 | 12466.15 | 2025-02-01 13:15:00 | 11842.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-31 12:15:00 | 12465.80 | 2025-02-03 09:15:00 | 11219.22 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-31 12:45:00 | 12468.85 | 2025-02-03 09:15:00 | 11221.97 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-31 14:00:00 | 12466.15 | 2025-02-03 09:15:00 | 11219.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-01 12:30:00 | 12370.05 | 2025-02-03 09:15:00 | 11133.05 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-10 12:45:00 | 12120.00 | 2025-02-12 09:15:00 | 11514.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 11987.95 | 2025-02-12 09:15:00 | 11388.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 12:45:00 | 12120.00 | 2025-02-17 09:15:00 | 10908.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 11987.95 | 2025-02-17 11:15:00 | 10789.16 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-04 14:00:00 | 12133.15 | 2025-03-05 09:15:00 | 12892.00 | STOP_HIT | 1.00 | -6.25% |
| SELL | retest2 | 2025-03-04 14:30:00 | 12108.50 | 2025-03-05 09:15:00 | 12892.00 | STOP_HIT | 1.00 | -6.47% |
| SELL | retest2 | 2025-03-11 14:00:00 | 12201.60 | 2025-03-17 09:15:00 | 12666.95 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2025-03-11 14:30:00 | 12194.00 | 2025-03-17 09:15:00 | 12666.95 | STOP_HIT | 1.00 | -3.88% |
| SELL | retest2 | 2025-03-11 15:00:00 | 12179.00 | 2025-03-17 09:15:00 | 12666.95 | STOP_HIT | 1.00 | -4.01% |
| SELL | retest2 | 2025-03-13 10:45:00 | 12209.50 | 2025-03-17 09:15:00 | 12666.95 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2025-03-13 13:15:00 | 12063.75 | 2025-03-17 09:15:00 | 12666.95 | STOP_HIT | 1.00 | -5.00% |
| SELL | retest2 | 2025-03-13 14:00:00 | 12028.05 | 2025-03-17 09:15:00 | 12666.95 | STOP_HIT | 1.00 | -5.31% |
| SELL | retest2 | 2025-03-18 09:15:00 | 11886.90 | 2025-03-24 09:15:00 | 12525.95 | STOP_HIT | 1.00 | -5.38% |
| SELL | retest2 | 2025-03-18 11:00:00 | 12047.20 | 2025-03-24 09:15:00 | 12525.95 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2025-03-21 13:45:00 | 12024.00 | 2025-03-24 09:15:00 | 12525.95 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2025-03-27 09:15:00 | 12023.35 | 2025-03-27 14:15:00 | 13115.20 | STOP_HIT | 1.00 | -9.08% |
| SELL | retest2 | 2025-04-04 14:30:00 | 12018.75 | 2025-04-07 09:15:00 | 10816.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 11736.90 | 2025-04-07 09:15:00 | 10563.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-01 09:15:00 | 19241.00 | 2025-09-02 10:15:00 | 18900.00 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-09-01 13:15:00 | 19152.00 | 2025-09-02 10:15:00 | 18900.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-09-02 09:15:00 | 19115.00 | 2025-09-02 10:15:00 | 18900.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-09-03 10:45:00 | 19160.00 | 2025-09-04 13:15:00 | 19095.00 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-09-04 09:15:00 | 19319.00 | 2025-09-04 14:15:00 | 18839.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-09-10 09:30:00 | 19368.00 | 2025-09-19 12:15:00 | 19026.00 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-09-22 09:15:00 | 19240.00 | 2025-09-22 14:15:00 | 19090.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-09-22 14:45:00 | 19244.00 | 2025-09-23 10:15:00 | 18948.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-09-26 12:15:00 | 19701.00 | 2025-09-26 14:15:00 | 19069.00 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2025-09-26 12:45:00 | 19696.00 | 2025-09-26 14:15:00 | 19069.00 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2026-01-08 14:30:00 | 19256.00 | 2026-01-08 15:15:00 | 18293.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 14:30:00 | 19256.00 | 2026-01-09 11:15:00 | 17330.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-04 10:30:00 | 19293.00 | 2026-02-06 09:15:00 | 21560.00 | STOP_HIT | 1.00 | -11.75% |
| SELL | retest2 | 2026-02-04 12:30:00 | 18988.00 | 2026-02-06 09:15:00 | 21560.00 | STOP_HIT | 1.00 | -13.55% |
