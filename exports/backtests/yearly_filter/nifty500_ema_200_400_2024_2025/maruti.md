# Maruti Suzuki India Ltd. (MARUTI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 13733.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 28 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 6
- **Winners / losers:** 2 / 20
- **Target hits / Stop hits / Partials:** 2 / 20 / 0
- **Avg / median % per leg:** -0.29% / -1.08%
- **Sum % (uncompounded):** -6.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 2 | 13.3% | 2 | 13 | 0 | 0.41% | 6.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 2 | 13.3% | 2 | 13 | 0 | 0.41% | 6.2% |
| SELL (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.80% | -12.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.80% | -12.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 22 | 2 | 9.1% | 2 | 20 | 0 | -0.29% | -6.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 15:15:00 | 12145.00 | 12430.84 | 12432.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 13:15:00 | 12123.45 | 12368.49 | 12392.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 13:15:00 | 12334.70 | 12330.46 | 12368.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-12 14:00:00 | 12334.70 | 12330.46 | 12368.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 12393.35 | 12331.09 | 12369.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 15:00:00 | 12393.35 | 12331.09 | 12369.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 12375.00 | 12331.52 | 12369.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 09:15:00 | 12360.00 | 12331.52 | 12369.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 10:30:00 | 12347.85 | 12314.44 | 12354.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 12:45:00 | 12353.15 | 12315.02 | 12354.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 15:00:00 | 12339.25 | 12315.81 | 12354.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 12329.05 | 12315.94 | 12354.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 12450.00 | 12315.94 | 12354.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 12552.00 | 12318.29 | 12355.61 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-20 09:15:00 | 12552.00 | 12318.29 | 12355.61 | SL hit (close>static) qty=1.00 sl=12412.95 alert=retest2 |

### Cycle 2 — BUY (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 09:15:00 | 12670.30 | 12389.29 | 12389.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 09:15:00 | 13156.55 | 12415.38 | 12402.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 12:15:00 | 12595.65 | 12638.62 | 12529.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-04 13:00:00 | 12595.65 | 12638.62 | 12529.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 12615.20 | 12637.48 | 12529.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-09 09:15:00 | 12696.35 | 12621.93 | 12529.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 09:15:00 | 12487.30 | 12653.42 | 12559.17 | SL hit (close<static) qty=1.00 sl=12512.50 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 09:15:00 | 11979.55 | 12480.97 | 12483.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 12:15:00 | 11655.00 | 12426.37 | 12455.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 12:15:00 | 11331.15 | 11317.53 | 11636.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-06 12:30:00 | 11338.05 | 11317.53 | 11636.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 11431.00 | 11099.94 | 11358.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:00:00 | 11431.00 | 11099.94 | 11358.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 11558.05 | 11104.50 | 11359.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:30:00 | 11533.25 | 11104.50 | 11359.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 09:15:00 | 12033.40 | 11520.81 | 11520.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 09:15:00 | 12199.00 | 11710.57 | 11626.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 09:15:00 | 12397.00 | 12436.13 | 12129.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 10:00:00 | 12397.00 | 12436.13 | 12129.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 11956.60 | 12416.34 | 12167.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 10:00:00 | 11956.60 | 12416.34 | 12167.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 10:15:00 | 11841.25 | 12410.62 | 12166.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 11:00:00 | 11841.25 | 12410.62 | 12166.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-18 09:15:00 | 11624.80 | 12001.93 | 12003.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 13:15:00 | 11454.35 | 11891.92 | 11939.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 11807.00 | 11721.66 | 11829.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-15 09:15:00 | 11807.00 | 11721.66 | 11829.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 11807.00 | 11721.66 | 11829.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 10:00:00 | 11807.00 | 11721.66 | 11829.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 10:15:00 | 11875.00 | 11723.18 | 11830.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 11:00:00 | 11875.00 | 11723.18 | 11830.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 11:15:00 | 11890.00 | 11724.84 | 11830.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 11:30:00 | 11873.00 | 11724.84 | 11830.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 11785.00 | 11720.89 | 11812.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 14:30:00 | 11666.00 | 11743.96 | 11816.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 15:15:00 | 11650.00 | 11743.96 | 11816.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 11:00:00 | 11720.00 | 11742.29 | 11814.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 11915.00 | 11752.97 | 11815.16 | SL hit (close>static) qty=1.00 sl=11875.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 09:15:00 | 12503.00 | 11872.23 | 11871.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 10:15:00 | 12563.00 | 11879.10 | 11874.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 12325.00 | 12346.75 | 12169.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:45:00 | 12319.00 | 12346.75 | 12169.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 12165.00 | 12345.85 | 12192.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:30:00 | 12172.00 | 12345.85 | 12192.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 12221.00 | 12344.61 | 12192.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 11:15:00 | 12257.00 | 12344.61 | 12192.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 11:45:00 | 12238.00 | 12343.65 | 12193.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 10:15:00 | 12252.00 | 12339.56 | 12194.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 10:15:00 | 12131.00 | 12337.49 | 12194.41 | SL hit (close<static) qty=1.00 sl=12132.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 14511.00 | 16031.42 | 16031.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 09:15:00 | 14425.00 | 15985.22 | 16008.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 13626.00 | 13304.14 | 14062.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:00:00 | 13626.00 | 13304.14 | 14062.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 13748.00 | 13312.84 | 13750.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 13525.00 | 13318.99 | 13749.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 14:15:00 | 13535.00 | 13323.08 | 13746.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 13424.00 | 13328.09 | 13745.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:30:00 | 13539.00 | 13340.03 | 13732.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 13740.00 | 13351.78 | 13729.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 13788.00 | 13351.78 | 13729.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 13725.00 | 13355.49 | 13729.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:30:00 | 13848.00 | 13355.49 | 13729.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 13761.00 | 13359.53 | 13729.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:00:00 | 13761.00 | 13359.53 | 13729.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 13774.00 | 13363.65 | 13729.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:45:00 | 13776.00 | 13363.65 | 13729.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 13833.00 | 13368.32 | 13729.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:00:00 | 13833.00 | 13368.32 | 13729.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 13817.00 | 13372.79 | 13730.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 13740.00 | 13380.69 | 13730.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 14:15:00 | 13746.00 | 13396.90 | 13730.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-07-09 09:15:00 | 12492.25 | 2024-08-05 09:15:00 | 12371.55 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-07-22 09:30:00 | 12468.00 | 2024-08-05 09:15:00 | 12371.55 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-07-23 12:30:00 | 12536.50 | 2024-08-05 09:15:00 | 12371.55 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-07-25 09:30:00 | 12465.10 | 2024-08-05 09:15:00 | 12371.55 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-07-25 12:00:00 | 12507.00 | 2024-08-05 09:15:00 | 12371.55 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-07-25 12:45:00 | 12480.90 | 2024-08-05 09:15:00 | 12371.55 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-07-25 14:45:00 | 12490.00 | 2024-08-05 09:15:00 | 12371.55 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-07-26 09:45:00 | 12519.90 | 2024-08-05 09:15:00 | 12371.55 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-09-13 09:15:00 | 12360.00 | 2024-09-20 09:15:00 | 12552.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-09-19 10:30:00 | 12347.85 | 2024-09-20 09:15:00 | 12552.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-09-19 12:45:00 | 12353.15 | 2024-09-20 09:15:00 | 12552.00 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-09-19 15:00:00 | 12339.25 | 2024-09-20 09:15:00 | 12552.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-10-09 09:15:00 | 12696.35 | 2024-10-15 09:15:00 | 12487.30 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-04-25 14:30:00 | 11666.00 | 2025-04-30 09:15:00 | 11915.00 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-04-25 15:15:00 | 11650.00 | 2025-04-30 09:15:00 | 11915.00 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-04-28 11:00:00 | 11720.00 | 2025-04-30 09:15:00 | 11915.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-06-02 11:15:00 | 12257.00 | 2025-06-03 10:15:00 | 12131.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-06-02 11:45:00 | 12238.00 | 2025-06-03 10:15:00 | 12131.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-06-03 10:15:00 | 12252.00 | 2025-06-03 10:15:00 | 12131.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-06-04 10:15:00 | 12232.00 | 2025-06-05 09:15:00 | 12062.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-06-06 10:15:00 | 12322.00 | 2025-08-18 09:15:00 | 13554.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-13 09:45:00 | 12295.00 | 2025-08-18 09:15:00 | 13524.50 | TARGET_HIT | 1.00 | 10.00% |
