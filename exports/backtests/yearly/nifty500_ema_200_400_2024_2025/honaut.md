# Honeywell Automation India Ltd. (HONAUT)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 30210.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 17 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 11
- **Target hits / Stop hits / Partials:** 2 / 15 / 5
- **Avg / median % per leg:** 1.42% / 2.37%
- **Sum % (uncompounded):** 31.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.58% | -15.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.58% | -15.5% |
| SELL (all) | 16 | 11 | 68.8% | 2 | 9 | 5 | 2.92% | 46.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 11 | 68.8% | 2 | 9 | 5 | 2.92% | 46.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 22 | 11 | 50.0% | 2 | 15 | 5 | 1.42% | 31.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 11:15:00 | 50930.60 | 52421.15 | 52423.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 13:15:00 | 50800.00 | 52390.82 | 52408.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-10 09:15:00 | 50139.00 | 49749.81 | 50619.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-10 09:30:00 | 50201.20 | 49749.81 | 50619.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 50027.60 | 49817.78 | 50569.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 11:00:00 | 49630.00 | 49816.91 | 50539.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 12:00:00 | 49618.95 | 49814.94 | 50534.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 12:30:00 | 49625.60 | 49813.52 | 50530.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 13:30:00 | 49634.50 | 49813.02 | 50526.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 50481.45 | 49797.64 | 50476.86 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-18 12:15:00 | 50641.65 | 49806.04 | 50477.68 | SL hit (close>static) qty=1.00 sl=50595.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 15:15:00 | 37200.00 | 35757.34 | 35751.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 10:15:00 | 37375.00 | 35787.64 | 35767.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 14:15:00 | 37685.00 | 37749.30 | 37071.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 15:00:00 | 37685.00 | 37749.30 | 37071.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 38655.00 | 39733.68 | 38857.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:45:00 | 38735.00 | 39733.68 | 38857.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 38135.00 | 39717.78 | 38854.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 14:30:00 | 38195.00 | 39717.78 | 38854.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 38400.00 | 39570.59 | 38820.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:30:00 | 38350.00 | 39570.59 | 38820.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 38645.00 | 39554.43 | 38820.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:00:00 | 38645.00 | 39554.43 | 38820.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 38700.00 | 39545.93 | 38819.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 14:00:00 | 38900.00 | 39485.83 | 38813.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 38965.00 | 39471.26 | 38813.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 15:15:00 | 39235.00 | 39429.25 | 38811.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 37450.00 | 39407.64 | 38806.75 | SL hit (close<static) qty=1.00 sl=38645.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 35700.00 | 38345.30 | 38350.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 35015.00 | 36225.95 | 36610.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 35965.00 | 35806.77 | 36325.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 35965.00 | 35806.77 | 36325.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 35965.00 | 35806.77 | 36325.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 35965.00 | 35806.77 | 36325.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 35875.00 | 35807.45 | 36323.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 35400.00 | 35807.45 | 36323.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 15:15:00 | 35535.00 | 35731.83 | 36251.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 11:15:00 | 35600.00 | 35728.19 | 36242.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 12:15:00 | 35600.00 | 35727.01 | 36238.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 36000.00 | 35693.84 | 36192.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:30:00 | 36300.00 | 35693.84 | 36192.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 10:15:00 | 33758.25 | 35268.75 | 35835.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 10:15:00 | 33820.00 | 35268.75 | 35835.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 10:15:00 | 33820.00 | 35268.75 | 35835.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 15:15:00 | 33630.00 | 34887.73 | 35543.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 34560.00 | 34017.69 | 34818.05 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 34560.00 | 34017.69 | 34818.05 | SL hit (close>ema200) qty=0.50 sl=34017.69 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-08-06 09:15:00 | 52969.90 | 2024-08-06 09:15:00 | 52331.20 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-08-21 09:15:00 | 53017.10 | 2024-08-21 14:15:00 | 52399.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-08-21 11:45:00 | 52894.85 | 2024-08-21 14:15:00 | 52399.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-10-16 11:00:00 | 49630.00 | 2024-10-18 12:15:00 | 50641.65 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-10-16 12:00:00 | 49618.95 | 2024-10-18 12:15:00 | 50641.65 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2024-10-16 12:30:00 | 49625.60 | 2024-10-18 12:15:00 | 50641.65 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-10-16 13:30:00 | 49634.50 | 2024-10-18 12:15:00 | 50641.65 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-10-23 12:15:00 | 50167.65 | 2024-10-23 12:15:00 | 50700.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-10-25 10:00:00 | 50063.30 | 2024-10-30 09:15:00 | 45056.97 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 14:00:00 | 38900.00 | 2025-08-04 09:15:00 | 37450.00 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2025-08-01 09:15:00 | 38965.00 | 2025-08-04 09:15:00 | 37450.00 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2025-08-01 15:15:00 | 39235.00 | 2025-08-04 09:15:00 | 37450.00 | STOP_HIT | 1.00 | -4.55% |
| SELL | retest2 | 2025-11-25 09:15:00 | 35400.00 | 2025-12-11 10:15:00 | 33758.25 | PARTIAL | 0.50 | 4.64% |
| SELL | retest2 | 2025-11-26 15:15:00 | 35535.00 | 2025-12-11 10:15:00 | 33820.00 | PARTIAL | 0.50 | 4.83% |
| SELL | retest2 | 2025-11-27 11:15:00 | 35600.00 | 2025-12-11 10:15:00 | 33820.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 12:15:00 | 35600.00 | 2025-12-17 15:15:00 | 33630.00 | PARTIAL | 0.50 | 5.53% |
| SELL | retest2 | 2025-11-25 09:15:00 | 35400.00 | 2026-01-05 09:15:00 | 34560.00 | STOP_HIT | 0.50 | 2.37% |
| SELL | retest2 | 2025-11-26 15:15:00 | 35535.00 | 2026-01-05 09:15:00 | 34560.00 | STOP_HIT | 0.50 | 2.74% |
| SELL | retest2 | 2025-11-27 11:15:00 | 35600.00 | 2026-01-05 09:15:00 | 34560.00 | STOP_HIT | 0.50 | 2.92% |
| SELL | retest2 | 2025-11-27 12:15:00 | 35600.00 | 2026-01-05 09:15:00 | 34560.00 | STOP_HIT | 0.50 | 2.92% |
| SELL | retest2 | 2026-02-04 12:15:00 | 33175.00 | 2026-02-13 09:15:00 | 31516.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 12:15:00 | 33175.00 | 2026-03-04 09:15:00 | 29857.50 | TARGET_HIT | 0.50 | 10.00% |
