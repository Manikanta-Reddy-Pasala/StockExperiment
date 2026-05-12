# Godfrey Phillips India Ltd. (GODFRYPHLP)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2424.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 40 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 36 |
| PARTIAL | 1 |
| TARGET_HIT | 5 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 37 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 31
- **Target hits / Stop hits / Partials:** 5 / 31 / 1
- **Avg / median % per leg:** -0.55% / -1.62%
- **Sum % (uncompounded):** -20.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 4 | 14.8% | 4 | 23 | 0 | -0.11% | -2.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 27 | 4 | 14.8% | 4 | 23 | 0 | -0.11% | -2.9% |
| SELL (all) | 10 | 2 | 20.0% | 1 | 8 | 1 | -1.76% | -17.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 2 | 20.0% | 1 | 8 | 1 | -1.76% | -17.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 37 | 6 | 16.2% | 5 | 31 | 1 | -0.55% | -20.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 09:15:00 | 715.52 | 573.35 | 573.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-10 09:15:00 | 732.83 | 624.52 | 602.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 09:15:00 | 683.42 | 693.20 | 661.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-12 09:45:00 | 681.43 | 693.20 | 661.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 15:15:00 | 707.55 | 720.77 | 697.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-25 09:30:00 | 709.73 | 720.68 | 698.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-25 12:15:00 | 678.98 | 719.84 | 697.92 | SL hit (close<static) qty=1.00 sl=695.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 12:15:00 | 689.40 | 700.08 | 700.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 683.37 | 699.77 | 699.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 701.78 | 699.63 | 699.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 14:15:00 | 701.78 | 699.63 | 699.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 14:15:00 | 701.78 | 699.63 | 699.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 15:00:00 | 701.78 | 699.63 | 699.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 15:15:00 | 700.00 | 699.64 | 699.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:15:00 | 700.00 | 699.64 | 699.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 699.77 | 699.64 | 699.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 12:15:00 | 695.33 | 699.63 | 699.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-01 14:45:00 | 695.83 | 699.70 | 699.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-01 15:15:00 | 695.02 | 699.70 | 699.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-02 09:30:00 | 695.58 | 699.61 | 699.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 12:15:00 | 716.95 | 699.66 | 699.84 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-01-02 12:15:00 | 716.95 | 699.66 | 699.84 | SL hit (close>static) qty=1.00 sl=705.90 alert=retest2 |

### Cycle 3 — BUY (started 2024-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 15:15:00 | 718.33 | 700.18 | 700.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-03 09:15:00 | 720.78 | 700.39 | 700.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 12:15:00 | 707.72 | 715.05 | 708.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 12:15:00 | 707.72 | 715.05 | 708.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 707.72 | 715.05 | 708.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:00:00 | 707.72 | 715.05 | 708.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 13:15:00 | 706.30 | 714.96 | 708.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:30:00 | 703.97 | 714.96 | 708.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 14:15:00 | 710.30 | 714.91 | 708.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 09:45:00 | 715.28 | 714.89 | 708.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 11:00:00 | 714.68 | 714.88 | 708.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 13:45:00 | 713.03 | 714.77 | 708.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 15:15:00 | 714.00 | 714.69 | 708.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 15:15:00 | 714.00 | 714.68 | 708.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 09:15:00 | 704.60 | 714.68 | 708.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 701.50 | 714.55 | 708.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 701.50 | 714.55 | 708.90 | SL hit (close<static) qty=1.00 sl=702.67 alert=retest2 |

### Cycle 4 — SELL (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 11:15:00 | 1922.65 | 2084.40 | 2084.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-03 09:15:00 | 1897.22 | 2076.20 | 2080.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 10:15:00 | 2013.00 | 2012.86 | 2044.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-11 10:45:00 | 2020.00 | 2012.86 | 2044.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 2020.35 | 2010.24 | 2039.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:00:00 | 2016.67 | 2010.31 | 2039.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 14:15:00 | 1915.84 | 2002.36 | 2033.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-20 14:15:00 | 1815.00 | 1981.74 | 2020.40 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 2199.75 | 1768.57 | 1767.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 10:15:00 | 2238.00 | 1922.28 | 1861.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 2708.67 | 2709.06 | 2490.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 09:30:00 | 2700.17 | 2709.06 | 2490.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 2629.67 | 2743.09 | 2612.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 2617.17 | 2743.09 | 2612.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 3305.00 | 3437.78 | 3302.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 3298.00 | 3437.78 | 3302.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 3326.00 | 3436.67 | 3302.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 13:00:00 | 3366.00 | 3420.71 | 3300.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 13:30:00 | 3336.90 | 3418.14 | 3307.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 15:15:00 | 3295.00 | 3410.27 | 3322.26 | SL hit (close<static) qty=1.00 sl=3300.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 12:15:00 | 3103.80 | 3283.85 | 3284.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 3028.00 | 3276.20 | 3280.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 2206.60 | 2200.20 | 2441.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 11:45:00 | 2212.20 | 2200.20 | 2441.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 2386.80 | 2170.85 | 2365.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:00:00 | 2386.80 | 2170.85 | 2365.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 2382.50 | 2172.96 | 2365.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:30:00 | 2388.30 | 2172.96 | 2365.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 2478.90 | 2176.00 | 2366.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 2478.90 | 2176.00 | 2366.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 2112.30 | 2024.65 | 2128.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:45:00 | 2118.40 | 2024.65 | 2128.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 11:15:00 | 2130.70 | 2026.48 | 2128.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 12:00:00 | 2130.70 | 2026.48 | 2128.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 12:15:00 | 2106.30 | 2027.27 | 2128.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 13:15:00 | 2090.60 | 2027.27 | 2128.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:30:00 | 2094.70 | 2029.85 | 2127.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 2230.90 | 2037.18 | 2127.76 | SL hit (close>static) qty=1.00 sl=2146.40 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-10-25 09:30:00 | 709.73 | 2023-10-25 12:15:00 | 678.98 | STOP_HIT | 1.00 | -4.33% |
| BUY | retest2 | 2023-10-26 13:45:00 | 709.92 | 2023-11-01 09:15:00 | 780.91 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-27 09:15:00 | 714.00 | 2023-11-01 14:15:00 | 785.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-08 10:30:00 | 711.33 | 2023-11-10 14:15:00 | 692.22 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2023-12-06 10:15:00 | 709.57 | 2023-12-08 10:15:00 | 698.33 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2023-12-06 10:45:00 | 711.67 | 2023-12-08 10:15:00 | 698.33 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2023-12-06 12:15:00 | 708.80 | 2023-12-08 10:15:00 | 698.33 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2023-12-13 10:00:00 | 708.92 | 2023-12-14 12:15:00 | 695.73 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2023-12-14 09:15:00 | 701.90 | 2023-12-14 12:15:00 | 695.73 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2023-12-14 10:45:00 | 703.38 | 2023-12-14 12:15:00 | 695.73 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2023-12-14 11:30:00 | 702.95 | 2023-12-14 12:15:00 | 695.73 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2023-12-14 12:00:00 | 701.83 | 2023-12-14 12:15:00 | 695.73 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2023-12-15 11:45:00 | 699.62 | 2023-12-18 09:15:00 | 693.10 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-12-15 14:00:00 | 698.97 | 2023-12-18 09:15:00 | 693.10 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2023-12-22 12:15:00 | 695.33 | 2024-01-02 12:15:00 | 716.95 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2024-01-01 14:45:00 | 695.83 | 2024-01-02 12:15:00 | 716.95 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2024-01-01 15:15:00 | 695.02 | 2024-01-02 12:15:00 | 716.95 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2024-01-02 09:30:00 | 695.58 | 2024-01-02 12:15:00 | 716.95 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2024-01-17 09:45:00 | 715.28 | 2024-01-18 09:15:00 | 701.50 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-01-17 11:00:00 | 714.68 | 2024-01-18 09:15:00 | 701.50 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-01-17 13:45:00 | 713.03 | 2024-01-18 09:15:00 | 701.50 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-01-17 15:15:00 | 714.00 | 2024-01-18 09:15:00 | 701.50 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-01-18 14:45:00 | 710.00 | 2024-01-31 09:15:00 | 781.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-19 09:15:00 | 715.43 | 2024-01-31 09:15:00 | 786.97 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-16 11:00:00 | 2016.67 | 2024-12-18 14:15:00 | 1915.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 11:00:00 | 2016.67 | 2024-12-20 14:15:00 | 1815.00 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-09-29 13:00:00 | 3366.00 | 2025-10-09 15:15:00 | 3295.00 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-10-01 13:30:00 | 3336.90 | 2025-10-09 15:15:00 | 3295.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-10-10 09:30:00 | 3343.00 | 2025-10-13 09:15:00 | 3292.30 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-10-10 11:00:00 | 3330.50 | 2025-10-13 09:15:00 | 3292.30 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-10-14 09:15:00 | 3349.80 | 2025-10-14 10:15:00 | 3305.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-10-15 09:15:00 | 3368.10 | 2025-10-23 09:15:00 | 3231.80 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2025-10-17 09:15:00 | 3398.10 | 2025-10-23 09:15:00 | 3231.80 | STOP_HIT | 1.00 | -4.89% |
| SELL | retest2 | 2026-04-15 13:15:00 | 2090.60 | 2026-04-17 09:15:00 | 2230.90 | STOP_HIT | 1.00 | -6.71% |
| SELL | retest2 | 2026-04-16 09:30:00 | 2094.70 | 2026-04-17 09:15:00 | 2230.90 | STOP_HIT | 1.00 | -6.50% |
| SELL | retest2 | 2026-04-22 09:15:00 | 2102.40 | 2026-04-22 09:15:00 | 2163.20 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2026-04-24 13:30:00 | 2088.40 | 2026-04-29 10:15:00 | 2174.90 | STOP_HIT | 1.00 | -4.14% |
