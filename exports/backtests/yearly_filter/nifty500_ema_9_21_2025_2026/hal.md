# Hindustan Aeronautics Ltd. (HAL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 4790.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 85 |
| ALERT1 | 57 |
| ALERT2 | 57 |
| ALERT2_SKIP | 29 |
| ALERT3 | 159 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 57 |
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 57 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 65 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 47
- **Target hits / Stop hits / Partials:** 2 / 56 / 7
- **Avg / median % per leg:** 0.34% / -0.72%
- **Sum % (uncompounded):** 22.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 4 | 16.7% | 2 | 22 | 0 | -0.08% | -2.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 4 | 16.7% | 2 | 22 | 0 | -0.08% | -2.0% |
| SELL (all) | 41 | 14 | 34.1% | 0 | 34 | 7 | 0.59% | 24.4% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.23% | 8.5% |
| SELL @ 3rd Alert (retest2) | 39 | 12 | 30.8% | 0 | 33 | 6 | 0.41% | 15.9% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.23% | 8.5% |
| retest2 (combined) | 63 | 16 | 25.4% | 2 | 55 | 6 | 0.22% | 13.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 10:15:00 | 4598.00 | 4504.48 | 4491.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 11:15:00 | 4633.20 | 4530.23 | 4504.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 09:15:00 | 5025.80 | 5047.99 | 4930.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 09:45:00 | 5011.50 | 5047.99 | 4930.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 4920.00 | 5011.52 | 4969.72 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 4879.80 | 4948.03 | 4949.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 4855.20 | 4929.47 | 4941.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 4940.00 | 4919.18 | 4933.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 4940.00 | 4919.18 | 4933.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 4940.00 | 4919.18 | 4933.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 4940.00 | 4919.18 | 4933.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 4956.00 | 4926.54 | 4935.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 4956.00 | 4926.54 | 4935.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 4931.40 | 4927.51 | 4935.37 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 13:15:00 | 4986.80 | 4947.77 | 4943.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 14:15:00 | 5000.00 | 4958.22 | 4948.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 09:15:00 | 4976.00 | 5011.91 | 4991.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 4976.00 | 5011.91 | 4991.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 4976.00 | 5011.91 | 4991.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:45:00 | 4978.50 | 5011.91 | 4991.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 4998.00 | 5009.13 | 4991.79 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 09:15:00 | 4956.70 | 4980.97 | 4984.15 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 11:15:00 | 5017.00 | 4989.53 | 4987.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 5087.90 | 5019.78 | 5004.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 5025.00 | 5030.50 | 5019.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 5025.00 | 5030.50 | 5019.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 5025.00 | 5030.50 | 5019.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:45:00 | 5011.40 | 5030.50 | 5019.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 5028.10 | 5030.02 | 5020.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:45:00 | 5018.50 | 5030.02 | 5020.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 5025.00 | 5032.00 | 5023.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:45:00 | 5028.00 | 5032.00 | 5023.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 5019.40 | 5029.48 | 5023.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:30:00 | 5019.00 | 5029.48 | 5023.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 5020.50 | 5027.69 | 5023.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:15:00 | 5017.90 | 5027.69 | 5023.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 10:15:00 | 4985.80 | 5015.25 | 5017.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 4976.60 | 5000.63 | 5010.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 15:15:00 | 5005.00 | 5000.92 | 5008.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 15:15:00 | 5005.00 | 5000.92 | 5008.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 5005.00 | 5000.92 | 5008.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 5044.00 | 5000.92 | 5008.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 5029.70 | 5006.68 | 5010.40 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 10:15:00 | 5040.80 | 5013.50 | 5013.16 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 4991.90 | 5010.83 | 5012.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 4977.90 | 5004.25 | 5009.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 11:15:00 | 5011.00 | 4995.45 | 5002.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 11:15:00 | 5011.00 | 4995.45 | 5002.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 5011.00 | 4995.45 | 5002.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:45:00 | 5009.00 | 4995.45 | 5002.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 5029.00 | 5002.16 | 5004.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:00:00 | 5029.00 | 5002.16 | 5004.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 13:15:00 | 5023.00 | 5006.33 | 5006.22 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 12:15:00 | 4977.00 | 5005.54 | 5007.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 4926.30 | 4982.13 | 4995.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 4957.00 | 4956.88 | 4977.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 13:30:00 | 4948.80 | 4956.88 | 4977.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 5027.90 | 4972.34 | 4979.19 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 5017.40 | 4987.87 | 4985.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 12:15:00 | 5032.70 | 4996.84 | 4989.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 5008.50 | 5025.39 | 5007.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 5008.50 | 5025.39 | 5007.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 5008.50 | 5025.39 | 5007.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:45:00 | 5010.00 | 5025.39 | 5007.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 4994.10 | 5019.14 | 5006.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:45:00 | 4999.50 | 5019.14 | 5006.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 4981.40 | 5011.59 | 5004.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:00:00 | 4981.40 | 5011.59 | 5004.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 4994.80 | 5002.69 | 5001.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 4994.80 | 5002.69 | 5001.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 15:15:00 | 4982.00 | 4998.55 | 4999.82 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 11:15:00 | 5018.50 | 5002.76 | 5001.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 5115.00 | 5028.39 | 5014.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 5030.00 | 5080.79 | 5064.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 5030.00 | 5080.79 | 5064.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 5030.00 | 5080.79 | 5064.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 5042.10 | 5080.79 | 5064.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 5052.20 | 5075.07 | 5063.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:15:00 | 5055.10 | 5067.86 | 5061.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:45:00 | 5054.70 | 5066.63 | 5061.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 11:15:00 | 5041.50 | 5056.72 | 5057.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 5041.50 | 5056.72 | 5057.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 4954.90 | 5033.04 | 5046.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 09:15:00 | 5059.00 | 5015.03 | 5032.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 5059.00 | 5015.03 | 5032.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 5059.00 | 5015.03 | 5032.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:00:00 | 5059.00 | 5015.03 | 5032.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 5056.00 | 5023.23 | 5035.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 11:00:00 | 5056.00 | 5023.23 | 5035.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 4996.30 | 5019.98 | 5031.62 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 5059.30 | 5034.84 | 5033.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 5097.80 | 5057.16 | 5045.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 12:15:00 | 5063.70 | 5071.85 | 5056.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 12:15:00 | 5063.70 | 5071.85 | 5056.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 5063.70 | 5071.85 | 5056.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:00:00 | 5063.70 | 5071.85 | 5056.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 5060.00 | 5069.48 | 5056.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:45:00 | 5060.00 | 5069.48 | 5056.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 5052.60 | 5066.10 | 5056.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 15:00:00 | 5052.60 | 5066.10 | 5056.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 5043.30 | 5061.54 | 5055.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 5038.00 | 5061.54 | 5055.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 5030.60 | 5055.41 | 5053.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:45:00 | 5024.90 | 5055.41 | 5053.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 5028.00 | 5049.93 | 5051.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 12:15:00 | 4999.90 | 5039.92 | 5046.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 4921.90 | 4921.75 | 4961.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 12:00:00 | 4921.90 | 4921.75 | 4961.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 4975.00 | 4932.01 | 4955.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 4975.00 | 4932.01 | 4955.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 4960.00 | 4937.61 | 4956.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 5011.20 | 4937.61 | 4956.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 5004.90 | 4951.06 | 4960.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:30:00 | 5006.80 | 4951.06 | 4960.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 5042.00 | 4980.43 | 4973.10 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 10:15:00 | 4908.00 | 4975.85 | 4978.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 15:15:00 | 4883.00 | 4920.53 | 4947.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 10:15:00 | 4823.50 | 4822.54 | 4866.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:30:00 | 4821.00 | 4822.54 | 4866.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 4879.30 | 4829.00 | 4849.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:45:00 | 4875.90 | 4829.00 | 4849.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 4877.00 | 4838.60 | 4851.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 11:45:00 | 4864.00 | 4840.58 | 4851.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 13:00:00 | 4865.00 | 4845.47 | 4852.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 14:15:00 | 4866.00 | 4850.43 | 4854.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 14:15:00 | 4896.70 | 4859.69 | 4858.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 14:15:00 | 4896.70 | 4859.69 | 4858.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 11:15:00 | 4904.00 | 4881.20 | 4869.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 4875.10 | 4879.98 | 4870.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:00:00 | 4875.10 | 4879.98 | 4870.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 4852.40 | 4874.47 | 4868.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:45:00 | 4848.10 | 4874.47 | 4868.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 4866.40 | 4872.85 | 4868.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 15:15:00 | 4877.90 | 4872.85 | 4868.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 14:00:00 | 4885.30 | 4901.64 | 4895.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 10:15:00 | 4942.50 | 4994.91 | 4996.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 4942.50 | 4994.91 | 4996.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 12:15:00 | 4918.00 | 4971.26 | 4985.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 4892.00 | 4884.87 | 4915.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:00:00 | 4892.00 | 4884.87 | 4915.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 4902.10 | 4888.06 | 4909.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:00:00 | 4902.10 | 4888.06 | 4909.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 4923.30 | 4894.96 | 4905.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:00:00 | 4923.30 | 4894.96 | 4905.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 4908.00 | 4897.57 | 4906.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:15:00 | 4897.50 | 4897.57 | 4906.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 4900.00 | 4898.05 | 4905.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 13:30:00 | 4887.30 | 4898.98 | 4904.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 4881.50 | 4898.50 | 4903.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 09:15:00 | 4642.94 | 4695.89 | 4758.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 09:15:00 | 4637.43 | 4695.89 | 4758.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-21 10:15:00 | 4699.60 | 4696.63 | 4752.85 | SL hit (close>ema200) qty=0.50 sl=4696.63 alert=retest2 |

### Cycle 21 — BUY (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 14:15:00 | 4761.90 | 4758.67 | 4758.52 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 4691.00 | 4745.00 | 4752.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 12:15:00 | 4673.50 | 4697.29 | 4716.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 4487.90 | 4481.26 | 4538.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 12:45:00 | 4478.50 | 4481.26 | 4538.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 4514.60 | 4497.53 | 4528.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 4503.80 | 4497.53 | 4528.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 10:00:00 | 4505.30 | 4526.15 | 4531.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 4569.10 | 4542.03 | 4538.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 4569.10 | 4542.03 | 4538.34 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 4481.50 | 4529.26 | 4533.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 4436.90 | 4486.59 | 4508.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 4523.10 | 4485.00 | 4503.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 4523.10 | 4485.00 | 4503.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 4523.10 | 4485.00 | 4503.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 4523.10 | 4485.00 | 4503.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 4499.40 | 4487.88 | 4503.41 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 4535.50 | 4513.81 | 4512.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 13:15:00 | 4563.00 | 4530.90 | 4522.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 4522.10 | 4543.26 | 4531.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 4522.10 | 4543.26 | 4531.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 4522.10 | 4543.26 | 4531.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 4522.10 | 4543.26 | 4531.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 4532.00 | 4541.01 | 4531.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 13:15:00 | 4561.00 | 4541.56 | 4533.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 12:15:00 | 4495.50 | 4526.03 | 4529.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 12:15:00 | 4495.50 | 4526.03 | 4529.50 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 4549.00 | 4531.15 | 4530.80 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 09:15:00 | 4490.90 | 4523.10 | 4527.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 10:15:00 | 4476.90 | 4513.86 | 4522.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 4479.30 | 4472.98 | 4494.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 10:15:00 | 4511.00 | 4472.98 | 4494.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 4526.60 | 4483.70 | 4497.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:45:00 | 4529.20 | 4483.70 | 4497.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 4483.40 | 4483.64 | 4496.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 12:15:00 | 4473.10 | 4483.64 | 4496.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:30:00 | 4433.00 | 4386.02 | 4427.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 11:15:00 | 4495.20 | 4449.00 | 4447.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 11:15:00 | 4495.20 | 4449.00 | 4447.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 13:15:00 | 4542.40 | 4475.97 | 4460.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 4492.00 | 4493.57 | 4473.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 09:45:00 | 4493.10 | 4493.57 | 4473.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 4494.40 | 4492.36 | 4476.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 12:45:00 | 4510.70 | 4504.87 | 4483.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 09:15:00 | 4454.00 | 4513.01 | 4510.35 | SL hit (close<static) qty=1.00 sl=4472.40 alert=retest2 |

### Cycle 30 — SELL (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 10:15:00 | 4464.10 | 4503.23 | 4506.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 13:15:00 | 4436.10 | 4475.58 | 4491.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 4532.70 | 4480.84 | 4489.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 4532.70 | 4480.84 | 4489.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 4532.70 | 4480.84 | 4489.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 10:30:00 | 4511.00 | 4488.48 | 4492.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 4521.70 | 4490.78 | 4489.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 09:15:00 | 4521.70 | 4490.78 | 4489.70 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 4480.00 | 4489.78 | 4491.08 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 4501.40 | 4493.42 | 4492.59 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 13:15:00 | 4480.00 | 4491.35 | 4491.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 4457.10 | 4484.50 | 4488.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 10:15:00 | 4417.30 | 4407.14 | 4434.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 10:45:00 | 4422.90 | 4407.14 | 4434.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 4392.10 | 4379.88 | 4401.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 4380.60 | 4379.88 | 4401.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 4424.50 | 4395.44 | 4393.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 4424.50 | 4395.44 | 4393.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 15:15:00 | 4435.00 | 4403.35 | 4396.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 4441.60 | 4449.75 | 4434.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 4441.60 | 4449.75 | 4434.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 4443.00 | 4448.40 | 4435.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:00:00 | 4443.00 | 4448.40 | 4435.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 4449.00 | 4448.52 | 4436.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:45:00 | 4441.60 | 4448.52 | 4436.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 4451.40 | 4449.39 | 4440.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 4438.00 | 4449.39 | 4440.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 4447.00 | 4451.11 | 4443.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:45:00 | 4446.30 | 4451.11 | 4443.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 4431.90 | 4447.27 | 4442.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 4431.90 | 4447.27 | 4442.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 4431.80 | 4444.18 | 4441.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 4455.60 | 4444.18 | 4441.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 4433.90 | 4440.57 | 4440.11 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 4396.60 | 4431.78 | 4436.15 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 4450.80 | 4435.65 | 4434.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 4464.10 | 4446.07 | 4440.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 4848.50 | 4872.00 | 4831.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 10:00:00 | 4848.50 | 4872.00 | 4831.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 4867.00 | 4891.70 | 4873.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:00:00 | 4867.00 | 4891.70 | 4873.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 4855.00 | 4884.36 | 4871.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:00:00 | 4855.00 | 4884.36 | 4871.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 4864.80 | 4880.45 | 4870.78 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 4811.00 | 4857.08 | 4861.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 4795.00 | 4837.45 | 4851.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 4770.70 | 4754.21 | 4781.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 4770.70 | 4754.21 | 4781.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 4770.70 | 4754.21 | 4781.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:30:00 | 4766.70 | 4754.21 | 4781.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 4792.00 | 4761.77 | 4782.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:00:00 | 4792.00 | 4761.77 | 4782.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 4776.00 | 4764.62 | 4782.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:00:00 | 4767.70 | 4767.23 | 4780.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:00:00 | 4766.00 | 4766.98 | 4779.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:15:00 | 4753.70 | 4781.63 | 4782.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 4815.00 | 4771.81 | 4775.84 | SL hit (close>static) qty=1.00 sl=4805.10 alert=retest2 |

### Cycle 39 — BUY (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 10:15:00 | 4789.00 | 4765.25 | 4763.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 4844.20 | 4799.65 | 4782.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 4842.00 | 4850.93 | 4825.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:45:00 | 4842.10 | 4850.93 | 4825.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 4826.90 | 4841.90 | 4833.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 4826.90 | 4841.90 | 4833.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 4815.70 | 4836.66 | 4831.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:30:00 | 4816.00 | 4836.66 | 4831.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 4835.70 | 4836.47 | 4831.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:15:00 | 4843.90 | 4836.47 | 4831.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:45:00 | 4841.10 | 4838.34 | 4833.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 15:15:00 | 4845.00 | 4838.34 | 4833.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 4810.00 | 4833.74 | 4832.46 | SL hit (close<static) qty=1.00 sl=4812.40 alert=retest2 |

### Cycle 40 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 4772.90 | 4821.57 | 4827.05 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 15:15:00 | 4852.80 | 4819.65 | 4816.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 4879.90 | 4831.70 | 4822.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 14:15:00 | 4831.00 | 4848.56 | 4836.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 14:15:00 | 4831.00 | 4848.56 | 4836.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 4831.00 | 4848.56 | 4836.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 4831.00 | 4848.56 | 4836.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 4832.00 | 4845.25 | 4836.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 4769.90 | 4845.25 | 4836.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 4759.30 | 4816.82 | 4824.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 4734.30 | 4800.31 | 4815.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 4766.20 | 4765.65 | 4789.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 4766.20 | 4765.65 | 4789.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 4766.20 | 4765.65 | 4789.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:15:00 | 4739.20 | 4762.40 | 4786.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 13:00:00 | 4739.90 | 4755.99 | 4778.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 15:15:00 | 4742.00 | 4753.50 | 4773.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 10:15:00 | 4794.00 | 4762.97 | 4773.01 | SL hit (close>static) qty=1.00 sl=4793.90 alert=retest2 |

### Cycle 43 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 4867.80 | 4792.54 | 4785.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 4972.00 | 4872.58 | 4839.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 14:15:00 | 4878.00 | 4898.34 | 4868.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 15:00:00 | 4878.00 | 4898.34 | 4868.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 4873.80 | 4893.43 | 4869.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 4892.10 | 4893.43 | 4869.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:30:00 | 4885.00 | 4889.98 | 4871.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 13:15:00 | 4855.60 | 4877.54 | 4870.08 | SL hit (close<static) qty=1.00 sl=4865.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 14:15:00 | 4852.00 | 4866.47 | 4866.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 12:15:00 | 4847.20 | 4861.29 | 4864.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 11:15:00 | 4839.20 | 4835.66 | 4847.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 11:15:00 | 4839.20 | 4835.66 | 4847.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 4839.20 | 4835.66 | 4847.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:45:00 | 4845.20 | 4835.66 | 4847.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 4682.00 | 4668.07 | 4694.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:45:00 | 4679.30 | 4668.07 | 4694.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 4680.00 | 4670.45 | 4693.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:45:00 | 4677.00 | 4670.45 | 4693.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 4689.40 | 4672.91 | 4690.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:00:00 | 4689.40 | 4672.91 | 4690.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 4697.20 | 4677.76 | 4690.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 11:30:00 | 4665.40 | 4675.81 | 4685.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 13:30:00 | 4667.50 | 4673.70 | 4682.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 4693.00 | 4687.69 | 4687.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 4693.00 | 4687.69 | 4687.27 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 4654.00 | 4682.73 | 4685.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 4629.30 | 4672.04 | 4680.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 13:15:00 | 4597.10 | 4595.01 | 4623.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:45:00 | 4593.00 | 4595.01 | 4623.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 4622.30 | 4600.47 | 4623.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:45:00 | 4623.30 | 4600.47 | 4623.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 4625.00 | 4605.37 | 4623.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 4689.00 | 4605.37 | 4623.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 4750.00 | 4634.30 | 4635.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:00:00 | 4750.00 | 4634.30 | 4635.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 4772.00 | 4661.84 | 4647.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 12:15:00 | 4780.00 | 4702.30 | 4669.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 14:15:00 | 4751.50 | 4854.40 | 4817.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 14:15:00 | 4751.50 | 4854.40 | 4817.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 4751.50 | 4854.40 | 4817.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:30:00 | 4734.00 | 4854.40 | 4817.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 4730.10 | 4829.54 | 4809.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 4732.30 | 4829.54 | 4809.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 4742.00 | 4795.50 | 4796.78 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 15:15:00 | 4814.00 | 4770.52 | 4764.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 09:15:00 | 4829.80 | 4782.37 | 4770.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 4775.60 | 4802.33 | 4790.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 4775.60 | 4802.33 | 4790.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 4775.60 | 4802.33 | 4790.61 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 12:15:00 | 4759.10 | 4781.05 | 4782.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 15:15:00 | 4740.40 | 4763.92 | 4773.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 4470.00 | 4455.68 | 4504.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 09:30:00 | 4472.00 | 4455.68 | 4504.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 4507.60 | 4468.20 | 4502.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:00:00 | 4507.60 | 4468.20 | 4502.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 4501.20 | 4474.80 | 4502.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:30:00 | 4514.80 | 4474.80 | 4502.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 4504.00 | 4480.64 | 4502.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:45:00 | 4505.60 | 4480.64 | 4502.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 4517.40 | 4487.99 | 4503.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:00:00 | 4517.40 | 4487.99 | 4503.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 4512.00 | 4492.79 | 4504.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 4508.00 | 4492.79 | 4504.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 4523.00 | 4492.42 | 4496.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 4523.00 | 4492.42 | 4496.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 10:15:00 | 4527.80 | 4499.50 | 4499.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 14:15:00 | 4540.10 | 4518.21 | 4509.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 4522.00 | 4522.46 | 4512.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 10:00:00 | 4522.00 | 4522.46 | 4512.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 4505.50 | 4519.07 | 4512.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 4505.50 | 4519.07 | 4512.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 4515.70 | 4518.39 | 4512.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 12:15:00 | 4525.20 | 4518.39 | 4512.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 11:15:00 | 4498.70 | 4516.93 | 4515.93 | SL hit (close<static) qty=1.00 sl=4504.60 alert=retest2 |

### Cycle 52 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 4493.00 | 4512.14 | 4513.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 4476.90 | 4502.63 | 4508.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 4462.60 | 4456.96 | 4477.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 09:30:00 | 4456.00 | 4456.96 | 4477.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 4501.80 | 4467.86 | 4478.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:00:00 | 4501.80 | 4467.86 | 4478.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 4490.30 | 4472.34 | 4479.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:30:00 | 4496.00 | 4472.34 | 4479.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 4491.00 | 4477.83 | 4480.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 4491.00 | 4477.83 | 4480.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 15:15:00 | 4522.00 | 4486.66 | 4484.72 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 09:15:00 | 4466.40 | 4482.61 | 4483.06 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 4489.10 | 4483.91 | 4483.61 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 11:15:00 | 4466.00 | 4480.33 | 4482.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 13:15:00 | 4464.00 | 4476.62 | 4480.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 4309.30 | 4297.72 | 4346.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 15:00:00 | 4309.30 | 4297.72 | 4346.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 4303.20 | 4294.33 | 4315.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 4296.00 | 4294.33 | 4315.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 4320.20 | 4300.04 | 4314.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 4320.20 | 4300.04 | 4314.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 4328.20 | 4305.67 | 4315.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:30:00 | 4331.50 | 4305.67 | 4315.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 4314.30 | 4311.24 | 4316.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 4347.90 | 4311.24 | 4316.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 4352.30 | 4319.45 | 4319.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 4346.00 | 4319.45 | 4319.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 4329.60 | 4321.48 | 4320.39 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 12:15:00 | 4282.60 | 4313.98 | 4317.18 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 4335.00 | 4319.12 | 4317.02 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 4257.40 | 4308.31 | 4312.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 4249.00 | 4285.75 | 4300.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 4252.40 | 4237.30 | 4255.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 4252.40 | 4237.30 | 4255.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 4252.40 | 4237.30 | 4255.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:45:00 | 4253.10 | 4237.30 | 4255.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 4256.90 | 4241.22 | 4255.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 4265.70 | 4241.22 | 4255.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 4235.70 | 4240.12 | 4253.53 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 4288.10 | 4263.69 | 4261.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 4296.30 | 4274.45 | 4266.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 14:15:00 | 4422.00 | 4428.01 | 4399.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:30:00 | 4417.50 | 4428.01 | 4399.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 4413.20 | 4432.96 | 4417.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 4422.60 | 4432.96 | 4417.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 4412.90 | 4428.95 | 4417.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 4439.10 | 4428.95 | 4417.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 13:15:00 | 4394.10 | 4423.97 | 4419.94 | SL hit (close<static) qty=1.00 sl=4401.10 alert=retest2 |

### Cycle 62 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 4376.50 | 4414.47 | 4415.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 4366.70 | 4398.44 | 4408.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 4373.80 | 4350.28 | 4372.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 4373.80 | 4350.28 | 4372.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 4373.80 | 4350.28 | 4372.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 4360.70 | 4350.28 | 4372.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 4391.30 | 4358.49 | 4374.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 4388.50 | 4358.49 | 4374.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 4391.70 | 4365.13 | 4375.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:45:00 | 4391.60 | 4365.13 | 4375.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 4380.00 | 4375.07 | 4377.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:00:00 | 4380.00 | 4375.07 | 4377.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 4384.60 | 4376.98 | 4378.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:00:00 | 4384.60 | 4376.98 | 4378.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 14:15:00 | 4392.60 | 4381.09 | 4380.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 15:15:00 | 4400.00 | 4384.87 | 4381.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 13:15:00 | 4388.90 | 4395.39 | 4389.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 13:15:00 | 4388.90 | 4395.39 | 4389.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 4388.90 | 4395.39 | 4389.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:00:00 | 4388.90 | 4395.39 | 4389.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 4418.30 | 4399.97 | 4392.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:30:00 | 4391.70 | 4399.97 | 4392.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 4515.00 | 4504.15 | 4479.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:45:00 | 4532.50 | 4513.29 | 4496.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 12:00:00 | 4527.00 | 4527.65 | 4509.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 15:15:00 | 4469.90 | 4503.95 | 4503.27 | SL hit (close<static) qty=1.00 sl=4478.60 alert=retest2 |

### Cycle 64 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 4494.00 | 4501.96 | 4502.43 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 10:15:00 | 4515.50 | 4504.67 | 4503.62 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 12:15:00 | 4489.90 | 4500.99 | 4502.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 4463.50 | 4493.49 | 4498.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 4484.90 | 4476.69 | 4485.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 4484.90 | 4476.69 | 4485.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 4484.90 | 4476.69 | 4485.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:45:00 | 4490.10 | 4476.69 | 4485.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 4503.60 | 4482.07 | 4487.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 4503.60 | 4482.07 | 4487.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 4515.20 | 4488.70 | 4489.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 4524.70 | 4488.70 | 4489.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 4525.20 | 4496.00 | 4493.10 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 11:15:00 | 4450.70 | 4484.19 | 4488.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 12:15:00 | 4434.60 | 4474.27 | 4483.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 4463.10 | 4459.85 | 4472.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 4463.10 | 4459.85 | 4472.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 4463.10 | 4459.85 | 4472.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 4453.80 | 4459.85 | 4472.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 4482.60 | 4464.40 | 4473.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:45:00 | 4484.80 | 4464.40 | 4473.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 4481.30 | 4467.78 | 4473.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:45:00 | 4494.00 | 4467.78 | 4473.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 4467.40 | 4471.96 | 4474.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:45:00 | 4488.50 | 4471.96 | 4474.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 4430.10 | 4462.15 | 4469.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:00:00 | 4408.20 | 4439.30 | 4456.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 12:15:00 | 4483.10 | 4454.95 | 4454.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 12:15:00 | 4483.10 | 4454.95 | 4454.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 14:15:00 | 4499.00 | 4468.81 | 4460.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 4448.70 | 4470.10 | 4463.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 4448.70 | 4470.10 | 4463.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 4448.70 | 4470.10 | 4463.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 4439.00 | 4470.10 | 4463.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 4445.70 | 4465.22 | 4461.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:30:00 | 4451.00 | 4465.22 | 4461.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 4431.40 | 4458.45 | 4458.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 4402.40 | 4447.24 | 4453.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 4343.00 | 4309.53 | 4355.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 10:00:00 | 4343.00 | 4309.53 | 4355.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 4351.00 | 4322.87 | 4344.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 15:00:00 | 4351.00 | 4322.87 | 4344.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 4366.80 | 4331.65 | 4346.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:15:00 | 4354.00 | 4331.65 | 4346.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 4342.20 | 4336.56 | 4346.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:30:00 | 4347.00 | 4336.56 | 4346.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 4325.30 | 4334.30 | 4344.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 13:15:00 | 4306.50 | 4331.90 | 4342.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 14:30:00 | 4307.10 | 4323.17 | 4336.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 15:00:00 | 4301.70 | 4323.17 | 4336.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 10:30:00 | 4303.90 | 4318.27 | 4330.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 4342.10 | 4317.72 | 4326.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 4342.10 | 4317.72 | 4326.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 4367.70 | 4327.72 | 4329.93 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 4367.70 | 4327.72 | 4329.93 | SL hit (close>static) qty=1.00 sl=4347.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 4447.40 | 4351.66 | 4340.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 4490.30 | 4379.38 | 4354.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 4578.60 | 4591.84 | 4540.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 13:00:00 | 4578.60 | 4591.84 | 4540.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 4485.40 | 4600.90 | 4571.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 4485.40 | 4600.90 | 4571.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 4506.60 | 4582.04 | 4565.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 4421.00 | 4582.04 | 4565.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 4425.10 | 4550.65 | 4552.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 4375.00 | 4515.52 | 4536.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 4348.90 | 4334.30 | 4413.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 4348.90 | 4334.30 | 4413.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 4413.30 | 4352.31 | 4408.29 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 4469.70 | 4436.56 | 4434.50 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 09:15:00 | 4218.50 | 4397.82 | 4417.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 3965.70 | 4204.16 | 4297.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 4068.00 | 4053.15 | 4122.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 15:00:00 | 4068.00 | 4053.15 | 4122.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 4112.00 | 4072.99 | 4110.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 4112.00 | 4072.99 | 4110.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 4125.20 | 4083.43 | 4111.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:00:00 | 4125.20 | 4083.43 | 4111.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 4128.10 | 4092.37 | 4112.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 14:15:00 | 4133.10 | 4092.37 | 4112.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 4151.00 | 4108.51 | 4117.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:15:00 | 4175.00 | 4108.51 | 4117.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 4162.20 | 4129.63 | 4125.78 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 4114.20 | 4130.15 | 4131.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 4101.10 | 4123.08 | 4127.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 13:15:00 | 4171.60 | 4123.25 | 4125.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 13:15:00 | 4171.60 | 4123.25 | 4125.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 4171.60 | 4123.25 | 4125.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:00:00 | 4171.60 | 4123.25 | 4125.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 4148.10 | 4128.22 | 4127.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 09:15:00 | 4196.00 | 4149.87 | 4137.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 15:15:00 | 4204.50 | 4206.59 | 4178.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-16 09:15:00 | 4202.10 | 4206.59 | 4178.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 4190.60 | 4203.39 | 4179.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:45:00 | 4192.80 | 4203.39 | 4179.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 4186.90 | 4200.09 | 4179.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:00:00 | 4186.90 | 4200.09 | 4179.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 4189.90 | 4198.05 | 4180.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:45:00 | 4186.60 | 4198.05 | 4180.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 4197.20 | 4232.31 | 4218.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 13:00:00 | 4243.20 | 4230.35 | 4220.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 11:15:00 | 4177.00 | 4222.29 | 4222.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 4177.00 | 4222.29 | 4222.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 4172.00 | 4212.23 | 4218.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 4200.00 | 4180.89 | 4197.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 10:15:00 | 4200.00 | 4180.89 | 4197.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 4200.00 | 4180.89 | 4197.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:00:00 | 4200.00 | 4180.89 | 4197.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 4207.00 | 4186.11 | 4198.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 4207.00 | 4186.11 | 4198.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 4173.40 | 4183.57 | 4195.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 4121.00 | 4178.52 | 4190.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 14:15:00 | 3914.95 | 3956.86 | 3980.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 10:15:00 | 3949.00 | 3944.65 | 3967.88 | SL hit (close>ema200) qty=0.50 sl=3944.65 alert=retest2 |

### Cycle 79 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 3922.80 | 3917.74 | 3917.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 13:15:00 | 3951.10 | 3924.41 | 3920.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 13:15:00 | 3972.20 | 3975.66 | 3956.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 14:00:00 | 3972.20 | 3975.66 | 3956.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 3943.80 | 3973.01 | 3960.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:45:00 | 3919.70 | 3973.01 | 3960.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 3965.80 | 3971.57 | 3960.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 11:30:00 | 3976.70 | 3971.25 | 3961.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 12:15:00 | 3974.50 | 3971.25 | 3961.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 09:45:00 | 3974.60 | 3996.86 | 3992.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 3974.80 | 3996.86 | 3992.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 3970.20 | 3991.53 | 3990.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:30:00 | 4005.40 | 3996.75 | 3992.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 3898.50 | 3985.66 | 3990.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 3898.50 | 3985.66 | 3990.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 3892.30 | 3923.49 | 3950.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 3910.00 | 3909.01 | 3933.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 3910.00 | 3909.01 | 3933.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 3910.00 | 3909.01 | 3933.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 3926.50 | 3909.01 | 3933.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 3930.60 | 3913.01 | 3931.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 3930.60 | 3913.01 | 3931.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 3939.80 | 3918.37 | 3931.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 3951.10 | 3918.37 | 3931.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 3927.20 | 3920.13 | 3931.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 3914.10 | 3920.13 | 3931.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 3964.70 | 3932.27 | 3935.21 | SL hit (close>static) qty=1.00 sl=3941.50 alert=retest2 |

### Cycle 81 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 3965.00 | 3938.82 | 3937.92 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 3885.00 | 3939.79 | 3941.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 3858.70 | 3907.29 | 3925.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 3868.00 | 3864.65 | 3895.88 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 11:45:00 | 3820.10 | 3852.31 | 3884.69 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 3629.09 | 3780.97 | 3836.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 3687.90 | 3662.84 | 3718.10 | SL hit (close>ema200) qty=0.50 sl=3662.84 alert=retest1 |

### Cycle 83 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 3660.00 | 3614.37 | 3609.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 3681.10 | 3627.72 | 3616.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 3594.60 | 3621.09 | 3614.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 3594.60 | 3621.09 | 3614.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 3594.60 | 3621.09 | 3614.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 3594.60 | 3621.09 | 3614.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 3610.20 | 3618.92 | 3613.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:30:00 | 3594.00 | 3618.92 | 3613.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 11:15:00 | 3632.40 | 3621.61 | 3615.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:15:00 | 3642.00 | 3621.61 | 3615.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 12:15:00 | 3580.00 | 3613.29 | 3612.24 | SL hit (close<static) qty=1.00 sl=3605.00 alert=retest2 |

### Cycle 84 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 4350.00 | 4361.87 | 4362.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 4319.40 | 4353.37 | 4358.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 4305.00 | 4289.59 | 4315.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 4305.00 | 4289.59 | 4315.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 4305.00 | 4289.59 | 4315.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 4339.00 | 4289.59 | 4315.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 4306.20 | 4292.91 | 4314.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 4317.30 | 4292.91 | 4314.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 4313.50 | 4298.84 | 4313.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:45:00 | 4307.90 | 4298.84 | 4313.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 4308.70 | 4300.81 | 4313.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:30:00 | 4305.00 | 4302.67 | 4312.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 4322.90 | 4307.89 | 4313.46 | SL hit (close>static) qty=1.00 sl=4318.00 alert=retest2 |

### Cycle 85 — BUY (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 14:15:00 | 4339.00 | 4316.37 | 4315.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 4533.30 | 4375.02 | 4350.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 13:15:00 | 4620.40 | 4623.14 | 4570.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 14:00:00 | 4620.40 | 4623.14 | 4570.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 09:15:00 | 4465.60 | 2025-05-13 09:15:00 | 4582.80 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-06-12 09:15:00 | 5055.10 | 2025-06-12 11:15:00 | 5041.50 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-06-12 09:45:00 | 5054.70 | 2025-06-12 11:15:00 | 5041.50 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-06-27 11:45:00 | 4864.00 | 2025-06-27 14:15:00 | 4896.70 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-06-27 13:00:00 | 4865.00 | 2025-06-27 14:15:00 | 4896.70 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-06-27 14:15:00 | 4866.00 | 2025-06-27 14:15:00 | 4896.70 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-06-30 15:15:00 | 4877.90 | 2025-07-10 10:15:00 | 4942.50 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest2 | 2025-07-02 14:00:00 | 4885.30 | 2025-07-10 10:15:00 | 4942.50 | STOP_HIT | 1.00 | 1.17% |
| SELL | retest2 | 2025-07-15 13:30:00 | 4887.30 | 2025-07-21 09:15:00 | 4642.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-16 09:15:00 | 4881.50 | 2025-07-21 09:15:00 | 4637.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-15 13:30:00 | 4887.30 | 2025-07-21 10:15:00 | 4699.60 | STOP_HIT | 0.50 | 3.84% |
| SELL | retest2 | 2025-07-16 09:15:00 | 4881.50 | 2025-07-21 10:15:00 | 4699.60 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2025-07-30 10:15:00 | 4503.80 | 2025-07-31 12:15:00 | 4569.10 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-07-31 10:00:00 | 4505.30 | 2025-07-31 12:15:00 | 4569.10 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-08-06 13:15:00 | 4561.00 | 2025-08-07 12:15:00 | 4495.50 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-08-11 12:15:00 | 4473.10 | 2025-08-13 11:15:00 | 4495.20 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-08-12 14:30:00 | 4433.00 | 2025-08-13 11:15:00 | 4495.20 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-08-14 12:45:00 | 4510.70 | 2025-08-19 09:15:00 | 4454.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-08-20 10:30:00 | 4511.00 | 2025-08-22 09:15:00 | 4521.70 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-08-29 12:15:00 | 4380.60 | 2025-09-01 14:15:00 | 4424.50 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-25 14:00:00 | 4767.70 | 2025-09-29 09:15:00 | 4815.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-09-25 15:00:00 | 4766.00 | 2025-09-29 09:15:00 | 4815.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-09-26 13:15:00 | 4753.70 | 2025-09-29 09:15:00 | 4815.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-09-29 11:30:00 | 4759.60 | 2025-10-01 09:15:00 | 4797.60 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-09-30 14:15:00 | 4728.20 | 2025-10-01 10:15:00 | 4789.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-10-07 13:15:00 | 4843.90 | 2025-10-08 09:15:00 | 4810.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-10-07 14:45:00 | 4841.10 | 2025-10-08 09:15:00 | 4810.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-10-07 15:15:00 | 4845.00 | 2025-10-08 09:15:00 | 4810.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-10-14 11:15:00 | 4739.20 | 2025-10-15 10:15:00 | 4794.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-10-14 13:00:00 | 4739.90 | 2025-10-15 10:15:00 | 4794.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-10-14 15:15:00 | 4742.00 | 2025-10-15 10:15:00 | 4794.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-10-20 09:15:00 | 4892.10 | 2025-10-20 13:15:00 | 4855.60 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-10-20 10:30:00 | 4885.00 | 2025-10-20 13:15:00 | 4855.60 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-11-03 11:30:00 | 4665.40 | 2025-11-04 10:15:00 | 4693.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-11-03 13:30:00 | 4667.50 | 2025-11-04 10:15:00 | 4693.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-12-01 12:15:00 | 4525.20 | 2025-12-02 11:15:00 | 4498.70 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-12-29 09:15:00 | 4439.10 | 2025-12-29 13:15:00 | 4394.10 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-01-07 14:45:00 | 4532.50 | 2026-01-08 15:15:00 | 4469.90 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-01-08 12:00:00 | 4527.00 | 2026-01-08 15:15:00 | 4469.90 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-01-16 13:00:00 | 4408.20 | 2026-01-19 12:15:00 | 4483.10 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-01-23 13:15:00 | 4306.50 | 2026-01-27 15:15:00 | 4367.70 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-01-23 14:30:00 | 4307.10 | 2026-01-27 15:15:00 | 4367.70 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2026-01-23 15:00:00 | 4301.70 | 2026-01-27 15:15:00 | 4367.70 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-01-27 10:30:00 | 4303.90 | 2026-01-27 15:15:00 | 4367.70 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-02-18 13:00:00 | 4243.20 | 2026-02-19 11:15:00 | 4177.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-02-23 09:15:00 | 4121.00 | 2026-02-27 14:15:00 | 3914.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 09:15:00 | 4121.00 | 2026-03-02 10:15:00 | 3949.00 | STOP_HIT | 0.50 | 4.17% |
| BUY | retest2 | 2026-03-10 11:30:00 | 3976.70 | 2026-03-13 09:15:00 | 3898.50 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2026-03-10 12:15:00 | 3974.50 | 2026-03-13 09:15:00 | 3898.50 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-03-12 09:45:00 | 3974.60 | 2026-03-13 09:15:00 | 3898.50 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-03-12 10:15:00 | 3974.80 | 2026-03-13 09:15:00 | 3898.50 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2026-03-12 11:30:00 | 4005.40 | 2026-03-13 09:15:00 | 3898.50 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-03-17 12:15:00 | 3914.10 | 2026-03-17 13:15:00 | 3964.70 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest1 | 2026-03-20 11:45:00 | 3820.10 | 2026-03-23 09:15:00 | 3629.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-20 11:45:00 | 3820.10 | 2026-03-24 12:15:00 | 3687.90 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2026-03-25 11:15:00 | 3679.50 | 2026-03-30 14:15:00 | 3495.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 14:15:00 | 3684.00 | 2026-03-30 14:15:00 | 3499.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 14:45:00 | 3682.70 | 2026-03-30 14:15:00 | 3498.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 11:15:00 | 3679.50 | 2026-04-01 09:15:00 | 3633.90 | STOP_HIT | 0.50 | 1.24% |
| SELL | retest2 | 2026-03-25 14:15:00 | 3684.00 | 2026-04-01 09:15:00 | 3633.90 | STOP_HIT | 0.50 | 1.36% |
| SELL | retest2 | 2026-03-25 14:45:00 | 3682.70 | 2026-04-01 09:15:00 | 3633.90 | STOP_HIT | 0.50 | 1.33% |
| BUY | retest2 | 2026-04-02 12:15:00 | 3642.00 | 2026-04-02 12:15:00 | 3580.00 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2026-04-02 14:00:00 | 3643.60 | 2026-04-09 09:15:00 | 4007.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 14:45:00 | 3658.20 | 2026-04-09 09:15:00 | 4024.02 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-27 14:30:00 | 4305.00 | 2026-04-28 09:15:00 | 4322.90 | STOP_HIT | 1.00 | -0.42% |
