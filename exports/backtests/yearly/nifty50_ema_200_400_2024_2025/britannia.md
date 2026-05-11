# BRITANNIA (BRITANNIA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 5516.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 7 |
| ALERT3 | 17 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 16 |
| PARTIAL | 12 |
| TARGET_HIT | 4 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 24 / 8
- **Target hits / Stop hits / Partials:** 4 / 16 / 12
- **Avg / median % per leg:** 3.27% / 4.37%
- **Sum % (uncompounded):** 104.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 8 | 66.7% | 4 | 4 | 4 | 4.29% | 51.5% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.12% | -8.5% |
| SELL (all) | 20 | 16 | 80.0% | 0 | 12 | 8 | 2.65% | 53.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 16 | 80.0% | 0 | 12 | 8 | 2.65% | 53.0% |
| retest1 (combined) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| retest2 (combined) | 24 | 16 | 66.7% | 0 | 16 | 8 | 1.85% | 44.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-02 14:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 14:30:00 | 5775.85 | 5694.90 | 5517.71 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 15:15:00 | 5766.00 | 5694.90 | 5517.71 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-05 10:00:00 | 5780.55 | 5696.45 | 5520.25 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-05 11:30:00 | 5755.00 | 5697.39 | 5522.48 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 11:15:00 | 6042.75 | 5816.06 | 5701.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 09:15:00 | 6064.64 | 5826.03 | 5709.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 09:15:00 | 6054.30 | 5826.03 | 5709.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 09:15:00 | 6069.58 | 5826.03 | 5709.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-09-27 12:15:00 | 6330.50 | 6014.55 | 5858.85 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 2 — SELL (started 2024-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 13:15:00 | 5618.20 | 5885.26 | 5885.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 11:15:00 | 5593.65 | 5872.87 | 5879.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 4928.45 | 4878.91 | 5087.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-09 10:00:00 | 4928.45 | 4878.91 | 5087.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 5024.50 | 4890.78 | 5030.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:45:00 | 5031.95 | 4890.78 | 5030.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 5030.50 | 4892.17 | 5030.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:30:00 | 5030.60 | 4892.17 | 5030.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 5013.25 | 4893.37 | 5030.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 09:15:00 | 5002.00 | 4937.27 | 5039.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 13:00:00 | 5007.00 | 4940.36 | 5038.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 14:15:00 | 5043.20 | 4942.15 | 5038.83 | SL hit (close>static) qty=1.00 sl=5032.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 14:15:00 | 5343.60 | 4914.34 | 4913.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 5361.05 | 4931.18 | 4922.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 10:15:00 | 5520.00 | 5524.42 | 5407.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 11:15:00 | 5615.00 | 5715.84 | 5612.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 5615.00 | 5715.84 | 5612.76 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 09:15:00 | 5412.50 | 5574.88 | 5575.53 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 11:15:00 | 5662.50 | 5576.69 | 5576.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 09:15:00 | 5690.00 | 5577.90 | 5576.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 14:15:00 | 5931.00 | 5948.54 | 5817.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 5844.00 | 5953.72 | 5856.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 5844.00 | 5953.72 | 5856.80 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 14:15:00 | 5816.50 | 5877.86 | 5877.98 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 5941.50 | 5878.11 | 5878.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 5960.00 | 5878.92 | 5878.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 5863.00 | 5880.03 | 5879.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 11:15:00 | 5863.00 | 5880.03 | 5879.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 5863.00 | 5880.03 | 5879.05 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 11:15:00 | 5833.00 | 5877.67 | 5877.88 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 5955.00 | 5878.32 | 5878.20 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 5826.00 | 5877.86 | 5877.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 12:15:00 | 5820.50 | 5877.29 | 5877.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 5891.00 | 5875.44 | 5876.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 11:15:00 | 5891.00 | 5875.44 | 5876.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 5891.00 | 5875.44 | 5876.73 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 11:15:00 | 6010.00 | 5878.55 | 5878.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 13:15:00 | 6039.50 | 5881.66 | 5879.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 12:15:00 | 5955.00 | 5977.79 | 5940.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 5992.00 | 5978.06 | 5941.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 5992.00 | 5978.06 | 5941.08 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 5712.00 | 5934.85 | 5935.12 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 6071.00 | 5925.92 | 5925.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 13:15:00 | 6120.00 | 5929.32 | 5927.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 14:15:00 | 5992.00 | 6019.06 | 5979.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 15:15:00 | 6000.00 | 6018.87 | 5980.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 6000.00 | 6018.87 | 5980.06 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2026-03-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 13:15:00 | 5806.50 | 5955.73 | 5955.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 5707.00 | 5939.33 | 5947.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 5682.00 | 5669.91 | 5769.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 10:15:00 | 5740.00 | 5676.24 | 5765.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 5740.00 | 5676.24 | 5765.66 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-02 14:30:00 | 5775.85 | 2024-09-11 11:15:00 | 6042.75 | PARTIAL | 0.50 | 4.62% |
| BUY | retest1 | 2024-08-02 15:15:00 | 5766.00 | 2024-09-12 09:15:00 | 6064.64 | PARTIAL | 0.50 | 5.18% |
| BUY | retest1 | 2024-08-05 10:00:00 | 5780.55 | 2024-09-12 09:15:00 | 6054.30 | PARTIAL | 0.50 | 4.74% |
| BUY | retest1 | 2024-08-05 11:30:00 | 5755.00 | 2024-09-12 09:15:00 | 6069.58 | PARTIAL | 0.50 | 5.47% |
| BUY | retest1 | 2024-08-02 14:30:00 | 5775.85 | 2024-09-27 12:15:00 | 6330.50 | TARGET_HIT | 0.50 | 9.60% |
| BUY | retest1 | 2024-08-02 15:15:00 | 5766.00 | 2024-09-30 09:15:00 | 6353.44 | TARGET_HIT | 0.50 | 10.19% |
| BUY | retest1 | 2024-08-05 10:00:00 | 5780.55 | 2024-09-30 09:15:00 | 6342.60 | TARGET_HIT | 0.50 | 9.72% |
| BUY | retest1 | 2024-08-05 11:30:00 | 5755.00 | 2024-09-30 14:15:00 | 6358.61 | TARGET_HIT | 0.50 | 10.49% |
| BUY | retest2 | 2024-10-15 09:30:00 | 6028.00 | 2024-10-18 11:15:00 | 5905.95 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-10-15 11:00:00 | 6025.00 | 2024-10-18 11:15:00 | 5905.95 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-10-17 10:15:00 | 6052.85 | 2024-10-18 11:15:00 | 5905.95 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-10-17 11:15:00 | 6029.20 | 2024-10-18 11:15:00 | 5905.95 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-01-29 09:15:00 | 5002.00 | 2025-01-29 14:15:00 | 5043.20 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-01-29 13:00:00 | 5007.00 | 2025-01-29 14:15:00 | 5043.20 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-02-04 13:00:00 | 5002.60 | 2025-02-04 15:15:00 | 5040.30 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-02-05 09:15:00 | 4967.65 | 2025-02-24 09:15:00 | 4719.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 14:15:00 | 4927.15 | 2025-02-24 09:15:00 | 4712.00 | PARTIAL | 0.50 | 4.37% |
| SELL | retest2 | 2025-02-06 09:45:00 | 4942.35 | 2025-02-24 09:15:00 | 4704.97 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2025-02-06 10:15:00 | 4942.70 | 2025-02-28 10:15:00 | 4680.79 | PARTIAL | 0.50 | 5.30% |
| SELL | retest2 | 2025-02-06 11:00:00 | 4933.10 | 2025-02-28 10:15:00 | 4695.23 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2025-02-07 10:15:00 | 4915.80 | 2025-02-28 10:15:00 | 4695.56 | PARTIAL | 0.50 | 4.48% |
| SELL | retest2 | 2025-02-10 11:45:00 | 4960.00 | 2025-02-28 10:15:00 | 4686.44 | PARTIAL | 0.50 | 5.52% |
| SELL | retest2 | 2025-02-17 15:15:00 | 4952.60 | 2025-02-28 12:15:00 | 4670.01 | PARTIAL | 0.50 | 5.71% |
| SELL | retest2 | 2025-02-05 09:15:00 | 4967.65 | 2025-03-13 09:15:00 | 4829.00 | STOP_HIT | 0.50 | 2.79% |
| SELL | retest2 | 2025-02-05 14:15:00 | 4927.15 | 2025-03-13 09:15:00 | 4829.00 | STOP_HIT | 0.50 | 1.99% |
| SELL | retest2 | 2025-02-06 09:45:00 | 4942.35 | 2025-03-13 09:15:00 | 4829.00 | STOP_HIT | 0.50 | 2.29% |
| SELL | retest2 | 2025-02-06 10:15:00 | 4942.70 | 2025-03-13 09:15:00 | 4829.00 | STOP_HIT | 0.50 | 2.30% |
| SELL | retest2 | 2025-02-06 11:00:00 | 4933.10 | 2025-03-13 09:15:00 | 4829.00 | STOP_HIT | 0.50 | 2.11% |
| SELL | retest2 | 2025-02-07 10:15:00 | 4915.80 | 2025-03-13 09:15:00 | 4829.00 | STOP_HIT | 0.50 | 1.77% |
| SELL | retest2 | 2025-02-10 11:45:00 | 4960.00 | 2025-03-13 09:15:00 | 4829.00 | STOP_HIT | 0.50 | 2.64% |
| SELL | retest2 | 2025-02-17 15:15:00 | 4952.60 | 2025-03-13 09:15:00 | 4829.00 | STOP_HIT | 0.50 | 2.50% |
| SELL | retest2 | 2025-03-28 11:15:00 | 4953.25 | 2025-04-04 09:15:00 | 5106.50 | STOP_HIT | 1.00 | -3.09% |
