# ABB India Ltd. (ABB)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 7010.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 2 |
| ALERT3 | 83 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 97 |
| PARTIAL | 10 |
| TARGET_HIT | 2 |
| STOP_HIT | 95 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 107 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 24 / 83
- **Target hits / Stop hits / Partials:** 2 / 95 / 10
- **Avg / median % per leg:** -0.58% / -1.09%
- **Sum % (uncompounded):** -62.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 1 | 2.5% | 1 | 39 | 0 | -1.76% | -70.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 40 | 1 | 2.5% | 1 | 39 | 0 | -1.76% | -70.5% |
| SELL (all) | 67 | 23 | 34.3% | 1 | 56 | 10 | 0.12% | 8.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 67 | 23 | 34.3% | 1 | 56 | 10 | 0.12% | 8.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 107 | 24 | 22.4% | 2 | 95 | 10 | -0.58% | -62.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 11:15:00 | 4033.05 | 4282.30 | 4283.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 12:15:00 | 4013.40 | 4279.62 | 4281.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-13 10:15:00 | 4228.60 | 4227.95 | 4251.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-13 11:00:00 | 4228.60 | 4227.95 | 4251.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 4274.20 | 4228.16 | 4250.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-16 10:00:00 | 4274.20 | 4228.16 | 4250.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 10:15:00 | 4266.00 | 4228.53 | 4250.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 10:30:00 | 4237.80 | 4235.38 | 4253.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-20 11:15:00 | 4025.91 | 4213.62 | 4240.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-11-03 12:15:00 | 4143.10 | 4130.27 | 4183.49 | SL hit (close>ema200) qty=0.50 sl=4130.27 alert=retest2 |

### Cycle 2 — BUY (started 2023-11-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 13:15:00 | 4255.15 | 4213.94 | 4213.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 10:15:00 | 4304.35 | 4221.93 | 4218.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 10:15:00 | 4609.95 | 4616.32 | 4488.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-02 10:30:00 | 4612.80 | 4616.32 | 4488.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 4724.70 | 4745.16 | 4625.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 10:15:00 | 4781.15 | 4745.16 | 4625.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 09:15:00 | 4772.00 | 4741.89 | 4631.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 12:30:00 | 4770.00 | 4742.53 | 4633.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-01 11:15:00 | 4451.10 | 4735.55 | 4640.57 | SL hit (close<static) qty=1.00 sl=4613.80 alert=retest2 |

### Cycle 3 — SELL (started 2024-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 15:15:00 | 7527.35 | 7815.82 | 7817.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 7482.00 | 7812.50 | 7815.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 09:15:00 | 7788.80 | 7763.91 | 7788.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 09:15:00 | 7788.80 | 7763.91 | 7788.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 7788.80 | 7763.91 | 7788.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:45:00 | 7836.50 | 7763.91 | 7788.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 7719.60 | 7763.47 | 7788.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 11:30:00 | 7690.00 | 7763.01 | 7788.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 12:30:00 | 7684.15 | 7762.33 | 7787.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 13:15:00 | 7691.05 | 7762.33 | 7787.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 15:00:00 | 7684.15 | 7760.66 | 7786.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 7778.75 | 7760.23 | 7786.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:45:00 | 7752.50 | 7760.23 | 7786.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 7792.00 | 7760.54 | 7786.03 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-16 11:15:00 | 7841.95 | 7761.35 | 7786.31 | SL hit (close>static) qty=1.00 sl=7795.45 alert=retest2 |

### Cycle 4 — BUY (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 10:15:00 | 8055.45 | 7800.91 | 7800.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 10:15:00 | 8130.15 | 7818.38 | 7809.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 10:15:00 | 7850.00 | 7901.71 | 7856.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 10:15:00 | 7850.00 | 7901.71 | 7856.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 7850.00 | 7901.71 | 7856.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:00:00 | 7850.00 | 7901.71 | 7856.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 11:15:00 | 7863.10 | 7901.33 | 7856.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:45:00 | 7830.00 | 7901.33 | 7856.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 12:15:00 | 7744.75 | 7899.77 | 7856.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 13:00:00 | 7744.75 | 7899.77 | 7856.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 13:15:00 | 7715.85 | 7897.94 | 7855.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 13:30:00 | 7728.00 | 7897.94 | 7855.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 8089.95 | 8230.19 | 8066.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:45:00 | 8053.85 | 8230.19 | 8066.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 8062.05 | 8228.52 | 8066.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:45:00 | 8085.15 | 8228.52 | 8066.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 8053.45 | 8226.77 | 8066.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:00:00 | 8053.45 | 8226.77 | 8066.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 7878.05 | 8223.30 | 8065.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:00:00 | 7878.05 | 8223.30 | 8065.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2024-11-01 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-01 18:15:00 | 7410.00 | 7943.31 | 7943.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 7309.45 | 7937.00 | 7940.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 10:15:00 | 7347.70 | 7321.50 | 7560.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-26 10:45:00 | 7376.85 | 7321.50 | 7560.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 7582.95 | 7351.67 | 7539.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:00:00 | 7582.95 | 7351.67 | 7539.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 7558.90 | 7353.73 | 7539.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 12:00:00 | 7547.35 | 7355.66 | 7539.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 13:00:00 | 7551.90 | 7357.61 | 7539.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 13:45:00 | 7533.65 | 7359.49 | 7539.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 10:15:00 | 7662.70 | 7368.27 | 7540.17 | SL hit (close>static) qty=1.00 sl=7635.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 6040.00 | 5617.45 | 5615.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 6047.00 | 5702.36 | 5660.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 5885.00 | 5922.32 | 5816.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 13:00:00 | 5885.00 | 5922.32 | 5816.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 5897.50 | 5956.52 | 5864.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 5870.50 | 5956.52 | 5864.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 5875.00 | 5951.70 | 5865.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:30:00 | 5871.00 | 5951.70 | 5865.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 5866.50 | 5950.85 | 5865.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 5888.00 | 5950.85 | 5865.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 11:15:00 | 5859.50 | 5948.57 | 5866.03 | SL hit (close<static) qty=1.00 sl=5862.50 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 11:15:00 | 5687.50 | 5819.20 | 5819.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 5663.00 | 5813.26 | 5816.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 5233.30 | 5204.92 | 5363.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 11:00:00 | 5233.30 | 5204.92 | 5363.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 5360.80 | 5210.63 | 5355.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:45:00 | 5359.80 | 5210.63 | 5355.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 5370.50 | 5212.22 | 5355.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:00:00 | 5370.50 | 5212.22 | 5355.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 5366.60 | 5213.76 | 5355.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 15:00:00 | 5340.00 | 5215.02 | 5355.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 09:45:00 | 5348.70 | 5217.61 | 5355.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 10:30:00 | 5343.60 | 5218.98 | 5355.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:00:00 | 5340.60 | 5220.19 | 5355.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 5345.00 | 5221.43 | 5355.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:15:00 | 5339.10 | 5221.43 | 5355.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 14:00:00 | 5335.50 | 5222.57 | 5354.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 5388.20 | 5226.76 | 5355.09 | SL hit (close>static) qty=1.00 sl=5375.00 alert=retest2 |

### Cycle 8 — BUY (started 2026-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 15:15:00 | 5662.00 | 5141.64 | 5139.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 5830.00 | 5148.49 | 5143.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 12:15:00 | 5997.00 | 6017.35 | 5770.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 13:00:00 | 5997.00 | 6017.35 | 5770.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-08-14 11:15:00 | 4375.00 | 2023-08-25 12:15:00 | 4251.75 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2023-08-14 12:15:00 | 4383.10 | 2023-08-25 12:15:00 | 4251.75 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2023-08-14 14:30:00 | 4367.95 | 2023-08-25 12:15:00 | 4251.75 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2023-08-24 09:15:00 | 4374.75 | 2023-08-25 12:15:00 | 4251.75 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2023-08-29 13:00:00 | 4327.05 | 2023-09-04 10:15:00 | 4258.10 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2023-09-01 13:30:00 | 4300.10 | 2023-09-04 10:15:00 | 4258.10 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2023-09-04 09:30:00 | 4299.90 | 2023-09-04 10:15:00 | 4258.10 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2023-09-04 13:30:00 | 4300.00 | 2023-09-13 09:15:00 | 4265.90 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2023-09-13 11:45:00 | 4318.65 | 2023-09-22 09:15:00 | 4223.40 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2023-09-13 13:00:00 | 4322.55 | 2023-09-22 09:15:00 | 4223.40 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2023-09-13 13:30:00 | 4316.85 | 2023-09-22 09:15:00 | 4223.40 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2023-09-21 11:00:00 | 4311.70 | 2023-09-22 09:15:00 | 4223.40 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2023-10-18 10:30:00 | 4237.80 | 2023-10-20 11:15:00 | 4025.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-18 10:30:00 | 4237.80 | 2023-11-03 12:15:00 | 4143.10 | STOP_HIT | 0.50 | 2.23% |
| SELL | retest2 | 2023-11-08 11:00:00 | 4248.85 | 2023-11-08 15:15:00 | 4280.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2023-11-08 12:30:00 | 4249.00 | 2023-11-08 15:15:00 | 4280.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2023-11-09 11:45:00 | 4244.40 | 2023-11-09 14:15:00 | 4338.00 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-01-24 10:15:00 | 4781.15 | 2024-02-01 11:15:00 | 4451.10 | STOP_HIT | 1.00 | -6.90% |
| BUY | retest2 | 2024-01-29 09:15:00 | 4772.00 | 2024-02-01 11:15:00 | 4451.10 | STOP_HIT | 1.00 | -6.72% |
| BUY | retest2 | 2024-01-29 12:30:00 | 4770.00 | 2024-02-01 11:15:00 | 4451.10 | STOP_HIT | 1.00 | -6.69% |
| BUY | retest2 | 2024-02-21 09:15:00 | 4805.75 | 2024-02-22 09:15:00 | 5286.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-29 10:30:00 | 7970.00 | 2024-08-01 10:15:00 | 7825.80 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-07-29 11:15:00 | 7963.05 | 2024-08-01 10:15:00 | 7825.80 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-07-29 12:30:00 | 7969.85 | 2024-08-01 10:15:00 | 7825.80 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-07-29 13:15:00 | 7977.00 | 2024-08-01 10:15:00 | 7825.80 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-07-31 11:45:00 | 7945.05 | 2024-08-02 09:15:00 | 7686.25 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2024-07-31 14:15:00 | 7926.20 | 2024-08-02 09:15:00 | 7686.25 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2024-08-01 09:15:00 | 7929.75 | 2024-08-02 09:15:00 | 7686.25 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2024-08-01 09:45:00 | 7934.75 | 2024-08-02 09:15:00 | 7686.25 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2024-08-07 14:15:00 | 7895.00 | 2024-08-12 09:15:00 | 7785.30 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-08-07 15:15:00 | 7887.10 | 2024-08-12 09:15:00 | 7785.30 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-08-08 09:30:00 | 7889.95 | 2024-08-12 09:15:00 | 7785.30 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-08-08 10:00:00 | 7928.85 | 2024-08-12 09:15:00 | 7785.30 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-08-20 09:15:00 | 7837.30 | 2024-08-20 09:15:00 | 7783.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-08-21 10:30:00 | 7844.95 | 2024-08-23 14:15:00 | 7788.80 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-08-21 12:00:00 | 7840.30 | 2024-08-23 14:15:00 | 7788.80 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-08-23 13:30:00 | 7832.25 | 2024-08-23 14:15:00 | 7788.80 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-08-29 15:15:00 | 7878.00 | 2024-09-02 11:15:00 | 7782.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-08-30 09:45:00 | 7888.00 | 2024-09-02 11:15:00 | 7782.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-08-30 10:15:00 | 7890.10 | 2024-09-02 11:15:00 | 7782.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-09-02 10:30:00 | 7882.00 | 2024-09-02 11:15:00 | 7782.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-09-13 11:30:00 | 7690.00 | 2024-09-16 11:15:00 | 7841.95 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-09-13 12:30:00 | 7684.15 | 2024-09-16 11:15:00 | 7841.95 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-09-13 13:15:00 | 7691.05 | 2024-09-16 11:15:00 | 7841.95 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-09-13 15:00:00 | 7684.15 | 2024-09-16 11:15:00 | 7841.95 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-09-16 14:15:00 | 7741.00 | 2024-09-17 09:15:00 | 7816.75 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-09-16 15:00:00 | 7756.55 | 2024-09-17 09:15:00 | 7816.75 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-09-18 15:15:00 | 7769.00 | 2024-09-19 11:15:00 | 7380.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-18 15:15:00 | 7769.00 | 2024-09-20 15:15:00 | 7744.00 | STOP_HIT | 0.50 | 0.32% |
| SELL | retest2 | 2024-12-03 12:00:00 | 7547.35 | 2024-12-04 10:15:00 | 7662.70 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-12-03 13:00:00 | 7551.90 | 2024-12-04 10:15:00 | 7662.70 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-12-03 13:45:00 | 7533.65 | 2024-12-04 10:15:00 | 7662.70 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-12-05 09:30:00 | 7536.35 | 2024-12-10 11:15:00 | 7691.00 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-12-19 10:45:00 | 7470.00 | 2024-12-20 12:15:00 | 7096.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-19 10:45:00 | 7470.00 | 2024-12-30 13:15:00 | 6723.00 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-04 09:15:00 | 5888.00 | 2025-07-04 11:15:00 | 5859.50 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-07-09 09:45:00 | 5880.00 | 2025-07-10 10:15:00 | 5843.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-07-09 11:00:00 | 5888.00 | 2025-07-10 10:15:00 | 5843.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-07-09 11:30:00 | 5877.50 | 2025-07-10 10:15:00 | 5843.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-09-15 15:00:00 | 5340.00 | 2025-09-17 09:15:00 | 5388.20 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-09-16 09:45:00 | 5348.70 | 2025-09-17 09:15:00 | 5388.20 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-09-16 10:30:00 | 5343.60 | 2025-09-17 09:15:00 | 5388.20 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-09-16 12:00:00 | 5340.60 | 2025-09-17 09:15:00 | 5388.20 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-09-16 13:15:00 | 5339.10 | 2025-09-17 09:15:00 | 5388.20 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-09-16 14:00:00 | 5335.50 | 2025-09-17 09:15:00 | 5388.20 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-09-17 14:00:00 | 5339.20 | 2025-09-17 14:15:00 | 5390.90 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-09-23 09:45:00 | 5315.50 | 2025-10-29 11:15:00 | 5294.00 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2025-10-07 12:30:00 | 5227.00 | 2025-10-29 11:15:00 | 5294.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-10-07 14:30:00 | 5228.00 | 2025-10-29 11:15:00 | 5294.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-10-16 10:00:00 | 5224.50 | 2025-11-07 09:15:00 | 5049.72 | PARTIAL | 0.50 | 3.35% |
| SELL | retest2 | 2025-10-20 10:45:00 | 5227.00 | 2025-11-07 10:15:00 | 4965.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 10:45:00 | 5211.50 | 2025-11-07 10:15:00 | 4966.60 | PARTIAL | 0.50 | 4.70% |
| SELL | retest2 | 2025-10-28 11:30:00 | 5216.50 | 2025-11-07 10:15:00 | 4963.27 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2025-10-29 09:30:00 | 5222.00 | 2025-11-07 10:15:00 | 4965.65 | PARTIAL | 0.50 | 4.91% |
| SELL | retest2 | 2025-11-03 11:45:00 | 5216.50 | 2025-11-10 09:15:00 | 4955.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 10:00:00 | 5224.50 | 2025-11-20 09:15:00 | 5130.50 | STOP_HIT | 0.50 | 1.80% |
| SELL | retest2 | 2025-10-20 10:45:00 | 5227.00 | 2025-11-20 09:15:00 | 5130.50 | STOP_HIT | 0.50 | 1.85% |
| SELL | retest2 | 2025-10-28 10:45:00 | 5211.50 | 2025-11-20 09:15:00 | 5130.50 | STOP_HIT | 0.50 | 1.55% |
| SELL | retest2 | 2025-10-28 11:30:00 | 5216.50 | 2025-11-20 09:15:00 | 5130.50 | STOP_HIT | 0.50 | 1.65% |
| SELL | retest2 | 2025-10-29 09:30:00 | 5222.00 | 2025-11-20 09:15:00 | 5130.50 | STOP_HIT | 0.50 | 1.75% |
| SELL | retest2 | 2025-11-03 11:45:00 | 5216.50 | 2025-11-20 09:15:00 | 5130.50 | STOP_HIT | 0.50 | 1.65% |
| SELL | retest2 | 2025-11-07 09:15:00 | 5048.50 | 2025-12-01 13:15:00 | 5185.00 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-11-26 15:15:00 | 5190.00 | 2025-12-01 13:15:00 | 5185.00 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-11-27 10:30:00 | 5195.50 | 2025-12-01 15:15:00 | 5186.00 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-11-28 12:15:00 | 5192.50 | 2025-12-02 13:15:00 | 5193.50 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-12-01 10:45:00 | 5167.00 | 2025-12-11 09:15:00 | 5237.50 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-12-01 11:45:00 | 5163.50 | 2025-12-11 09:15:00 | 5237.50 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-12-01 14:45:00 | 5169.00 | 2025-12-11 09:15:00 | 5237.50 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-12-02 11:45:00 | 5168.50 | 2025-12-11 09:15:00 | 5237.50 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-12-03 09:15:00 | 5153.50 | 2025-12-11 09:15:00 | 5237.50 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-12-04 13:30:00 | 5163.00 | 2025-12-11 09:15:00 | 5237.50 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-12-05 09:30:00 | 5161.50 | 2025-12-11 09:15:00 | 5237.50 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-12-05 12:15:00 | 5159.00 | 2025-12-11 09:15:00 | 5237.50 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-12-08 12:30:00 | 5108.50 | 2025-12-12 14:15:00 | 5282.50 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2025-12-09 13:45:00 | 5113.00 | 2025-12-12 14:15:00 | 5282.50 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2025-12-09 15:15:00 | 5108.00 | 2025-12-12 14:15:00 | 5282.50 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2025-12-10 09:45:00 | 5114.00 | 2025-12-12 14:15:00 | 5282.50 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-12-17 12:15:00 | 5194.00 | 2025-12-24 10:15:00 | 5243.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-12-22 11:45:00 | 5195.00 | 2025-12-24 10:15:00 | 5243.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-12-22 12:30:00 | 5186.50 | 2025-12-24 10:15:00 | 5243.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-12-23 14:30:00 | 5175.50 | 2025-12-24 10:15:00 | 5243.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-12-29 11:15:00 | 5179.00 | 2026-01-02 15:15:00 | 5219.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-29 12:30:00 | 5172.00 | 2026-01-02 15:15:00 | 5219.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-29 13:00:00 | 5179.00 | 2026-01-02 15:15:00 | 5219.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-29 14:00:00 | 5176.50 | 2026-01-02 15:15:00 | 5219.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-01-08 14:45:00 | 5087.50 | 2026-01-20 09:15:00 | 4833.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 14:45:00 | 5087.50 | 2026-01-28 12:15:00 | 5036.00 | STOP_HIT | 0.50 | 1.01% |
