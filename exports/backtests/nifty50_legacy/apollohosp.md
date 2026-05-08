# APOLLOHOSP (APOLLOHOSP)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4998 bars)
- **Last close:** 8097.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 9 |
| ALERT3 | 9 |
| PENDING | 22 |
| PENDING_CANCEL | 8 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 8 / 8
- **Target hits / Stop hits / Partials:** 0 / 12 / 4
- **Avg / median % per leg:** 5.82% / 11.41%
- **Sum % (uncompounded):** 93.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 8 | 61.5% | 0 | 9 | 4 | 7.72% | 100.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 8 | 61.5% | 0 | 9 | 4 | 7.72% | 100.3% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.38% | -7.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.38% | -7.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 16 | 8 | 50.0% | 0 | 12 | 4 | 5.82% | 93.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 09:15:00 | 5090.05 | 5016.73 | 5016.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 10:15:00 | 5120.65 | 5017.76 | 5016.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 11:15:00 | 4986.25 | 5030.57 | 5023.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 11:15:00 | 4986.25 | 5030.57 | 5023.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 11:15:00 | 4986.25 | 5030.57 | 5023.75 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-10-05 09:15:00 | 5064.15 | 5031.13 | 5024.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 10:15:00 | 5056.30 | 5031.38 | 5024.35 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-10-06 12:15:00 | 5059.15 | 5033.23 | 5025.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-06 13:15:00 | 5066.25 | 5033.56 | 5025.82 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-10-10 10:15:00 | 5072.00 | 5035.08 | 5027.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-10 11:15:00 | 5067.00 | 5035.40 | 5027.21 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-10-12 10:15:00 | 5070.50 | 5042.46 | 5031.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-10-12 11:15:00 | 5048.10 | 5042.51 | 5031.45 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2023-10-12 13:15:00 | 4937.50 | 5041.30 | 5030.95 | SL hit (close<static) qty=1.00 sl=4983.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-12 13:15:00 | 4937.50 | 5041.30 | 5030.95 | SL hit (close<static) qty=1.00 sl=4983.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-12 13:15:00 | 4937.50 | 5041.30 | 5030.95 | SL hit (close<static) qty=1.00 sl=4983.35 alert=retest2 |

### Cycle 2 — SELL (started 2023-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 14:15:00 | 4966.45 | 5023.28 | 5023.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 09:15:00 | 4922.80 | 5021.75 | 5022.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-03 09:15:00 | 5100.00 | 4952.42 | 4983.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 09:15:00 | 5100.00 | 4952.42 | 4983.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 09:15:00 | 5100.00 | 4952.42 | 4983.15 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-11-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 14:15:00 | 5295.00 | 5009.18 | 5009.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 13:15:00 | 5313.60 | 5065.71 | 5039.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 5400.95 | 5417.85 | 5295.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 09:15:00 | 5439.65 | 5417.92 | 5296.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 5439.65 | 5417.92 | 5296.91 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-12-21 11:15:00 | 5497.90 | 5418.98 | 5298.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 12:15:00 | 5485.00 | 5419.64 | 5299.58 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 09:15:00 | 6307.75 | 5732.18 | 5558.61 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-28 12:15:00 | 6368.90 | 6411.80 | 6118.81 | SL hit (close<ema200) qty=0.50 sl=6411.80 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-08 11:15:00 | 5860.50 | 6169.21 | 6170.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-08 12:15:00 | 5841.10 | 6165.94 | 6168.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 14:15:00 | 5961.45 | 5950.10 | 6022.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-07 10:15:00 | 6011.70 | 5951.48 | 6021.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 6011.70 | 5951.48 | 6021.79 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 15:15:00 | 6259.55 | 6066.47 | 6066.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 6292.90 | 6068.72 | 6067.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 6098.00 | 6100.72 | 6085.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 12:15:00 | 6092.60 | 6100.64 | 6085.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 6092.60 | 6100.64 | 6085.47 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-07-03 09:15:00 | 6138.80 | 6100.83 | 6085.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 10:15:00 | 6163.40 | 6101.45 | 6086.25 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 14:15:00 | 7087.91 | 6778.58 | 6615.59 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-03 11:15:00 | 6895.60 | 6985.27 | 6802.35 | SL hit (close<ema200) qty=0.50 sl=6985.27 alert=retest2 |

### Cycle 6 — SELL (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 15:15:00 | 6767.00 | 7070.06 | 7071.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 6741.20 | 7025.97 | 7047.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 11:15:00 | 6951.25 | 6927.18 | 6986.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 11:15:00 | 6951.25 | 6927.18 | 6986.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 6951.25 | 6927.18 | 6986.08 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-02-06 09:15:00 | 6876.55 | 6928.13 | 6985.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 10:15:00 | 6866.40 | 6927.51 | 6984.51 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-15 13:15:00 | 6869.50 | 6576.58 | 6598.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-15 14:15:00 | 6905.00 | 6579.84 | 6600.20 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-04-16 12:15:00 | 7010.00 | 6598.54 | 6609.16 | SL hit (close>static) qty=1.00 sl=6998.30 alert=retest2 |

### Cycle 7 — BUY (started 2025-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 11:15:00 | 7036.50 | 6623.10 | 6621.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 14:15:00 | 7075.00 | 6635.70 | 6627.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 6855.50 | 6862.91 | 6771.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 6758.50 | 6861.69 | 6772.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 6758.50 | 6861.69 | 6772.54 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 6911.50 | 6853.90 | 6771.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 6926.00 | 6854.62 | 6772.40 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-04 14:15:00 | 6855.00 | 6924.66 | 6858.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 15:15:00 | 6855.00 | 6923.97 | 6858.16 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:15:00 | 7883.25 | 7344.47 | 7241.57 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 11:15:00 | 7964.90 | 7483.55 | 7330.46 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 7716.50 | 7735.07 | 7574.51 | SL hit (close<ema200) qty=0.50 sl=7735.07 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 7716.50 | 7735.07 | 7574.51 | SL hit (close<ema200) qty=0.50 sl=7735.07 alert=retest2 |

### Cycle 8 — SELL (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 09:15:00 | 7455.50 | 7636.15 | 7636.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 7426.00 | 7627.53 | 7632.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 7259.50 | 7163.12 | 7294.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 10:15:00 | 7309.50 | 7164.58 | 7294.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 7309.50 | 7164.58 | 7294.08 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-09 13:15:00 | 7256.00 | 7206.04 | 7301.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 14:15:00 | 7254.00 | 7206.52 | 7301.24 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-13 09:15:00 | 7255.00 | 7209.21 | 7298.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-13 10:15:00 | 7274.50 | 7209.86 | 7298.32 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 7312.50 | 7212.69 | 7298.00 | SL hit (close>static) qty=1.00 sl=7309.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-16 10:15:00 | 7235.00 | 7219.29 | 7297.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:15:00 | 7255.00 | 7219.65 | 7297.02 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-11 09:15:00 | 7563.50 | 7086.31 | 7169.63 | SL hit (close>static) qty=1.00 sl=7309.50 alert=retest2 |

### Cycle 9 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 7660.00 | 7240.65 | 7239.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 10:15:00 | 7670.00 | 7305.46 | 7273.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 7512.00 | 7544.08 | 7431.95 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-13 11:15:00 | 7598.00 | 7544.62 | 7433.34 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-13 12:15:00 | 7516.00 | 7544.34 | 7433.75 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-13 13:15:00 | 7572.50 | 7544.62 | 7434.44 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-13 14:15:00 | 7550.00 | 7544.67 | 7435.02 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 10:15:00 | 7460.50 | 7543.92 | 7436.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 7460.50 | 7543.92 | 7436.28 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-18 09:15:00 | 7571.00 | 7538.59 | 7440.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-18 10:15:00 | 7548.00 | 7538.69 | 7440.80 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-18 11:15:00 | 7558.50 | 7538.88 | 7441.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-18 12:15:00 | 7498.00 | 7538.48 | 7441.67 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-25 10:15:00 | 7573.00 | 7481.17 | 7425.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 11:15:00 | 7588.50 | 7482.24 | 7426.14 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-27 10:15:00 | 7600.50 | 7488.01 | 7430.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:15:00 | 7585.00 | 7488.98 | 7431.50 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 7427.50 | 7490.90 | 7433.90 | SL hit (close<static) qty=1.00 sl=7430.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 7427.50 | 7490.90 | 7433.90 | SL hit (close<static) qty=1.00 sl=7430.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-10 11:15:00 | 7553.00 | 7445.70 | 7420.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-10 12:15:00 | 7535.00 | 7446.59 | 7421.03 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-15 09:15:00 | 7649.00 | 7455.12 | 7426.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 10:15:00 | 7656.00 | 7457.12 | 7427.90 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-16 14:15:00 | 7550.00 | 7472.20 | 7437.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 15:15:00 | 7555.00 | 7473.02 | 7437.84 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-10-05 10:15:00 | 5056.30 | 2023-10-12 13:15:00 | 4937.50 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2023-10-06 13:15:00 | 5066.25 | 2023-10-12 13:15:00 | 4937.50 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2023-10-10 11:15:00 | 5067.00 | 2023-10-12 13:15:00 | 4937.50 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2023-12-21 12:15:00 | 5485.00 | 2024-01-23 09:15:00 | 6307.75 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2023-12-21 12:15:00 | 5485.00 | 2024-02-28 12:15:00 | 6368.90 | STOP_HIT | 0.50 | 16.11% |
| BUY | retest2 | 2024-07-03 10:15:00 | 6163.40 | 2024-09-12 14:15:00 | 7087.91 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-07-03 10:15:00 | 6163.40 | 2024-10-03 11:15:00 | 6895.60 | STOP_HIT | 0.50 | 11.88% |
| SELL | retest2 | 2025-02-06 10:15:00 | 6866.40 | 2025-04-16 12:15:00 | 7010.00 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-05-12 10:15:00 | 6926.00 | 2025-08-18 09:15:00 | 7883.25 | PARTIAL | 0.50 | 13.82% |
| BUY | retest2 | 2025-06-04 15:15:00 | 6855.00 | 2025-08-22 11:15:00 | 7964.90 | PARTIAL | 0.50 | 16.19% |
| BUY | retest2 | 2025-05-12 10:15:00 | 6926.00 | 2025-09-22 13:15:00 | 7716.50 | STOP_HIT | 0.50 | 11.41% |
| BUY | retest2 | 2025-06-04 15:15:00 | 6855.00 | 2025-09-22 13:15:00 | 7716.50 | STOP_HIT | 0.50 | 12.57% |
| SELL | retest2 | 2026-01-09 14:15:00 | 7254.00 | 2026-01-13 14:15:00 | 7312.50 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-01-16 11:15:00 | 7255.00 | 2026-02-11 09:15:00 | 7563.50 | STOP_HIT | 1.00 | -4.25% |
| BUY | retest2 | 2026-03-25 11:15:00 | 7588.50 | 2026-03-30 09:15:00 | 7427.50 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2026-03-27 11:15:00 | 7585.00 | 2026-03-30 09:15:00 | 7427.50 | STOP_HIT | 1.00 | -2.08% |
