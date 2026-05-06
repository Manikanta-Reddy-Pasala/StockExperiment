# APOLLOHOSP (APOLLOHOSP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4998 bars)
- **Last close:** 7760.50
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 6 |
| ALERT3 | 8 |
| PENDING | 27 |
| PENDING_CANCEL | 9 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 9 / 11
- **Target hits / Stop hits / Partials:** 0 / 16 / 4
- **Avg / median % per leg:** 5.32% / 0.00%
- **Sum % (uncompounded):** 106.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 9 | 45.0% | 0 | 16 | 4 | 5.32% | 106.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 9 | 45.0% | 0 | 16 | 4 | 5.32% | 106.5% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 20 | 9 | 45.0% | 0 | 16 | 4 | 5.32% | 106.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 15:15:00 | 5107.00 | 5011.75 | 5011.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 09:15:00 | 5125.65 | 5012.88 | 5011.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 11:15:00 | 4986.25 | 5030.44 | 5021.35 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 11:15:00 | 4986.25 | 5030.44 | 5021.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 11:15:00 | 4986.25 | 5030.44 | 5021.35 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2023-10-05 09:15:00 | 5064.15 | 5031.00 | 5021.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 10:15:00 | 5056.30 | 5031.25 | 5022.03 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-10-06 12:15:00 | 5059.15 | 5033.12 | 5023.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-06 13:15:00 | 5066.25 | 5033.45 | 5023.60 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-10-10 10:15:00 | 5072.00 | 5034.98 | 5024.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-10 11:15:00 | 5067.00 | 5035.29 | 5025.12 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-10-12 10:15:00 | 5070.50 | 5042.37 | 5029.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-10-12 11:15:00 | 5048.10 | 5042.43 | 5029.51 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2023-10-12 13:15:00 | 4983.35 | 5041.21 | 5029.03 | SL hit qty=1.00 sl=4983.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-12 13:15:00 | 4983.35 | 5041.21 | 5029.03 | SL hit qty=1.00 sl=4983.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-12 13:15:00 | 4983.35 | 5041.21 | 5029.03 | SL hit qty=1.00 sl=4983.35 alert=retest2 |

### Cycle 2 — SELL (started 2023-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 10:15:00 | 4880.00 | 5020.29 | 5020.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 12:15:00 | 4846.75 | 5017.44 | 5018.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-03 09:15:00 | 5100.00 | 4952.39 | 4981.98 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 09:15:00 | 5100.00 | 4952.39 | 4981.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 09:15:00 | 5100.00 | 4952.39 | 4981.98 | EMA400 retest candle locked |

### Cycle 3 — BUY (started 2023-11-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 14:15:00 | 5295.00 | 5009.16 | 5008.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 13:15:00 | 5313.60 | 5065.69 | 5038.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 5400.95 | 5417.85 | 5295.28 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 09:15:00 | 5439.65 | 5417.91 | 5296.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 5439.65 | 5417.91 | 5296.53 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2023-12-21 11:15:00 | 5497.90 | 5418.98 | 5298.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 12:15:00 | 5485.00 | 5419.64 | 5299.20 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-01-23 09:15:00 | 6307.75 | 5732.17 | 5558.42 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2024-05-08 11:15:00 | 5860.50 | 6169.21 | 6170.01 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2024-06-24 15:15:00 | 6259.55 | 6066.47 | 6066.21 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 15:15:00 | 6259.55 | 6066.47 | 6066.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 6292.90 | 6068.72 | 6067.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 6098.00 | 6100.72 | 6085.43 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 12:15:00 | 6092.60 | 6100.64 | 6085.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 6092.60 | 6100.64 | 6085.47 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-07-03 09:15:00 | 6138.80 | 6100.83 | 6085.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 10:15:00 | 6163.40 | 6101.45 | 6086.25 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-09-12 14:15:00 | 7087.91 | 6778.58 | 6615.59 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2025-01-20 15:15:00 | 6767.00 | 7070.06 | 7071.07 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2025-02-17 09:15:00 | 6163.40 | 6762.49 | 6882.98 | SL hit qty=0.50 sl=6163.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-03 09:15:00 | 6128.55 | 6528.28 | 6717.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-03 10:15:00 | 6127.00 | 6524.28 | 6714.32 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 6085.00 | 6417.83 | 6620.43 | SL hit qty=1.00 sl=6085.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-11 14:15:00 | 6128.85 | 6401.56 | 6607.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 15:15:00 | 6134.80 | 6398.91 | 6604.84 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-12 09:15:00 | 6085.00 | 6395.75 | 6602.23 | SL hit qty=1.00 sl=6085.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-12 13:15:00 | 6123.10 | 6384.24 | 6592.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 14:15:00 | 6140.25 | 6381.81 | 6590.07 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 6143.75 | 6379.44 | 6587.84 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-03-13 11:15:00 | 6085.00 | 6371.87 | 6580.92 | SL hit qty=1.00 sl=6085.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-17 09:15:00 | 6146.85 | 6358.99 | 6569.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-03-17 10:15:00 | 6140.75 | 6356.82 | 6567.11 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-03-17 11:15:00 | 6160.70 | 6354.87 | 6565.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 12:15:00 | 6154.00 | 6352.87 | 6563.03 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-17 11:15:00 | 7036.50 | 6623.10 | 6621.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 11:15:00 | 7036.50 | 6623.10 | 6621.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 14:15:00 | 7075.00 | 6635.70 | 6627.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 6855.50 | 6862.91 | 6771.80 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 6758.50 | 6861.69 | 6772.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 6758.50 | 6861.69 | 6772.54 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 6911.50 | 6853.90 | 6771.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 6926.00 | 6854.62 | 6772.40 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-04 14:15:00 | 6855.00 | 6924.66 | 6858.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 15:15:00 | 6855.00 | 6923.97 | 6858.16 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-08-18 09:15:00 | 7883.25 | 7344.47 | 7241.57 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-08-22 11:15:00 | 7964.90 | 7483.55 | 7330.46 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2025-11-20 09:15:00 | 7455.50 | 7636.15 | 7636.88 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2025-12-17 13:15:00 | 6926.00 | 7283.71 | 7414.82 | SL hit qty=0.50 sl=6926.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 09:15:00 | 6855.00 | 7195.75 | 7277.80 | SL hit qty=0.50 sl=6855.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-21 12:15:00 | 6845.00 | 7184.48 | 7270.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-21 13:15:00 | 6825.50 | 7180.90 | 7268.67 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-22 09:15:00 | 6890.50 | 7171.16 | 7262.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-22 10:15:00 | 6825.00 | 7167.71 | 7260.27 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-28 09:15:00 | 6854.00 | 7100.88 | 7216.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 10:15:00 | 6883.50 | 7098.71 | 7214.72 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-30 09:15:00 | 6962.00 | 7067.09 | 7191.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:15:00 | 6946.50 | 7065.89 | 7189.88 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 6950.00 | 7064.73 | 7188.68 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-03 09:15:00 | 7034.50 | 7048.49 | 7173.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 7110.50 | 7049.11 | 7172.78 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 7660.00 | 7240.65 | 7239.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 7660.00 | 7240.65 | 7239.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 7660.00 | 7240.65 | 7239.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 7660.00 | 7240.65 | 7239.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 10:15:00 | 7670.00 | 7305.46 | 7273.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 7512.00 | 7544.08 | 7431.95 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-03-13 11:15:00 | 7598.00 | 7544.62 | 7433.34 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-13 12:15:00 | 7516.00 | 7544.34 | 7433.75 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-13 13:15:00 | 7572.50 | 7544.62 | 7434.44 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-13 14:15:00 | 7550.00 | 7544.67 | 7435.02 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 10:15:00 | 7460.50 | 7543.92 | 7436.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 7460.50 | 7543.92 | 7436.28 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-18 09:15:00 | 7571.00 | 7538.59 | 7440.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-18 10:15:00 | 7548.00 | 7538.69 | 7440.80 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-18 11:15:00 | 7558.50 | 7538.88 | 7441.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-18 12:15:00 | 7498.00 | 7538.48 | 7441.67 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-25 10:15:00 | 7573.00 | 7481.17 | 7425.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 11:15:00 | 7588.50 | 7482.24 | 7426.14 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-27 10:15:00 | 7600.50 | 7488.01 | 7430.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:15:00 | 7585.00 | 7488.98 | 7431.50 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 7430.00 | 7490.90 | 7433.90 | SL hit qty=1.00 sl=7430.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 7430.00 | 7490.90 | 7433.90 | SL hit qty=1.00 sl=7430.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-10 11:15:00 | 7553.00 | 7445.70 | 7420.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-10 12:15:00 | 7535.00 | 7446.59 | 7421.03 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-15 09:15:00 | 7649.00 | 7455.12 | 7426.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 10:15:00 | 7656.00 | 7457.12 | 7427.90 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-16 14:15:00 | 7550.00 | 7472.20 | 7437.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 15:15:00 | 7555.00 | 7473.02 | 7437.84 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-10-05 10:15:00 | 5056.30 | 2023-10-12 13:15:00 | 4983.35 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2023-10-06 13:15:00 | 5066.25 | 2023-10-12 13:15:00 | 4983.35 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2023-10-10 11:15:00 | 5067.00 | 2023-10-12 13:15:00 | 4983.35 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2023-12-21 12:15:00 | 5485.00 | 2024-01-23 09:15:00 | 6307.75 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2023-12-21 12:15:00 | 5485.00 | 2024-06-24 15:15:00 | 6259.55 | STOP_HIT | 0.50 | 14.12% |
| BUY | retest2 | 2024-07-03 10:15:00 | 6163.40 | 2024-09-12 14:15:00 | 7087.91 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-07-03 10:15:00 | 6163.40 | 2025-02-17 09:15:00 | 6163.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2025-03-03 10:15:00 | 6127.00 | 2025-03-11 09:15:00 | 6085.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-03-11 15:15:00 | 6134.80 | 2025-03-12 09:15:00 | 6085.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-03-12 14:15:00 | 6140.25 | 2025-03-13 11:15:00 | 6085.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-03-17 12:15:00 | 6154.00 | 2025-04-17 11:15:00 | 7036.50 | STOP_HIT | 1.00 | 14.34% |
| BUY | retest2 | 2025-05-12 10:15:00 | 6926.00 | 2025-08-18 09:15:00 | 7883.25 | PARTIAL | 0.50 | 13.82% |
| BUY | retest2 | 2025-06-04 15:15:00 | 6855.00 | 2025-08-22 11:15:00 | 7964.90 | PARTIAL | 0.50 | 16.19% |
| BUY | retest2 | 2025-05-12 10:15:00 | 6926.00 | 2025-12-17 13:15:00 | 6926.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2025-06-04 15:15:00 | 6855.00 | 2026-01-21 09:15:00 | 6855.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2026-01-28 10:15:00 | 6883.50 | 2026-02-18 12:15:00 | 7660.00 | STOP_HIT | 1.00 | 11.28% |
| BUY | retest2 | 2026-01-30 10:15:00 | 6946.50 | 2026-02-18 12:15:00 | 7660.00 | STOP_HIT | 1.00 | 10.27% |
| BUY | retest2 | 2026-02-03 10:15:00 | 7110.50 | 2026-02-18 12:15:00 | 7660.00 | STOP_HIT | 1.00 | 7.73% |
| BUY | retest2 | 2026-03-25 11:15:00 | 7588.50 | 2026-03-30 09:15:00 | 7430.00 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2026-03-27 11:15:00 | 7585.00 | 2026-03-30 09:15:00 | 7430.00 | STOP_HIT | 1.00 | -2.04% |
