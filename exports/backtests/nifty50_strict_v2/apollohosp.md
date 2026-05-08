# APOLLOHOSP (APOLLOHOSP)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4998 bars)
- **Last close:** 8097.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 7 |
| ALERT3 | 7 |
| PENDING | 18 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 0 |
| TARGET_HIT | 5 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 5 / 4
- **Target hits / Stop hits / Partials:** 5 / 4 / 0
- **Avg / median % per leg:** 4.53% / 8.87%
- **Sum % (uncompounded):** 40.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 4 | 2 | 0 | 5.97% | 35.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 4 | 66.7% | 4 | 2 | 0 | 5.97% | 35.8% |
| SELL (all) | 3 | 1 | 33.3% | 1 | 2 | 0 | 1.65% | 4.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 1 | 33.3% | 1 | 2 | 0 | 1.65% | 4.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 5 | 55.6% | 5 | 4 | 0 | 4.53% | 40.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 15:15:00 | 5281.90 | 5028.50 | 5028.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 11:15:00 | 5290.00 | 5060.82 | 5044.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 5400.95 | 5417.85 | 5299.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 09:15:00 | 5439.65 | 5417.91 | 5300.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 5439.65 | 5417.91 | 5300.53 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-12-21 11:15:00 | 5497.90 | 5418.98 | 5302.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 12:15:00 | 5485.00 | 5419.63 | 5303.15 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2024-01-19 13:15:00 | 6033.50 | 5719.12 | 5551.27 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-08 11:15:00 | 5860.50 | 6169.21 | 6170.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-08 12:15:00 | 5841.10 | 6165.94 | 6168.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 14:15:00 | 5961.45 | 5950.10 | 6022.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-07 10:15:00 | 6011.70 | 5951.48 | 6021.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 6011.70 | 5951.48 | 6021.87 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 15:15:00 | 6259.55 | 6066.47 | 6066.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 6292.90 | 6068.72 | 6067.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 6098.00 | 6100.72 | 6085.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 12:15:00 | 6092.60 | 6100.64 | 6085.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 6092.60 | 6100.64 | 6085.51 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-07-03 09:15:00 | 6138.80 | 6100.83 | 6085.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 10:15:00 | 6163.40 | 6101.45 | 6086.29 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2024-08-02 10:15:00 | 6779.74 | 6402.76 | 6283.73 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 15:15:00 | 6767.00 | 7070.06 | 7071.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 6741.20 | 7025.97 | 7047.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 11:15:00 | 6951.25 | 6927.18 | 6986.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 11:15:00 | 6951.25 | 6927.18 | 6986.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 6951.25 | 6927.18 | 6986.08 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-02-06 09:15:00 | 6876.55 | 6928.13 | 6985.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 10:15:00 | 6866.40 | 6927.51 | 6984.51 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2025-02-17 09:15:00 | 6179.76 | 6762.49 | 6882.98 | Target hit (10%) qty=1.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-15 13:15:00 | 6869.50 | 6576.58 | 6598.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-15 14:15:00 | 6905.00 | 6579.84 | 6600.20 | ENTRY2 sustain failed after 60m |

### Cycle 5 — BUY (started 2025-04-17 11:15:00)

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
| Target hit | 2025-07-01 09:15:00 | 7540.50 | 7015.55 | 6941.45 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-07-07 13:15:00 | 7618.60 | 7156.64 | 7028.31 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-11-20 09:15:00)

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

### Cycle 7 — BUY (started 2026-02-18 12:15:00)

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
| BUY | retest2 | 2023-12-21 12:15:00 | 5485.00 | 2024-01-19 13:15:00 | 6033.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-03 10:15:00 | 6163.40 | 2024-08-02 10:15:00 | 6779.74 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-06 10:15:00 | 6866.40 | 2025-02-17 09:15:00 | 6179.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-12 10:15:00 | 6926.00 | 2025-07-01 09:15:00 | 7540.50 | TARGET_HIT | 1.00 | 8.87% |
| BUY | retest2 | 2025-06-04 15:15:00 | 6855.00 | 2025-07-07 13:15:00 | 7618.60 | TARGET_HIT | 1.00 | 11.14% |
| SELL | retest2 | 2026-01-09 14:15:00 | 7254.00 | 2026-01-13 14:15:00 | 7312.50 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-01-16 11:15:00 | 7255.00 | 2026-02-11 09:15:00 | 7563.50 | STOP_HIT | 1.00 | -4.25% |
| BUY | retest2 | 2026-03-25 11:15:00 | 7588.50 | 2026-03-30 09:15:00 | 7427.50 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2026-03-27 11:15:00 | 7585.00 | 2026-03-30 09:15:00 | 7427.50 | STOP_HIT | 1.00 | -2.08% |
