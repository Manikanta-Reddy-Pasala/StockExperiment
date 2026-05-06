# BAJAJ-AUTO (BAJAJ-AUTO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4997 bars)
- **Last close:** 10319.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 6 |
| ALERT3 | 10 |
| PENDING | 25 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 2 |
| ENTRY2 | 18 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 11 / 13
- **Target hits / Stop hits / Partials:** 2 / 17 / 5
- **Avg / median % per leg:** 5.14% / -0.01%
- **Sum % (uncompounded):** 123.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 7 | 41.2% | 2 | 12 | 3 | 5.41% | 92.0% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 22.50% | 45.0% |
| BUY @ 3rd Alert (retest2) | 15 | 5 | 33.3% | 1 | 12 | 2 | 3.13% | 47.0% |
| SELL (all) | 7 | 4 | 57.1% | 0 | 5 | 2 | 4.50% | 31.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.76% | -1.8% |
| SELL @ 3rd Alert (retest2) | 6 | 4 | 66.7% | 0 | 4 | 2 | 5.54% | 33.2% |
| retest1 (combined) | 3 | 2 | 66.7% | 1 | 1 | 1 | 14.41% | 43.2% |
| retest2 (combined) | 21 | 9 | 42.9% | 1 | 16 | 4 | 3.82% | 80.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 11:15:00 | 5091.95 | 4732.65 | 4732.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 12:15:00 | 5100.50 | 4736.31 | 4734.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 15:15:00 | 4906.05 | 4911.66 | 4842.00 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2023-10-05 09:15:00 | 4954.95 | 4912.09 | 4842.57 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-05 10:15:00 | 4962.05 | 4912.58 | 4843.16 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2023-11-21 09:15:00 | 5706.36 | 5349.58 | 5186.03 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Target hit — 30% from entry | 2023-12-18 09:15:00 | 6450.66 | 5906.81 | 5613.43 | Target hit (30%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 9416.00 | 9500.12 | 9294.00 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-07-18 12:15:00 | 9496.15 | 9500.08 | 9295.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 13:15:00 | 9577.00 | 9500.85 | 9296.42 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-19 13:15:00 | 9448.70 | 9502.87 | 9304.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-07-19 14:15:00 | 9390.65 | 9501.75 | 9304.93 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-07-22 09:15:00 | 9292.95 | 9499.09 | 9305.55 | SL hit qty=1.00 sl=9292.95 alert=retest2 |
| Cross detected — sustain check pending | 2024-07-23 10:15:00 | 9470.10 | 9492.10 | 9309.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 11:15:00 | 9471.60 | 9491.90 | 9310.37 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-07-23 12:15:00 | 9292.95 | 9490.05 | 9310.35 | SL hit qty=1.00 sl=9292.95 alert=retest2 |
| Cross detected — sustain check pending | 2024-07-26 09:15:00 | 9429.95 | 9459.25 | 9309.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 10:15:00 | 9465.30 | 9459.31 | 9310.53 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-08-30 14:15:00 | 10885.09 | 9968.21 | 9700.05 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2024-09-23 14:15:00 | 12304.89 | 11037.10 | 10466.61 | Target hit (30%) qty=0.50 alert=retest2 |

### Cycle 2 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 9400.80 | 10773.45 | 10779.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 09:15:00 | 9200.00 | 10073.77 | 10352.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 8775.00 | 8707.60 | 9054.50 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 9058.85 | 8724.13 | 9039.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 9058.85 | 8724.13 | 9039.52 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-03 10:15:00 | 8966.90 | 8726.54 | 9039.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 11:15:00 | 8979.30 | 8729.06 | 9038.86 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-04 13:15:00 | 8998.10 | 8749.74 | 9035.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 14:15:00 | 8915.95 | 8751.39 | 9035.19 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-03-04 09:15:00 | 7632.40 | 8516.26 | 8773.07 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-03-04 09:15:00 | 7578.56 | 8516.26 | 8773.07 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-22 14:15:00 | 8706.50 | 8164.81 | 8163.11 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-22 14:15:00 | 8706.50 | 8164.81 | 8163.11 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 14:15:00 | 8706.50 | 8164.81 | 8163.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 8813.00 | 8176.93 | 8169.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 8444.00 | 8496.44 | 8377.55 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 8444.00 | 8496.44 | 8377.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 8444.00 | 8496.44 | 8377.55 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-13 14:15:00 | 8464.00 | 8493.69 | 8379.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 15:15:00 | 8483.00 | 8493.59 | 8379.61 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 8350.00 | 8498.91 | 8397.96 | SL hit qty=1.00 sl=8350.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-27 09:15:00 | 8459.50 | 8462.34 | 8394.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:15:00 | 8453.00 | 8462.24 | 8394.32 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-27 12:15:00 | 8476.50 | 8462.27 | 8395.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 13:15:00 | 8471.00 | 8462.36 | 8395.39 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-27 15:15:00 | 8470.00 | 8462.06 | 8395.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-30 09:15:00 | 8450.00 | 8461.94 | 8396.17 | ENTRY2 sustain failed after 3960m |
| Cross detected — sustain check pending | 2025-06-30 10:15:00 | 8471.50 | 8462.03 | 8396.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-30 11:15:00 | 8451.00 | 8461.92 | 8396.82 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 8350.00 | 8458.08 | 8396.48 | SL hit qty=1.00 sl=8350.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 8350.00 | 8458.08 | 8396.48 | SL hit qty=1.00 sl=8350.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-07 10:15:00 | 8465.00 | 8443.38 | 8396.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 11:15:00 | 8456.00 | 8443.51 | 8397.08 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 8367.50 | 8443.38 | 8398.17 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 8350.00 | 8442.08 | 8397.74 | SL hit qty=1.00 sl=8350.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 8321.00 | 8363.90 | 8364.00 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 13:15:00 | 8405.50 | 8364.19 | 8364.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 14:15:00 | 8445.00 | 8365.00 | 8364.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 8326.00 | 8365.95 | 8365.00 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 10:15:00 | 8326.00 | 8365.95 | 8365.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 8326.00 | 8365.95 | 8365.00 | EMA400 retest candle locked |

### Cycle 6 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 8303.00 | 8363.81 | 8363.94 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 13:15:00 | 8394.00 | 8364.29 | 8364.17 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 14:15:00 | 8285.50 | 8363.45 | 8363.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 8062.50 | 8359.72 | 8361.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 8245.00 | 8242.48 | 8291.48 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-08-08 13:15:00 | 8203.50 | 8241.65 | 8289.86 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-08 14:15:00 | 8225.00 | 8241.49 | 8289.53 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-11 09:15:00 | 8178.50 | 8240.69 | 8288.65 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 10:15:00 | 8143.00 | 8239.72 | 8287.93 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 8274.00 | 8239.11 | 8286.66 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 8286.66 | 8239.11 | 8286.66 | SL hit qty=1.00 sl=8286.66 alert=retest1 |
| Cross detected — sustain check pending | 2025-08-11 15:15:00 | 8235.00 | 8239.07 | 8286.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-12 09:15:00 | 8293.50 | 8239.62 | 8286.44 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-08-12 13:15:00 | 8237.50 | 8240.09 | 8285.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:15:00 | 8196.00 | 8239.65 | 8285.31 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-13 10:15:00 | 8292.00 | 8239.70 | 8284.65 | SL hit qty=1.00 sl=8292.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-14 09:15:00 | 8217.00 | 8240.46 | 8283.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 10:15:00 | 8212.00 | 8240.17 | 8283.35 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 8292.00 | 8242.54 | 8283.25 | SL hit qty=1.00 sl=8292.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 13:15:00 | 8827.00 | 8322.51 | 8321.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 8913.50 | 8469.45 | 8403.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 10:15:00 | 8872.50 | 8904.58 | 8716.31 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 14:15:00 | 8697.50 | 8892.25 | 8726.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 8697.50 | 8892.25 | 8726.14 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-06 14:15:00 | 8789.00 | 8831.02 | 8719.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 15:15:00 | 8794.00 | 8830.66 | 8720.00 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-09 11:15:00 | 8794.00 | 8829.57 | 8728.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 12:15:00 | 8791.00 | 8829.19 | 8728.83 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 8668.00 | 8933.14 | 8850.75 | SL hit qty=1.00 sl=8668.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 8668.00 | 8933.14 | 8850.75 | SL hit qty=1.00 sl=8668.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-11 11:15:00 | 8866.00 | 8904.44 | 8842.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:15:00 | 8900.00 | 8904.40 | 8842.35 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-18 10:15:00 | 8795.50 | 8978.18 | 8932.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:15:00 | 8800.00 | 8976.41 | 8932.18 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 8816.50 | 8974.82 | 8931.61 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-19 09:15:00 | 8884.50 | 8969.28 | 8929.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 10:15:00 | 8943.00 | 8969.02 | 8929.73 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-02-25 10:15:00 | 10120.00 | 9648.48 | 9468.33 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 8800.00 | 9570.49 | 9497.22 | SL hit qty=0.50 sl=8800.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 8799.00 | 9570.49 | 9497.22 | SL hit qty=1.00 sl=8799.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-16 11:15:00 | 8925.00 | 9564.07 | 9494.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 12:15:00 | 8950.00 | 9557.96 | 9491.65 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-20 15:15:00 | 9060.00 | 9433.00 | 9434.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 15:15:00 | 9060.00 | 9433.00 | 9434.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 15:15:00 | 9060.00 | 9433.00 | 9434.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 8836.50 | 9427.06 | 9431.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 9446.50 | 9187.25 | 9290.88 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 9446.50 | 9187.25 | 9290.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 9446.50 | 9187.25 | 9290.88 | EMA400 retest candle locked |

### Cycle 11 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 10:15:00 | 9769.50 | 9371.94 | 9371.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 14:15:00 | 9817.00 | 9399.20 | 9385.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 9374.00 | 9488.95 | 9440.08 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 9374.00 | 9488.95 | 9440.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 9374.00 | 9488.95 | 9440.08 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-30 10:15:00 | 9678.00 | 9490.83 | 9441.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 11:15:00 | 9757.00 | 9493.48 | 9442.84 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-05 10:15:00 | 4962.05 | 2023-11-21 09:15:00 | 5706.36 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2023-10-05 10:15:00 | 4962.05 | 2023-12-18 09:15:00 | 6450.66 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2024-07-18 13:15:00 | 9577.00 | 2024-07-22 09:15:00 | 9292.95 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2024-07-23 11:15:00 | 9471.60 | 2024-07-23 12:15:00 | 9292.95 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-07-26 10:15:00 | 9465.30 | 2024-08-30 14:15:00 | 10885.09 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-07-26 10:15:00 | 9465.30 | 2024-09-23 14:15:00 | 12304.89 | TARGET_HIT | 0.50 | 30.00% |
| SELL | retest2 | 2025-02-03 11:15:00 | 8979.30 | 2025-03-04 09:15:00 | 7632.40 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-02-04 14:15:00 | 8915.95 | 2025-03-04 09:15:00 | 7578.56 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-02-03 11:15:00 | 8979.30 | 2025-05-22 14:15:00 | 8706.50 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2025-02-04 14:15:00 | 8915.95 | 2025-05-22 14:15:00 | 8706.50 | STOP_HIT | 0.50 | 2.35% |
| BUY | retest2 | 2025-06-13 15:15:00 | 8483.00 | 2025-06-20 09:15:00 | 8350.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-06-27 10:15:00 | 8453.00 | 2025-07-01 09:15:00 | 8350.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-06-27 13:15:00 | 8471.00 | 2025-07-01 09:15:00 | 8350.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-07-07 11:15:00 | 8456.00 | 2025-07-08 10:15:00 | 8350.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest1 | 2025-08-11 10:15:00 | 8143.00 | 2025-08-11 14:15:00 | 8286.66 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-08-12 14:15:00 | 8196.00 | 2025-08-13 10:15:00 | 8292.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-08-14 10:15:00 | 8212.00 | 2025-08-18 09:15:00 | 8292.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-10-06 15:15:00 | 8794.00 | 2025-11-07 09:15:00 | 8668.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-10-09 12:15:00 | 8791.00 | 2025-11-07 09:15:00 | 8668.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-11-11 12:15:00 | 8900.00 | 2026-02-25 10:15:00 | 10120.00 | PARTIAL | 0.50 | 13.71% |
| BUY | retest2 | 2025-11-11 12:15:00 | 8900.00 | 2026-03-16 10:15:00 | 8800.00 | STOP_HIT | 0.50 | -1.12% |
| BUY | retest2 | 2025-12-18 11:15:00 | 8800.00 | 2026-03-16 10:15:00 | 8799.00 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-12-19 10:15:00 | 8943.00 | 2026-03-20 15:15:00 | 9060.00 | STOP_HIT | 1.00 | 1.31% |
| BUY | retest2 | 2026-03-16 12:15:00 | 8950.00 | 2026-03-20 15:15:00 | 9060.00 | STOP_HIT | 1.00 | 1.23% |
