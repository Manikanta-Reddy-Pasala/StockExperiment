# BAJAJ-AUTO (BAJAJ-AUTO)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:15:00 (4997 bars)
- **Last close:** 10711.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 6 |
| ALERT3 | 10 |
| PENDING | 26 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 2 |
| ENTRY2 | 18 |
| PARTIAL | 1 |
| TARGET_HIT | 7 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 13
- **Target hits / Stop hits / Partials:** 7 / 13 / 1
- **Avg / median % per leg:** 2.14% / -1.66%
- **Sum % (uncompounded):** 44.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 6 | 37.5% | 5 | 10 | 1 | 2.23% | 35.7% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| BUY @ 3rd Alert (retest2) | 14 | 4 | 28.6% | 4 | 10 | 0 | 1.48% | 20.7% |
| SELL (all) | 5 | 2 | 40.0% | 2 | 3 | 0 | 1.83% | 9.2% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.85% | -1.8% |
| SELL @ 3rd Alert (retest2) | 4 | 2 | 50.0% | 2 | 2 | 0 | 2.75% | 11.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.38% | 13.2% |
| retest2 (combined) | 18 | 6 | 33.3% | 6 | 12 | 0 | 1.76% | 31.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 12:15:00 | 5209.95 | 4764.31 | 4762.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-19 09:15:00 | 5326.00 | 4997.44 | 4918.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-15 10:15:00 | 8116.65 | 8158.23 | 7665.51 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-03-15 14:15:00 | 8355.10 | 8161.00 | 7676.65 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-15 15:15:00 | 8350.70 | 8162.89 | 7680.01 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 09:15:00 | 8768.24 | 8218.64 | 7743.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-03-27 11:15:00 | 9185.77 | 8388.44 | 7900.21 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 9416.00 | 9500.12 | 9294.01 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-07-18 12:15:00 | 9496.15 | 9500.08 | 9295.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 13:15:00 | 9577.00 | 9500.85 | 9296.43 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-19 13:15:00 | 9448.70 | 9502.87 | 9304.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-07-19 14:15:00 | 9390.65 | 9501.75 | 9304.94 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-07-23 10:15:00 | 9470.10 | 9492.10 | 9309.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 11:15:00 | 9471.60 | 9491.90 | 9310.38 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-07-24 12:15:00 | 9279.50 | 9479.32 | 9311.10 | SL hit (close<static) qty=1.00 sl=9292.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-24 12:15:00 | 9279.50 | 9479.32 | 9311.10 | SL hit (close<static) qty=1.00 sl=9292.95 alert=retest2 |
| Cross detected — sustain check pending | 2024-07-26 09:15:00 | 9429.95 | 9459.25 | 9309.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 10:15:00 | 9465.30 | 9459.31 | 9310.54 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Target hit | 2024-08-23 14:15:00 | 10411.83 | 9689.85 | 9523.27 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 9400.80 | 10773.45 | 10779.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 09:15:00 | 9200.00 | 10073.77 | 10352.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 8775.00 | 8707.60 | 9054.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 9058.85 | 8724.13 | 9039.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 9058.85 | 8724.13 | 9039.52 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-02-03 10:15:00 | 8966.90 | 8726.54 | 9039.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 11:15:00 | 8979.30 | 8729.06 | 9038.86 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-04 13:15:00 | 8998.10 | 8749.74 | 9035.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 14:15:00 | 8915.95 | 8751.39 | 9035.19 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Target hit | 2025-02-28 09:15:00 | 8081.37 | 8619.66 | 8841.36 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-02-28 11:15:00 | 8024.36 | 8608.31 | 8833.45 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 14:15:00 | 8706.50 | 8164.81 | 8163.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 8813.00 | 8176.93 | 8169.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 8444.00 | 8496.44 | 8377.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 8444.00 | 8496.44 | 8377.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 8444.00 | 8496.44 | 8377.55 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-06-13 14:15:00 | 8464.00 | 8493.69 | 8379.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 15:15:00 | 8483.00 | 8493.59 | 8379.61 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 8328.00 | 8498.91 | 8397.96 | SL hit (close<static) qty=1.00 sl=8350.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-27 09:15:00 | 8459.50 | 8462.34 | 8394.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:15:00 | 8453.00 | 8462.24 | 8394.32 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-27 12:15:00 | 8476.50 | 8462.27 | 8395.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 13:15:00 | 8471.00 | 8462.36 | 8395.39 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-27 15:15:00 | 8470.00 | 8462.06 | 8395.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-30 09:15:00 | 8450.00 | 8461.94 | 8396.17 | ENTRY2 sustain failed after 3960m |
| Cross detected — sustain check pending | 2025-06-30 10:15:00 | 8471.50 | 8462.03 | 8396.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-30 11:15:00 | 8451.00 | 8461.92 | 8396.82 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-07-01 10:15:00 | 8336.00 | 8456.86 | 8396.18 | SL hit (close<static) qty=1.00 sl=8350.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-01 10:15:00 | 8336.00 | 8456.86 | 8396.18 | SL hit (close<static) qty=1.00 sl=8350.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-07 10:15:00 | 8465.00 | 8443.38 | 8396.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 11:15:00 | 8456.00 | 8443.51 | 8397.08 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 8367.50 | 8443.38 | 8398.17 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 8312.00 | 8442.08 | 8397.74 | SL hit (close<static) qty=1.00 sl=8350.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 8321.00 | 8363.90 | 8364.00 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 13:15:00 | 8405.50 | 8364.19 | 8364.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 14:15:00 | 8445.00 | 8365.00 | 8364.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 8326.00 | 8365.95 | 8365.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 10:15:00 | 8326.00 | 8365.95 | 8365.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 8326.00 | 8365.95 | 8365.00 | EMA400 retest candle locked (from upside) |

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
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 8245.00 | 8242.48 | 8291.48 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-08-08 13:15:00 | 8203.50 | 8241.65 | 8289.86 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-08 14:15:00 | 8225.00 | 8241.49 | 8289.53 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-11 09:15:00 | 8178.50 | 8240.69 | 8288.65 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 10:15:00 | 8143.00 | 8239.72 | 8287.93 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 8274.00 | 8239.11 | 8286.66 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-08-11 15:15:00 | 8235.00 | 8239.07 | 8286.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-12 09:15:00 | 8293.50 | 8239.62 | 8286.44 | ENTRY2 sustain failed after 1080m |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 8293.50 | 8239.62 | 8286.44 | SL hit (close>ema400) qty=1.00 sl=8286.44 alert=retest1 |
| Cross detected — sustain check pending | 2025-08-12 13:15:00 | 8237.50 | 8240.09 | 8285.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:15:00 | 8196.00 | 8239.65 | 8285.31 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-14 09:15:00 | 8217.00 | 8240.46 | 8283.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 10:15:00 | 8212.00 | 8240.17 | 8283.35 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 8573.00 | 8242.54 | 8283.25 | SL hit (close>static) qty=1.00 sl=8292.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 8573.00 | 8242.54 | 8283.25 | SL hit (close>static) qty=1.00 sl=8292.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 13:15:00 | 8827.00 | 8322.51 | 8321.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 8913.50 | 8469.45 | 8403.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 10:15:00 | 8872.50 | 8904.58 | 8716.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 14:15:00 | 8697.50 | 8892.25 | 8726.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 8697.50 | 8892.25 | 8726.14 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-10-06 14:15:00 | 8789.00 | 8831.02 | 8719.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 15:15:00 | 8794.00 | 8830.66 | 8720.00 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-09 11:15:00 | 8794.00 | 8829.57 | 8728.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 12:15:00 | 8791.00 | 8829.19 | 8728.83 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 8620.00 | 8933.14 | 8850.75 | SL hit (close<static) qty=1.00 sl=8668.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 8620.00 | 8933.14 | 8850.75 | SL hit (close<static) qty=1.00 sl=8668.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-11 11:15:00 | 8866.00 | 8904.44 | 8842.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:15:00 | 8900.00 | 8904.40 | 8842.35 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-18 10:15:00 | 8795.50 | 8978.18 | 8932.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:15:00 | 8800.00 | 8976.41 | 8932.18 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Target hit | 2026-01-05 10:15:00 | 9680.00 | 9116.40 | 9023.03 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-01-07 14:15:00 | 9790.00 | 9207.02 | 9078.25 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 9181.00 | 9336.31 | 9185.82 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-22 09:15:00 | 9291.00 | 9322.83 | 9185.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-22 10:15:00 | 9213.50 | 9321.74 | 9185.68 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-22 12:15:00 | 9270.00 | 9320.19 | 9186.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 13:15:00 | 9286.50 | 9319.86 | 9186.75 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 9132.00 | 9662.71 | 9535.67 | SL hit (close<static) qty=1.00 sl=9180.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-18 14:15:00 | 9278.00 | 9498.72 | 9465.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 15:15:00 | 9271.00 | 9496.46 | 9464.85 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-19 09:15:00 | 9079.50 | 9492.31 | 9462.93 | SL hit (close<static) qty=1.00 sl=9180.00 alert=retest2 |

### Cycle 10 — SELL (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 15:15:00 | 9060.00 | 9433.00 | 9434.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 8836.50 | 9427.06 | 9431.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 9446.50 | 9187.25 | 9290.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 9446.50 | 9187.25 | 9290.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 9446.50 | 9187.25 | 9290.88 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 10:15:00 | 9769.50 | 9371.94 | 9371.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 14:15:00 | 9817.00 | 9399.20 | 9385.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 9374.00 | 9488.95 | 9440.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 9374.00 | 9488.95 | 9440.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 9374.00 | 9488.95 | 9440.08 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-04-30 10:15:00 | 9678.00 | 9490.83 | 9441.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 11:15:00 | 9757.00 | 9493.48 | 9442.84 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2026-05-07 09:15:00 | 10732.70 | 9644.39 | 9528.97 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-03-15 15:15:00 | 8350.70 | 2024-03-20 09:15:00 | 8768.24 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-03-15 15:15:00 | 8350.70 | 2024-03-27 11:15:00 | 9185.77 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-07-18 13:15:00 | 9577.00 | 2024-07-24 12:15:00 | 9279.50 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2024-07-23 11:15:00 | 9471.60 | 2024-07-24 12:15:00 | 9279.50 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-07-26 10:15:00 | 9465.30 | 2024-08-23 14:15:00 | 10411.83 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-03 11:15:00 | 8979.30 | 2025-02-28 09:15:00 | 8081.37 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-04 14:15:00 | 8915.95 | 2025-02-28 11:15:00 | 8024.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-13 15:15:00 | 8483.00 | 2025-06-20 09:15:00 | 8328.00 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-06-27 10:15:00 | 8453.00 | 2025-07-01 10:15:00 | 8336.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-06-27 13:15:00 | 8471.00 | 2025-07-01 10:15:00 | 8336.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-07-07 11:15:00 | 8456.00 | 2025-07-08 10:15:00 | 8312.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest1 | 2025-08-11 10:15:00 | 8143.00 | 2025-08-12 09:15:00 | 8293.50 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-08-12 14:15:00 | 8196.00 | 2025-08-18 09:15:00 | 8573.00 | STOP_HIT | 1.00 | -4.60% |
| SELL | retest2 | 2025-08-14 10:15:00 | 8212.00 | 2025-08-18 09:15:00 | 8573.00 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest2 | 2025-10-06 15:15:00 | 8794.00 | 2025-11-07 09:15:00 | 8620.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-10-09 12:15:00 | 8791.00 | 2025-11-07 09:15:00 | 8620.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-11-11 12:15:00 | 8900.00 | 2026-01-05 10:15:00 | 9680.00 | TARGET_HIT | 1.00 | 8.76% |
| BUY | retest2 | 2025-12-18 11:15:00 | 8800.00 | 2026-01-07 14:15:00 | 9790.00 | TARGET_HIT | 1.00 | 11.25% |
| BUY | retest2 | 2026-01-22 13:15:00 | 9286.50 | 2026-03-12 09:15:00 | 9132.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2026-03-18 15:15:00 | 9271.00 | 2026-03-19 09:15:00 | 9079.50 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2026-04-30 11:15:00 | 9757.00 | 2026-05-07 09:15:00 | 10732.70 | TARGET_HIT | 1.00 | 10.00% |
