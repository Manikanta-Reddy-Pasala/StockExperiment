# M&M (M&M.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4996 bars)
- **Last close:** 3300.80
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 6 |
| PENDING | 14 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 2 |
| ENTRY2 | 9 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 6 / 7
- **Target hits / Stop hits / Partials:** 1 / 8 / 4
- **Avg / median % per leg:** 6.93% / -1.00%
- **Sum % (uncompounded):** 90.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 4 | 40.0% | 1 | 7 | 2 | 6.28% | 62.8% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.17% | -4.3% |
| BUY @ 3rd Alert (retest2) | 8 | 4 | 50.0% | 1 | 5 | 2 | 8.40% | 67.2% |
| SELL (all) | 3 | 2 | 66.7% | 0 | 1 | 2 | 9.09% | 27.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 2 | 66.7% | 0 | 1 | 2 | 9.09% | 27.3% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.17% | -4.3% |
| retest2 (combined) | 11 | 6 | 54.5% | 1 | 6 | 4 | 8.58% | 94.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 11:15:00 | 1456.25 | 1538.08 | 1538.23 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 15:15:00 | 1559.75 | 1534.74 | 1534.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 09:15:00 | 1563.05 | 1537.33 | 1536.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-21 11:15:00 | 1631.80 | 1632.77 | 1596.81 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2023-12-26 09:15:00 | 1664.65 | 1633.44 | 1599.24 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 10:15:00 | 1662.05 | 1633.72 | 1599.56 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 11:15:00 | 1620.10 | 1652.63 | 1620.38 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-01-08 11:15:00 | 1620.38 | 1652.63 | 1620.38 | SL hit qty=1.00 sl=1620.38 alert=retest1 |
| Cross detected — sustain check pending | 2024-01-11 09:15:00 | 1638.50 | 1648.18 | 1620.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-11 10:15:00 | 1632.40 | 1648.02 | 1621.01 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-11 12:15:00 | 1640.95 | 1647.81 | 1621.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-11 13:15:00 | 1630.90 | 1647.64 | 1621.23 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-15 15:15:00 | 1635.55 | 1644.19 | 1621.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-16 09:15:00 | 1630.45 | 1644.05 | 1621.50 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2024-01-19 12:15:00 | 1645.70 | 1638.15 | 1620.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 13:15:00 | 1652.95 | 1638.30 | 1621.07 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-23 09:15:00 | 1620.00 | 1638.61 | 1621.48 | SL hit qty=1.00 sl=1620.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-01-25 15:15:00 | 1635.50 | 1634.60 | 1620.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 09:15:00 | 1638.50 | 1634.64 | 1621.07 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 5400m) |
| Cross detected — sustain check pending | 2024-01-29 11:15:00 | 1638.65 | 1634.65 | 1621.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 12:15:00 | 1638.80 | 1634.69 | 1621.30 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-30 11:15:00 | 1643.20 | 1635.00 | 1621.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-30 12:15:00 | 1636.40 | 1635.01 | 1621.92 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 1621.30 | 1634.83 | 1621.96 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-01-30 14:15:00 | 1620.00 | 1634.83 | 1621.96 | SL hit qty=1.00 sl=1620.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-30 14:15:00 | 1620.00 | 1634.83 | 1621.96 | SL hit qty=1.00 sl=1620.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-30 14:15:00 | 1620.00 | 1634.83 | 1621.96 | SL hit qty=1.00 sl=1620.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-01-31 09:15:00 | 1647.70 | 1634.80 | 1622.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 10:15:00 | 1649.00 | 1634.94 | 1622.21 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-02-22 14:15:00 | 1896.35 | 1716.96 | 1675.38 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2024-04-30 09:15:00 | 2143.70 | 2001.80 | 1919.43 | Target hit (30%) qty=0.50 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 2782.70 | 2975.60 | 2975.69 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 14:15:00 | 3174.00 | 2973.74 | 2973.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 09:15:00 | 3189.00 | 2977.81 | 2975.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-12 09:15:00 | 2978.55 | 3037.60 | 3009.41 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 09:15:00 | 2978.55 | 3037.60 | 3009.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 2978.55 | 3037.60 | 3009.41 | EMA400 retest candle locked |

### Cycle 5 — SELL (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 12:15:00 | 2756.00 | 2987.75 | 2987.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 09:15:00 | 2695.00 | 2967.89 | 2977.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 13:15:00 | 2788.40 | 2782.02 | 2855.77 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 09:15:00 | 2859.20 | 2787.51 | 2852.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 2859.20 | 2787.51 | 2852.59 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-03-21 14:15:00 | 2800.80 | 2790.33 | 2852.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 15:15:00 | 2786.00 | 2790.29 | 2852.09 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 2862.45 | 2695.57 | 2760.52 | SL hit qty=1.00 sl=2862.45 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 14:15:00 | 3071.40 | 2808.08 | 2806.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 3097.30 | 2816.09 | 2811.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 2977.50 | 2983.44 | 2922.72 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-06-02 10:15:00 | 3018.40 | 2983.46 | 2925.10 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 11:15:00 | 3017.40 | 2983.80 | 2925.56 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2989.30 | 3020.92 | 2962.15 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 2962.15 | 3020.92 | 2962.15 | SL hit qty=1.00 sl=2962.15 alert=retest1 |
| Cross detected — sustain check pending | 2025-06-13 13:15:00 | 3003.30 | 3019.86 | 2962.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 14:15:00 | 3006.00 | 3019.72 | 2963.00 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-09-04 09:15:00 | 3456.90 | 3271.11 | 3209.02 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 09:15:00 | 3380.00 | 3618.44 | 3619.39 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 7 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 3380.00 | 3618.44 | 3619.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 10:15:00 | 3349.00 | 3615.76 | 3618.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 3592.10 | 3575.65 | 3596.00 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 3592.10 | 3575.65 | 3596.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 3592.10 | 3575.65 | 3596.00 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-16 09:15:00 | 3529.40 | 3590.36 | 3599.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 10:15:00 | 3510.60 | 3589.57 | 3598.81 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-19 09:15:00 | 3502.40 | 3572.32 | 3588.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:15:00 | 3478.70 | 3571.39 | 3588.30 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-13 09:15:00 | 2984.01 | 3394.46 | 3477.41 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-13 12:15:00 | 2956.89 | 3381.99 | 3469.89 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-12-26 10:15:00 | 1662.05 | 2024-01-08 11:15:00 | 1620.38 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2024-01-19 13:15:00 | 1652.95 | 2024-01-23 09:15:00 | 1620.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-01-29 09:15:00 | 1638.50 | 2024-01-30 14:15:00 | 1620.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-01-29 12:15:00 | 1638.80 | 2024-01-30 14:15:00 | 1620.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-01-30 12:15:00 | 1636.40 | 2024-01-30 14:15:00 | 1620.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-01-31 10:15:00 | 1649.00 | 2024-02-22 14:15:00 | 1896.35 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-01-31 10:15:00 | 1649.00 | 2024-04-30 09:15:00 | 2143.70 | TARGET_HIT | 0.50 | 30.00% |
| SELL | retest2 | 2025-03-21 15:15:00 | 2786.00 | 2025-04-23 09:15:00 | 2862.45 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest1 | 2025-06-02 11:15:00 | 3017.40 | 2025-06-13 09:15:00 | 2962.15 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-06-13 14:15:00 | 3006.00 | 2025-09-04 09:15:00 | 3456.90 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-06-13 14:15:00 | 3006.00 | 2026-01-29 09:15:00 | 3380.00 | STOP_HIT | 0.50 | 12.44% |
| SELL | retest2 | 2026-02-16 10:15:00 | 3510.60 | 2026-03-13 09:15:00 | 2984.01 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-02-19 10:15:00 | 3478.70 | 2026-03-13 12:15:00 | 2956.89 | PARTIAL | 0.50 | 15.00% |
