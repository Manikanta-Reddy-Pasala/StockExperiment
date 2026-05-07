# DIVISLAB (DIVISLAB)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 6685.50
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 4 |
| ALERT3 | 5 |
| PENDING | 13 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 1 |
| ENTRY2 | 7 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / Stop hits / Partials:** 0 / 8 / 0
- **Avg / median % per leg:** -3.28% / -3.93%
- **Sum % (uncompounded):** -26.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 1 | 12.5% | 0 | 8 | 0 | -3.28% | -26.3% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.93% | -3.9% |
| SELL @ 3rd Alert (retest2) | 7 | 1 | 14.3% | 0 | 7 | 0 | -3.19% | -22.3% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.93% | -3.9% |
| retest2 (combined) | 7 | 1 | 14.3% | 0 | 7 | 0 | -3.19% | -22.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 12:15:00 | 5666.35 | 5820.58 | 5821.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 09:15:00 | 5610.45 | 5814.96 | 5818.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 12:15:00 | 5805.15 | 5782.29 | 5800.85 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 12:15:00 | 5805.15 | 5782.29 | 5800.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 5805.15 | 5782.29 | 5800.85 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-28 09:15:00 | 5470.50 | 5845.75 | 5844.44 | ENTRY2 cross detected — sustain check pending (75m) |

### Cycle 2 — SELL (started 2025-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 10:15:00 | 5435.00 | 5841.66 | 5842.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 11:15:00 | 5373.15 | 5837.00 | 5840.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 5737.25 | 5707.01 | 5763.45 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-03-13 14:15:00 | 5619.85 | 5705.38 | 5761.24 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-03-17 09:15:00 | 5650.95 | 5703.94 | 5759.96 | ENTRY1 sustain failed after 5460m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 14:15:00 | 5768.00 | 5703.16 | 5756.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 5768.00 | 5703.16 | 5756.27 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-01 10:15:00 | 5655.30 | 5754.52 | 5772.41 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 12:15:00 | 5562.10 | 5751.08 | 5770.50 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 5825.80 | 5737.29 | 5762.32 | SL hit (close>static) qty=1.00 sl=5773.45 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-04 09:15:00 | 5421.85 | 5735.83 | 5760.75 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 11:15:00 | 5504.80 | 5731.36 | 5758.25 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-04-16 09:15:00 | 5796.50 | 5651.46 | 5708.10 | SL hit (close>static) qty=1.00 sl=5773.45 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-17 12:15:00 | 5693.00 | 5659.77 | 5709.61 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-17 14:15:00 | 5663.50 | 5659.98 | 5709.22 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-04-21 09:15:00 | 5822.50 | 5661.23 | 5709.35 | SL hit (close>static) qty=1.00 sl=5773.45 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 5980.00 | 6491.92 | 6494.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 14:15:00 | 5959.50 | 6462.34 | 6479.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 09:15:00 | 6260.00 | 6145.58 | 6248.23 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 6260.00 | 6145.58 | 6248.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 6260.00 | 6145.58 | 6248.23 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-09-22 09:15:00 | 6139.00 | 6149.55 | 6246.73 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 11:15:00 | 6130.50 | 6149.31 | 6245.64 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-10-08 14:15:00 | 6119.00 | 6008.72 | 6129.07 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-10-09 09:15:00 | 6191.00 | 6011.49 | 6129.27 | ENTRY2 sustain failed after 1140m |
| Cross detected — sustain check pending | 2025-10-09 11:15:00 | 6154.50 | 6014.48 | 6129.60 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-10-09 12:15:00 | 6163.00 | 6015.96 | 6129.76 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-09 13:15:00 | 6142.00 | 6017.21 | 6129.82 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 15:15:00 | 6123.50 | 6019.41 | 6129.81 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-10-10 11:15:00 | 6386.00 | 6025.97 | 6131.46 | SL hit (close>static) qty=1.00 sl=6275.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 11:15:00 | 6386.00 | 6025.97 | 6131.46 | SL hit (close>static) qty=1.00 sl=6275.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-19 10:15:00 | 6150.00 | 6438.96 | 6426.08 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 12:15:00 | 6127.50 | 6432.81 | 6423.12 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-01-20 12:15:00 | 6072.50 | 6411.16 | 6412.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 6072.50 | 6411.16 | 6412.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 6023.50 | 6407.30 | 6410.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 6241.50 | 6230.99 | 6307.35 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-02-04 10:15:00 | 6073.00 | 6228.86 | 6303.28 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-04 12:15:00 | 6049.00 | 6225.37 | 6300.79 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 6286.50 | 6189.93 | 6269.28 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-02-11 11:15:00 | 6286.50 | 6189.93 | 6269.28 | SL hit (close>ema400) qty=1.00 sl=6269.28 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-12 12:15:00 | 6196.00 | 6197.05 | 6269.81 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:15:00 | 6194.00 | 6197.16 | 6269.15 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 6313.50 | 6199.08 | 6259.91 | SL hit (close>static) qty=1.00 sl=6307.50 alert=retest2 |

### Cycle 5 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 6233.00 | 6294.15 | 6294.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 6169.00 | 6292.73 | 6293.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 11:15:00 | 6091.00 | 6061.97 | 6149.44 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 6031.50 | 6063.78 | 6148.20 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-13 10:15:00 | 6060.00 | 6063.74 | 6147.76 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 6127.50 | 6064.95 | 6145.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 6127.50 | 6064.95 | 6145.88 | EMA400 retest candle locked |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-04-01 12:15:00 | 5562.10 | 2025-04-03 09:15:00 | 5825.80 | STOP_HIT | 1.00 | -4.74% |
| SELL | retest2 | 2025-04-04 11:15:00 | 5504.80 | 2025-04-16 09:15:00 | 5796.50 | STOP_HIT | 1.00 | -5.30% |
| SELL | retest2 | 2025-04-17 14:15:00 | 5663.50 | 2025-04-21 09:15:00 | 5822.50 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-09-22 11:15:00 | 6130.50 | 2025-10-10 11:15:00 | 6386.00 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2025-10-09 15:15:00 | 6123.50 | 2025-10-10 11:15:00 | 6386.00 | STOP_HIT | 1.00 | -4.29% |
| SELL | retest2 | 2026-01-19 12:15:00 | 6127.50 | 2026-01-20 12:15:00 | 6072.50 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest1 | 2026-02-04 12:15:00 | 6049.00 | 2026-02-11 11:15:00 | 6286.50 | STOP_HIT | 1.00 | -3.93% |
| SELL | retest2 | 2026-02-12 14:15:00 | 6194.00 | 2026-02-19 09:15:00 | 6313.50 | STOP_HIT | 1.00 | -1.93% |
