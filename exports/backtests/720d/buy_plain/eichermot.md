# EICHERMOT (EICHERMOT)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 7342.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 3 |
| PENDING | 10 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 5 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 4
- **Target hits / Stop hits / Partials:** 1 / 8 / 1
- **Avg / median % per leg:** 4.17% / 0.54%
- **Sum % (uncompounded):** 41.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 6 | 60.0% | 1 | 8 | 1 | 4.17% | 41.7% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 4 | 0 | 0.58% | 2.3% |
| BUY @ 3rd Alert (retest2) | 6 | 3 | 50.0% | 1 | 4 | 1 | 6.56% | 39.4% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 4 | 0 | 0.58% | 2.3% |
| retest2 (combined) | 6 | 3 | 50.0% | 1 | 4 | 1 | 6.56% | 39.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 5003.75 | 4808.13 | 4807.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 5014.20 | 4829.86 | 4819.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 4820.80 | 4851.30 | 4831.65 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 10:15:00 | 4820.80 | 4851.30 | 4831.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 4820.80 | 4851.30 | 4831.65 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-06 09:15:00 | 4864.50 | 4841.25 | 4829.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 10:15:00 | 4895.30 | 4841.78 | 4830.10 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-11 13:15:00 | 4803.45 | 4840.14 | 4830.63 | SL hit (close<static) qty=1.00 sl=4804.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-27 09:15:00 | 4935.45 | 4810.37 | 4816.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 10:15:00 | 4883.95 | 4811.10 | 4816.44 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-27 12:15:00 | 4877.55 | 4812.14 | 4816.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 13:15:00 | 4880.35 | 4812.81 | 4817.22 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-01 09:15:00 | 4784.85 | 4819.42 | 4820.36 | SL hit (close<static) qty=1.00 sl=4804.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-01 09:15:00 | 4784.85 | 4819.42 | 4820.36 | SL hit (close<static) qty=1.00 sl=4804.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-01 13:15:00 | 4882.05 | 4819.95 | 4820.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 14:15:00 | 4886.30 | 4820.61 | 4820.93 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-02 09:15:00 | 4897.40 | 4821.83 | 4821.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 09:15:00 | 4897.40 | 4821.83 | 4821.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 10:15:00 | 4960.10 | 4823.21 | 4822.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 10:15:00 | 4943.40 | 4960.93 | 4901.05 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-01-14 10:15:00 | 5004.25 | 4960.70 | 4902.99 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-14 11:15:00 | 4992.35 | 4961.01 | 4903.43 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-14 12:15:00 | 5037.10 | 4961.77 | 4904.10 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-14 13:15:00 | 5028.20 | 4962.43 | 4904.72 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-20 11:15:00 | 5015.45 | 4980.82 | 4921.64 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 12:15:00 | 5016.45 | 4981.18 | 4922.12 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-22 12:15:00 | 5006.50 | 4983.86 | 4927.52 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-22 13:15:00 | 5005.40 | 4984.07 | 4927.90 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-23 09:15:00 | 5025.20 | 4984.95 | 4929.18 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:15:00 | 5056.50 | 4985.66 | 4929.82 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 5055.50 | 5184.27 | 5069.78 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 5055.50 | 5184.27 | 5069.78 | SL hit (close<ema400) qty=1.00 sl=5069.78 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 5055.50 | 5184.27 | 5069.78 | SL hit (close<ema400) qty=1.00 sl=5069.78 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 5055.50 | 5184.27 | 5069.78 | SL hit (close<ema400) qty=1.00 sl=5069.78 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 5055.50 | 5184.27 | 5069.78 | SL hit (close<ema400) qty=1.00 sl=5069.78 alert=retest1 |

### Cycle 3 — BUY (started 2025-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 15:15:00 | 5130.00 | 4994.61 | 4993.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 5138.45 | 5007.79 | 5001.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 5104.50 | 5193.17 | 5114.83 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 5104.50 | 5193.17 | 5114.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 5104.50 | 5193.17 | 5114.83 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-08 09:15:00 | 5181.30 | 5186.24 | 5114.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 10:15:00 | 5183.00 | 5186.21 | 5114.34 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-08-18 09:15:00 | 5960.45 | 5601.62 | 5546.32 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Target hit — 30% from entry | 2025-09-08 11:15:00 | 6737.90 | 5994.74 | 5800.26 | Target hit (30%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-12-06 10:15:00 | 4895.30 | 2024-12-11 13:15:00 | 4803.45 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-12-27 10:15:00 | 4883.95 | 2025-01-01 09:15:00 | 4784.85 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-12-27 13:15:00 | 4880.35 | 2025-01-01 09:15:00 | 4784.85 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-01-01 14:15:00 | 4886.30 | 2025-01-02 09:15:00 | 4897.40 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest1 | 2025-01-14 13:15:00 | 5028.20 | 2025-02-11 09:15:00 | 5055.50 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest1 | 2025-01-20 12:15:00 | 5016.45 | 2025-02-11 09:15:00 | 5055.50 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest1 | 2025-01-22 13:15:00 | 5005.40 | 2025-02-11 09:15:00 | 5055.50 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest1 | 2025-01-23 10:15:00 | 5056.50 | 2025-02-11 09:15:00 | 5055.50 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-04-08 10:15:00 | 5183.00 | 2025-08-18 09:15:00 | 5960.45 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-08 10:15:00 | 5183.00 | 2025-09-08 11:15:00 | 6737.90 | TARGET_HIT | 0.50 | 30.00% |
