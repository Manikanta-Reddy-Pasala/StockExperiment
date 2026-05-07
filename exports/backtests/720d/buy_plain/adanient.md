# ADANIENT (ADANIENT)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 2519.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 2 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 1 |
| PENDING | 3 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 3 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** 0.40% / -2.91%
- **Sum % (uncompounded):** 1.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 3 | 0 | 0.40% | 1.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 1 | 33.3% | 0 | 3 | 0 | 0.40% | 1.2% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 3 | 1 | 33.3% | 0 | 3 | 0 | 0.40% | 1.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 14:15:00 | 3042.68 | 2984.22 | 2984.03 | EMA200 above EMA400 |

### Cycle 2 — BUY (started 2025-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 15:15:00 | 2371.36 | 2269.13 | 2268.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 11:15:00 | 2403.06 | 2275.61 | 2272.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 15:15:00 | 2278.29 | 2281.31 | 2275.86 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 2262.10 | 2281.12 | 2275.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 2262.10 | 2281.12 | 2275.79 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 2314.35 | 2272.83 | 2272.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 2343.92 | 2273.54 | 2272.41 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 2236.02 | 2443.03 | 2436.87 | SL hit (close<static) qty=1.00 sl=2244.36 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-19 14:15:00 | 2309.60 | 2341.97 | 2379.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 15:15:00 | 2308.05 | 2341.64 | 2379.26 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-25 13:15:00 | 2240.96 | 2328.90 | 2368.02 | SL hit (close<static) qty=1.00 sl=2244.36 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-11 09:15:00 | 2308.54 | 2272.40 | 2320.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 10:15:00 | 2333.36 | 2273.01 | 2320.70 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-24 14:15:00 | 2536.66 | 2348.59 | 2348.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 14:15:00 | 2536.66 | 2348.59 | 2348.48 | EMA200 above EMA400 |

### Cycle 4 — BUY (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 15:15:00 | 2281.60 | 2094.05 | 2093.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 2309.40 | 2096.19 | 2095.04 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 10:15:00 | 2343.92 | 2025-08-06 10:15:00 | 2236.02 | STOP_HIT | 1.00 | -4.60% |
| BUY | retest2 | 2025-08-19 15:15:00 | 2308.05 | 2025-08-25 13:15:00 | 2240.96 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest2 | 2025-09-11 10:15:00 | 2333.36 | 2025-09-24 14:15:00 | 2536.66 | STOP_HIT | 1.00 | 8.71% |
