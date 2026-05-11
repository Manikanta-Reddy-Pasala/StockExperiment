# Chalet Hotels Ltd. (CHALET)

## Backtest Summary

- **Window:** 2024-01-08 09:15:00 → 2026-05-08 15:25:00 (41842 bars)
- **Last close:** 787.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 7
- **Target hits / Stop hits / Partials:** 0 / 7 / 3
- **Avg / median % per leg:** -0.01% / 0.00%
- **Sum % (uncompounded):** -0.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.52% | -1.6% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.52% | -1.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 3 | 42.9% | 0 | 4 | 3 | 0.20% | 1.4% |
| SELL @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 0 | 4 | 3 | 0.20% | 1.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 3 | 30.0% | 0 | 7 | 3 | -0.01% | -0.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-01-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 09:35:00 | 733.05 | 727.83 | 0.00 | ORB-long ORB[718.40,729.20] vol=4.0x ATR=4.17 |
| Stop hit — per-position SL triggered | 2024-01-19 11:10:00 | 728.88 | 730.37 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-03-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-01 09:30:00 | 811.60 | 815.28 | 0.00 | ORB-short ORB[813.05,822.40] vol=2.0x ATR=4.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-01 09:45:00 | 805.37 | 813.02 | 0.00 | T1 1.5R @ 805.37 |
| Stop hit — per-position SL triggered | 2024-03-01 10:00:00 | 811.60 | 811.93 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-03-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 10:35:00 | 806.25 | 812.28 | 0.00 | ORB-short ORB[810.00,820.15] vol=1.9x ATR=2.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-05 10:45:00 | 802.25 | 811.36 | 0.00 | T1 1.5R @ 802.25 |
| Stop hit — per-position SL triggered | 2024-03-05 10:50:00 | 806.25 | 811.03 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-03-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-19 09:40:00 | 746.80 | 742.03 | 0.00 | ORB-long ORB[734.10,743.85] vol=2.0x ATR=3.55 |
| Stop hit — per-position SL triggered | 2024-03-19 09:50:00 | 743.25 | 742.42 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-03-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 10:15:00 | 821.20 | 816.27 | 0.00 | ORB-long ORB[809.40,819.80] vol=6.9x ATR=4.27 |
| Stop hit — per-position SL triggered | 2024-03-22 10:20:00 | 816.93 | 816.28 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-03-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-26 09:35:00 | 815.00 | 820.50 | 0.00 | ORB-short ORB[817.95,828.95] vol=1.7x ATR=4.04 |
| Stop hit — per-position SL triggered | 2024-03-26 09:50:00 | 819.04 | 819.95 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 10:25:00 | 859.20 | 864.55 | 0.00 | ORB-short ORB[866.10,874.10] vol=1.8x ATR=3.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 11:05:00 | 853.63 | 861.17 | 0.00 | T1 1.5R @ 853.63 |
| Stop hit — per-position SL triggered | 2024-05-09 11:10:00 | 859.20 | 861.16 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-19 09:35:00 | 733.05 | 2024-01-19 11:10:00 | 728.88 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2024-03-01 09:30:00 | 811.60 | 2024-03-01 09:45:00 | 805.37 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2024-03-01 09:30:00 | 811.60 | 2024-03-01 10:00:00 | 811.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-05 10:35:00 | 806.25 | 2024-03-05 10:45:00 | 802.25 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-03-05 10:35:00 | 806.25 | 2024-03-05 10:50:00 | 806.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-19 09:40:00 | 746.80 | 2024-03-19 09:50:00 | 743.25 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-03-22 10:15:00 | 821.20 | 2024-03-22 10:20:00 | 816.93 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-03-26 09:35:00 | 815.00 | 2024-03-26 09:50:00 | 819.04 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-05-09 10:25:00 | 859.20 | 2024-05-09 11:05:00 | 853.63 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-05-09 10:25:00 | 859.20 | 2024-05-09 11:10:00 | 859.20 | STOP_HIT | 0.50 | 0.00% |
