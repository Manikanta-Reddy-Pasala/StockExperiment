# Olectra Greentech Ltd. (OLECTRA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1386.00
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 6.33% / 0.00%
- **Sum % (uncompounded):** 44.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 6.33% | 44.3% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 6.33% | 44.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 4 | 2 | 6.33% | 44.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-16 00:00:00 | 1222.15 | 797.68 | 1136.43 | Stage2 pullback-breakout RSI=59 vol=2.7x ATR=71.04 |
| Stop hit — per-position SL triggered | 2023-08-30 00:00:00 | 1194.30 | 836.54 | 1179.01 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-08-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 00:00:00 | 1248.95 | 840.64 | 1185.67 | Stage2 pullback-breakout RSI=60 vol=1.6x ATR=58.21 |
| Stop hit — per-position SL triggered | 2023-09-13 00:00:00 | 1161.63 | 873.88 | 1208.14 | SL hit (bars_held=9) |

### Cycle 3 — BUY (started 2023-11-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 00:00:00 | 1200.70 | 977.77 | 1167.44 | Stage2 pullback-breakout RSI=59 vol=2.7x ATR=37.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-20 00:00:00 | 1276.11 | 983.07 | 1181.97 | T1 booked 50% @ 1276.11 |
| Stop hit — per-position SL triggered | 2023-11-30 00:00:00 | 1200.70 | 999.57 | 1202.72 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2023-12-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 00:00:00 | 1346.80 | 1044.03 | 1248.37 | Stage2 pullback-breakout RSI=68 vol=7.0x ATR=51.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-05 00:00:00 | 1450.69 | 1060.14 | 1299.13 | T1 booked 50% @ 1450.69 |
| Target hit | 2024-03-04 00:00:00 | 1948.90 | 1329.18 | 1955.74 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-03-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 00:00:00 | 1925.25 | 1392.50 | 1819.64 | Stage2 pullback-breakout RSI=57 vol=2.7x ATR=101.63 |
| Stop hit — per-position SL triggered | 2024-04-12 00:00:00 | 1826.05 | 1439.66 | 1858.65 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-16 00:00:00 | 1222.15 | 2023-08-30 00:00:00 | 1194.30 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest1 | 2023-08-31 00:00:00 | 1248.95 | 2023-09-13 00:00:00 | 1161.63 | STOP_HIT | 1.00 | -6.99% |
| BUY | retest1 | 2023-11-16 00:00:00 | 1200.70 | 2023-11-20 00:00:00 | 1276.11 | PARTIAL | 0.50 | 6.28% |
| BUY | retest1 | 2023-11-16 00:00:00 | 1200.70 | 2023-11-30 00:00:00 | 1200.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-29 00:00:00 | 1346.80 | 2024-01-05 00:00:00 | 1450.69 | PARTIAL | 0.50 | 7.71% |
| BUY | retest1 | 2023-12-29 00:00:00 | 1346.80 | 2024-03-04 00:00:00 | 1948.90 | TARGET_HIT | 0.50 | 44.71% |
| BUY | retest1 | 2024-03-27 00:00:00 | 1925.25 | 2024-04-12 00:00:00 | 1826.05 | STOP_HIT | 1.00 | -5.15% |
