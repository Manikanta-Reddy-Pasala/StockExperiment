# Kirloskar Oil Eng Ltd. (KIRLOSENG)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2024-06-07 15:25:00 (1446 bars)
- **Last close:** 1245.00
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
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 2
- **Avg / median % per leg:** 0.45% / 0.72%
- **Sum % (uncompounded):** 1.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.54% | 1.1% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.54% | 1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.36% | 0.7% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.36% | 0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 2 | 2 | 0.45% | 1.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:40:00 | 1213.25 | 1205.46 | 0.00 | ORB-long ORB[1195.00,1211.30] vol=1.8x ATR=8.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 10:05:00 | 1226.39 | 1213.34 | 0.00 | T1 1.5R @ 1226.39 |
| Stop hit — per-position SL triggered | 2024-05-16 12:40:00 | 1213.25 | 1219.09 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:00:00 | 1184.85 | 1199.34 | 0.00 | ORB-short ORB[1201.40,1218.00] vol=2.3x ATR=5.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 10:20:00 | 1176.33 | 1192.45 | 0.00 | T1 1.5R @ 1176.33 |
| Stop hit — per-position SL triggered | 2024-05-31 10:25:00 | 1184.85 | 1191.67 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 09:40:00 | 1213.25 | 2024-05-16 10:05:00 | 1226.39 | PARTIAL | 0.50 | 1.08% |
| BUY | retest1 | 2024-05-16 09:40:00 | 1213.25 | 2024-05-16 12:40:00 | 1213.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-31 10:00:00 | 1184.85 | 2024-05-31 10:20:00 | 1176.33 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2024-05-31 10:00:00 | 1184.85 | 2024-05-31 10:25:00 | 1184.85 | STOP_HIT | 0.50 | 0.00% |
