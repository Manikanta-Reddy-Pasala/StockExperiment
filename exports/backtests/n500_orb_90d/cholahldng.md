# Cholamandalam Financial Holdings Ltd. (CHOLAHLDNG)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1785.00
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
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 2
- **Avg / median % per leg:** -0.08% / 0.00%
- **Sum % (uncompounded):** -0.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.13% | -0.5% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.13% | -0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | -0.02% | -0.0% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | -0.02% | -0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.08% | -0.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:45:00 | 1750.00 | 1740.51 | 0.00 | ORB-long ORB[1725.60,1748.50] vol=4.6x ATR=4.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:55:00 | 1756.45 | 1744.22 | 0.00 | T1 1.5R @ 1756.45 |
| Stop hit — per-position SL triggered | 2026-02-17 12:20:00 | 1750.00 | 1747.56 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:10:00 | 1755.30 | 1758.24 | 0.00 | ORB-short ORB[1756.80,1769.90] vol=1.5x ATR=4.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 13:25:00 | 1748.98 | 1755.85 | 0.00 | T1 1.5R @ 1748.98 |
| Stop hit — per-position SL triggered | 2026-02-18 14:50:00 | 1755.30 | 1754.23 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-04-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:05:00 | 1562.30 | 1565.47 | 0.00 | ORB-short ORB[1562.40,1578.00] vol=7.4x ATR=6.37 |
| Stop hit — per-position SL triggered | 2026-04-22 12:00:00 | 1568.67 | 1565.39 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-05-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:35:00 | 1759.80 | 1749.83 | 0.00 | ORB-long ORB[1735.90,1758.50] vol=3.6x ATR=8.26 |
| Stop hit — per-position SL triggered | 2026-05-07 10:45:00 | 1751.54 | 1750.13 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:40:00 | 1786.80 | 1772.49 | 0.00 | ORB-long ORB[1757.00,1778.70] vol=1.9x ATR=7.65 |
| Stop hit — per-position SL triggered | 2026-05-08 09:45:00 | 1779.15 | 1773.75 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 10:45:00 | 1750.00 | 2026-02-17 10:55:00 | 1756.45 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-02-17 10:45:00 | 1750.00 | 2026-02-17 12:20:00 | 1750.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 11:10:00 | 1755.30 | 2026-02-18 13:25:00 | 1748.98 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-18 11:10:00 | 1755.30 | 2026-02-18 14:50:00 | 1755.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-22 10:05:00 | 1562.30 | 2026-04-22 12:00:00 | 1568.67 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-05-07 10:35:00 | 1759.80 | 2026-05-07 10:45:00 | 1751.54 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-05-08 09:40:00 | 1786.80 | 2026-05-08 09:45:00 | 1779.15 | STOP_HIT | 1.00 | -0.43% |
