# BHARTIARTL (BHARTIARTL)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1829.60
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
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 2 |
| PENDING | 11 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 1 |
| ENTRY2 | 4 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 2 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 0 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 0
- **Avg / median % per leg:** -1.16% / -0.28%
- **Sum % (uncompounded):** -2.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.16% | -2.3% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.03% | -2.0% |
| SELL @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.28% | -0.3% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.03% | -2.0% |
| retest2 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.28% | -0.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 12:15:00 | 1556.70 | 1600.98 | 1601.00 | EMA200 below EMA400 |

### Cycle 2 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 1583.65 | 1601.30 | 1601.38 | EMA200 below EMA400 |

### Cycle 3 — SELL (started 2025-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 15:15:00 | 1594.05 | 1601.23 | 1601.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 12:15:00 | 1592.80 | 1646.29 | 1634.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 13:15:00 | 1633.95 | 1633.44 | 1629.35 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 13:15:00 | 1633.95 | 1633.44 | 1629.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 1633.95 | 1633.44 | 1629.35 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-03-07 14:15:00 | 1630.00 | 1633.41 | 1629.35 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-03-07 15:15:00 | 1637.95 | 1633.45 | 1629.39 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-03-10 15:15:00 | 1628.00 | 1633.76 | 1629.69 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-03-11 09:15:00 | 1643.20 | 1633.85 | 1629.76 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-03-18 12:15:00 | 1629.85 | 1636.79 | 1631.98 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 14:15:00 | 1630.00 | 1636.63 | 1631.94 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-03-19 10:15:00 | 1634.55 | 1636.55 | 1631.97 | SL hit (close>static) qty=1.00 sl=1634.45 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-19 12:15:00 | 1630.05 | 1636.44 | 1631.96 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-03-19 13:15:00 | 1633.05 | 1636.41 | 1631.97 | ENTRY2 sustain failed after 60m |

### Cycle 4 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 2002.80 | 2056.72 | 2056.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 1997.40 | 2055.12 | 2055.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 10:15:00 | 2022.20 | 2019.76 | 2035.58 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-02-05 09:15:00 | 2008.90 | 2019.88 | 2035.18 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-02-05 10:15:00 | 2020.10 | 2019.88 | 2035.10 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-05 12:15:00 | 1999.70 | 2019.59 | 2034.80 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 14:15:00 | 1993.60 | 2019.01 | 2034.36 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 2034.10 | 2018.71 | 2033.91 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-02-06 11:15:00 | 2034.10 | 2018.71 | 2033.91 | SL hit (close>ema400) qty=1.00 sl=2033.91 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-10 13:15:00 | 2011.20 | 2021.03 | 2033.95 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 15:15:00 | 2011.50 | 2020.85 | 2033.73 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-02-12 11:15:00 | 2019.90 | 2020.25 | 2032.79 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-02-12 12:15:00 | 2020.60 | 2020.25 | 2032.73 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-12 14:15:00 | 2014.50 | 2020.21 | 2032.59 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-02-13 09:15:00 | 2023.30 | 2020.21 | 2032.46 | ENTRY2 sustain failed after 1140m |
| Cross detected — sustain check pending | 2026-02-13 10:15:00 | 2010.40 | 2020.11 | 2032.35 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 12:15:00 | 2009.90 | 2019.86 | 2032.10 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-02-19 09:15:00 | 2015.30 | 2019.84 | 2030.63 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:15:00 | 2005.70 | 2019.65 | 2030.43 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-03-18 14:15:00 | 1630.00 | 2025-03-19 10:15:00 | 1634.55 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-05 14:15:00 | 1993.60 | 2026-02-06 11:15:00 | 2034.10 | STOP_HIT | 1.00 | -2.03% |
