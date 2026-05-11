# IRCON International Ltd. (IRCON)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 158.99
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 9 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 10
- **Target hits / Stop hits / Partials:** 1 / 15 / 6
- **Avg / median % per leg:** 0.23% / 0.85%
- **Sum % (uncompounded):** 5.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 22 | 12 | 54.5% | 1 | 15 | 6 | 0.23% | 5.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 12 | 54.5% | 1 | 15 | 6 | 0.23% | 5.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 22 | 12 | 54.5% | 1 | 15 | 6 | 0.23% | 5.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 14:15:00 | 164.90 | 183.40 | 183.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 164.50 | 183.21 | 183.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 12:15:00 | 177.67 | 173.06 | 177.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 12:15:00 | 177.67 | 173.06 | 177.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 177.67 | 173.06 | 177.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:00:00 | 177.67 | 173.06 | 177.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 175.35 | 173.08 | 177.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 10:30:00 | 172.30 | 173.14 | 176.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:15:00 | 173.75 | 172.44 | 176.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:00:00 | 173.74 | 172.45 | 176.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:45:00 | 173.20 | 172.46 | 176.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 175.02 | 172.53 | 176.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 175.11 | 172.53 | 176.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 185.63 | 172.48 | 175.66 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 185.63 | 172.48 | 175.66 | SL hit (close>static) qty=1.00 sl=177.74 alert=retest2 |

### Cycle 2 — BUY (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 15:15:00 | 177.40 | 166.17 | 166.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 09:15:00 | 178.04 | 166.29 | 166.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 09:15:00 | 166.76 | 167.89 | 167.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 166.76 | 167.89 | 167.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 166.76 | 167.89 | 167.06 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 160.85 | 166.35 | 166.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 158.47 | 166.21 | 166.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 14:15:00 | 164.17 | 162.86 | 164.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 15:15:00 | 164.60 | 162.88 | 164.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 164.60 | 162.88 | 164.43 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 160.55 | 144.72 | 144.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 12:15:00 | 163.05 | 146.62 | 145.66 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-03 10:30:00 | 172.30 | 2025-09-15 09:15:00 | 185.63 | STOP_HIT | 1.00 | -7.74% |
| SELL | retest2 | 2025-09-09 10:15:00 | 173.75 | 2025-09-15 09:15:00 | 185.63 | STOP_HIT | 1.00 | -6.84% |
| SELL | retest2 | 2025-09-09 11:00:00 | 173.74 | 2025-09-15 09:15:00 | 185.63 | STOP_HIT | 1.00 | -6.84% |
| SELL | retest2 | 2025-09-09 11:45:00 | 173.20 | 2025-09-15 09:15:00 | 185.63 | STOP_HIT | 1.00 | -7.18% |
| SELL | retest2 | 2025-09-15 11:15:00 | 182.48 | 2025-09-17 14:15:00 | 187.34 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-09-16 09:15:00 | 182.40 | 2025-09-17 14:15:00 | 187.34 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-09-16 10:30:00 | 182.10 | 2025-09-17 14:15:00 | 187.34 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-09-22 13:45:00 | 182.20 | 2025-09-25 14:15:00 | 173.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 13:45:00 | 182.20 | 2025-10-07 12:15:00 | 176.31 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2025-09-24 11:15:00 | 177.00 | 2025-10-07 14:15:00 | 182.70 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2025-09-25 09:30:00 | 176.23 | 2025-10-07 14:15:00 | 182.70 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2025-10-09 09:30:00 | 176.78 | 2025-10-17 13:15:00 | 168.18 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2025-10-09 10:00:00 | 177.03 | 2025-10-20 10:15:00 | 167.94 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2025-10-31 09:45:00 | 171.35 | 2025-11-06 15:15:00 | 162.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 09:30:00 | 170.97 | 2025-11-07 09:15:00 | 162.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-09 09:30:00 | 176.78 | 2025-11-17 09:15:00 | 169.90 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest2 | 2025-10-09 10:00:00 | 177.03 | 2025-11-17 09:15:00 | 169.90 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2025-10-31 09:45:00 | 171.35 | 2025-11-17 09:15:00 | 169.90 | STOP_HIT | 0.50 | 0.85% |
| SELL | retest2 | 2025-11-03 09:30:00 | 170.97 | 2025-11-17 09:15:00 | 169.90 | STOP_HIT | 0.50 | 0.63% |
| SELL | retest2 | 2025-11-17 11:00:00 | 169.30 | 2025-11-24 09:15:00 | 160.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 11:00:00 | 169.30 | 2025-12-05 09:15:00 | 152.37 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-23 13:30:00 | 171.46 | 2025-12-26 09:15:00 | 177.98 | STOP_HIT | 1.00 | -3.80% |
