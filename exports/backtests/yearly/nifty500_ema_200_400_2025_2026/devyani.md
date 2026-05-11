# Devyani International Ltd. (DEVYANI)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3162 bars)
- **Last close:** 118.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 12
- **Target hits / Stop hits / Partials:** 0 / 12 / 0
- **Avg / median % per leg:** -1.93% / -1.60%
- **Sum % (uncompounded):** -23.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.93% | -23.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.93% | -23.2% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.93% | -23.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 14:15:00 | 166.28 | 170.15 | 170.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 09:15:00 | 165.82 | 170.07 | 170.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 172.91 | 169.83 | 169.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 172.91 | 169.83 | 169.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 172.91 | 169.83 | 169.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 170.66 | 169.83 | 169.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 173.31 | 169.87 | 170.00 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 12:15:00 | 173.73 | 170.14 | 170.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 175.65 | 170.46 | 170.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 165.48 | 172.27 | 171.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 165.48 | 172.27 | 171.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 165.48 | 172.27 | 171.38 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 161.21 | 170.55 | 170.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 14:15:00 | 161.10 | 170.05 | 170.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 14:15:00 | 139.12 | 138.25 | 147.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 145.33 | 138.45 | 146.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 145.33 | 138.45 | 146.73 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-06 11:30:00 | 171.63 | 2025-06-12 13:15:00 | 169.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-06-06 14:45:00 | 171.50 | 2025-06-12 13:15:00 | 169.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-06-09 09:45:00 | 171.79 | 2025-06-12 13:15:00 | 169.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-06-12 09:30:00 | 171.74 | 2025-06-12 13:15:00 | 169.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-06-17 13:00:00 | 168.58 | 2025-06-23 09:15:00 | 163.12 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2025-06-19 14:00:00 | 168.67 | 2025-06-23 09:15:00 | 163.12 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2025-06-20 11:30:00 | 168.22 | 2025-06-23 09:15:00 | 163.12 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2025-06-23 12:30:00 | 168.23 | 2025-06-23 13:15:00 | 165.99 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-06-25 12:30:00 | 172.10 | 2025-06-30 11:15:00 | 169.23 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-06-26 11:00:00 | 171.98 | 2025-06-30 11:15:00 | 169.23 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-06-27 09:15:00 | 171.64 | 2025-06-30 11:15:00 | 169.23 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-06-27 09:45:00 | 171.61 | 2025-06-30 11:15:00 | 169.23 | STOP_HIT | 1.00 | -1.39% |
