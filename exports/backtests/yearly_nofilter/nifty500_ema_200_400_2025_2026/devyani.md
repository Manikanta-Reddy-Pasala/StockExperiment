# Devyani International Ltd. (DEVYANI)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 118.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 25 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 13
- **Target hits / Stop hits / Partials:** 1 / 13 / 1
- **Avg / median % per leg:** -0.90% / -1.60%
- **Sum % (uncompounded):** -13.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.93% | -23.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.93% | -23.2% |
| SELL (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 3.22% | 9.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 2 | 66.7% | 1 | 1 | 1 | 3.22% | 9.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 2 | 13.3% | 1 | 13 | 1 | -0.90% | -13.5% |

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
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 12:15:00 | 172.25 | 172.37 | 171.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 13:15:00 | 171.65 | 172.36 | 171.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 171.65 | 172.36 | 171.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:45:00 | 172.02 | 172.36 | 171.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 171.30 | 172.35 | 171.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:30:00 | 171.76 | 172.35 | 171.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 171.85 | 172.34 | 171.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 172.47 | 172.34 | 171.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 171.17 | 172.33 | 171.44 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 162.55 | 170.69 | 170.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 162.25 | 170.60 | 170.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 10:15:00 | 165.74 | 163.56 | 166.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 10:15:00 | 165.74 | 163.56 | 166.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 165.74 | 163.56 | 166.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 165.74 | 163.56 | 166.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 166.84 | 163.59 | 166.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:45:00 | 167.85 | 163.59 | 166.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 170.90 | 163.66 | 166.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:00:00 | 170.90 | 163.66 | 166.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 168.79 | 163.71 | 166.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 14:45:00 | 167.05 | 163.74 | 166.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 09:15:00 | 175.99 | 164.24 | 166.69 | SL hit (close>static) qty=1.00 sl=171.95 alert=retest2 |

### Cycle 4 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 175.42 | 168.54 | 168.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 180.05 | 168.91 | 168.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 10:15:00 | 175.88 | 176.14 | 173.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 11:00:00 | 175.88 | 176.14 | 173.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 173.22 | 176.14 | 173.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 173.22 | 176.14 | 173.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 173.48 | 176.12 | 173.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 171.31 | 176.12 | 173.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 169.06 | 175.98 | 173.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:30:00 | 169.23 | 175.98 | 173.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 15:15:00 | 165.00 | 171.74 | 171.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 13:15:00 | 164.24 | 171.41 | 171.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 14:15:00 | 139.12 | 137.97 | 146.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-22 15:00:00 | 139.12 | 137.97 | 146.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 145.33 | 138.19 | 146.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 14:00:00 | 140.63 | 141.02 | 145.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 13:15:00 | 133.60 | 140.21 | 144.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 11:15:00 | 126.57 | 137.65 | 142.67 | Target hit (10%) qty=0.50 alert=retest2 |


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
| SELL | retest2 | 2025-08-19 14:45:00 | 167.05 | 2025-08-21 09:15:00 | 175.99 | STOP_HIT | 1.00 | -5.35% |
| SELL | retest2 | 2026-01-05 14:00:00 | 140.63 | 2026-01-09 13:15:00 | 133.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 14:00:00 | 140.63 | 2026-01-20 11:15:00 | 126.57 | TARGET_HIT | 0.50 | 10.00% |
