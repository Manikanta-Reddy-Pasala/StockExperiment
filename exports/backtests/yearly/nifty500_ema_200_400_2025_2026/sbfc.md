# SBFC Finance Ltd. (SBFC)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 98.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 2 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 3 |
| TARGET_HIT | 5 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 6
- **Target hits / Stop hits / Partials:** 5 / 6 / 3
- **Avg / median % per leg:** 3.93% / 5.00%
- **Sum % (uncompounded):** 55.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 2 | 6 | 0 | 1.25% | 10.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 2 | 25.0% | 2 | 6 | 0 | 1.25% | 10.0% |
| SELL (all) | 6 | 6 | 100.0% | 3 | 0 | 3 | 7.50% | 45.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 6 | 100.0% | 3 | 0 | 3 | 7.50% | 45.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 8 | 57.1% | 5 | 6 | 3 | 3.93% | 55.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 10:15:00 | 105.79 | 107.54 | 107.55 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 14:15:00 | 109.30 | 107.55 | 107.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 15:15:00 | 109.99 | 107.58 | 107.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 108.22 | 108.24 | 107.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 108.22 | 108.24 | 107.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 108.22 | 108.24 | 107.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 108.66 | 108.24 | 107.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 107.30 | 108.23 | 107.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:00:00 | 107.30 | 108.23 | 107.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 107.59 | 108.23 | 107.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:45:00 | 107.54 | 108.23 | 107.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 12:15:00 | 107.28 | 108.22 | 107.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 12:30:00 | 107.43 | 108.22 | 107.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 106.88 | 108.21 | 107.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 106.88 | 108.21 | 107.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 108.48 | 108.13 | 107.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 14:00:00 | 108.66 | 107.99 | 107.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 09:15:00 | 109.15 | 108.01 | 107.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 13:00:00 | 108.53 | 108.02 | 107.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 13:30:00 | 108.57 | 108.02 | 107.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 108.34 | 108.41 | 108.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:00:00 | 108.34 | 108.41 | 108.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 108.31 | 108.42 | 108.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:00:00 | 108.31 | 108.42 | 108.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 106.87 | 108.41 | 108.09 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 106.87 | 108.41 | 108.09 | SL hit (close<static) qty=1.00 sl=107.55 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 106.89 | 109.68 | 109.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 106.60 | 109.64 | 109.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 10:15:00 | 106.91 | 106.49 | 107.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 10:15:00 | 106.91 | 106.49 | 107.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 106.91 | 106.49 | 107.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:30:00 | 107.66 | 106.49 | 107.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 105.39 | 104.02 | 105.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:45:00 | 106.26 | 104.02 | 105.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 108.34 | 104.06 | 105.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 108.34 | 104.06 | 105.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 107.42 | 104.10 | 105.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 108.87 | 104.10 | 105.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 107.48 | 104.21 | 105.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 15:00:00 | 107.48 | 104.21 | 105.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 107.30 | 104.24 | 105.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 104.56 | 104.24 | 105.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:30:00 | 105.79 | 104.27 | 105.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:00:00 | 105.50 | 104.28 | 105.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 99.33 | 104.21 | 105.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 100.50 | 104.21 | 105.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 100.22 | 104.21 | 105.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-27 09:15:00 | 94.10 | 103.71 | 105.25 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-10-01 14:00:00 | 108.66 | 2025-10-13 09:15:00 | 106.87 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-10-03 09:15:00 | 109.15 | 2025-10-13 09:15:00 | 106.87 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-10-03 13:00:00 | 108.53 | 2025-10-13 09:15:00 | 106.87 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-10-03 13:30:00 | 108.57 | 2025-10-13 09:15:00 | 106.87 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-10-15 09:15:00 | 108.22 | 2025-11-03 09:15:00 | 119.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-15 10:00:00 | 108.18 | 2025-11-03 09:15:00 | 119.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-21 13:45:00 | 107.80 | 2025-11-24 11:15:00 | 106.11 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-11-24 10:30:00 | 107.81 | 2025-11-24 11:15:00 | 106.11 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2026-01-20 09:15:00 | 104.56 | 2026-01-21 10:15:00 | 99.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 10:30:00 | 105.79 | 2026-01-21 10:15:00 | 100.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 12:00:00 | 105.50 | 2026-01-21 10:15:00 | 100.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 09:15:00 | 104.56 | 2026-01-27 09:15:00 | 94.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-20 10:30:00 | 105.79 | 2026-01-27 09:15:00 | 95.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-20 12:00:00 | 105.50 | 2026-01-27 09:15:00 | 94.95 | TARGET_HIT | 0.50 | 10.00% |
