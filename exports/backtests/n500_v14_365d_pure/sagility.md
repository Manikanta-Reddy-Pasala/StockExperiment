# Sagility Ltd. (SAGILITY)

## Backtest Summary

- **Window:** 2024-11-12 09:15:00 → 2026-05-08 15:15:00 (2564 bars)
- **Last close:** 44.44
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 0 |
| TARGET_HIT | 3 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 15
- **Target hits / Stop hits / Partials:** 3 / 15 / 0
- **Avg / median % per leg:** -1.61% / -3.49%
- **Sum % (uncompounded):** -28.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |
| SELL (all) | 15 | 0 | 0.0% | 0 | 15 | 0 | -3.93% | -58.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 0 | 0.0% | 0 | 15 | 0 | -3.93% | -58.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 18 | 3 | 16.7% | 3 | 15 | 0 | -1.61% | -28.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 13:15:00 | 46.45 | 43.06 | 43.05 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 10:15:00 | 39.83 | 43.07 | 43.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 09:15:00 | 38.98 | 42.66 | 42.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 10:15:00 | 41.41 | 40.87 | 41.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-17 11:00:00 | 41.41 | 40.87 | 41.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 41.55 | 40.87 | 41.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:00:00 | 41.55 | 40.87 | 41.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 41.58 | 40.88 | 41.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:30:00 | 40.56 | 40.95 | 41.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 10:45:00 | 40.76 | 40.93 | 41.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 40.66 | 40.93 | 41.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 13:30:00 | 40.76 | 40.93 | 41.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 41.53 | 40.88 | 41.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 41.53 | 40.88 | 41.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 41.66 | 40.89 | 41.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:00:00 | 41.66 | 40.89 | 41.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 41.84 | 40.89 | 41.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:00:00 | 41.84 | 40.89 | 41.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 12:15:00 | 41.78 | 40.90 | 41.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:45:00 | 41.92 | 40.90 | 41.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 41.50 | 40.91 | 41.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-25 13:15:00 | 42.59 | 40.98 | 41.60 | SL hit (close>static) qty=1.00 sl=42.07 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 13:15:00 | 42.59 | 40.98 | 41.60 | SL hit (close>static) qty=1.00 sl=42.07 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 13:15:00 | 42.59 | 40.98 | 41.60 | SL hit (close>static) qty=1.00 sl=42.07 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 13:15:00 | 42.59 | 40.98 | 41.60 | SL hit (close>static) qty=1.00 sl=42.07 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 13:15:00 | 41.30 | 41.09 | 41.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 10:15:00 | 41.29 | 41.11 | 41.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 11:45:00 | 41.30 | 41.11 | 41.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 12:30:00 | 41.28 | 41.11 | 41.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 41.66 | 41.12 | 41.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 41.66 | 41.12 | 41.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 43.25 | 41.14 | 41.59 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-02 10:15:00 | 43.25 | 41.14 | 41.59 | SL hit (close>static) qty=1.00 sl=41.83 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-02 10:15:00 | 43.25 | 41.14 | 41.59 | SL hit (close>static) qty=1.00 sl=41.83 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-02 10:15:00 | 43.25 | 41.14 | 41.59 | SL hit (close>static) qty=1.00 sl=41.83 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-02 10:15:00 | 43.25 | 41.14 | 41.59 | SL hit (close>static) qty=1.00 sl=41.83 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-07-02 11:00:00 | 43.25 | 41.14 | 41.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 42.97 | 41.16 | 41.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 14:00:00 | 42.54 | 41.33 | 41.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 14:45:00 | 42.46 | 41.34 | 41.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:30:00 | 42.17 | 41.43 | 41.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 43.64 | 41.54 | 41.73 | SL hit (close>static) qty=1.00 sl=43.37 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 43.64 | 41.54 | 41.73 | SL hit (close>static) qty=1.00 sl=43.37 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 43.64 | 41.54 | 41.73 | SL hit (close>static) qty=1.00 sl=43.37 alert=retest2 |

### Cycle 3 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 45.32 | 41.94 | 41.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 45.49 | 42.15 | 42.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 11:15:00 | 43.04 | 43.14 | 42.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 12:00:00 | 43.04 | 43.14 | 42.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 42.44 | 43.17 | 42.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 15:00:00 | 42.44 | 43.17 | 42.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 41.50 | 43.15 | 42.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 11:45:00 | 42.66 | 43.13 | 42.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-01 09:15:00 | 46.93 | 43.35 | 42.82 | Target hit (10%) qty=1.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:45:00 | 42.79 | 44.35 | 43.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 10:15:00 | 42.73 | 44.35 | 43.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-11 09:15:00 | 47.07 | 44.33 | 43.94 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-09-11 09:15:00 | 47.00 | 44.33 | 43.94 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 13:15:00 | 48.33 | 50.14 | 50.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 47.40 | 50.08 | 50.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 14:15:00 | 41.70 | 40.92 | 43.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-27 15:00:00 | 41.70 | 40.92 | 43.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 43.02 | 41.20 | 43.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 11:30:00 | 42.61 | 41.45 | 43.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 09:30:00 | 42.35 | 41.81 | 43.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 12:15:00 | 42.58 | 41.86 | 43.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:00:00 | 42.56 | 41.87 | 43.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 42.45 | 41.79 | 42.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:45:00 | 42.79 | 41.79 | 42.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 42.73 | 41.80 | 42.77 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-08 09:15:00 | 43.84 | 41.87 | 42.78 | SL hit (close>static) qty=1.00 sl=43.78 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-08 09:15:00 | 43.84 | 41.87 | 42.78 | SL hit (close>static) qty=1.00 sl=43.78 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-08 09:15:00 | 43.84 | 41.87 | 42.78 | SL hit (close>static) qty=1.00 sl=43.78 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-08 09:15:00 | 43.84 | 41.87 | 42.78 | SL hit (close>static) qty=1.00 sl=43.78 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-19 12:30:00 | 40.56 | 2025-06-25 13:15:00 | 42.59 | STOP_HIT | 1.00 | -5.00% |
| SELL | retest2 | 2025-06-20 10:45:00 | 40.76 | 2025-06-25 13:15:00 | 42.59 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest2 | 2025-06-20 12:15:00 | 40.66 | 2025-06-25 13:15:00 | 42.59 | STOP_HIT | 1.00 | -4.75% |
| SELL | retest2 | 2025-06-20 13:30:00 | 40.76 | 2025-06-25 13:15:00 | 42.59 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest2 | 2025-06-27 13:15:00 | 41.30 | 2025-07-02 10:15:00 | 43.25 | STOP_HIT | 1.00 | -4.72% |
| SELL | retest2 | 2025-06-30 10:15:00 | 41.29 | 2025-07-02 10:15:00 | 43.25 | STOP_HIT | 1.00 | -4.75% |
| SELL | retest2 | 2025-06-30 11:45:00 | 41.30 | 2025-07-02 10:15:00 | 43.25 | STOP_HIT | 1.00 | -4.72% |
| SELL | retest2 | 2025-06-30 12:30:00 | 41.28 | 2025-07-02 10:15:00 | 43.25 | STOP_HIT | 1.00 | -4.77% |
| SELL | retest2 | 2025-07-03 14:00:00 | 42.54 | 2025-07-10 09:15:00 | 43.64 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-07-03 14:45:00 | 42.46 | 2025-07-10 09:15:00 | 43.64 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-07-07 09:30:00 | 42.17 | 2025-07-10 09:15:00 | 43.64 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest2 | 2025-07-29 11:45:00 | 42.66 | 2025-08-01 09:15:00 | 46.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-09 09:45:00 | 42.79 | 2025-09-11 09:15:00 | 47.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-09 10:15:00 | 42.73 | 2025-09-11 09:15:00 | 47.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-10 11:30:00 | 42.61 | 2026-05-08 09:15:00 | 43.84 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2026-04-20 09:30:00 | 42.35 | 2026-05-08 09:15:00 | 43.84 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2026-04-21 12:15:00 | 42.58 | 2026-05-08 09:15:00 | 43.84 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2026-04-23 10:00:00 | 42.56 | 2026-05-08 09:15:00 | 43.84 | STOP_HIT | 1.00 | -3.01% |
