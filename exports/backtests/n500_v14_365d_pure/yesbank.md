# Yes Bank Ltd. (YESBANK)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 22.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 19
- **Target hits / Stop hits / Partials:** 0 / 19 / 0
- **Avg / median % per leg:** -1.65% / -1.72%
- **Sum % (uncompounded):** -31.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.70% | -15.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.70% | -15.3% |
| SELL (all) | 10 | 0 | 0.0% | 0 | 10 | 0 | -1.61% | -16.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 0 | 0.0% | 0 | 10 | 0 | -1.61% | -16.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 0 | 0.0% | 0 | 19 | 0 | -1.65% | -31.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 20.45 | 17.95 | 17.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 21.78 | 18.70 | 18.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 20.44 | 20.51 | 19.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 14:00:00 | 20.44 | 20.51 | 19.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 20.08 | 20.43 | 19.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 12:45:00 | 20.20 | 20.25 | 19.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:00:00 | 20.14 | 20.24 | 19.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 20.25 | 20.24 | 19.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:45:00 | 20.16 | 20.26 | 19.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 19.96 | 20.22 | 19.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 12:45:00 | 19.99 | 20.22 | 19.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 10:15:00 | 19.85 | 20.19 | 19.91 | SL hit (close<static) qty=1.00 sl=19.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 19.71 | 20.16 | 19.90 | SL hit (close<static) qty=1.00 sl=19.74 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 19.71 | 20.16 | 19.90 | SL hit (close<static) qty=1.00 sl=19.74 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 19.71 | 20.16 | 19.90 | SL hit (close<static) qty=1.00 sl=19.74 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 19.71 | 20.16 | 19.90 | SL hit (close<static) qty=1.00 sl=19.74 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 12:45:00 | 19.98 | 20.13 | 19.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 15:00:00 | 19.97 | 20.13 | 19.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 19.89 | 20.14 | 19.95 | SL hit (close<static) qty=1.00 sl=19.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 19.89 | 20.14 | 19.95 | SL hit (close<static) qty=1.00 sl=19.90 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 10:30:00 | 19.97 | 20.12 | 19.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 19.70 | 20.12 | 19.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-24 11:15:00 | 19.70 | 20.12 | 19.95 | SL hit (close<static) qty=1.00 sl=19.90 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:45:00 | 19.97 | 20.11 | 19.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 12:15:00 | 19.38 | 20.06 | 19.93 | SL hit (close<static) qty=1.00 sl=19.43 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 18.61 | 19.82 | 19.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 18.53 | 19.79 | 19.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 14:15:00 | 19.36 | 19.29 | 19.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 15:00:00 | 19.36 | 19.29 | 19.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 19.47 | 19.30 | 19.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 19.44 | 19.30 | 19.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 19.60 | 19.30 | 19.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:30:00 | 19.65 | 19.30 | 19.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 19.56 | 19.30 | 19.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:15:00 | 19.50 | 19.30 | 19.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 13:15:00 | 19.50 | 19.31 | 19.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 19.66 | 19.32 | 19.49 | SL hit (close>static) qty=1.00 sl=19.61 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 19.66 | 19.32 | 19.49 | SL hit (close>static) qty=1.00 sl=19.61 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:00:00 | 19.51 | 19.33 | 19.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 19.27 | 19.33 | 19.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 19.43 | 19.28 | 19.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 10:45:00 | 19.41 | 19.28 | 19.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 19.46 | 19.28 | 19.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:45:00 | 19.48 | 19.28 | 19.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 19.21 | 19.28 | 19.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:45:00 | 19.15 | 19.28 | 19.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:15:00 | 19.15 | 19.28 | 19.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:15:00 | 19.17 | 19.28 | 19.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:45:00 | 19.18 | 19.28 | 19.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 19.55 | 19.28 | 19.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-01 13:15:00 | 19.55 | 19.28 | 19.45 | SL hit (close>static) qty=1.00 sl=19.47 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 13:15:00 | 19.55 | 19.28 | 19.45 | SL hit (close>static) qty=1.00 sl=19.47 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 13:15:00 | 19.55 | 19.28 | 19.45 | SL hit (close>static) qty=1.00 sl=19.47 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 13:15:00 | 19.55 | 19.28 | 19.45 | SL hit (close>static) qty=1.00 sl=19.47 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-09-01 13:45:00 | 19.50 | 19.28 | 19.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 19.56 | 19.28 | 19.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 19.56 | 19.28 | 19.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 19.68 | 19.29 | 19.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 19.68 | 19.29 | 19.45 | SL hit (close>static) qty=1.00 sl=19.61 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 19.68 | 19.29 | 19.45 | SL hit (close>static) qty=1.00 sl=19.61 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-09-02 10:45:00 | 19.66 | 19.29 | 19.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 12:15:00 | 20.39 | 19.58 | 19.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 20.75 | 19.61 | 19.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 12:15:00 | 22.43 | 22.47 | 21.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 13:00:00 | 22.43 | 22.47 | 21.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 22.16 | 22.61 | 22.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 22.16 | 22.61 | 22.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 22.00 | 22.61 | 22.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 22.00 | 22.61 | 22.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 21.93 | 22.54 | 22.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:00:00 | 21.93 | 22.54 | 22.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 14:15:00 | 21.39 | 21.98 | 21.98 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 15:15:00 | 22.79 | 21.98 | 21.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 09:15:00 | 23.15 | 21.99 | 21.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 22.17 | 22.45 | 22.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 22.17 | 22.45 | 22.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 22.17 | 22.45 | 22.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 22.02 | 22.45 | 22.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 22.02 | 22.44 | 22.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 22.02 | 22.44 | 22.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 21.96 | 22.44 | 22.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 21.96 | 22.44 | 22.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 21.16 | 22.09 | 22.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 20.92 | 21.94 | 22.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 14:15:00 | 19.01 | 19.01 | 19.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 14:45:00 | 19.06 | 19.01 | 19.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 19.73 | 19.03 | 19.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:00:00 | 19.73 | 19.03 | 19.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 19.98 | 19.04 | 19.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:45:00 | 19.98 | 19.04 | 19.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 19.82 | 19.05 | 19.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 12:45:00 | 19.73 | 19.06 | 19.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 13:15:00 | 20.05 | 19.12 | 19.79 | SL hit (close>static) qty=1.00 sl=20.01 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 13:30:00 | 19.73 | 19.24 | 19.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 20.07 | 19.26 | 19.81 | SL hit (close>static) qty=1.00 sl=20.01 alert=retest2 |

### Cycle 7 — BUY (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 14:15:00 | 22.93 | 20.10 | 20.10 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-25 12:45:00 | 20.20 | 2025-07-10 10:15:00 | 19.85 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-06-26 12:00:00 | 20.14 | 2025-07-11 10:15:00 | 19.71 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-06-27 09:15:00 | 20.25 | 2025-07-11 10:15:00 | 19.71 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-07-04 09:45:00 | 20.16 | 2025-07-11 10:15:00 | 19.71 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-07-08 12:45:00 | 19.99 | 2025-07-11 10:15:00 | 19.71 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-07-14 12:45:00 | 19.98 | 2025-07-23 09:15:00 | 19.89 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-07-14 15:00:00 | 19.97 | 2025-07-23 09:15:00 | 19.89 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-07-24 10:30:00 | 19.97 | 2025-07-24 11:15:00 | 19.70 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-07-24 14:45:00 | 19.97 | 2025-07-28 12:15:00 | 19.38 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2025-08-21 12:15:00 | 19.50 | 2025-08-25 09:15:00 | 19.66 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-08-21 13:15:00 | 19.50 | 2025-08-25 09:15:00 | 19.66 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-08-25 14:00:00 | 19.51 | 2025-09-01 13:15:00 | 19.55 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-08-26 09:15:00 | 19.27 | 2025-09-01 13:15:00 | 19.55 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-08-29 13:45:00 | 19.15 | 2025-09-01 13:15:00 | 19.55 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-08-29 14:15:00 | 19.15 | 2025-09-01 13:15:00 | 19.55 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-09-01 10:15:00 | 19.17 | 2025-09-02 10:15:00 | 19.68 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-09-01 10:45:00 | 19.18 | 2025-09-02 10:15:00 | 19.68 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2026-04-16 12:45:00 | 19.73 | 2026-04-17 13:15:00 | 20.05 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-04-21 13:30:00 | 19.73 | 2026-04-22 10:15:00 | 20.07 | STOP_HIT | 1.00 | -1.72% |
