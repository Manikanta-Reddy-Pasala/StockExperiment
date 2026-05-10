# Jaiprakash Power Ventures Ltd. (JPPOWER)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 19.02
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 11
- **Target hits / Stop hits / Partials:** 0 / 12 / 1
- **Avg / median % per leg:** -2.08% / -2.09%
- **Sum % (uncompounded):** -27.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.59% | -10.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.59% | -10.4% |
| SELL (all) | 9 | 2 | 22.2% | 0 | 8 | 1 | -1.86% | -16.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 2 | 22.2% | 0 | 8 | 1 | -1.86% | -16.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 2 | 15.4% | 0 | 12 | 1 | -2.08% | -27.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 13:15:00 | 15.61 | 14.86 | 14.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 15.87 | 14.96 | 14.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 20.54 | 20.69 | 18.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 09:30:00 | 20.32 | 20.69 | 18.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 19.36 | 20.66 | 19.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 15:15:00 | 19.59 | 20.66 | 19.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 19.22 | 20.63 | 19.29 | SL hit (close<static) qty=1.00 sl=19.26 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 12:45:00 | 19.55 | 20.28 | 19.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 14:15:00 | 19.09 | 20.26 | 19.26 | SL hit (close<static) qty=1.00 sl=19.26 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:45:00 | 19.60 | 19.71 | 19.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 13:15:00 | 19.19 | 19.69 | 19.18 | SL hit (close<static) qty=1.00 sl=19.26 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 20.02 | 19.14 | 19.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 11:15:00 | 19.21 | 19.21 | 19.05 | SL hit (close<static) qty=1.00 sl=19.26 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 19.11 | 19.22 | 19.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 19.11 | 19.22 | 19.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 19.05 | 19.22 | 19.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 19.10 | 19.22 | 19.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 19.04 | 19.22 | 19.07 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 18.09 | 18.97 | 18.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 13:15:00 | 17.86 | 18.93 | 18.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 12:15:00 | 18.60 | 18.57 | 18.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-03 12:45:00 | 18.59 | 18.57 | 18.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 18.70 | 18.44 | 18.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 18.70 | 18.44 | 18.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 18.59 | 18.44 | 18.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 14:30:00 | 18.54 | 18.45 | 18.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 10:15:00 | 17.61 | 18.33 | 18.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 18.23 | 18.21 | 18.45 | SL hit (close>ema200) qty=0.50 sl=18.21 alert=retest2 |

### Cycle 3 — BUY (started 2025-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 13:15:00 | 20.04 | 18.46 | 18.45 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 17.73 | 18.56 | 18.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 13:15:00 | 17.62 | 18.53 | 18.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 10:15:00 | 18.20 | 18.05 | 18.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-01 11:00:00 | 18.20 | 18.05 | 18.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 15.12 | 14.48 | 15.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 15.41 | 14.48 | 15.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 15.22 | 14.50 | 15.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 12:45:00 | 15.28 | 14.50 | 15.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 15.06 | 14.52 | 15.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 14:45:00 | 15.17 | 14.52 | 15.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 15.04 | 14.53 | 15.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 14.50 | 14.80 | 15.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 10:15:00 | 14.49 | 14.80 | 15.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 09:30:00 | 14.51 | 14.77 | 15.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 15.35 | 14.81 | 15.25 | SL hit (close>static) qty=1.00 sl=15.32 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 15.35 | 14.81 | 15.25 | SL hit (close>static) qty=1.00 sl=15.32 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 15.35 | 14.81 | 15.25 | SL hit (close>static) qty=1.00 sl=15.32 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 10:15:00 | 19.26 | 15.61 | 15.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 11:15:00 | 19.60 | 15.65 | 15.62 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-15 09:30:00 | 14.64 | 2025-05-16 13:15:00 | 15.10 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2025-05-22 14:00:00 | 14.67 | 2025-05-23 10:15:00 | 14.79 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-05-22 14:45:00 | 14.65 | 2025-05-23 10:15:00 | 14.79 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-05-23 10:15:00 | 14.66 | 2025-05-23 10:15:00 | 14.79 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-08-04 15:15:00 | 19.59 | 2025-08-05 09:15:00 | 19.22 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-08-08 12:45:00 | 19.55 | 2025-08-08 14:15:00 | 19.09 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-08-21 09:45:00 | 19.60 | 2025-08-21 13:15:00 | 19.19 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-09-08 09:15:00 | 20.02 | 2025-09-09 11:15:00 | 19.21 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2025-10-10 14:30:00 | 18.54 | 2025-10-20 10:15:00 | 17.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 14:30:00 | 18.54 | 2025-10-29 11:15:00 | 18.23 | STOP_HIT | 0.50 | 1.67% |
| SELL | retest2 | 2026-03-30 09:15:00 | 14.50 | 2026-04-07 09:15:00 | 15.35 | STOP_HIT | 1.00 | -5.86% |
| SELL | retest2 | 2026-03-30 10:15:00 | 14.49 | 2026-04-07 09:15:00 | 15.35 | STOP_HIT | 1.00 | -5.94% |
| SELL | retest2 | 2026-04-01 09:30:00 | 14.51 | 2026-04-07 09:15:00 | 15.35 | STOP_HIT | 1.00 | -5.79% |
