# IFCI Ltd. (IFCI)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 64.27
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 4 |
| ALERT3 | 24 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 16
- **Target hits / Stop hits / Partials:** 0 / 18 / 2
- **Avg / median % per leg:** -1.46% / -1.45%
- **Sum % (uncompounded):** -29.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -4.10% | -20.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -4.10% | -20.5% |
| SELL (all) | 15 | 4 | 26.7% | 0 | 13 | 2 | -0.58% | -8.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 4 | 26.7% | 0 | 13 | 2 | -0.58% | -8.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 20 | 4 | 20.0% | 0 | 18 | 2 | -1.46% | -29.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 58.45 | 46.24 | 46.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 60.08 | 46.73 | 46.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 11:15:00 | 58.73 | 59.07 | 54.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 11:45:00 | 58.74 | 59.07 | 54.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 58.64 | 61.31 | 59.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 15:00:00 | 58.64 | 61.31 | 59.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 58.60 | 61.28 | 59.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 59.46 | 61.28 | 59.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 58.83 | 61.24 | 59.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:00:00 | 58.83 | 61.24 | 59.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 58.57 | 61.21 | 59.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:45:00 | 58.75 | 61.21 | 59.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 58.81 | 60.96 | 59.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 09:45:00 | 59.39 | 60.91 | 59.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 11:15:00 | 59.40 | 60.89 | 59.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 14:15:00 | 58.61 | 60.81 | 59.02 | SL hit (close<static) qty=1.00 sl=58.64 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 53.44 | 57.86 | 57.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 52.83 | 57.76 | 57.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 56.32 | 54.40 | 55.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 56.32 | 54.40 | 55.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 56.32 | 54.40 | 55.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:30:00 | 57.61 | 54.40 | 55.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 55.86 | 54.41 | 55.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:30:00 | 56.26 | 54.41 | 55.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 55.88 | 54.53 | 55.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:15:00 | 55.85 | 54.53 | 55.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 56.18 | 55.85 | 56.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:30:00 | 55.72 | 55.85 | 56.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 15:15:00 | 52.93 | 55.53 | 55.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 55.40 | 55.39 | 55.80 | SL hit (close>ema200) qty=0.50 sl=55.39 alert=retest2 |

### Cycle 3 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 58.41 | 56.14 | 56.14 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 12:15:00 | 55.34 | 56.14 | 56.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 13:15:00 | 55.13 | 56.13 | 56.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 15:15:00 | 56.31 | 55.99 | 56.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 15:15:00 | 56.31 | 55.99 | 56.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 56.31 | 55.99 | 56.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 56.45 | 55.99 | 56.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 56.30 | 56.00 | 56.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:15:00 | 56.93 | 56.00 | 56.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 55.94 | 56.03 | 56.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 11:30:00 | 55.79 | 56.03 | 56.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 13:00:00 | 55.80 | 56.03 | 56.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 14:00:00 | 55.75 | 56.01 | 56.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 09:30:00 | 55.73 | 56.01 | 56.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 56.06 | 56.00 | 56.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 56.06 | 56.00 | 56.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 56.60 | 56.01 | 56.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-28 12:15:00 | 56.60 | 56.01 | 56.06 | SL hit (close>static) qty=1.00 sl=56.41 alert=retest2 |

### Cycle 5 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 58.65 | 56.12 | 56.12 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 09:15:00 | 54.48 | 56.14 | 56.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 09:15:00 | 53.84 | 56.04 | 56.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 55.80 | 55.75 | 55.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 55.80 | 55.75 | 55.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 55.80 | 55.75 | 55.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 55.80 | 55.75 | 55.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 55.60 | 55.75 | 55.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:45:00 | 55.71 | 55.75 | 55.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 52.76 | 50.32 | 52.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 52.76 | 50.32 | 52.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 53.15 | 50.34 | 52.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 14:30:00 | 52.42 | 50.37 | 52.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 09:45:00 | 52.40 | 50.42 | 52.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:15:00 | 52.37 | 50.80 | 52.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 53.97 | 50.86 | 52.20 | SL hit (close>static) qty=1.00 sl=53.90 alert=retest2 |

### Cycle 7 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 62.00 | 52.87 | 52.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 64.10 | 56.11 | 54.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 13:15:00 | 58.90 | 59.33 | 57.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-23 14:00:00 | 58.90 | 59.33 | 57.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 57.15 | 59.30 | 57.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 58.80 | 57.03 | 56.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 10:45:00 | 57.96 | 57.03 | 56.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 14:30:00 | 58.06 | 57.04 | 56.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 09:15:00 | 54.80 | 57.02 | 56.66 | SL hit (close<static) qty=1.00 sl=56.09 alert=retest2 |

### Cycle 8 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 51.18 | 56.35 | 56.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 50.51 | 56.24 | 56.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 55.14 | 54.28 | 55.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 55.14 | 54.28 | 55.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 55.14 | 54.28 | 55.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:00:00 | 55.14 | 54.28 | 55.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 55.34 | 54.29 | 55.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:30:00 | 55.32 | 54.29 | 55.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 55.30 | 54.30 | 55.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:00:00 | 55.00 | 54.32 | 55.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:45:00 | 54.96 | 54.35 | 55.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 10:15:00 | 55.79 | 54.36 | 55.16 | SL hit (close>static) qty=1.00 sl=55.59 alert=retest2 |

### Cycle 9 — BUY (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 10:15:00 | 62.61 | 55.83 | 55.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 13:15:00 | 63.59 | 58.31 | 57.34 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-26 11:30:00 | 58.12 | 2025-05-26 12:15:00 | 58.45 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-07-30 09:45:00 | 59.39 | 2025-07-30 14:15:00 | 58.61 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-07-30 11:15:00 | 59.40 | 2025-07-30 14:15:00 | 58.61 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-09-25 11:30:00 | 55.72 | 2025-09-29 15:15:00 | 52.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 11:30:00 | 55.72 | 2025-10-01 15:15:00 | 55.40 | STOP_HIT | 0.50 | 0.57% |
| SELL | retest2 | 2025-10-03 10:00:00 | 55.72 | 2025-10-03 11:15:00 | 57.06 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-10-24 11:30:00 | 55.79 | 2025-10-28 12:15:00 | 56.60 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-10-24 13:00:00 | 55.80 | 2025-10-28 12:15:00 | 56.60 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-10-27 14:00:00 | 55.75 | 2025-10-28 12:15:00 | 56.60 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-10-28 09:30:00 | 55.73 | 2025-10-28 12:15:00 | 56.60 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-12-23 14:30:00 | 52.42 | 2025-12-31 09:15:00 | 53.97 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-12-24 09:45:00 | 52.40 | 2025-12-31 09:15:00 | 53.97 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-12-29 10:15:00 | 52.37 | 2025-12-31 09:15:00 | 53.97 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2025-12-31 15:15:00 | 52.46 | 2026-01-08 15:15:00 | 49.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 15:15:00 | 52.46 | 2026-01-12 09:15:00 | 51.60 | STOP_HIT | 0.50 | 1.64% |
| BUY | retest2 | 2026-03-13 09:15:00 | 58.80 | 2026-03-16 09:15:00 | 54.80 | STOP_HIT | 1.00 | -6.80% |
| BUY | retest2 | 2026-03-13 10:45:00 | 57.96 | 2026-03-16 09:15:00 | 54.80 | STOP_HIT | 1.00 | -5.45% |
| BUY | retest2 | 2026-03-13 14:30:00 | 58.06 | 2026-03-16 09:15:00 | 54.80 | STOP_HIT | 1.00 | -5.61% |
| SELL | retest2 | 2026-04-08 14:00:00 | 55.00 | 2026-04-09 10:15:00 | 55.79 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-04-09 09:45:00 | 54.96 | 2026-04-09 10:15:00 | 55.79 | STOP_HIT | 1.00 | -1.51% |
