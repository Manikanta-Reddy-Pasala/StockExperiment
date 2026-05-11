# Welspun Living Ltd. (WELSPUNLIV)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 134.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 33 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 31 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 25
- **Target hits / Stop hits / Partials:** 0 / 31 / 3
- **Avg / median % per leg:** -2.94% / -2.08%
- **Sum % (uncompounded):** -100.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.09% | -12.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.09% | -12.5% |
| SELL (all) | 28 | 9 | 32.1% | 0 | 25 | 3 | -3.13% | -87.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 9 | 32.1% | 0 | 25 | 3 | -3.13% | -87.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 34 | 9 | 26.5% | 0 | 31 | 3 | -2.94% | -100.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 13:15:00 | 125.44 | 136.41 | 136.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 122.28 | 136.09 | 136.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 124.55 | 118.87 | 124.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 124.55 | 118.87 | 124.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 124.55 | 118.87 | 124.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 125.59 | 118.87 | 124.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 124.12 | 118.92 | 124.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:30:00 | 123.19 | 119.24 | 124.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 122.60 | 119.33 | 124.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 13:15:00 | 123.19 | 119.37 | 124.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 121.71 | 119.48 | 124.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 124.18 | 119.72 | 124.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 124.80 | 119.72 | 124.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 124.15 | 119.76 | 124.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:45:00 | 124.30 | 119.76 | 124.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 124.09 | 119.80 | 124.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:00:00 | 124.09 | 119.80 | 124.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 124.02 | 119.85 | 124.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:30:00 | 124.23 | 119.85 | 124.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 124.44 | 119.89 | 124.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:00:00 | 124.44 | 119.89 | 124.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 124.56 | 119.94 | 124.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 15:15:00 | 123.99 | 119.94 | 124.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 128.72 | 120.06 | 124.11 | SL hit (close>static) qty=1.00 sl=125.91 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 12:15:00 | 131.90 | 123.45 | 123.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 14:15:00 | 132.44 | 123.62 | 123.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 14:15:00 | 131.50 | 131.57 | 128.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-24 15:00:00 | 131.50 | 131.57 | 128.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 129.54 | 133.94 | 130.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 128.47 | 133.94 | 130.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 131.35 | 133.91 | 130.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 11:15:00 | 131.51 | 133.91 | 130.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 11:45:00 | 131.68 | 133.89 | 130.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 15:00:00 | 132.00 | 133.83 | 130.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 14:15:00 | 131.51 | 134.65 | 132.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 133.29 | 134.40 | 132.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 133.29 | 134.40 | 132.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 131.66 | 134.36 | 132.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 131.39 | 134.36 | 132.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 131.84 | 134.34 | 132.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:15:00 | 131.54 | 134.34 | 132.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 132.15 | 134.10 | 132.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 13:00:00 | 132.15 | 134.10 | 132.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 132.01 | 134.04 | 132.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 132.00 | 134.02 | 132.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 132.14 | 134.00 | 132.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:30:00 | 132.10 | 134.00 | 132.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 131.82 | 133.85 | 132.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 131.82 | 133.85 | 132.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 131.49 | 133.82 | 132.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 129.95 | 133.82 | 132.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 130.28 | 133.75 | 132.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-06 13:15:00 | 129.22 | 133.63 | 132.31 | SL hit (close<static) qty=1.00 sl=129.54 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 11:15:00 | 122.00 | 131.23 | 131.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 13:15:00 | 120.56 | 131.04 | 131.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 10:15:00 | 127.78 | 126.73 | 128.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 10:15:00 | 127.78 | 126.73 | 128.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 127.78 | 126.73 | 128.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:30:00 | 128.09 | 126.73 | 128.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 128.19 | 126.75 | 128.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:00:00 | 128.19 | 126.75 | 128.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 126.16 | 126.72 | 128.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 125.15 | 126.71 | 128.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 10:15:00 | 125.69 | 126.46 | 128.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 13:30:00 | 124.96 | 126.46 | 128.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:45:00 | 125.77 | 126.40 | 128.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 143.59 | 126.15 | 128.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 143.59 | 126.15 | 128.00 | SL hit (close>static) qty=1.00 sl=129.38 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 10:15:00 | 138.70 | 129.68 | 129.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 13:15:00 | 140.15 | 129.98 | 129.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 11:15:00 | 132.69 | 135.26 | 133.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 11:15:00 | 132.69 | 135.26 | 133.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 132.69 | 135.26 | 133.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:00:00 | 132.69 | 135.26 | 133.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 132.60 | 135.24 | 133.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:00:00 | 132.60 | 135.24 | 133.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 135.30 | 135.20 | 133.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:30:00 | 132.58 | 135.15 | 133.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 130.81 | 135.11 | 133.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 11:15:00 | 131.65 | 135.11 | 133.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 12:15:00 | 127.70 | 134.99 | 133.05 | SL hit (close<static) qty=1.00 sl=129.23 alert=retest2 |

### Cycle 5 — SELL (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 13:15:00 | 116.76 | 131.42 | 131.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 116.47 | 130.36 | 130.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 11:15:00 | 118.73 | 118.50 | 123.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 12:00:00 | 118.73 | 118.50 | 123.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 123.74 | 118.53 | 122.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:30:00 | 122.50 | 118.83 | 122.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 13:15:00 | 122.36 | 118.96 | 122.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 120.75 | 119.35 | 122.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 15:15:00 | 122.35 | 119.51 | 122.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 122.35 | 119.54 | 122.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:15:00 | 124.24 | 119.54 | 122.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 124.90 | 119.59 | 122.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 124.90 | 119.59 | 122.79 | SL hit (close>static) qty=1.00 sl=124.89 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 132.32 | 124.89 | 124.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 133.79 | 125.06 | 124.95 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-11 09:30:00 | 123.19 | 2025-09-17 09:15:00 | 128.72 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest2 | 2025-09-11 11:30:00 | 122.60 | 2025-09-17 09:15:00 | 128.72 | STOP_HIT | 1.00 | -4.99% |
| SELL | retest2 | 2025-09-11 13:15:00 | 123.19 | 2025-09-17 09:15:00 | 128.72 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest2 | 2025-09-12 09:15:00 | 121.71 | 2025-09-17 09:15:00 | 128.72 | STOP_HIT | 1.00 | -5.76% |
| SELL | retest2 | 2025-09-16 15:15:00 | 123.99 | 2025-09-17 09:15:00 | 128.72 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2025-09-22 09:15:00 | 123.90 | 2025-09-26 09:15:00 | 117.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 12:00:00 | 123.81 | 2025-09-26 09:15:00 | 117.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 13:30:00 | 123.60 | 2025-09-26 09:15:00 | 117.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 09:15:00 | 123.90 | 2025-10-09 09:15:00 | 119.07 | STOP_HIT | 0.50 | 3.90% |
| SELL | retest2 | 2025-09-22 12:00:00 | 123.81 | 2025-10-09 09:15:00 | 119.07 | STOP_HIT | 0.50 | 3.83% |
| SELL | retest2 | 2025-09-22 13:30:00 | 123.60 | 2025-10-09 09:15:00 | 119.07 | STOP_HIT | 0.50 | 3.67% |
| SELL | retest2 | 2025-10-15 09:30:00 | 121.41 | 2025-10-15 14:15:00 | 125.18 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-10-17 13:00:00 | 121.46 | 2025-10-23 09:15:00 | 129.99 | STOP_HIT | 1.00 | -7.02% |
| SELL | retest2 | 2025-10-17 15:15:00 | 121.45 | 2025-10-23 09:15:00 | 129.99 | STOP_HIT | 1.00 | -7.03% |
| BUY | retest2 | 2025-12-09 11:15:00 | 131.51 | 2026-01-06 13:15:00 | 129.22 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-12-09 11:45:00 | 131.68 | 2026-01-06 13:15:00 | 129.22 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-12-09 15:00:00 | 132.00 | 2026-01-06 13:15:00 | 129.22 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-12-29 14:15:00 | 131.51 | 2026-01-06 13:15:00 | 129.22 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-01-08 09:30:00 | 131.49 | 2026-01-08 10:15:00 | 128.78 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2026-01-29 09:15:00 | 125.15 | 2026-02-03 09:15:00 | 143.59 | STOP_HIT | 1.00 | -14.73% |
| SELL | retest2 | 2026-01-30 10:15:00 | 125.69 | 2026-02-03 09:15:00 | 143.59 | STOP_HIT | 1.00 | -14.24% |
| SELL | retest2 | 2026-01-30 13:30:00 | 124.96 | 2026-02-03 09:15:00 | 143.59 | STOP_HIT | 1.00 | -14.91% |
| SELL | retest2 | 2026-02-01 11:45:00 | 125.77 | 2026-02-03 09:15:00 | 143.59 | STOP_HIT | 1.00 | -14.17% |
| SELL | retest2 | 2026-02-04 09:15:00 | 138.80 | 2026-02-04 09:15:00 | 148.88 | STOP_HIT | 1.00 | -7.26% |
| SELL | retest2 | 2026-02-05 09:30:00 | 140.86 | 2026-02-06 10:15:00 | 138.70 | STOP_HIT | 1.00 | 1.53% |
| SELL | retest2 | 2026-02-05 10:00:00 | 142.02 | 2026-02-06 10:15:00 | 138.70 | STOP_HIT | 1.00 | 2.34% |
| SELL | retest2 | 2026-02-05 10:45:00 | 141.12 | 2026-02-06 10:15:00 | 138.70 | STOP_HIT | 1.00 | 1.71% |
| BUY | retest2 | 2026-02-25 11:15:00 | 131.65 | 2026-02-25 12:15:00 | 127.70 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2026-04-09 09:30:00 | 122.50 | 2026-04-15 09:15:00 | 124.90 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-04-09 13:15:00 | 122.36 | 2026-04-15 09:15:00 | 124.90 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2026-04-13 09:15:00 | 120.75 | 2026-04-15 09:15:00 | 124.90 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2026-04-13 15:15:00 | 122.35 | 2026-04-15 09:15:00 | 124.90 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2026-04-16 11:45:00 | 124.04 | 2026-04-17 09:15:00 | 126.42 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-04-16 14:45:00 | 123.92 | 2026-04-17 09:15:00 | 126.42 | STOP_HIT | 1.00 | -2.02% |
