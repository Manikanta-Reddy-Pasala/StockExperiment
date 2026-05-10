# Indian Overseas Bank (IOB)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 34.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 12
- **Target hits / Stop hits / Partials:** 2 / 14 / 4
- **Avg / median % per leg:** 0.89% / -1.13%
- **Sum % (uncompounded):** 17.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.99% | -15.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.99% | -15.9% |
| SELL (all) | 12 | 8 | 66.7% | 2 | 6 | 4 | 2.81% | 33.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 8 | 66.7% | 2 | 6 | 4 | 2.81% | 33.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 20 | 8 | 40.0% | 2 | 14 | 4 | 0.89% | 17.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 39.90 | 38.50 | 38.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 40.20 | 38.62 | 38.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 38.92 | 39.21 | 38.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 14:15:00 | 38.92 | 39.21 | 38.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 38.92 | 39.21 | 38.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 38.92 | 39.21 | 38.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 38.86 | 39.21 | 38.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 38.59 | 39.21 | 38.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 38.47 | 39.20 | 38.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 38.65 | 39.20 | 38.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 38.49 | 39.19 | 38.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:15:00 | 38.41 | 39.19 | 38.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 38.30 | 39.10 | 38.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 39.14 | 39.10 | 38.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:45:00 | 38.93 | 39.23 | 38.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 10:00:00 | 38.82 | 39.32 | 39.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:00:00 | 38.83 | 39.27 | 39.04 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 40.15 | 39.28 | 39.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 39.60 | 39.28 | 39.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 38.98 | 39.35 | 39.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:00:00 | 38.98 | 39.35 | 39.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 39.32 | 39.35 | 39.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:15:00 | 39.76 | 39.33 | 39.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 10:15:00 | 38.81 | 39.76 | 39.43 | SL hit (close<static) qty=1.00 sl=38.93 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 38.06 | 39.70 | 39.40 | SL hit (close<static) qty=1.00 sl=38.22 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 38.06 | 39.70 | 39.40 | SL hit (close<static) qty=1.00 sl=38.22 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 38.06 | 39.70 | 39.40 | SL hit (close<static) qty=1.00 sl=38.22 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 38.06 | 39.70 | 39.40 | SL hit (close<static) qty=1.00 sl=38.22 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 13:30:00 | 39.48 | 39.67 | 39.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 39.43 | 39.63 | 39.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 10:15:00 | 38.90 | 39.58 | 39.38 | SL hit (close<static) qty=1.00 sl=38.93 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 10:15:00 | 38.90 | 39.58 | 39.38 | SL hit (close<static) qty=1.00 sl=38.93 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 39.62 | 39.50 | 39.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 39.80 | 39.58 | 39.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:45:00 | 39.45 | 39.58 | 39.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 39.46 | 39.60 | 39.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 39.21 | 39.60 | 39.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 38.91 | 39.60 | 39.43 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 38.91 | 39.60 | 39.43 | SL hit (close<static) qty=1.00 sl=38.93 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 38.91 | 39.60 | 39.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 38.89 | 39.59 | 39.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:45:00 | 38.99 | 39.59 | 39.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 39.45 | 39.47 | 39.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:30:00 | 39.40 | 39.47 | 39.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 39.47 | 39.47 | 39.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 14:30:00 | 39.44 | 39.47 | 39.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 39.22 | 39.47 | 39.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:00:00 | 39.22 | 39.47 | 39.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 39.21 | 39.46 | 39.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:15:00 | 39.11 | 39.46 | 39.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 39.02 | 39.40 | 39.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 39.02 | 39.40 | 39.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 38.10 | 39.30 | 39.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 09:15:00 | 37.94 | 39.23 | 39.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 36.33 | 36.31 | 37.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:45:00 | 36.24 | 36.31 | 37.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 37.32 | 36.33 | 37.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:15:00 | 37.26 | 36.33 | 37.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 37.13 | 36.34 | 37.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 10:45:00 | 36.87 | 36.35 | 37.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 13:15:00 | 36.86 | 36.39 | 37.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 14:15:00 | 35.03 | 36.31 | 37.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 14:15:00 | 35.02 | 36.31 | 37.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 12:15:00 | 36.59 | 36.16 | 37.02 | SL hit (close>ema200) qty=0.50 sl=36.16 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 12:15:00 | 36.59 | 36.16 | 37.02 | SL hit (close>ema200) qty=0.50 sl=36.16 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:00:00 | 36.76 | 35.52 | 35.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 11:45:00 | 36.71 | 35.62 | 36.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 34.92 | 35.82 | 36.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 13:15:00 | 34.87 | 35.80 | 36.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 33.08 | 35.48 | 35.85 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 33.04 | 35.48 | 35.85 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 34.70 | 33.38 | 34.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:45:00 | 34.57 | 33.38 | 34.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 34.52 | 33.39 | 34.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 15:00:00 | 34.39 | 33.42 | 34.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:30:00 | 34.27 | 33.44 | 34.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 11:00:00 | 34.41 | 33.45 | 34.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 34.80 | 33.50 | 34.29 | SL hit (close>static) qty=1.00 sl=34.78 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 34.80 | 33.50 | 34.29 | SL hit (close>static) qty=1.00 sl=34.78 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 34.80 | 33.50 | 34.29 | SL hit (close>static) qty=1.00 sl=34.78 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 33.83 | 33.57 | 34.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 34.35 | 33.58 | 34.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 10:45:00 | 34.38 | 33.58 | 34.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 11:15:00 | 34.42 | 33.59 | 34.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:45:00 | 34.39 | 33.59 | 34.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 12:15:00 | 34.42 | 33.60 | 34.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 12:30:00 | 34.47 | 33.60 | 34.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 35.14 | 33.64 | 34.31 | SL hit (close>static) qty=1.00 sl=34.78 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 34.65 | 34.52 | 34.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 34.90 | 34.52 | 34.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 34.91 | 34.52 | 34.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:45:00 | 34.98 | 34.52 | 34.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 34.87 | 34.61 | 34.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:30:00 | 34.85 | 34.61 | 34.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-09-30 09:15:00 | 39.14 | 2025-11-06 10:15:00 | 38.81 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-10-09 09:45:00 | 38.93 | 2025-11-07 09:15:00 | 38.06 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-10-14 10:00:00 | 38.82 | 2025-11-07 09:15:00 | 38.06 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-10-15 10:00:00 | 38.83 | 2025-11-07 09:15:00 | 38.06 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-10-20 10:15:00 | 39.76 | 2025-11-07 09:15:00 | 38.06 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest2 | 2025-11-07 13:30:00 | 39.48 | 2025-11-13 10:15:00 | 38.90 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-11-12 09:15:00 | 39.43 | 2025-11-13 10:15:00 | 38.90 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-11-17 09:15:00 | 39.62 | 2025-11-21 09:15:00 | 38.91 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-01-05 10:45:00 | 36.87 | 2026-01-09 14:15:00 | 35.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 13:15:00 | 36.86 | 2026-01-09 14:15:00 | 35.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 10:45:00 | 36.87 | 2026-01-14 12:15:00 | 36.59 | STOP_HIT | 0.50 | 0.76% |
| SELL | retest2 | 2026-01-06 13:15:00 | 36.86 | 2026-01-14 12:15:00 | 36.59 | STOP_HIT | 0.50 | 0.73% |
| SELL | retest2 | 2026-02-23 10:00:00 | 36.76 | 2026-03-02 09:15:00 | 34.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 11:45:00 | 36.71 | 2026-03-02 13:15:00 | 34.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 10:00:00 | 36.76 | 2026-03-09 09:15:00 | 33.08 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 11:45:00 | 36.71 | 2026-03-09 09:15:00 | 33.04 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-08 15:00:00 | 34.39 | 2026-04-10 09:15:00 | 34.80 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-04-09 09:30:00 | 34.27 | 2026-04-10 09:15:00 | 34.80 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-04-09 11:00:00 | 34.41 | 2026-04-10 09:15:00 | 34.80 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-04-13 09:15:00 | 33.83 | 2026-04-15 09:15:00 | 35.14 | STOP_HIT | 1.00 | -3.87% |
