# IRB Infrastructure Developers Ltd. (IRB)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 21.46
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
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 25 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 41 |
| PARTIAL | 13 |
| TARGET_HIT | 0 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 28
- **Target hits / Stop hits / Partials:** 0 / 41 / 13
- **Avg / median % per leg:** -0.09% / -1.01%
- **Sum % (uncompounded):** -4.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 0 | 0.0% | 0 | 11 | 0 | -2.18% | -24.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 0 | 0.0% | 0 | 11 | 0 | -2.18% | -24.0% |
| SELL (all) | 43 | 26 | 60.5% | 0 | 30 | 13 | 0.45% | 19.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 43 | 26 | 60.5% | 0 | 30 | 13 | 0.45% | 19.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 54 | 26 | 48.1% | 0 | 41 | 13 | -0.09% | -4.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 25.74 | 23.93 | 23.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 26.04 | 24.32 | 24.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 24.99 | 25.19 | 24.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 24.99 | 25.19 | 24.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 24.99 | 25.19 | 24.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:30:00 | 25.01 | 25.19 | 24.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 11:15:00 | 25.06 | 25.19 | 24.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 15:00:00 | 25.08 | 25.18 | 24.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 14:00:00 | 25.01 | 25.16 | 24.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 24.77 | 25.15 | 24.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:00:00 | 24.77 | 25.15 | 24.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 24.65 | 25.14 | 24.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:45:00 | 24.63 | 25.14 | 24.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 24.75 | 25.14 | 24.73 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-18 15:15:00 | 24.18 | 25.08 | 24.72 | SL hit (close<static) qty=1.00 sl=24.31 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-18 15:15:00 | 24.18 | 25.08 | 24.72 | SL hit (close<static) qty=1.00 sl=24.31 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-18 15:15:00 | 24.18 | 25.08 | 24.72 | SL hit (close<static) qty=1.00 sl=24.31 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-18 15:15:00 | 24.18 | 25.08 | 24.72 | SL hit (close<static) qty=1.00 sl=24.31 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 13:15:00 | 24.85 | 24.95 | 24.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 24.60 | 24.94 | 24.69 | SL hit (close<static) qty=1.00 sl=24.63 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 11:30:00 | 24.91 | 24.94 | 24.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 12:00:00 | 24.88 | 24.94 | 24.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 13:15:00 | 24.46 | 24.93 | 24.69 | SL hit (close<static) qty=1.00 sl=24.63 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 13:15:00 | 24.46 | 24.93 | 24.69 | SL hit (close<static) qty=1.00 sl=24.63 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 13:00:00 | 24.89 | 24.91 | 24.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 24.90 | 24.91 | 24.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:30:00 | 24.79 | 24.91 | 24.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 24.53 | 24.91 | 24.71 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 24.53 | 24.91 | 24.71 | SL hit (close<static) qty=1.00 sl=24.63 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 24.53 | 24.91 | 24.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 24.51 | 24.91 | 24.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 24.51 | 24.91 | 24.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 24.85 | 24.90 | 24.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 24.95 | 24.89 | 24.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 24.65 | 24.89 | 24.71 | SL hit (close<static) qty=1.00 sl=24.69 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:45:00 | 25.05 | 24.89 | 24.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 13:15:00 | 24.68 | 24.88 | 24.71 | SL hit (close<static) qty=1.00 sl=24.69 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 09:45:00 | 24.94 | 24.87 | 24.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 11:15:00 | 24.53 | 24.86 | 24.71 | SL hit (close<static) qty=1.00 sl=24.69 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 10:15:00 | 24.29 | 24.60 | 24.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 11:15:00 | 24.19 | 24.60 | 24.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 10:15:00 | 22.07 | 21.95 | 22.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 11:00:00 | 22.07 | 21.95 | 22.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 21.87 | 21.36 | 21.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:00:00 | 21.87 | 21.36 | 21.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 21.70 | 21.35 | 21.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 21.61 | 21.37 | 21.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 10:15:00 | 21.58 | 21.38 | 21.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 13:30:00 | 21.61 | 21.39 | 21.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 22.60 | 21.41 | 21.83 | SL hit (close>static) qty=1.00 sl=21.91 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 22.60 | 21.41 | 21.83 | SL hit (close>static) qty=1.00 sl=21.91 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 22.60 | 21.41 | 21.83 | SL hit (close>static) qty=1.00 sl=21.91 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:30:00 | 21.54 | 21.82 | 21.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 21.60 | 21.79 | 21.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 14:15:00 | 21.56 | 21.79 | 21.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 21.53 | 21.79 | 21.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 13:30:00 | 21.53 | 21.74 | 21.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 22.59 | 21.75 | 21.89 | SL hit (close>static) qty=1.00 sl=21.91 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 22.59 | 21.75 | 21.89 | SL hit (close>static) qty=1.00 sl=22.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 22.59 | 21.75 | 21.89 | SL hit (close>static) qty=1.00 sl=22.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 22.59 | 21.75 | 21.89 | SL hit (close>static) qty=1.00 sl=22.30 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 10:30:00 | 21.55 | 21.82 | 21.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 21.50 | 21.75 | 21.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:30:00 | 21.71 | 21.75 | 21.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 20.47 | 21.60 | 21.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 11:15:00 | 21.25 | 21.17 | 21.46 | SL hit (close>ema200) qty=0.50 sl=21.17 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 21.30 | 21.18 | 21.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 11:30:00 | 21.18 | 21.18 | 21.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 12:15:00 | 21.22 | 21.18 | 21.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 12:45:00 | 21.21 | 21.18 | 21.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 13:15:00 | 21.21 | 21.19 | 21.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 21.40 | 21.14 | 21.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 15:00:00 | 21.40 | 21.14 | 21.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 21.43 | 21.14 | 21.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:15:00 | 21.19 | 21.14 | 21.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 21.21 | 21.14 | 21.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 15:15:00 | 21.10 | 21.14 | 21.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 09:30:00 | 21.09 | 21.13 | 21.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 10:15:00 | 21.11 | 21.13 | 21.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 13:30:00 | 21.08 | 21.13 | 21.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 21.32 | 21.14 | 21.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:30:00 | 21.30 | 21.14 | 21.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 20.16 | 21.11 | 21.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 20.23 | 21.11 | 21.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 20.15 | 21.09 | 21.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 20.15 | 21.09 | 21.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 20.12 | 20.92 | 21.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 20.05 | 20.90 | 21.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 20.04 | 20.90 | 21.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 20.05 | 20.90 | 21.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 20.03 | 20.90 | 21.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 20.67 | 20.62 | 20.96 | SL hit (close>ema200) qty=0.50 sl=20.62 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 20.67 | 20.62 | 20.96 | SL hit (close>ema200) qty=0.50 sl=20.62 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 20.67 | 20.62 | 20.96 | SL hit (close>ema200) qty=0.50 sl=20.62 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 20.67 | 20.62 | 20.96 | SL hit (close>ema200) qty=0.50 sl=20.62 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 20.67 | 20.62 | 20.96 | SL hit (close>ema200) qty=0.50 sl=20.62 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 20.67 | 20.62 | 20.96 | SL hit (close>ema200) qty=0.50 sl=20.62 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 20.67 | 20.62 | 20.96 | SL hit (close>ema200) qty=0.50 sl=20.62 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 20.67 | 20.62 | 20.96 | SL hit (close>ema200) qty=0.50 sl=20.62 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 20.67 | 20.62 | 20.96 | SL hit (close>ema200) qty=0.50 sl=20.62 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:30:00 | 21.22 | 20.67 | 20.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 12:15:00 | 21.29 | 20.68 | 20.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 09:15:00 | 22.04 | 20.73 | 20.94 | SL hit (close>static) qty=1.00 sl=21.68 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-09 09:15:00 | 22.04 | 20.73 | 20.94 | SL hit (close>static) qty=1.00 sl=21.68 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 15:15:00 | 21.95 | 21.12 | 21.12 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 20.61 | 21.12 | 21.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 09:15:00 | 20.35 | 21.09 | 21.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 20.89 | 20.88 | 20.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 20.89 | 20.88 | 20.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 20.89 | 20.88 | 20.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:45:00 | 20.94 | 20.88 | 20.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 20.76 | 20.62 | 20.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 12:45:00 | 20.67 | 20.70 | 20.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 09:15:00 | 20.58 | 20.71 | 20.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 10:00:00 | 20.58 | 20.70 | 20.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 15:00:00 | 20.61 | 20.69 | 20.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 20.67 | 20.69 | 20.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 20.86 | 20.69 | 20.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 20.83 | 20.69 | 20.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 20.71 | 20.69 | 20.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 20.64 | 20.69 | 20.83 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 20.91 | 20.69 | 20.83 | SL hit (close>static) qty=1.00 sl=20.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 20.91 | 20.69 | 20.83 | SL hit (close>static) qty=1.00 sl=20.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 20.91 | 20.69 | 20.83 | SL hit (close>static) qty=1.00 sl=20.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 20.91 | 20.69 | 20.83 | SL hit (close>static) qty=1.00 sl=20.85 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 10:00:00 | 20.56 | 20.70 | 20.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 20.55 | 20.71 | 20.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 10:00:00 | 20.43 | 20.71 | 20.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 10:00:00 | 20.57 | 20.66 | 20.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 20.40 | 20.65 | 20.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 21.08 | 20.65 | 20.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 22.41 | 20.66 | 20.78 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 22.41 | 20.66 | 20.78 | SL hit (close>static) qty=1.00 sl=20.94 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 22.41 | 20.66 | 20.78 | SL hit (close>static) qty=1.00 sl=20.94 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 22.41 | 20.66 | 20.78 | SL hit (close>static) qty=1.00 sl=20.94 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 22.41 | 20.66 | 20.78 | SL hit (close>static) qty=1.00 sl=20.94 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-03-30 09:30:00 | 22.52 | 20.66 | 20.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 22.38 | 20.68 | 20.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 13:30:00 | 22.04 | 20.73 | 20.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 22.12 | 20.76 | 20.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 22.06 | 20.82 | 20.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 20.94 | 20.84 | 20.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 20.96 | 20.84 | 20.86 | SL hit (close>static) qty=0.50 sl=20.84 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 21.01 | 20.84 | 20.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 20.96 | 20.84 | 20.86 | SL hit (close>static) qty=0.50 sl=20.84 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 20.96 | 20.84 | 20.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 20.96 | 20.84 | 20.86 | SL hit (close>static) qty=0.50 sl=20.84 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 13:15:00 | 21.53 | 20.89 | 20.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 14:15:00 | 21.68 | 20.89 | 20.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 21.39 | 21.48 | 21.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:00:00 | 21.39 | 21.48 | 21.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-13 10:30:00 | 25.01 | 2025-06-18 15:15:00 | 24.18 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2025-06-13 11:15:00 | 25.06 | 2025-06-18 15:15:00 | 24.18 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2025-06-13 15:00:00 | 25.08 | 2025-06-18 15:15:00 | 24.18 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2025-06-16 14:00:00 | 25.01 | 2025-06-18 15:15:00 | 24.18 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2025-06-23 13:15:00 | 24.85 | 2025-06-24 09:15:00 | 24.60 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-06-24 11:30:00 | 24.91 | 2025-06-24 13:15:00 | 24.46 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-06-24 12:00:00 | 24.88 | 2025-06-24 13:15:00 | 24.46 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-06-25 13:00:00 | 24.89 | 2025-07-01 09:15:00 | 24.53 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-07-02 09:15:00 | 24.95 | 2025-07-02 09:15:00 | 24.65 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-07-02 11:45:00 | 25.05 | 2025-07-03 13:15:00 | 24.68 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-07-07 09:45:00 | 24.94 | 2025-07-07 11:15:00 | 24.53 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-10-24 09:15:00 | 21.61 | 2025-10-27 09:15:00 | 22.60 | STOP_HIT | 1.00 | -4.58% |
| SELL | retest2 | 2025-10-24 10:15:00 | 21.58 | 2025-10-27 09:15:00 | 22.60 | STOP_HIT | 1.00 | -4.73% |
| SELL | retest2 | 2025-10-24 13:30:00 | 21.61 | 2025-10-27 09:15:00 | 22.60 | STOP_HIT | 1.00 | -4.58% |
| SELL | retest2 | 2025-11-11 09:30:00 | 21.54 | 2025-11-17 09:15:00 | 22.59 | STOP_HIT | 1.00 | -4.87% |
| SELL | retest2 | 2025-11-12 14:15:00 | 21.56 | 2025-11-17 09:15:00 | 22.59 | STOP_HIT | 1.00 | -4.78% |
| SELL | retest2 | 2025-11-13 09:15:00 | 21.53 | 2025-11-17 09:15:00 | 22.59 | STOP_HIT | 1.00 | -4.92% |
| SELL | retest2 | 2025-11-14 13:30:00 | 21.53 | 2025-11-17 09:15:00 | 22.59 | STOP_HIT | 1.00 | -4.92% |
| SELL | retest2 | 2025-11-24 10:30:00 | 21.55 | 2025-12-09 09:15:00 | 20.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 10:30:00 | 21.55 | 2025-12-23 11:15:00 | 21.25 | STOP_HIT | 0.50 | 1.39% |
| SELL | retest2 | 2025-12-24 11:30:00 | 21.18 | 2026-01-12 09:15:00 | 20.16 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2025-12-24 12:15:00 | 21.22 | 2026-01-12 09:15:00 | 20.23 | PARTIAL | 0.50 | 4.64% |
| SELL | retest2 | 2025-12-24 12:45:00 | 21.21 | 2026-01-12 11:15:00 | 20.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-26 13:15:00 | 21.21 | 2026-01-12 11:15:00 | 20.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 15:15:00 | 21.10 | 2026-01-20 15:15:00 | 20.12 | PARTIAL | 0.50 | 4.64% |
| SELL | retest2 | 2026-01-07 09:30:00 | 21.09 | 2026-01-21 09:15:00 | 20.05 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2026-01-07 10:15:00 | 21.11 | 2026-01-21 09:15:00 | 20.04 | PARTIAL | 0.50 | 5.09% |
| SELL | retest2 | 2026-01-07 13:30:00 | 21.08 | 2026-01-21 09:15:00 | 20.05 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2026-01-08 10:30:00 | 21.30 | 2026-01-21 09:15:00 | 20.03 | PARTIAL | 0.50 | 5.98% |
| SELL | retest2 | 2025-12-24 11:30:00 | 21.18 | 2026-01-30 10:15:00 | 20.67 | STOP_HIT | 0.50 | 2.41% |
| SELL | retest2 | 2025-12-24 12:15:00 | 21.22 | 2026-01-30 10:15:00 | 20.67 | STOP_HIT | 0.50 | 2.59% |
| SELL | retest2 | 2025-12-24 12:45:00 | 21.21 | 2026-01-30 10:15:00 | 20.67 | STOP_HIT | 0.50 | 2.55% |
| SELL | retest2 | 2025-12-26 13:15:00 | 21.21 | 2026-01-30 10:15:00 | 20.67 | STOP_HIT | 0.50 | 2.55% |
| SELL | retest2 | 2026-01-05 15:15:00 | 21.10 | 2026-01-30 10:15:00 | 20.67 | STOP_HIT | 0.50 | 2.04% |
| SELL | retest2 | 2026-01-07 09:30:00 | 21.09 | 2026-01-30 10:15:00 | 20.67 | STOP_HIT | 0.50 | 1.99% |
| SELL | retest2 | 2026-01-07 10:15:00 | 21.11 | 2026-01-30 10:15:00 | 20.67 | STOP_HIT | 0.50 | 2.08% |
| SELL | retest2 | 2026-01-07 13:30:00 | 21.08 | 2026-01-30 10:15:00 | 20.67 | STOP_HIT | 0.50 | 1.94% |
| SELL | retest2 | 2026-01-08 10:30:00 | 21.30 | 2026-01-30 10:15:00 | 20.67 | STOP_HIT | 0.50 | 2.96% |
| SELL | retest2 | 2026-02-06 09:30:00 | 21.22 | 2026-02-09 09:15:00 | 22.04 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2026-02-06 12:15:00 | 21.29 | 2026-02-09 09:15:00 | 22.04 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2026-03-13 12:45:00 | 20.67 | 2026-03-18 09:15:00 | 20.91 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-03-16 09:15:00 | 20.58 | 2026-03-18 09:15:00 | 20.91 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-03-16 10:00:00 | 20.58 | 2026-03-18 09:15:00 | 20.91 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-03-16 15:00:00 | 20.61 | 2026-03-18 09:15:00 | 20.91 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-03-19 10:00:00 | 20.56 | 2026-03-30 09:15:00 | 22.41 | STOP_HIT | 1.00 | -9.00% |
| SELL | retest2 | 2026-03-23 09:15:00 | 20.55 | 2026-03-30 09:15:00 | 22.41 | STOP_HIT | 1.00 | -9.05% |
| SELL | retest2 | 2026-03-23 10:00:00 | 20.43 | 2026-03-30 09:15:00 | 22.41 | STOP_HIT | 1.00 | -9.69% |
| SELL | retest2 | 2026-03-27 10:00:00 | 20.57 | 2026-03-30 09:15:00 | 22.41 | STOP_HIT | 1.00 | -8.95% |
| SELL | retest2 | 2026-03-30 13:30:00 | 22.04 | 2026-04-02 09:15:00 | 20.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-30 13:30:00 | 22.04 | 2026-04-02 09:15:00 | 20.96 | STOP_HIT | 0.50 | 4.90% |
| SELL | retest2 | 2026-04-01 10:15:00 | 22.12 | 2026-04-02 09:15:00 | 21.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 22.12 | 2026-04-02 09:15:00 | 20.96 | STOP_HIT | 0.50 | 5.24% |
| SELL | retest2 | 2026-04-01 13:30:00 | 22.06 | 2026-04-02 09:15:00 | 20.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 13:30:00 | 22.06 | 2026-04-02 09:15:00 | 20.96 | STOP_HIT | 0.50 | 4.99% |
