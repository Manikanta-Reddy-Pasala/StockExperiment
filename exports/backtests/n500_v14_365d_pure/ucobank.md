# UCO Bank (UCOBANK)

## Backtest Summary

- **Window:** 2025-01-15 09:15:00 → 2026-05-08 15:15:00 (2263 bars)
- **Last close:** 26.79
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
| ALERT2_SKIP | 0 |
| ALERT3 | 18 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / Stop hits / Partials:** 0 / 6 / 2
- **Avg / median % per leg:** 0.67% / 1.37%
- **Sum % (uncompounded):** 5.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.94% | -7.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.94% | -7.8% |
| SELL (all) | 4 | 4 | 100.0% | 0 | 2 | 2 | 3.28% | 13.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 0 | 2 | 2 | 3.28% | 13.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 4 | 50.0% | 0 | 6 | 2 | 0.67% | 5.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 31.85 | 30.18 | 30.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 13:15:00 | 32.03 | 30.21 | 30.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 31.55 | 31.68 | 31.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-07 09:45:00 | 31.58 | 31.68 | 31.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 31.15 | 31.73 | 31.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:30:00 | 31.20 | 31.73 | 31.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 31.13 | 31.72 | 31.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:45:00 | 31.21 | 31.72 | 31.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 31.24 | 31.72 | 31.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:30:00 | 31.20 | 31.72 | 31.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 31.16 | 31.71 | 31.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:45:00 | 31.20 | 31.71 | 31.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 31.20 | 31.71 | 31.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 31.20 | 31.71 | 31.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 31.10 | 31.70 | 31.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 31.27 | 31.70 | 31.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 11:15:00 | 30.97 | 31.68 | 31.19 | SL hit (close<static) qty=1.00 sl=31.01 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 31.71 | 31.59 | 31.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:00:00 | 31.40 | 31.57 | 31.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 30.71 | 31.54 | 31.20 | SL hit (close<static) qty=1.00 sl=31.01 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 30.71 | 31.54 | 31.20 | SL hit (close<static) qty=1.00 sl=31.01 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 09:30:00 | 31.36 | 31.41 | 31.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 31.31 | 31.41 | 31.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:45:00 | 31.32 | 31.41 | 31.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 31.09 | 31.40 | 31.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:00:00 | 31.09 | 31.40 | 31.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 31.09 | 31.40 | 31.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:30:00 | 31.04 | 31.40 | 31.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 30.90 | 31.39 | 31.16 | SL hit (close<static) qty=1.00 sl=31.01 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 30.90 | 31.39 | 31.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:30:00 | 30.92 | 31.39 | 31.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 31.00 | 31.33 | 31.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 31.00 | 31.33 | 31.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 30.81 | 31.32 | 31.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 30.81 | 31.32 | 31.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 30.96 | 31.29 | 31.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:30:00 | 30.97 | 31.29 | 31.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 14:15:00 | 29.42 | 30.99 | 31.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 28.97 | 30.96 | 30.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 11:15:00 | 29.58 | 29.50 | 30.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 11:30:00 | 29.63 | 29.50 | 30.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 30.06 | 29.51 | 30.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 30.06 | 29.51 | 30.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 30.22 | 29.52 | 30.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:00:00 | 30.22 | 29.52 | 30.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 30.30 | 29.53 | 30.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:45:00 | 30.30 | 29.53 | 30.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 30.20 | 29.59 | 30.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:00:00 | 30.20 | 29.59 | 30.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 30.10 | 29.60 | 30.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 14:00:00 | 29.88 | 29.61 | 30.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 11:45:00 | 29.99 | 29.63 | 30.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 28.39 | 29.56 | 29.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 28.49 | 29.56 | 29.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 12:15:00 | 29.47 | 29.46 | 29.87 | SL hit (close>ema200) qty=0.50 sl=29.46 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 12:15:00 | 29.47 | 29.46 | 29.87 | SL hit (close>ema200) qty=0.50 sl=29.46 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-13 09:15:00 | 31.27 | 2025-11-13 11:15:00 | 30.97 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-11-17 09:15:00 | 31.71 | 2025-11-21 09:15:00 | 30.71 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2025-11-19 10:00:00 | 31.40 | 2025-11-21 09:15:00 | 30.71 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-11-26 09:30:00 | 31.36 | 2025-11-27 09:15:00 | 30.90 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-01-06 14:00:00 | 29.88 | 2026-01-12 09:15:00 | 28.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 11:45:00 | 29.99 | 2026-01-12 09:15:00 | 28.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 14:00:00 | 29.88 | 2026-01-14 12:15:00 | 29.47 | STOP_HIT | 0.50 | 1.37% |
| SELL | retest2 | 2026-01-07 11:45:00 | 29.99 | 2026-01-14 12:15:00 | 29.47 | STOP_HIT | 0.50 | 1.73% |
