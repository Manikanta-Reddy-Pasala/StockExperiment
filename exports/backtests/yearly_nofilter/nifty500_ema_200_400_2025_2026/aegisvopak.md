# Aegis Vopak Terminals Ltd. (AEGISVOPAK)

## Backtest Summary

- **Window:** 2025-06-02 09:15:00 → 2026-05-08 15:15:00 (1619 bars)
- **Last close:** 211.15
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
| ALERT2_SKIP | 2 |
| ALERT3 | 11 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 8 |
| TARGET_HIT | 4 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 12 / 8
- **Target hits / Stop hits / Partials:** 4 / 8 / 8
- **Avg / median % per leg:** 2.14% / 5.00%
- **Sum % (uncompounded):** 42.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.54% | -13.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.54% | -13.6% |
| SELL (all) | 17 | 12 | 70.6% | 4 | 5 | 8 | 3.32% | 56.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 17 | 12 | 70.6% | 4 | 5 | 8 | 3.32% | 56.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 20 | 12 | 60.0% | 4 | 8 | 8 | 2.14% | 42.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 11:15:00 | 278.10 | 249.87 | 249.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 15:15:00 | 283.50 | 259.80 | 255.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 10:15:00 | 260.95 | 270.37 | 263.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 10:15:00 | 260.95 | 270.37 | 263.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 260.95 | 270.37 | 263.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 262.35 | 270.37 | 263.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 259.55 | 270.26 | 263.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:00:00 | 259.55 | 270.26 | 263.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 260.95 | 268.18 | 263.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 259.45 | 268.18 | 263.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 262.10 | 268.12 | 263.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:15:00 | 260.85 | 268.12 | 263.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 261.10 | 268.05 | 263.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:00:00 | 261.10 | 268.05 | 263.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 261.10 | 267.98 | 263.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:30:00 | 262.30 | 267.98 | 263.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 272.30 | 267.97 | 263.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 10:15:00 | 274.30 | 268.06 | 263.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 12:15:00 | 262.50 | 268.10 | 263.92 | SL hit (close<static) qty=1.00 sl=262.60 alert=retest2 |

### Cycle 2 — SELL (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 12:15:00 | 243.80 | 262.00 | 262.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 15:15:00 | 242.20 | 261.44 | 261.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 258.10 | 257.92 | 259.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 14:15:00 | 258.10 | 257.92 | 259.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 258.10 | 257.92 | 259.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:45:00 | 259.15 | 257.92 | 259.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 258.50 | 257.93 | 259.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 256.00 | 257.93 | 259.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 262.50 | 257.87 | 259.70 | SL hit (close>static) qty=1.00 sl=260.90 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-12 10:15:00 | 274.30 | 2025-11-18 12:15:00 | 262.50 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2025-12-01 09:15:00 | 274.90 | 2025-12-05 10:15:00 | 261.90 | STOP_HIT | 1.00 | -4.73% |
| BUY | retest2 | 2025-12-02 14:45:00 | 274.50 | 2025-12-05 10:15:00 | 261.90 | STOP_HIT | 1.00 | -4.59% |
| SELL | retest2 | 2025-12-18 09:15:00 | 256.00 | 2025-12-19 09:15:00 | 262.50 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-12-19 12:45:00 | 257.00 | 2025-12-26 09:15:00 | 244.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-22 11:15:00 | 257.40 | 2025-12-26 09:15:00 | 244.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-22 12:15:00 | 257.60 | 2025-12-26 09:15:00 | 244.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-19 12:45:00 | 257.00 | 2026-01-09 12:15:00 | 231.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-22 11:15:00 | 257.40 | 2026-01-09 12:15:00 | 231.66 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-22 12:15:00 | 257.60 | 2026-01-09 12:15:00 | 231.84 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-06 15:00:00 | 218.00 | 2026-02-13 09:15:00 | 207.86 | PARTIAL | 0.50 | 4.65% |
| SELL | retest2 | 2026-02-09 12:30:00 | 218.80 | 2026-02-13 09:15:00 | 207.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-09 15:00:00 | 218.80 | 2026-02-13 09:15:00 | 207.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 09:30:00 | 218.79 | 2026-02-13 11:15:00 | 207.10 | PARTIAL | 0.50 | 5.34% |
| SELL | retest2 | 2026-02-06 15:00:00 | 218.00 | 2026-02-23 09:15:00 | 230.08 | STOP_HIT | 0.50 | -5.54% |
| SELL | retest2 | 2026-02-09 12:30:00 | 218.80 | 2026-02-23 09:15:00 | 230.08 | STOP_HIT | 0.50 | -5.16% |
| SELL | retest2 | 2026-02-09 15:00:00 | 218.80 | 2026-02-23 09:15:00 | 230.08 | STOP_HIT | 0.50 | -5.16% |
| SELL | retest2 | 2026-02-11 09:30:00 | 218.79 | 2026-02-23 09:15:00 | 230.08 | STOP_HIT | 0.50 | -5.16% |
| SELL | retest2 | 2026-03-04 09:15:00 | 209.75 | 2026-03-04 14:15:00 | 199.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-04 09:15:00 | 209.75 | 2026-03-05 10:15:00 | 188.78 | TARGET_HIT | 0.50 | 10.00% |
