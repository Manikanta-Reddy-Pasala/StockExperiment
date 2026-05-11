# APOLLOHOSP (APOLLOHOSP)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3143 bars)
- **Last close:** 8100.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 4 |
| ALERT3 | 24 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 7
- **Target hits / Stop hits / Partials:** 2 / 7 / 0
- **Avg / median % per leg:** 0.79% / -1.59%
- **Sum % (uncompounded):** 7.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 2 | 3 | 0 | 2.96% | 14.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 2 | 3 | 0 | 2.96% | 14.8% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.94% | -7.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.94% | -7.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 2 | 22.2% | 2 | 7 | 0 | 0.79% | 7.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 7072.00 | 7574.90 | 7576.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 7040.00 | 7541.72 | 7559.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 09:15:00 | 7563.50 | 7478.86 | 7524.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 7563.50 | 7478.86 | 7524.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 7563.50 | 7478.86 | 7524.73 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 12:15:00 | 7769.00 | 7554.35 | 7553.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 7782.50 | 7558.79 | 7555.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 7556.50 | 7602.08 | 7578.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 09:15:00 | 7556.50 | 7602.08 | 7578.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 7556.50 | 7602.08 | 7578.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:00:00 | 7556.50 | 7602.08 | 7578.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 7570.50 | 7601.77 | 7578.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 11:00:00 | 7570.50 | 7601.77 | 7578.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 7497.50 | 7600.73 | 7578.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:00:00 | 7497.50 | 7600.73 | 7578.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 7626.50 | 7616.40 | 7588.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:30:00 | 7595.50 | 7616.40 | 7588.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 7611.50 | 7616.35 | 7588.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 12:00:00 | 7659.00 | 7616.78 | 7589.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 7570.00 | 7638.10 | 7603.25 | SL hit (close<static) qty=1.00 sl=7579.50 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 13:15:00 | 7417.00 | 7575.63 | 7576.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 7365.00 | 7573.54 | 7575.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 7541.00 | 7531.81 | 7553.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 09:15:00 | 7541.00 | 7531.81 | 7553.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 7541.00 | 7531.81 | 7553.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 7541.00 | 7531.81 | 7553.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 7573.00 | 7532.22 | 7553.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:00:00 | 7573.00 | 7532.22 | 7553.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 11:15:00 | 7588.00 | 7532.78 | 7553.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 12:00:00 | 7588.00 | 7532.78 | 7553.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 12:15:00 | 7599.00 | 7533.44 | 7553.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 12:30:00 | 7599.50 | 7533.44 | 7553.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 7600.50 | 7535.61 | 7554.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 7600.50 | 7535.61 | 7554.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 7585.00 | 7536.10 | 7554.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:30:00 | 7618.00 | 7536.10 | 7554.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 7547.00 | 7536.84 | 7554.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 7430.00 | 7536.84 | 7554.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 14:00:00 | 7527.50 | 7473.31 | 7513.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 11:30:00 | 7528.50 | 7475.23 | 7513.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 12:30:00 | 7527.00 | 7475.74 | 7513.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 13:15:00 | 7510.00 | 7476.08 | 7513.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 7648.50 | 7478.53 | 7514.08 | SL hit (close>static) qty=1.00 sl=7584.50 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 14:15:00 | 7673.00 | 7542.68 | 7542.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 12:15:00 | 7750.00 | 7550.12 | 7546.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 7570.50 | 7605.45 | 7576.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 7570.50 | 7605.45 | 7576.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 7570.50 | 7605.45 | 7576.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 7570.50 | 7605.45 | 7576.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 7581.00 | 7605.21 | 7576.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:00:00 | 7602.00 | 7605.17 | 7576.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-02 09:15:00 | 7059.00 | 2025-06-03 09:15:00 | 6838.00 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2025-06-05 09:30:00 | 6902.00 | 2025-06-06 09:15:00 | 6841.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-06-06 13:15:00 | 6906.50 | 2025-07-03 12:15:00 | 7597.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-09 10:15:00 | 6904.00 | 2025-07-03 12:15:00 | 7594.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-09 12:00:00 | 7659.00 | 2026-03-12 11:15:00 | 7570.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-03-30 09:15:00 | 7430.00 | 2026-04-15 09:15:00 | 7648.50 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2026-04-10 14:00:00 | 7527.50 | 2026-04-15 09:15:00 | 7648.50 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-04-13 11:30:00 | 7528.50 | 2026-04-15 09:15:00 | 7648.50 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-04-13 12:30:00 | 7527.00 | 2026-04-15 09:15:00 | 7648.50 | STOP_HIT | 1.00 | -1.61% |
