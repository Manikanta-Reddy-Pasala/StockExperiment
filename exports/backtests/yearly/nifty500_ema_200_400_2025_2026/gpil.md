# Godawari Power & Ispat Ltd. (GPIL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 295.00
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
| ALERT2_SKIP | 1 |
| ALERT3 | 13 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 0 |
| TARGET_HIT | 11 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 14
- **Target hits / Stop hits / Partials:** 11 / 14 / 0
- **Avg / median % per leg:** 2.91% / -1.63%
- **Sum % (uncompounded):** 72.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 11 | 50.0% | 11 | 11 | 0 | 3.84% | 84.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 11 | 50.0% | 11 | 11 | 0 | 3.84% | 84.5% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.96% | -11.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.96% | -11.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 11 | 44.0% | 11 | 14 | 0 | 2.91% | 72.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 14:15:00 | 178.65 | 189.19 | 189.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 176.20 | 189.06 | 189.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 187.64 | 186.93 | 187.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 14:15:00 | 187.64 | 186.93 | 187.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 187.64 | 186.93 | 187.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 15:00:00 | 187.64 | 186.93 | 187.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 190.00 | 186.97 | 187.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 188.99 | 186.97 | 187.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 189.35 | 186.99 | 187.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 14:00:00 | 187.43 | 187.19 | 188.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 187.96 | 187.24 | 188.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 187.60 | 185.46 | 186.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 195.09 | 185.62 | 186.74 | SL hit (close>static) qty=1.00 sl=191.94 alert=retest2 |

### Cycle 2 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 192.23 | 187.68 | 187.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 15:15:00 | 194.40 | 188.02 | 187.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 15:15:00 | 188.10 | 188.14 | 187.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:15:00 | 189.95 | 188.14 | 187.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 188.71 | 188.16 | 187.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 11:30:00 | 189.52 | 188.20 | 187.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 12:00:00 | 189.76 | 188.20 | 187.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 13:00:00 | 189.66 | 188.21 | 187.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 193.38 | 189.38 | 188.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 190.99 | 189.40 | 188.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 12:30:00 | 194.58 | 189.50 | 188.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 09:15:00 | 186.22 | 190.30 | 189.19 | SL hit (close<static) qty=1.00 sl=187.92 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 15:15:00 | 230.20 | 246.41 | 246.49 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 14:15:00 | 256.34 | 245.73 | 245.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 09:15:00 | 261.00 | 245.99 | 245.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 10:15:00 | 256.35 | 257.03 | 252.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-12 11:00:00 | 256.35 | 257.03 | 252.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 254.60 | 257.67 | 253.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 254.60 | 257.67 | 253.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 251.20 | 257.60 | 253.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:00:00 | 251.20 | 257.60 | 253.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 250.45 | 257.53 | 253.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 12:00:00 | 250.45 | 257.53 | 253.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 248.25 | 254.03 | 252.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:30:00 | 249.70 | 254.03 | 252.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 249.00 | 253.98 | 252.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 253.15 | 253.70 | 252.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 13:15:00 | 246.98 | 253.70 | 252.09 | SL hit (close<static) qty=1.00 sl=247.05 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 13:00:00 | 187.69 | 2025-05-15 09:15:00 | 206.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-12 14:15:00 | 187.66 | 2025-05-15 09:15:00 | 206.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-06 14:45:00 | 188.00 | 2025-06-13 12:15:00 | 184.70 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-06-09 10:30:00 | 187.76 | 2025-06-13 12:15:00 | 184.70 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-06-11 09:15:00 | 187.84 | 2025-06-16 09:15:00 | 182.75 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-06-12 14:15:00 | 186.82 | 2025-06-16 09:15:00 | 182.75 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-06-17 09:30:00 | 186.59 | 2025-06-17 11:15:00 | 184.04 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-06-30 14:00:00 | 187.43 | 2025-07-17 09:15:00 | 195.09 | STOP_HIT | 1.00 | -4.09% |
| SELL | retest2 | 2025-07-01 09:15:00 | 187.96 | 2025-07-17 09:15:00 | 195.09 | STOP_HIT | 1.00 | -3.79% |
| SELL | retest2 | 2025-07-16 09:15:00 | 187.60 | 2025-07-17 09:15:00 | 195.09 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2025-07-29 11:30:00 | 189.52 | 2025-08-07 09:15:00 | 186.22 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-07-29 12:00:00 | 189.76 | 2025-08-07 09:15:00 | 186.22 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-07-29 13:00:00 | 189.66 | 2025-08-07 09:15:00 | 186.22 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-08-04 09:15:00 | 193.38 | 2025-08-07 09:15:00 | 186.22 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2025-08-04 12:30:00 | 194.58 | 2025-08-07 09:15:00 | 186.22 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2025-08-11 13:00:00 | 196.20 | 2025-08-19 13:15:00 | 215.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-11 14:15:00 | 195.06 | 2025-08-19 13:15:00 | 214.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-01 09:15:00 | 253.15 | 2026-02-01 13:15:00 | 246.98 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2026-02-02 15:15:00 | 251.00 | 2026-02-18 09:15:00 | 276.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-03 10:45:00 | 250.87 | 2026-02-18 09:15:00 | 275.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-09 10:45:00 | 251.74 | 2026-02-18 10:15:00 | 276.91 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-12 13:15:00 | 266.65 | 2026-04-08 09:15:00 | 293.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-20 10:45:00 | 267.55 | 2026-04-08 09:15:00 | 294.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-20 12:00:00 | 266.50 | 2026-04-08 09:15:00 | 293.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-25 09:15:00 | 271.60 | 2026-04-15 09:15:00 | 298.76 | TARGET_HIT | 1.00 | 10.00% |
