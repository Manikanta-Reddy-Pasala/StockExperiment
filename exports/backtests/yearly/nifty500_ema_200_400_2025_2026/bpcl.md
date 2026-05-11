# Bharat Petroleum Corporation Ltd. (BPCL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3164 bars)
- **Last close:** 303.20
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
| ALERT2_SKIP | 1 |
| ALERT3 | 13 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 0 |
| TARGET_HIT | 6 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 8
- **Target hits / Stop hits / Partials:** 6 / 8 / 0
- **Avg / median % per leg:** 3.09% / -0.16%
- **Sum % (uncompounded):** 43.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 6 | 54.5% | 6 | 5 | 0 | 4.30% | 47.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 6 | 54.5% | 6 | 5 | 0 | 4.30% | 47.3% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.34% | -4.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.34% | -4.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 6 | 42.9% | 6 | 8 | 0 | 3.09% | 43.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 15:15:00 | 313.00 | 303.19 | 303.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 315.30 | 303.31 | 303.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 11:15:00 | 311.70 | 312.05 | 308.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 11:30:00 | 311.50 | 312.05 | 308.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 309.80 | 311.96 | 308.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 310.00 | 311.96 | 308.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 309.50 | 311.94 | 308.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:00:00 | 309.50 | 311.94 | 308.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 309.60 | 311.92 | 308.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:30:00 | 308.30 | 311.92 | 308.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 309.45 | 314.97 | 310.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 11:30:00 | 310.30 | 314.90 | 310.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:15:00 | 310.80 | 314.74 | 311.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 12:15:00 | 310.30 | 314.66 | 311.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 10:45:00 | 311.30 | 314.59 | 311.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 313.00 | 314.57 | 311.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:30:00 | 311.45 | 314.57 | 311.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 312.25 | 314.55 | 311.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:45:00 | 312.30 | 314.55 | 311.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 311.60 | 314.51 | 311.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 311.30 | 314.51 | 311.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 312.45 | 314.49 | 311.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 11:15:00 | 315.05 | 314.49 | 311.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 309.80 | 314.40 | 311.42 | SL hit (close<static) qty=1.00 sl=310.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 316.40 | 324.67 | 324.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 314.55 | 323.90 | 324.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 322.85 | 318.96 | 321.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 322.85 | 318.96 | 321.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 322.85 | 318.96 | 321.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 322.50 | 318.96 | 321.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 324.60 | 319.02 | 321.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:45:00 | 325.70 | 319.02 | 321.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 320.65 | 319.13 | 321.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:30:00 | 319.70 | 319.15 | 321.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 09:30:00 | 319.75 | 319.12 | 321.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:00:00 | 319.10 | 319.05 | 320.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 10:15:00 | 323.80 | 319.09 | 320.94 | SL hit (close>static) qty=1.00 sl=323.20 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 12:15:00 | 329.90 | 322.38 | 322.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 13:15:00 | 330.90 | 322.46 | 322.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 12:15:00 | 332.15 | 332.18 | 328.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 13:00:00 | 332.15 | 332.18 | 328.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 334.00 | 333.32 | 329.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 335.60 | 333.24 | 329.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-03 12:15:00 | 369.16 | 339.08 | 333.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 09:15:00 | 325.15 | 364.39 | 364.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 323.85 | 363.99 | 364.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 14:15:00 | 310.40 | 307.34 | 325.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-15 15:00:00 | 310.40 | 307.34 | 325.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-13 11:30:00 | 310.30 | 2025-06-23 09:15:00 | 309.80 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-06-16 10:15:00 | 310.80 | 2025-07-04 11:15:00 | 341.33 | TARGET_HIT | 1.00 | 9.82% |
| BUY | retest2 | 2025-06-16 12:15:00 | 310.30 | 2025-07-04 11:15:00 | 341.88 | TARGET_HIT | 1.00 | 10.18% |
| BUY | retest2 | 2025-06-19 10:45:00 | 311.30 | 2025-07-04 11:15:00 | 341.33 | TARGET_HIT | 1.00 | 9.65% |
| BUY | retest2 | 2025-06-20 11:15:00 | 315.05 | 2025-07-04 11:15:00 | 342.43 | TARGET_HIT | 1.00 | 8.69% |
| BUY | retest2 | 2025-06-24 09:15:00 | 322.60 | 2025-07-08 10:15:00 | 354.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-05 11:00:00 | 315.60 | 2025-08-07 12:15:00 | 308.10 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2025-08-05 14:30:00 | 315.25 | 2025-08-07 12:15:00 | 308.10 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-09-12 11:30:00 | 319.70 | 2025-09-17 10:15:00 | 323.80 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-09-15 09:30:00 | 319.75 | 2025-09-17 10:15:00 | 323.80 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-09-16 11:00:00 | 319.10 | 2025-09-17 10:15:00 | 323.80 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-10-27 09:15:00 | 335.60 | 2025-11-03 12:15:00 | 369.16 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-09 11:45:00 | 336.10 | 2026-03-10 10:15:00 | 325.40 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2026-03-09 12:45:00 | 335.75 | 2026-03-10 10:15:00 | 325.40 | STOP_HIT | 1.00 | -3.08% |
