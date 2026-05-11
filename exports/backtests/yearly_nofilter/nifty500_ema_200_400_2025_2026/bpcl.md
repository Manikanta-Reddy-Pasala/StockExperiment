# Bharat Petroleum Corporation Ltd. (BPCL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 303.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 13 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 0 |
| TARGET_HIT | 5 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 13
- **Target hits / Stop hits / Partials:** 5 / 13 / 0
- **Avg / median % per leg:** 1.51% / -1.21%
- **Sum % (uncompounded):** 27.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 5 | 33.3% | 5 | 10 | 0 | 2.08% | 31.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 5 | 33.3% | 5 | 10 | 0 | 2.08% | 31.2% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.34% | -4.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.34% | -4.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 18 | 5 | 27.8% | 5 | 13 | 0 | 1.51% | 27.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 310.55 | 322.59 | 322.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 308.45 | 322.34 | 322.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 322.85 | 318.95 | 320.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 322.85 | 318.95 | 320.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 322.85 | 318.95 | 320.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 322.50 | 318.95 | 320.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 324.60 | 319.01 | 320.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:45:00 | 325.70 | 319.01 | 320.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 320.70 | 319.09 | 320.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 320.70 | 319.09 | 320.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 320.00 | 319.10 | 320.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 320.30 | 319.10 | 320.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 320.65 | 319.12 | 320.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:30:00 | 319.70 | 319.14 | 320.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 09:30:00 | 319.75 | 319.11 | 320.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:00:00 | 319.10 | 319.05 | 320.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 10:15:00 | 323.80 | 319.09 | 320.34 | SL hit (close>static) qty=1.00 sl=323.20 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 14:15:00 | 330.65 | 321.44 | 321.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 09:15:00 | 335.30 | 322.86 | 322.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 12:15:00 | 332.15 | 332.18 | 327.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 13:00:00 | 332.15 | 332.18 | 327.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 334.00 | 333.32 | 329.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 335.60 | 333.24 | 329.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-03 12:15:00 | 369.16 | 339.07 | 333.19 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 09:15:00 | 325.15 | 364.39 | 364.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 323.85 | 363.99 | 364.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 14:15:00 | 310.40 | 307.34 | 325.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-15 15:00:00 | 310.40 | 307.34 | 325.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-13 11:30:00 | 310.30 | 2025-07-04 11:15:00 | 341.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-16 10:15:00 | 310.80 | 2025-07-04 11:15:00 | 341.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-16 12:15:00 | 310.30 | 2025-07-04 11:15:00 | 341.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-19 10:45:00 | 311.30 | 2025-07-04 11:15:00 | 342.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 11:15:00 | 330.80 | 2025-08-01 09:15:00 | 323.75 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-07-31 11:45:00 | 330.05 | 2025-08-01 09:15:00 | 323.75 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-07-31 12:45:00 | 330.30 | 2025-08-01 09:15:00 | 323.75 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-07-31 14:00:00 | 330.05 | 2025-08-01 09:15:00 | 323.75 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-08-12 09:15:00 | 322.10 | 2025-08-14 13:15:00 | 318.20 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-08-13 12:45:00 | 321.75 | 2025-08-14 13:15:00 | 318.20 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-08-19 15:15:00 | 321.90 | 2025-08-22 09:15:00 | 318.30 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-08-20 12:15:00 | 321.95 | 2025-08-22 09:15:00 | 318.30 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-09-12 11:30:00 | 319.70 | 2025-09-17 10:15:00 | 323.80 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-09-15 09:30:00 | 319.75 | 2025-09-17 10:15:00 | 323.80 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-09-16 11:00:00 | 319.10 | 2025-09-17 10:15:00 | 323.80 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-10-27 09:15:00 | 335.60 | 2025-11-03 12:15:00 | 369.16 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-09 11:45:00 | 336.10 | 2026-03-10 10:15:00 | 325.40 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2026-03-09 12:45:00 | 335.75 | 2026-03-10 10:15:00 | 325.40 | STOP_HIT | 1.00 | -3.08% |
