# Indian Oil Corporation Ltd. (IOC)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 144.88
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
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 13 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 12
- **Target hits / Stop hits / Partials:** 1 / 12 / 0
- **Avg / median % per leg:** -1.12% / -1.63%
- **Sum % (uncompounded):** -14.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 1 | 7.7% | 1 | 12 | 0 | -1.12% | -14.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 1 | 7.7% | 1 | 12 | 0 | -1.12% | -14.5% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 1 | 7.7% | 1 | 12 | 0 | -1.12% | -14.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 139.76 | 143.82 | 143.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 139.40 | 143.75 | 143.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 142.94 | 141.63 | 142.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 142.94 | 141.63 | 142.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 142.94 | 141.63 | 142.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 143.33 | 141.63 | 142.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 144.93 | 141.66 | 142.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:45:00 | 144.90 | 141.66 | 142.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 142.81 | 141.88 | 142.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 142.84 | 141.88 | 142.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 143.08 | 141.90 | 142.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:45:00 | 143.46 | 141.90 | 142.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 142.83 | 141.91 | 142.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:45:00 | 143.03 | 141.91 | 142.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 144.05 | 141.98 | 142.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 144.55 | 141.98 | 142.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 15:15:00 | 148.69 | 143.12 | 143.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 10:15:00 | 148.91 | 144.38 | 143.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 149.94 | 150.17 | 147.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 149.94 | 150.17 | 147.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 160.98 | 163.99 | 160.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 14:45:00 | 161.94 | 163.87 | 160.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:00:00 | 161.98 | 163.81 | 160.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:00:00 | 161.99 | 163.79 | 161.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:30:00 | 161.97 | 163.77 | 161.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 161.08 | 163.70 | 161.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 161.77 | 163.70 | 161.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 160.06 | 163.66 | 161.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 160.06 | 163.66 | 161.00 | SL hit (close<static) qty=1.00 sl=160.41 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 15:15:00 | 146.50 | 166.55 | 166.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 144.47 | 165.09 | 165.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-20 12:15:00 | 147.78 | 147.48 | 153.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-20 13:00:00 | 147.78 | 147.48 | 153.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-13 10:30:00 | 141.62 | 2025-06-23 09:15:00 | 136.78 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2025-06-13 11:45:00 | 141.50 | 2025-06-23 09:15:00 | 136.78 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest2 | 2025-06-16 13:30:00 | 141.80 | 2025-06-23 09:15:00 | 136.78 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest2 | 2025-06-24 09:15:00 | 144.13 | 2025-08-22 14:15:00 | 139.76 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2025-12-19 14:45:00 | 161.94 | 2025-12-26 09:15:00 | 160.06 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-12-24 11:00:00 | 161.98 | 2025-12-26 09:15:00 | 160.06 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-12-24 12:00:00 | 161.99 | 2025-12-26 09:15:00 | 160.06 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-12-24 12:30:00 | 161.97 | 2025-12-26 09:15:00 | 160.06 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-12-31 09:15:00 | 162.96 | 2026-01-08 09:15:00 | 160.30 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2026-01-06 12:45:00 | 163.00 | 2026-01-08 09:15:00 | 160.30 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2026-01-07 13:45:00 | 162.86 | 2026-01-08 09:15:00 | 160.30 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2026-01-28 10:30:00 | 162.84 | 2026-02-01 11:15:00 | 160.18 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2026-02-02 14:30:00 | 162.69 | 2026-02-06 09:15:00 | 178.96 | TARGET_HIT | 1.00 | 10.00% |
