# COALINDIA (COALINDIA)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 456.55
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 8 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 0 |
| TARGET_HIT | 5 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 7
- **Target hits / Stop hits / Partials:** 5 / 7 / 0
- **Avg / median % per leg:** 3.19% / -0.01%
- **Sum % (uncompounded):** 38.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 5 | 100.0% | 5 | 0 | 0 | 10.00% | 50.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 5 | 100.0% | 5 | 0 | 0 | 10.00% | 50.0% |
| SELL (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.67% | -11.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.67% | -11.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 5 | 41.7% | 5 | 7 | 0 | 3.19% | 38.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 12:15:00 | 402.70 | 384.05 | 384.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 405.35 | 384.80 | 384.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 419.30 | 419.57 | 408.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:15:00 | 418.55 | 419.57 | 408.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 412.45 | 422.91 | 413.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 09:45:00 | 418.90 | 422.11 | 413.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 14:45:00 | 418.25 | 421.66 | 413.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 10:45:00 | 417.90 | 421.52 | 413.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 14:45:00 | 417.85 | 421.33 | 413.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 424.80 | 423.96 | 416.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 431.20 | 423.97 | 416.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-12 10:15:00 | 460.79 | 431.12 | 422.24 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-12 10:15:00 | 460.08 | 431.12 | 422.24 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-12 10:15:00 | 459.69 | 431.12 | 422.24 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-12 10:15:00 | 459.64 | 431.12 | 422.24 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-13 09:15:00 | 474.32 | 433.23 | 423.57 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-11-13 14:00:00 | 383.85 | 2025-11-17 09:15:00 | 388.20 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-11-14 10:00:00 | 383.90 | 2025-11-17 09:15:00 | 388.20 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-11-18 15:00:00 | 384.10 | 2025-12-15 11:15:00 | 384.35 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-12-11 13:00:00 | 384.30 | 2025-12-15 11:15:00 | 384.35 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-12-15 09:15:00 | 380.65 | 2025-12-17 10:15:00 | 384.85 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-12-15 10:30:00 | 382.00 | 2025-12-23 09:15:00 | 397.40 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2025-12-16 10:15:00 | 381.40 | 2025-12-23 09:15:00 | 397.40 | STOP_HIT | 1.00 | -4.20% |
| BUY | retest2 | 2026-02-16 09:45:00 | 418.90 | 2026-03-12 10:15:00 | 460.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-18 14:45:00 | 418.25 | 2026-03-12 10:15:00 | 460.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-19 10:45:00 | 417.90 | 2026-03-12 10:15:00 | 459.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-19 14:45:00 | 417.85 | 2026-03-12 10:15:00 | 459.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-04 09:15:00 | 431.20 | 2026-03-13 09:15:00 | 474.32 | TARGET_HIT | 1.00 | 10.00% |
