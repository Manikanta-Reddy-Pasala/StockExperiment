# Tata Chemicals Ltd. (TATACHEM)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 782.00
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
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 8 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 2 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 2 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 0
- **Avg / median % per leg:** 4.62% / 10.00%
- **Sum % (uncompounded):** 9.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 1 | 50.0% | 1 | 1 | 0 | 4.62% | 9.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 1 | 50.0% | 1 | 1 | 0 | 4.62% | 9.2% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 2 | 1 | 50.0% | 1 | 1 | 0 | 4.62% | 9.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 14:15:00 | 905.20 | 865.01 | 864.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 12:15:00 | 908.95 | 867.01 | 865.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 904.10 | 905.85 | 890.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 09:45:00 | 904.30 | 905.85 | 890.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 906.65 | 920.39 | 905.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:15:00 | 910.40 | 920.39 | 905.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 910.40 | 920.29 | 905.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:30:00 | 912.00 | 920.29 | 905.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-29 14:15:00 | 1003.20 | 936.07 | 919.71 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 11:15:00 | 907.65 | 940.70 | 940.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 12:15:00 | 907.05 | 940.37 | 940.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 13:15:00 | 776.15 | 774.89 | 808.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 14:00:00 | 776.15 | 774.89 | 808.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 15:15:00 | 705.00 | 656.91 | 687.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:30:00 | 761.25 | 657.72 | 687.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 731.70 | 658.45 | 687.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:15:00 | 731.90 | 658.45 | 687.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 692.40 | 680.99 | 693.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:15:00 | 694.00 | 680.99 | 693.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 693.50 | 681.11 | 693.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:45:00 | 695.75 | 681.11 | 693.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 695.20 | 681.25 | 693.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 707.80 | 681.25 | 693.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 712.80 | 681.56 | 693.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 712.80 | 681.56 | 693.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 816.00 | 703.45 | 703.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 822.80 | 706.69 | 704.97 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-14 09:30:00 | 912.00 | 2025-07-29 14:15:00 | 1003.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-30 13:45:00 | 914.65 | 2025-10-10 11:15:00 | 907.65 | STOP_HIT | 1.00 | -0.77% |
