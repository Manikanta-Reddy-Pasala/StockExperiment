# Paradeep Phosphates Ltd. (PARADEEP)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 125.09
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
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 6 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 9 |
| TARGET_HIT | 9 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 0
- **Target hits / Stop hits / Partials:** 9 / 0 / 9
- **Avg / median % per leg:** 7.50% / 9.59%
- **Sum % (uncompounded):** 135.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 18 | 18 | 100.0% | 9 | 0 | 9 | 7.50% | 135.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 18 | 100.0% | 9 | 0 | 9 | 7.50% | 135.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 18 | 18 | 100.0% | 9 | 0 | 9 | 7.50% | 135.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 173.61 | 188.42 | 188.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 172.90 | 188.26 | 188.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 09:15:00 | 163.16 | 160.22 | 167.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:45:00 | 163.73 | 160.22 | 167.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 168.28 | 159.74 | 164.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:45:00 | 167.95 | 159.74 | 164.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 167.75 | 159.82 | 164.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 11:30:00 | 166.40 | 159.88 | 164.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 12:00:00 | 165.64 | 159.88 | 164.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 10:00:00 | 166.40 | 160.17 | 164.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:00:00 | 165.70 | 160.22 | 164.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 164.20 | 160.32 | 164.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:30:00 | 164.52 | 160.32 | 164.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 165.10 | 160.43 | 164.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 165.14 | 160.43 | 164.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 164.93 | 160.48 | 164.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:15:00 | 164.46 | 160.48 | 164.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:15:00 | 164.51 | 160.60 | 164.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:45:00 | 164.36 | 160.76 | 164.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 162.72 | 161.01 | 164.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 161.73 | 161.01 | 164.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 13:45:00 | 161.29 | 161.05 | 164.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 09:15:00 | 158.08 | 160.99 | 164.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 09:15:00 | 157.36 | 160.99 | 164.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 09:15:00 | 158.08 | 160.99 | 164.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 09:15:00 | 157.41 | 160.99 | 164.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:15:00 | 156.24 | 160.91 | 164.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:15:00 | 156.28 | 160.91 | 164.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:15:00 | 156.14 | 160.91 | 164.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 10:15:00 | 154.58 | 160.48 | 164.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 153.23 | 160.41 | 164.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-09 09:15:00 | 149.76 | 160.00 | 163.89 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-12-30 11:30:00 | 166.40 | 2026-01-06 09:15:00 | 158.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-30 12:00:00 | 165.64 | 2026-01-06 09:15:00 | 157.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 10:00:00 | 166.40 | 2026-01-06 09:15:00 | 158.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 11:00:00 | 165.70 | 2026-01-06 09:15:00 | 157.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 10:15:00 | 164.46 | 2026-01-06 11:15:00 | 156.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 13:15:00 | 164.51 | 2026-01-06 11:15:00 | 156.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-02 09:45:00 | 164.36 | 2026-01-06 11:15:00 | 156.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 09:15:00 | 162.72 | 2026-01-08 10:15:00 | 154.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 13:45:00 | 161.29 | 2026-01-08 11:15:00 | 153.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-30 11:30:00 | 166.40 | 2026-01-09 09:15:00 | 149.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-30 12:00:00 | 165.64 | 2026-01-09 09:15:00 | 149.76 | TARGET_HIT | 0.50 | 9.59% |
| SELL | retest2 | 2025-12-31 10:00:00 | 166.40 | 2026-01-09 12:15:00 | 149.08 | TARGET_HIT | 0.50 | 10.41% |
| SELL | retest2 | 2025-12-31 11:00:00 | 165.70 | 2026-01-09 12:15:00 | 149.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-01 10:15:00 | 164.46 | 2026-01-09 14:15:00 | 148.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-01 13:15:00 | 164.51 | 2026-01-09 14:15:00 | 148.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-02 09:45:00 | 164.36 | 2026-01-09 14:15:00 | 147.92 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-05 09:15:00 | 162.72 | 2026-01-12 09:15:00 | 146.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-05 13:45:00 | 161.29 | 2026-01-12 09:15:00 | 145.16 | TARGET_HIT | 0.50 | 10.00% |
