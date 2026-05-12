# Paradeep Phosphates Ltd. (PARADEEP)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
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
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 9 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 12 |
| PARTIAL | 13 |
| TARGET_HIT | 13 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 3
- **Target hits / Stop hits / Partials:** 13 / 3 / 13
- **Avg / median % per leg:** 5.88% / 5.00%
- **Sum % (uncompounded):** 170.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 8 | 72.7% | 4 | 3 | 4 | 3.24% | 35.7% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -8.11% | -24.3% |
| SELL (all) | 18 | 18 | 100.0% | 9 | 0 | 9 | 7.50% | 135.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 18 | 100.0% | 9 | 0 | 9 | 7.50% | 135.0% |
| retest1 (combined) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| retest2 (combined) | 21 | 18 | 85.7% | 9 | 3 | 9 | 5.27% | 110.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:15:00 | 164.96 | 163.61 | 151.13 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 14:15:00 | 164.94 | 163.72 | 151.49 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 15:15:00 | 164.87 | 163.72 | 151.56 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:45:00 | 165.72 | 163.18 | 152.97 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 09:15:00 | 173.21 | 162.62 | 155.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 09:15:00 | 173.19 | 162.62 | 155.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 09:15:00 | 173.11 | 162.62 | 155.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 09:15:00 | 174.01 | 162.62 | 155.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-07-21 09:15:00 | 181.46 | 165.11 | 157.84 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 2 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 178.60 | 189.02 | 189.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 176.42 | 188.70 | 188.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 09:15:00 | 163.16 | 160.22 | 167.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:45:00 | 163.73 | 160.22 | 167.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 168.28 | 159.74 | 164.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:45:00 | 167.95 | 159.74 | 164.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 167.75 | 159.82 | 164.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 11:30:00 | 166.40 | 159.88 | 164.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 12:00:00 | 165.64 | 159.88 | 164.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 10:00:00 | 166.40 | 160.17 | 165.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:00:00 | 165.70 | 160.22 | 165.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 164.20 | 160.32 | 165.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:30:00 | 164.52 | 160.32 | 165.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 165.10 | 160.43 | 165.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 165.14 | 160.43 | 165.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 164.93 | 160.48 | 165.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:15:00 | 164.46 | 160.48 | 165.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:15:00 | 164.51 | 160.60 | 165.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:45:00 | 164.36 | 160.76 | 164.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 162.72 | 161.01 | 164.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 161.73 | 161.01 | 164.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 13:45:00 | 161.29 | 161.05 | 164.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 09:15:00 | 158.08 | 160.99 | 164.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 09:15:00 | 157.36 | 160.99 | 164.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 09:15:00 | 158.08 | 160.99 | 164.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 09:15:00 | 157.41 | 160.99 | 164.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:15:00 | 156.24 | 160.91 | 164.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:15:00 | 156.28 | 160.91 | 164.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:15:00 | 156.14 | 160.91 | 164.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 10:15:00 | 154.58 | 160.48 | 164.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 153.23 | 160.41 | 164.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-09 09:15:00 | 149.76 | 160.00 | 163.93 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-24 09:15:00 | 164.96 | 2025-07-15 09:15:00 | 173.21 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-24 14:15:00 | 164.94 | 2025-07-15 09:15:00 | 173.19 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-24 15:15:00 | 164.87 | 2025-07-15 09:15:00 | 173.11 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-07-01 09:45:00 | 165.72 | 2025-07-15 09:15:00 | 174.01 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-24 09:15:00 | 164.96 | 2025-07-21 09:15:00 | 181.46 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-06-24 14:15:00 | 164.94 | 2025-07-21 09:15:00 | 181.43 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-06-24 15:15:00 | 164.87 | 2025-07-21 09:15:00 | 181.36 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-07-01 09:45:00 | 165.72 | 2025-07-21 09:15:00 | 182.29 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-09-30 09:15:00 | 198.50 | 2025-10-09 09:15:00 | 182.22 | STOP_HIT | 1.00 | -8.20% |
| BUY | retest2 | 2025-10-01 11:30:00 | 198.12 | 2025-10-09 09:15:00 | 182.22 | STOP_HIT | 1.00 | -8.03% |
| BUY | retest2 | 2025-10-07 09:30:00 | 198.30 | 2025-10-09 09:15:00 | 182.22 | STOP_HIT | 1.00 | -8.11% |
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
