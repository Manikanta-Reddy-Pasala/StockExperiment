# Paradeep Phosphates Ltd. (PARADEEP)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 125.09
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
| ALERT2_SKIP | 0 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 19 |
| PARTIAL | 13 |
| TARGET_HIT | 18 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 5
- **Target hits / Stop hits / Partials:** 18 / 5 / 13
- **Avg / median % per leg:** 5.98% / 9.59%
- **Sum % (uncompounded):** 215.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 13 | 72.2% | 9 | 5 | 4 | 4.45% | 80.1% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| BUY @ 3rd Alert (retest2) | 10 | 5 | 50.0% | 5 | 5 | 0 | 2.01% | 20.1% |
| SELL (all) | 18 | 18 | 100.0% | 9 | 0 | 9 | 7.50% | 135.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 18 | 100.0% | 9 | 0 | 9 | 7.50% | 135.0% |
| retest1 (combined) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| retest2 (combined) | 28 | 23 | 82.1% | 14 | 5 | 9 | 5.54% | 155.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 12:15:00 | 74.21 | 70.73 | 70.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 13:15:00 | 74.91 | 70.77 | 70.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 12:15:00 | 83.45 | 83.99 | 79.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-23 12:30:00 | 83.95 | 83.99 | 79.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 14:15:00 | 81.77 | 85.49 | 81.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-06 15:00:00 | 81.77 | 85.49 | 81.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 15:15:00 | 81.79 | 85.45 | 81.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 09:15:00 | 82.27 | 85.45 | 81.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 13:30:00 | 82.14 | 85.29 | 81.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 15:15:00 | 82.15 | 85.25 | 81.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-12 11:15:00 | 90.50 | 85.21 | 82.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 15:15:00 | 99.15 | 108.56 | 108.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 09:15:00 | 98.13 | 108.46 | 108.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 14:15:00 | 95.65 | 95.40 | 99.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-21 15:00:00 | 95.65 | 95.40 | 99.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 100.15 | 95.46 | 99.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 10:00:00 | 100.15 | 95.46 | 99.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 10:15:00 | 102.05 | 95.52 | 99.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 11:00:00 | 102.05 | 95.52 | 99.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-04-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 13:15:00 | 113.92 | 102.13 | 102.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-07 15:15:00 | 114.65 | 102.37 | 102.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 15:15:00 | 163.60 | 163.61 | 151.13 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:15:00 | 164.96 | 163.61 | 151.13 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 14:15:00 | 164.94 | 163.72 | 151.49 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 15:15:00 | 164.87 | 163.72 | 151.56 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:45:00 | 165.72 | 163.18 | 152.97 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 09:15:00 | 173.21 | 162.62 | 155.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 09:15:00 | 173.19 | 162.62 | 155.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 09:15:00 | 173.11 | 162.62 | 155.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 09:15:00 | 174.01 | 162.62 | 155.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-07-21 09:15:00 | 181.46 | 165.11 | 157.84 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 4 — SELL (started 2025-10-13 13:15:00)

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
| BUY | retest2 | 2024-08-07 09:15:00 | 82.27 | 2024-08-12 11:15:00 | 90.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-07 13:30:00 | 82.14 | 2024-08-12 11:15:00 | 90.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-07 15:15:00 | 82.15 | 2024-08-12 11:15:00 | 90.37 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-07 14:00:00 | 82.26 | 2024-10-10 09:15:00 | 90.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-23 09:15:00 | 87.20 | 2024-10-25 13:15:00 | 84.80 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2024-10-23 09:45:00 | 87.25 | 2024-10-25 13:15:00 | 84.80 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2024-10-28 10:00:00 | 87.11 | 2024-10-29 09:15:00 | 95.82 | TARGET_HIT | 1.00 | 10.00% |
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
