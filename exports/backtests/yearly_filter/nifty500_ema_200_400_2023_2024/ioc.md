# Indian Oil Corporation Ltd. (IOC)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 144.88
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 60 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 52 |
| PARTIAL | 3 |
| TARGET_HIT | 11 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 44
- **Target hits / Stop hits / Partials:** 11 / 44 / 3
- **Avg / median % per leg:** 0.60% / -1.18%
- **Sum % (uncompounded):** 34.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 44 | 8 | 18.2% | 8 | 36 | 0 | 0.27% | 11.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 44 | 8 | 18.2% | 8 | 36 | 0 | 0.27% | 11.9% |
| SELL (all) | 14 | 6 | 42.9% | 3 | 8 | 3 | 1.63% | 22.8% |
| SELL @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 3 | 0 | 3 | 7.50% | 45.0% |
| SELL @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.78% | -22.2% |
| retest1 (combined) | 6 | 6 | 100.0% | 3 | 0 | 3 | 7.50% | 45.0% |
| retest2 (combined) | 52 | 8 | 15.4% | 8 | 44 | 0 | -0.20% | -10.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 10:15:00 | 89.85 | 91.97 | 91.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 11:15:00 | 89.25 | 91.95 | 91.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-13 09:15:00 | 91.10 | 90.94 | 91.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-13 10:00:00 | 91.10 | 90.94 | 91.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 91.60 | 90.94 | 91.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 10:00:00 | 91.60 | 90.94 | 91.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 10:15:00 | 92.05 | 90.95 | 91.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 10:30:00 | 92.00 | 90.95 | 91.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 11:15:00 | 91.00 | 91.00 | 91.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 11:30:00 | 91.35 | 91.00 | 91.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 92.25 | 89.84 | 90.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-01 10:00:00 | 92.25 | 89.84 | 90.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 10:15:00 | 92.75 | 89.87 | 90.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-01 11:00:00 | 92.75 | 89.87 | 90.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 09:15:00 | 99.85 | 91.37 | 91.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-07 10:15:00 | 100.75 | 91.46 | 91.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 12:15:00 | 167.85 | 168.42 | 152.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-28 13:00:00 | 167.85 | 168.42 | 152.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 10:15:00 | 159.10 | 170.08 | 158.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-15 11:00:00 | 159.10 | 170.08 | 158.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 11:15:00 | 155.35 | 169.93 | 158.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-15 12:00:00 | 155.35 | 169.93 | 158.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 12:15:00 | 157.00 | 169.80 | 158.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-15 13:30:00 | 158.40 | 169.68 | 158.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-15 14:30:00 | 158.60 | 169.60 | 158.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 13:15:00 | 158.25 | 167.64 | 158.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 14:45:00 | 158.20 | 167.45 | 158.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-02 13:15:00 | 174.24 | 167.62 | 160.51 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 11:15:00 | 165.29 | 171.38 | 171.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 164.45 | 170.85 | 171.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 10:15:00 | 142.75 | 142.35 | 149.96 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 14:45:00 | 141.49 | 142.47 | 148.90 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:15:00 | 141.50 | 142.46 | 148.83 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 11:30:00 | 141.37 | 142.53 | 148.38 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 134.42 | 142.06 | 147.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 134.42 | 142.06 | 147.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 134.30 | 142.06 | 147.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-01-13 09:15:00 | 127.34 | 137.62 | 142.66 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 4 — BUY (started 2025-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 13:15:00 | 133.38 | 128.53 | 128.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 14:15:00 | 133.74 | 128.58 | 128.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 141.12 | 141.13 | 137.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 14:30:00 | 141.30 | 141.13 | 137.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 140.80 | 141.58 | 138.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:30:00 | 141.62 | 141.57 | 138.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 11:45:00 | 141.50 | 141.58 | 138.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 13:30:00 | 141.80 | 141.49 | 138.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 136.78 | 141.05 | 139.00 | SL hit (close<static) qty=1.00 sl=137.01 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-22 14:15:00)

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

### Cycle 6 — BUY (started 2025-09-19 15:15:00)

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

### Cycle 7 — SELL (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 15:15:00 | 146.50 | 166.55 | 166.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 144.47 | 165.09 | 165.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-20 12:15:00 | 147.78 | 147.48 | 153.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-20 13:00:00 | 147.78 | 147.48 | 153.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-03-15 13:30:00 | 158.40 | 2024-04-02 13:15:00 | 174.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-15 14:30:00 | 158.60 | 2024-04-02 13:15:00 | 174.08 | TARGET_HIT | 1.00 | 9.76% |
| BUY | retest2 | 2024-03-20 13:15:00 | 158.25 | 2024-04-02 13:15:00 | 174.02 | TARGET_HIT | 1.00 | 9.97% |
| BUY | retest2 | 2024-03-20 14:45:00 | 158.20 | 2024-04-02 14:15:00 | 174.46 | TARGET_HIT | 1.00 | 10.28% |
| BUY | retest2 | 2024-04-19 12:30:00 | 166.70 | 2024-05-07 11:15:00 | 161.00 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2024-04-19 14:45:00 | 166.40 | 2024-05-07 11:15:00 | 161.00 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2024-05-06 10:15:00 | 166.85 | 2024-05-07 11:15:00 | 161.00 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2024-05-06 13:15:00 | 166.45 | 2024-05-07 11:15:00 | 161.00 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2024-05-18 09:15:00 | 165.00 | 2024-05-30 09:15:00 | 163.35 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-05-30 10:30:00 | 164.80 | 2024-05-30 13:15:00 | 163.20 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-06-03 09:15:00 | 169.75 | 2024-06-04 10:15:00 | 160.80 | STOP_HIT | 1.00 | -5.27% |
| BUY | retest2 | 2024-06-06 09:15:00 | 164.65 | 2024-06-07 09:15:00 | 163.35 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-06-11 09:45:00 | 166.05 | 2024-06-25 11:15:00 | 164.65 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-06-11 10:30:00 | 166.04 | 2024-06-25 11:15:00 | 164.65 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-06-11 12:30:00 | 167.88 | 2024-06-25 11:15:00 | 164.65 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-06-24 10:15:00 | 166.23 | 2024-06-25 11:15:00 | 164.65 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-07-01 09:15:00 | 166.44 | 2024-07-19 15:15:00 | 165.44 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-07-22 09:15:00 | 166.15 | 2024-07-23 12:15:00 | 165.07 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-07-23 14:30:00 | 166.31 | 2024-07-30 09:15:00 | 182.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-24 09:15:00 | 166.97 | 2024-07-30 09:15:00 | 183.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-25 10:15:00 | 168.80 | 2024-07-30 09:15:00 | 185.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-06 14:45:00 | 168.18 | 2024-08-13 13:15:00 | 164.77 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-08-07 09:15:00 | 170.63 | 2024-08-13 13:15:00 | 164.77 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest2 | 2024-08-12 10:15:00 | 168.43 | 2024-08-13 13:15:00 | 164.77 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2024-08-19 14:15:00 | 170.33 | 2024-09-18 12:15:00 | 168.30 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-08-20 09:15:00 | 172.95 | 2024-09-18 12:15:00 | 168.30 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2024-09-11 14:30:00 | 170.16 | 2024-09-18 12:15:00 | 168.30 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-09-12 09:15:00 | 171.05 | 2024-09-18 12:15:00 | 168.30 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-09-27 09:30:00 | 172.58 | 2024-10-03 12:15:00 | 171.58 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-09-27 10:00:00 | 172.96 | 2024-10-03 12:15:00 | 171.58 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-10-03 12:30:00 | 172.70 | 2024-10-03 13:15:00 | 171.58 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest1 | 2024-12-12 14:45:00 | 141.49 | 2024-12-19 09:15:00 | 134.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-12-13 10:15:00 | 141.50 | 2024-12-19 09:15:00 | 134.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-12-17 11:30:00 | 141.37 | 2024-12-19 09:15:00 | 134.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-12-12 14:45:00 | 141.49 | 2025-01-13 09:15:00 | 127.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2024-12-13 10:15:00 | 141.50 | 2025-01-13 09:15:00 | 127.35 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2024-12-17 11:30:00 | 141.37 | 2025-01-13 09:15:00 | 127.23 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-12 11:30:00 | 124.51 | 2025-03-20 10:15:00 | 129.74 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2025-03-12 13:45:00 | 124.90 | 2025-03-20 10:15:00 | 129.74 | STOP_HIT | 1.00 | -3.88% |
| SELL | retest2 | 2025-03-12 14:30:00 | 124.85 | 2025-03-20 10:15:00 | 129.74 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2025-03-13 09:15:00 | 124.95 | 2025-03-20 10:15:00 | 129.74 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2025-04-02 09:15:00 | 128.95 | 2025-04-02 12:15:00 | 130.54 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-04-04 13:30:00 | 129.22 | 2025-04-04 15:15:00 | 130.46 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-04-07 09:15:00 | 126.75 | 2025-04-08 11:15:00 | 130.51 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2025-04-08 09:45:00 | 128.88 | 2025-04-08 11:15:00 | 130.51 | STOP_HIT | 1.00 | -1.26% |
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
