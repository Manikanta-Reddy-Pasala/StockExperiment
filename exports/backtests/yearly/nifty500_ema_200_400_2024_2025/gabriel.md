# Gabriel India Ltd. (GABRIEL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1136.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 25 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 23 |
| PARTIAL | 2 |
| TARGET_HIT | 11 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 12
- **Target hits / Stop hits / Partials:** 11 / 12 / 2
- **Avg / median % per leg:** 2.19% / 5.00%
- **Sum % (uncompounded):** 54.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 9 | 56.2% | 9 | 7 | 0 | 4.23% | 67.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 9 | 56.2% | 9 | 7 | 0 | 4.23% | 67.7% |
| SELL (all) | 9 | 4 | 44.4% | 2 | 5 | 2 | -1.45% | -13.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 4 | 44.4% | 2 | 5 | 2 | -1.45% | -13.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 13 | 52.0% | 11 | 12 | 2 | 2.19% | 54.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 12:15:00 | 445.80 | 493.76 | 493.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 442.20 | 491.91 | 492.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 10:15:00 | 464.80 | 462.32 | 474.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 10:45:00 | 465.05 | 462.32 | 474.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 456.10 | 445.21 | 457.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 449.70 | 445.21 | 457.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 09:15:00 | 472.10 | 445.25 | 456.43 | SL hit (close>static) qty=1.00 sl=457.95 alert=retest2 |

### Cycle 2 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 514.95 | 465.54 | 465.37 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 11:15:00 | 435.65 | 470.61 | 470.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 12:15:00 | 434.00 | 470.24 | 470.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 12:15:00 | 483.65 | 448.19 | 457.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 12:15:00 | 483.65 | 448.19 | 457.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 483.65 | 448.19 | 457.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:00:00 | 483.65 | 448.19 | 457.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 455.95 | 448.27 | 457.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 14:15:00 | 450.30 | 448.27 | 457.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 15:00:00 | 451.00 | 448.30 | 457.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 09:15:00 | 443.55 | 448.33 | 457.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 13:00:00 | 448.70 | 448.94 | 457.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 480.90 | 449.24 | 457.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:00:00 | 480.90 | 449.24 | 457.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 483.05 | 449.58 | 457.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 11:00:00 | 483.05 | 449.58 | 457.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-03 15:15:00 | 491.00 | 451.28 | 458.15 | SL hit (close>static) qty=1.00 sl=489.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 10:15:00 | 472.90 | 463.76 | 463.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-12 11:15:00 | 475.60 | 463.88 | 463.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-17 09:15:00 | 464.25 | 467.67 | 465.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 09:15:00 | 464.25 | 467.67 | 465.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 464.25 | 467.67 | 465.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 09:15:00 | 483.15 | 465.83 | 465.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 15:00:00 | 480.35 | 466.29 | 465.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:15:00 | 478.90 | 466.61 | 465.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 14:15:00 | 478.85 | 469.66 | 467.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 466.10 | 469.90 | 467.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 10:00:00 | 466.10 | 469.90 | 467.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 10:15:00 | 462.70 | 469.83 | 467.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 11:00:00 | 462.70 | 469.83 | 467.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 11:15:00 | 462.80 | 469.76 | 467.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 11:45:00 | 462.15 | 469.76 | 467.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-28 12:15:00 | 460.20 | 469.66 | 467.24 | SL hit (close<static) qty=1.00 sl=461.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 10:15:00 | 1028.90 | 1175.02 | 1175.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 1025.30 | 1151.26 | 1163.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 12:15:00 | 1052.40 | 1038.28 | 1088.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:45:00 | 1053.90 | 1038.28 | 1088.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 1109.90 | 1038.99 | 1088.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:00:00 | 1109.90 | 1038.99 | 1088.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 1079.30 | 1039.39 | 1088.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:30:00 | 1122.60 | 1039.39 | 1088.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1074.60 | 1037.09 | 1072.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 1074.60 | 1037.09 | 1072.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1073.80 | 1037.45 | 1072.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 1066.50 | 1037.45 | 1072.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 13:15:00 | 1063.40 | 1038.07 | 1072.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 15:15:00 | 1013.17 | 1037.14 | 1068.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 15:15:00 | 1010.23 | 1037.14 | 1068.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-12 09:15:00 | 959.85 | 1033.27 | 1065.16 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 1013.90 | 944.99 | 944.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 1020.35 | 947.72 | 946.29 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-31 09:15:00 | 367.35 | 2024-05-31 09:15:00 | 358.50 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2024-06-03 09:15:00 | 366.45 | 2024-06-03 10:15:00 | 358.75 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-06-04 13:45:00 | 366.40 | 2024-06-11 11:15:00 | 403.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 15:15:00 | 365.00 | 2024-06-11 11:15:00 | 401.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 10:15:00 | 375.25 | 2024-06-11 11:15:00 | 412.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 11:00:00 | 375.70 | 2024-06-11 11:15:00 | 413.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 11:30:00 | 375.30 | 2024-06-11 11:15:00 | 412.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 13:00:00 | 375.15 | 2024-06-11 11:15:00 | 412.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-04 11:45:00 | 494.45 | 2024-10-04 13:15:00 | 488.65 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-12-04 09:15:00 | 449.70 | 2024-12-06 09:15:00 | 472.10 | STOP_HIT | 1.00 | -4.98% |
| SELL | retest2 | 2025-01-30 14:15:00 | 450.30 | 2025-02-03 15:15:00 | 491.00 | STOP_HIT | 1.00 | -9.04% |
| SELL | retest2 | 2025-01-30 15:00:00 | 451.00 | 2025-02-03 15:15:00 | 491.00 | STOP_HIT | 1.00 | -8.87% |
| SELL | retest2 | 2025-01-31 09:15:00 | 443.55 | 2025-02-03 15:15:00 | 491.00 | STOP_HIT | 1.00 | -10.70% |
| SELL | retest2 | 2025-02-01 13:00:00 | 448.70 | 2025-02-03 15:15:00 | 491.00 | STOP_HIT | 1.00 | -9.43% |
| BUY | retest2 | 2025-02-21 09:15:00 | 483.15 | 2025-02-28 12:15:00 | 460.20 | STOP_HIT | 1.00 | -4.75% |
| BUY | retest2 | 2025-02-21 15:00:00 | 480.35 | 2025-02-28 12:15:00 | 460.20 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2025-02-24 11:15:00 | 478.90 | 2025-02-28 12:15:00 | 460.20 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2025-02-27 14:15:00 | 478.85 | 2025-02-28 12:15:00 | 460.20 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2025-04-08 09:15:00 | 537.20 | 2025-04-21 10:15:00 | 575.08 | TARGET_HIT | 1.00 | 7.05% |
| BUY | retest2 | 2025-04-11 09:15:00 | 530.25 | 2025-04-23 09:15:00 | 583.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-11 10:45:00 | 522.80 | 2025-04-24 09:15:00 | 590.92 | TARGET_HIT | 1.00 | 13.03% |
| SELL | retest2 | 2026-01-05 11:15:00 | 1066.50 | 2026-01-08 15:15:00 | 1013.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 13:15:00 | 1063.40 | 2026-01-08 15:15:00 | 1010.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 11:15:00 | 1066.50 | 2026-01-12 09:15:00 | 959.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-05 13:15:00 | 1063.40 | 2026-01-12 09:15:00 | 957.06 | TARGET_HIT | 0.50 | 10.00% |
