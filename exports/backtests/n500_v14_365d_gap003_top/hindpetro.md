# Hindustan Petroleum Corporation Ltd. (HINDPETRO)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 387.50
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
| ALERT3 | 17 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 4 |
| TARGET_HIT | 4 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 4
- **Target hits / Stop hits / Partials:** 4 / 4 / 4
- **Avg / median % per leg:** 4.50% / 5.00%
- **Sum % (uncompounded):** 53.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 8 | 66.7% | 4 | 4 | 4 | 4.50% | 54.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 8 | 66.7% | 4 | 4 | 4 | 4.50% | 54.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 8 | 66.7% | 4 | 4 | 4 | 4.50% | 54.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 421.20 | 457.24 | 457.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 418.25 | 456.85 | 457.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 447.70 | 447.14 | 451.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 10:00:00 | 447.70 | 447.14 | 451.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 452.45 | 447.15 | 451.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 452.45 | 447.15 | 451.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 453.65 | 447.22 | 451.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 453.50 | 447.22 | 451.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 450.90 | 447.27 | 451.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 15:00:00 | 450.90 | 447.27 | 451.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 463.25 | 447.47 | 451.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:00:00 | 463.25 | 447.47 | 451.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 460.20 | 447.59 | 451.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 11:30:00 | 459.05 | 447.71 | 451.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 14:30:00 | 459.10 | 448.02 | 451.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 464.60 | 448.31 | 451.93 | SL hit (close>static) qty=1.00 sl=464.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 464.60 | 448.31 | 451.93 | SL hit (close>static) qty=1.00 sl=464.30 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 12:30:00 | 457.50 | 448.64 | 452.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 15:00:00 | 458.90 | 448.82 | 452.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 466.45 | 449.09 | 452.21 | SL hit (close>static) qty=1.00 sl=464.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 466.45 | 449.09 | 452.21 | SL hit (close>static) qty=1.00 sl=464.30 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 453.85 | 452.33 | 453.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:00:00 | 453.85 | 452.33 | 453.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 453.85 | 452.35 | 453.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:30:00 | 455.20 | 452.35 | 453.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 451.85 | 452.34 | 453.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:45:00 | 452.50 | 452.34 | 453.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 451.90 | 452.34 | 453.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:30:00 | 452.90 | 452.34 | 453.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 454.30 | 451.88 | 453.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 454.30 | 451.88 | 453.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 454.70 | 451.90 | 453.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 456.30 | 451.90 | 453.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 439.00 | 448.71 | 451.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:15:00 | 435.40 | 448.61 | 451.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 437.90 | 447.70 | 450.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 13:15:00 | 438.05 | 447.63 | 450.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:45:00 | 436.50 | 447.31 | 450.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 416.00 | 446.74 | 449.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 416.15 | 446.74 | 449.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 413.63 | 445.19 | 449.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 414.67 | 445.19 | 449.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 391.86 | 437.78 | 444.77 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 394.11 | 437.78 | 444.77 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 394.25 | 437.78 | 444.77 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 392.85 | 437.78 | 444.77 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 383.80 | 372.00 | 385.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:30:00 | 383.15 | 372.00 | 385.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 393.15 | 372.21 | 385.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:45:00 | 392.95 | 372.21 | 385.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 398.15 | 372.47 | 385.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:30:00 | 401.50 | 372.47 | 385.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 386.45 | 375.08 | 385.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:30:00 | 389.20 | 375.08 | 385.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 387.50 | 375.20 | 385.67 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-02-04 11:30:00 | 459.05 | 2026-02-05 09:15:00 | 464.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-02-04 14:30:00 | 459.10 | 2026-02-05 09:15:00 | 464.60 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-02-05 12:30:00 | 457.50 | 2026-02-06 09:15:00 | 466.45 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-02-05 15:00:00 | 458.90 | 2026-02-06 09:15:00 | 466.45 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-02-25 11:15:00 | 435.40 | 2026-03-02 09:15:00 | 416.00 | PARTIAL | 0.50 | 4.45% |
| SELL | retest2 | 2026-02-26 11:30:00 | 437.90 | 2026-03-02 09:15:00 | 416.15 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2026-02-26 13:15:00 | 438.05 | 2026-03-04 09:15:00 | 413.63 | PARTIAL | 0.50 | 5.57% |
| SELL | retest2 | 2026-02-27 10:45:00 | 436.50 | 2026-03-04 09:15:00 | 414.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:15:00 | 435.40 | 2026-03-09 09:15:00 | 391.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-26 11:30:00 | 437.90 | 2026-03-09 09:15:00 | 394.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-26 13:15:00 | 438.05 | 2026-03-09 09:15:00 | 394.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-27 10:45:00 | 436.50 | 2026-03-09 09:15:00 | 392.85 | TARGET_HIT | 0.50 | 10.00% |
