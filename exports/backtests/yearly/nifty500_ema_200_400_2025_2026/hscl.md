# Himadri Speciality Chemical Ltd. (HSCL)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 631.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 4 |
| ALERT3 | 32 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 28 |
| PARTIAL | 7 |
| TARGET_HIT | 0 |
| STOP_HIT | 28 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 35 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 25
- **Target hits / Stop hits / Partials:** 0 / 28 / 7
- **Avg / median % per leg:** -0.64% / -1.58%
- **Sum % (uncompounded):** -22.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.50% | -6.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.50% | -6.0% |
| SELL (all) | 31 | 10 | 32.3% | 0 | 24 | 7 | -0.53% | -16.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 31 | 10 | 32.3% | 0 | 24 | 7 | -0.53% | -16.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 35 | 10 | 28.6% | 0 | 28 | 7 | -0.64% | -22.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 450.45 | 474.40 | 474.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 449.85 | 473.93 | 474.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 483.50 | 466.83 | 470.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 483.50 | 466.83 | 470.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 483.50 | 466.83 | 470.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 483.50 | 466.83 | 470.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 474.80 | 466.91 | 470.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:30:00 | 471.85 | 467.18 | 470.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 470.45 | 467.36 | 470.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 10:00:00 | 471.20 | 467.20 | 469.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 10:00:00 | 472.25 | 467.35 | 469.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 472.00 | 467.40 | 469.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:45:00 | 472.85 | 467.40 | 469.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 469.55 | 467.49 | 469.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:30:00 | 469.80 | 467.49 | 469.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 469.75 | 467.52 | 469.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 469.75 | 467.52 | 469.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 469.95 | 467.54 | 469.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 470.35 | 467.54 | 469.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 472.30 | 467.59 | 469.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:45:00 | 472.40 | 467.59 | 469.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 472.70 | 467.64 | 469.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:30:00 | 472.95 | 467.64 | 469.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 470.00 | 467.71 | 469.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:45:00 | 470.10 | 467.71 | 469.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 467.45 | 467.71 | 469.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 464.00 | 467.71 | 469.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:15:00 | 448.26 | 466.14 | 468.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:15:00 | 446.93 | 466.14 | 468.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:15:00 | 447.64 | 466.14 | 468.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:15:00 | 448.64 | 466.14 | 468.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 11:15:00 | 471.35 | 466.19 | 468.87 | SL hit (close>ema200) qty=0.50 sl=466.19 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 486.05 | 466.51 | 466.51 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 15:15:00 | 452.60 | 466.66 | 466.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 15:15:00 | 450.85 | 465.75 | 466.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 454.50 | 451.26 | 457.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 454.50 | 451.26 | 457.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 454.50 | 451.26 | 457.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 456.30 | 451.26 | 457.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 467.80 | 451.43 | 457.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 467.80 | 451.43 | 457.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 463.15 | 451.54 | 457.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 12:15:00 | 458.15 | 451.54 | 457.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 474.50 | 452.11 | 457.34 | SL hit (close>static) qty=1.00 sl=467.90 alert=retest2 |

### Cycle 4 — BUY (started 2025-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 14:15:00 | 484.20 | 460.71 | 460.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 14:15:00 | 488.45 | 466.98 | 464.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 13:15:00 | 473.10 | 474.03 | 468.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 13:30:00 | 472.90 | 474.03 | 468.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 474.35 | 474.06 | 468.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 470.25 | 474.06 | 468.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 469.30 | 473.94 | 468.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 15:15:00 | 475.00 | 473.94 | 468.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 464.35 | 473.86 | 468.88 | SL hit (close<static) qty=1.00 sl=467.50 alert=retest2 |

### Cycle 5 — SELL (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 11:15:00 | 450.70 | 466.05 | 466.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 12:15:00 | 449.70 | 465.89 | 465.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 15:15:00 | 470.50 | 464.52 | 465.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 15:15:00 | 470.50 | 464.52 | 465.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 470.50 | 464.52 | 465.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 452.50 | 464.10 | 465.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 12:00:00 | 453.15 | 462.72 | 464.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 10:00:00 | 454.50 | 462.45 | 463.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 11:15:00 | 454.50 | 462.38 | 463.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 461.70 | 461.90 | 463.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 15:15:00 | 453.20 | 461.90 | 463.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 13:45:00 | 459.80 | 461.53 | 463.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 15:15:00 | 459.90 | 461.53 | 463.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 14:15:00 | 473.00 | 459.89 | 462.24 | SL hit (close>static) qty=1.00 sl=464.50 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 486.65 | 464.37 | 464.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 12:15:00 | 488.55 | 464.83 | 464.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 469.65 | 470.29 | 467.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 469.65 | 470.29 | 467.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 469.65 | 470.29 | 467.54 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 435.50 | 464.96 | 465.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 426.50 | 457.79 | 461.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 13:15:00 | 457.20 | 455.47 | 459.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-18 13:30:00 | 457.00 | 455.47 | 459.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 461.15 | 452.62 | 457.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:45:00 | 461.00 | 452.62 | 457.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 460.40 | 452.69 | 457.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:30:00 | 460.75 | 452.69 | 457.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 454.20 | 453.25 | 457.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 455.35 | 453.25 | 457.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 452.50 | 453.24 | 457.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 452.50 | 453.24 | 457.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 458.60 | 451.93 | 456.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:15:00 | 454.95 | 451.93 | 456.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 457.60 | 451.99 | 456.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:45:00 | 451.20 | 452.06 | 456.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:15:00 | 450.90 | 452.06 | 456.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 428.64 | 451.74 | 456.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 428.35 | 451.74 | 456.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 451.95 | 450.59 | 455.58 | SL hit (close>ema200) qty=0.50 sl=450.59 alert=retest2 |

### Cycle 8 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 494.15 | 459.05 | 459.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 505.90 | 463.20 | 461.18 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-10 14:30:00 | 471.85 | 2025-09-24 10:15:00 | 448.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-12 09:15:00 | 470.45 | 2025-09-24 10:15:00 | 446.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 10:00:00 | 471.20 | 2025-09-24 10:15:00 | 447.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 10:00:00 | 472.25 | 2025-09-24 10:15:00 | 448.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-10 14:30:00 | 471.85 | 2025-09-24 11:15:00 | 471.35 | STOP_HIT | 0.50 | 0.11% |
| SELL | retest2 | 2025-09-12 09:15:00 | 470.45 | 2025-09-24 11:15:00 | 471.35 | STOP_HIT | 0.50 | -0.19% |
| SELL | retest2 | 2025-09-17 10:00:00 | 471.20 | 2025-09-24 11:15:00 | 471.35 | STOP_HIT | 0.50 | -0.03% |
| SELL | retest2 | 2025-09-18 10:00:00 | 472.25 | 2025-09-24 11:15:00 | 471.35 | STOP_HIT | 0.50 | 0.19% |
| SELL | retest2 | 2025-09-22 09:15:00 | 464.00 | 2025-09-24 11:15:00 | 471.35 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-09-25 10:00:00 | 464.60 | 2025-09-30 13:15:00 | 441.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 10:00:00 | 464.60 | 2025-10-06 11:15:00 | 463.30 | STOP_HIT | 0.50 | 0.28% |
| SELL | retest2 | 2025-10-08 10:00:00 | 464.50 | 2025-10-15 10:15:00 | 472.60 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-10-13 13:45:00 | 464.20 | 2025-10-15 10:15:00 | 472.60 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-10-17 10:30:00 | 462.80 | 2025-10-23 11:15:00 | 470.45 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-12-04 12:15:00 | 458.15 | 2025-12-05 09:15:00 | 474.50 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2025-12-08 13:00:00 | 462.75 | 2025-12-11 15:15:00 | 464.25 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-12-10 15:15:00 | 459.75 | 2025-12-12 11:15:00 | 469.45 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-12-11 09:30:00 | 461.00 | 2025-12-12 11:15:00 | 469.45 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-12-11 12:30:00 | 457.25 | 2025-12-12 11:15:00 | 469.45 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2026-01-09 15:15:00 | 475.00 | 2026-01-12 09:15:00 | 464.35 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2026-01-13 09:15:00 | 475.00 | 2026-01-19 09:15:00 | 467.45 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-01-14 11:15:00 | 472.50 | 2026-01-19 09:15:00 | 467.45 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-01-16 09:15:00 | 472.65 | 2026-01-19 09:15:00 | 467.45 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-02-01 12:15:00 | 452.50 | 2026-02-16 14:15:00 | 473.00 | STOP_HIT | 1.00 | -4.53% |
| SELL | retest2 | 2026-02-05 12:00:00 | 453.15 | 2026-02-16 14:15:00 | 473.00 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2026-02-06 10:00:00 | 454.50 | 2026-02-16 14:15:00 | 473.00 | STOP_HIT | 1.00 | -4.07% |
| SELL | retest2 | 2026-02-06 11:15:00 | 454.50 | 2026-02-17 09:15:00 | 476.20 | STOP_HIT | 1.00 | -4.77% |
| SELL | retest2 | 2026-02-09 15:15:00 | 453.20 | 2026-02-17 09:15:00 | 476.20 | STOP_HIT | 1.00 | -5.08% |
| SELL | retest2 | 2026-02-10 13:45:00 | 459.80 | 2026-02-17 09:15:00 | 476.20 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2026-02-10 15:15:00 | 459.90 | 2026-02-17 09:15:00 | 476.20 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2026-04-01 13:45:00 | 451.20 | 2026-04-02 09:15:00 | 428.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 14:15:00 | 450.90 | 2026-04-02 09:15:00 | 428.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 13:45:00 | 451.20 | 2026-04-06 14:15:00 | 451.95 | STOP_HIT | 0.50 | -0.17% |
| SELL | retest2 | 2026-04-01 14:15:00 | 450.90 | 2026-04-06 14:15:00 | 451.95 | STOP_HIT | 0.50 | -0.23% |
| SELL | retest2 | 2026-04-07 15:00:00 | 452.00 | 2026-04-08 09:15:00 | 471.40 | STOP_HIT | 1.00 | -4.29% |
