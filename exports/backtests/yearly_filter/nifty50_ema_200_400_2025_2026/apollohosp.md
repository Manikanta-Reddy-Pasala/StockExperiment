# APOLLOHOSP (APOLLOHOSP)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 8100.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 25 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 26 |
| PARTIAL | 1 |
| TARGET_HIT | 2 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 4 / 20
- **Target hits / Stop hits / Partials:** 2 / 21 / 1
- **Avg / median % per leg:** -0.15% / -0.88%
- **Sum % (uncompounded):** -3.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 2 | 12.5% | 2 | 14 | 0 | -0.46% | -7.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 2 | 12.5% | 2 | 14 | 0 | -0.46% | -7.4% |
| SELL (all) | 8 | 2 | 25.0% | 0 | 7 | 1 | 0.46% | 3.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 2 | 25.0% | 0 | 7 | 1 | 0.46% | 3.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 24 | 4 | 16.7% | 2 | 21 | 1 | -0.15% | -3.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 09:15:00 | 7455.50 | 7636.03 | 7636.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 7426.50 | 7627.42 | 7632.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 7259.50 | 7162.88 | 7293.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-06 10:00:00 | 7259.50 | 7162.88 | 7293.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 7310.00 | 7164.35 | 7293.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:30:00 | 7280.50 | 7164.35 | 7293.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 7286.00 | 7165.56 | 7293.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 12:45:00 | 7271.00 | 7205.35 | 7301.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 15:00:00 | 7265.00 | 7207.95 | 7298.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:30:00 | 7274.00 | 7210.89 | 7297.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 7313.50 | 7212.46 | 7297.81 | SL hit (close>static) qty=1.00 sl=7311.50 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 7643.00 | 7232.96 | 7231.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 12:15:00 | 7660.00 | 7237.21 | 7233.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 7513.50 | 7543.36 | 7429.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 10:00:00 | 7513.50 | 7543.36 | 7429.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 7460.50 | 7543.21 | 7433.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:45:00 | 7461.50 | 7543.21 | 7433.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 7425.50 | 7540.31 | 7433.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:00:00 | 7425.50 | 7540.31 | 7433.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 7486.00 | 7539.77 | 7433.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 10:00:00 | 7542.00 | 7539.30 | 7434.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 13:00:00 | 7525.00 | 7538.46 | 7435.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 13:30:00 | 7522.50 | 7538.47 | 7436.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 09:15:00 | 7586.50 | 7537.62 | 7436.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 7357.00 | 7534.20 | 7439.14 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-19 09:15:00 | 7357.00 | 7534.20 | 7439.14 | SL hit (close<static) qty=1.00 sl=7413.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-02 09:15:00 | 7059.00 | 2025-06-03 09:15:00 | 6838.00 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2025-06-05 09:30:00 | 6902.00 | 2025-06-06 09:15:00 | 6841.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-06-06 13:15:00 | 6906.50 | 2025-07-03 12:15:00 | 7597.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-09 10:15:00 | 6904.00 | 2025-07-03 12:15:00 | 7594.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-03 14:15:00 | 7834.00 | 2025-11-07 12:15:00 | 7672.00 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-11-03 15:15:00 | 7835.00 | 2025-11-07 12:15:00 | 7672.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-11-04 11:45:00 | 7840.50 | 2025-11-07 12:15:00 | 7672.00 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2026-01-09 12:45:00 | 7271.00 | 2026-01-13 14:15:00 | 7313.50 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-01-12 15:00:00 | 7265.00 | 2026-01-13 14:15:00 | 7313.50 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2026-01-13 12:30:00 | 7274.00 | 2026-01-13 14:15:00 | 7313.50 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-01-14 09:30:00 | 7274.50 | 2026-01-20 14:15:00 | 6910.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 7274.50 | 2026-02-03 10:15:00 | 7110.50 | STOP_HIT | 0.50 | 2.25% |
| SELL | retest2 | 2026-02-06 13:30:00 | 7137.50 | 2026-02-09 09:15:00 | 7176.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-02-06 14:00:00 | 7137.50 | 2026-02-09 09:15:00 | 7176.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-02-06 15:15:00 | 7128.50 | 2026-02-09 09:15:00 | 7176.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2026-03-17 10:00:00 | 7542.00 | 2026-03-19 09:15:00 | 7357.00 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2026-03-17 13:00:00 | 7525.00 | 2026-03-19 09:15:00 | 7357.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-03-17 13:30:00 | 7522.50 | 2026-03-19 09:15:00 | 7357.00 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2026-03-18 09:15:00 | 7586.50 | 2026-03-19 09:15:00 | 7357.00 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2026-03-20 13:15:00 | 7442.00 | 2026-03-20 14:15:00 | 7365.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-03-24 12:45:00 | 7440.00 | 2026-03-30 10:15:00 | 7374.50 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-03-24 13:15:00 | 7441.00 | 2026-03-30 10:15:00 | 7374.50 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2026-03-25 09:15:00 | 7497.00 | 2026-03-30 10:15:00 | 7374.50 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2026-04-01 09:15:00 | 7514.00 | 2026-04-01 10:15:00 | 7306.50 | STOP_HIT | 1.00 | -2.76% |
