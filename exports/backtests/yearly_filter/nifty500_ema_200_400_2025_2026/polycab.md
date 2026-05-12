# Polycab India Ltd. (POLYCAB)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 9080.00
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
| ALERT2 | 6 |
| ALERT2_SKIP | 3 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 2 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 2 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 0
- **Target hits / Stop hits / Partials:** 2 / 0 / 0
- **Avg / median % per leg:** 10.00% / 10.00%
- **Sum % (uncompounded):** 20.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 13:15:00 | 5946.00 | 5571.86 | 5571.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 14:15:00 | 5979.50 | 5575.91 | 5573.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 5891.00 | 5953.68 | 5839.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 10:00:00 | 5891.00 | 5953.68 | 5839.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 5879.50 | 5952.95 | 5839.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 5841.00 | 5952.95 | 5839.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 5810.00 | 5950.64 | 5839.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 5810.00 | 5950.64 | 5839.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 5855.00 | 5949.69 | 5839.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:30:00 | 5871.00 | 5945.25 | 5839.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 11:30:00 | 5948.00 | 5944.96 | 5840.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-25 09:15:00 | 6458.10 | 5994.80 | 5875.27 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 7305.50 | 7393.33 | 7393.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 7270.00 | 7392.10 | 7392.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 7384.50 | 7349.88 | 7370.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 13:15:00 | 7384.50 | 7349.88 | 7370.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 7384.50 | 7349.88 | 7370.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:45:00 | 7368.50 | 7349.88 | 7370.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 7437.50 | 7350.75 | 7370.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 7437.50 | 7350.75 | 7370.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 7435.00 | 7351.59 | 7371.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 7595.00 | 7351.59 | 7371.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 09:15:00 | 7653.00 | 7390.11 | 7389.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 7805.00 | 7459.35 | 7427.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 09:15:00 | 7536.50 | 7571.57 | 7498.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:30:00 | 7535.00 | 7571.57 | 7498.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 7486.50 | 7569.94 | 7498.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:00:00 | 7486.50 | 7569.94 | 7498.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 7424.50 | 7568.49 | 7498.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:00:00 | 7424.50 | 7568.49 | 7498.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 7517.50 | 7567.82 | 7498.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 7397.00 | 7567.82 | 7498.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 7395.00 | 7566.10 | 7498.21 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 12:15:00 | 7038.50 | 7442.15 | 7443.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 14:15:00 | 6991.50 | 7433.49 | 7439.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 7437.50 | 7224.68 | 7319.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 7437.50 | 7224.68 | 7319.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 7437.50 | 7224.68 | 7319.05 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 7812.50 | 7390.98 | 7390.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 10:15:00 | 7827.50 | 7472.70 | 7434.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 11:15:00 | 7959.50 | 7964.14 | 7741.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-10 12:00:00 | 7959.50 | 7964.14 | 7741.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 7774.00 | 7960.13 | 7742.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:00:00 | 7774.00 | 7960.13 | 7742.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 7722.00 | 7957.77 | 7741.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:45:00 | 7752.00 | 7957.77 | 7741.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 7725.00 | 7955.45 | 7741.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 7619.50 | 7955.45 | 7741.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2026-03-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 15:15:00 | 6800.00 | 7592.16 | 7592.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 6769.50 | 7416.95 | 7497.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 7540.00 | 7345.80 | 7451.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 7540.00 | 7345.80 | 7451.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 7540.00 | 7345.80 | 7451.73 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2026-04-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 11:15:00 | 8253.00 | 7531.77 | 7529.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 15:15:00 | 8271.00 | 7733.81 | 7642.27 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-20 10:30:00 | 5871.00 | 2025-06-25 09:15:00 | 6458.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-20 11:30:00 | 5948.00 | 2025-06-27 09:15:00 | 6542.80 | TARGET_HIT | 1.00 | 10.00% |
