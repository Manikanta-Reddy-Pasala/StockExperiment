# HCLTECH (HCLTECH)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1182.30
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 2 |
| PENDING | 2 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 1 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 1
- **Target hits / Stop hits / Partials:** 0 / 2 / 1
- **Avg / median % per leg:** 5.46% / 4.47%
- **Sum % (uncompounded):** 16.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 0 | 2 | 1 | 5.46% | 16.4% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.07% | -3.1% |
| BUY @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 0 | 1 | 1 | 9.73% | 19.5% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.07% | -3.1% |
| retest2 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 9.73% | 19.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 1650.80 | 1606.38 | 1606.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 1668.90 | 1622.83 | 1615.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 1691.70 | 1692.97 | 1665.86 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-07-08 12:15:00 | 1714.50 | 1693.30 | 1666.43 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 13:15:00 | 1711.00 | 1693.48 | 1666.65 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1658.40 | 1692.30 | 1667.36 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 1658.40 | 1692.30 | 1667.36 | SL hit (close<ema400) qty=1.00 sl=1667.36 alert=retest1 |

### Cycle 2 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 1535.20 | 1495.73 | 1495.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 1539.50 | 1496.17 | 1495.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 1498.10 | 1501.08 | 1498.50 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 09:15:00 | 1498.10 | 1501.08 | 1498.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1498.10 | 1501.08 | 1498.50 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-11-10 09:15:00 | 1532.00 | 1501.90 | 1499.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 10:15:00 | 1541.80 | 1502.30 | 1499.22 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-02-03 09:15:00 | 1773.07 | 1674.42 | 1644.97 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1610.70 | 1675.51 | 1646.55 | SL hit (close<ema200) qty=0.50 sl=1675.51 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-08 13:15:00 | 1711.00 | 2025-07-10 09:15:00 | 1658.40 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2025-11-10 10:15:00 | 1541.80 | 2026-02-03 09:15:00 | 1773.07 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-11-10 10:15:00 | 1541.80 | 2026-02-04 09:15:00 | 1610.70 | STOP_HIT | 0.50 | 4.47% |
