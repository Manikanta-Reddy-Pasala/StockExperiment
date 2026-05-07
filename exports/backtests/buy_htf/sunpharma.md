# SUNPHARMA (SUNPHARMA)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2025-11-10 09:15:00 → 2026-05-07 15:15:00 (847 bars)
- **Last close:** 1832.40
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
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 2 |
| PENDING | 5 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / Stop hits / Partials:** 0 / 5 / 0
- **Avg / median % per leg:** -1.77% / -2.56%
- **Sum % (uncompounded):** -8.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 5 | 0 | -1.77% | -8.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 1 | 20.0% | 0 | 5 | 0 | -1.77% | -8.8% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 5 | 1 | 20.0% | 0 | 5 | 0 | -1.77% | -8.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 11:15:00 | 1762.10 | 1712.82 | 1712.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 09:15:00 | 1779.30 | 1718.87 | 1715.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 10:15:00 | 1759.70 | 1760.09 | 1741.09 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 14:15:00 | 1739.70 | 1759.89 | 1741.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 1739.70 | 1759.89 | 1741.37 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-20 09:15:00 | 1761.40 | 1759.81 | 1741.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 10:15:00 | 1766.70 | 1759.88 | 1741.64 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-24 09:15:00 | 1761.80 | 1760.75 | 1743.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 1761.70 | 1760.76 | 1743.33 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-25 09:15:00 | 1776.60 | 1760.90 | 1743.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 1782.40 | 1761.11 | 1744.11 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-30 15:15:00 | 1762.00 | 1765.88 | 1748.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 1777.80 | 1766.00 | 1748.33 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 2520m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 1740.10 | 1765.62 | 1748.40 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-01 13:15:00 | 1721.50 | 1765.19 | 1748.27 | SL hit (close<static) qty=1.00 sl=1737.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 13:15:00 | 1721.50 | 1765.19 | 1748.27 | SL hit (close<static) qty=1.00 sl=1737.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 13:15:00 | 1721.50 | 1765.19 | 1748.27 | SL hit (close<static) qty=1.00 sl=1737.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 13:15:00 | 1721.50 | 1765.19 | 1748.27 | SL hit (close<static) qty=1.00 sl=1737.90 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-29 14:15:00 | 1780.10 | 1713.68 | 1721.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 15:15:00 | 1773.00 | 1714.27 | 1721.89 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 1819.00 | 1729.19 | 1729.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 11:15:00 | 1819.00 | 1729.19 | 1729.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 1842.70 | 1734.75 | 1731.84 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-03-20 10:15:00 | 1766.70 | 2026-04-01 13:15:00 | 1721.50 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2026-03-24 10:15:00 | 1761.70 | 2026-04-01 13:15:00 | 1721.50 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2026-03-25 10:15:00 | 1782.40 | 2026-04-01 13:15:00 | 1721.50 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2026-04-01 09:15:00 | 1777.80 | 2026-04-01 13:15:00 | 1721.50 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2026-04-29 15:15:00 | 1773.00 | 2026-05-05 11:15:00 | 1819.00 | STOP_HIT | 1.00 | 2.59% |
