# Ather Energy Ltd. (ATHERENERG.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-05-06 09:15:00 → 2026-04-30 15:15:00 (1717 bars)
- **Last close:** 934.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT3 | 2 |
| ENTRY1 | 1 |
| ENTRY2 | 1 |
| EXIT | 1 |

## P&L

- **Trades closed:** 2
- **Trades open at end:** 0
- **Winners / losers:** 0 / 2
- **Target hits / EMA400 exits:** 0 / 2
- **Total realized P&L (per unit):** -59.05
- **Avg P&L per closed trade:** -29.52

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 596.40 | 651.00 | 651.23 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 12:15:00 | 696.35 | 650.88 | 650.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 14:15:00 | 713.90 | 652.01 | 651.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 12:15:00 | 686.70 | 690.55 | 675.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-24 15:15:00 | 706.00 | 690.70 | 675.42 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 12:15:00 | 679.15 | 692.54 | 678.22 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-02 14:15:00 | 707.05 | 692.56 | 678.37 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 680.10 | 692.48 | 678.54 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-04 11:15:00 | 677.00 | 692.32 | 678.53 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2026-02-24 15:15:00 | 706.00 | 2026-03-04 11:15:00 | 677.00 | EXIT_EMA400 | -29.00 |
| BUY | 2026-03-02 14:15:00 | 707.05 | 2026-03-04 11:15:00 | 677.00 | EXIT_EMA400 | -30.05 |
