# Sai Life Sciences Ltd. (SAILIFE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-12-18 09:15:00 → 2026-04-30 15:30:00 (2344 bars)
- **Last close:** 1069.45
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 17.43
- **Avg P&L per closed trade:** 3.49

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 12:15:00 | 736.00 | 703.16 | 703.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 13:15:00 | 739.95 | 703.53 | 703.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 13:15:00 | 713.15 | 721.73 | 713.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-15 10:15:00 | 725.40 | 708.03 | 707.40 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-22 12:15:00 | 709.95 | 714.12 | 710.84 | Close below EMA400 |

### Cycle 2 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 818.50 | 886.28 | 886.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 800.10 | 885.43 | 886.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 870.55 | 869.21 | 877.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-04 09:15:00 | 855.60 | 869.52 | 876.89 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 855.60 | 869.52 | 876.89 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-04 10:15:00 | 845.00 | 869.28 | 876.73 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 840.65 | 862.67 | 872.86 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-09 09:15:00 | 887.45 | 862.03 | 872.18 | Close above EMA400 |

### Cycle 3 — BUY (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 12:15:00 | 920.55 | 880.56 | 880.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 928.05 | 882.20 | 881.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 11:15:00 | 951.30 | 957.62 | 928.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-13 15:15:00 | 969.95 | 957.49 | 928.96 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 949.10 | 974.10 | 948.26 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-04-02 12:15:00 | 964.75 | 973.62 | 948.41 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-04-06 09:15:00 | 938.35 | 972.94 | 948.56 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-04-15 10:15:00 | 725.40 | 2025-04-15 14:15:00 | 779.40 | TARGET | 54.00 |
| SELL | 2026-02-04 09:15:00 | 855.60 | 2026-02-05 09:15:00 | 791.72 | TARGET | 63.88 |
| SELL | 2026-02-04 10:15:00 | 845.00 | 2026-02-09 09:15:00 | 887.45 | EXIT_EMA400 | -42.45 |
| BUY | 2026-03-13 15:15:00 | 969.95 | 2026-04-06 09:15:00 | 938.35 | EXIT_EMA400 | -31.60 |
| BUY | 2026-04-02 12:15:00 | 964.75 | 2026-04-06 09:15:00 | 938.35 | EXIT_EMA400 | -26.40 |
