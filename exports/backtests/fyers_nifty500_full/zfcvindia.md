# ZF Commercial Vehicle Control Systems India Ltd. (ZFCVINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 14604.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 1708.48
- **Avg P&L per closed trade:** 284.75

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 14:15:00 | 15216.00 | 15752.35 | 15753.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 14:15:00 | 15024.85 | 15703.26 | 15728.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 14938.50 | 14738.39 | 15108.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-22 14:15:00 | 14411.50 | 14772.00 | 15022.95 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 14935.00 | 14770.11 | 15013.32 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-26 11:15:00 | 14599.25 | 14769.73 | 15008.32 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 11779.95 | 11120.24 | 11821.10 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-07 12:15:00 | 11888.20 | 11127.88 | 11821.43 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 12769.90 | 11602.42 | 11598.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 13467.40 | 11640.56 | 11617.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 13:15:00 | 12109.00 | 12158.52 | 11919.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-11 09:15:00 | 12455.10 | 12159.80 | 11924.04 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-02 11:15:00 | 12183.00 | 12549.34 | 12252.47 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 12561.00 | 13487.73 | 13492.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 10:15:00 | 12540.00 | 13461.32 | 13478.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 10:15:00 | 13501.00 | 13342.13 | 13413.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-16 15:15:00 | 12976.00 | 13287.35 | 13365.97 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 13064.00 | 12886.36 | 13066.12 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-27 10:15:00 | 13050.00 | 12888.11 | 13032.59 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 12:15:00 | 14937.00 | 13153.09 | 13151.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 12:15:00 | 15258.00 | 13878.20 | 13577.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 14:15:00 | 14471.00 | 14576.60 | 14144.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-30 11:15:00 | 15093.00 | 14212.55 | 14083.08 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 14935.00 | 15220.58 | 14828.58 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-02 10:15:00 | 14828.00 | 15216.67 | 14828.58 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 14:15:00 | 13632.00 | 14577.79 | 14579.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 15:15:00 | 13577.00 | 14567.84 | 14574.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 14465.00 | 14120.10 | 14316.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-06 09:15:00 | 13800.00 | 14131.59 | 14309.65 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-08 09:15:00 | 14542.00 | 14082.46 | 14272.12 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 10:15:00 | 14999.00 | 14382.19 | 14380.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 14:15:00 | 15177.00 | 14410.75 | 14394.84 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-22 14:15:00 | 14411.50 | 2024-11-27 09:15:00 | 12577.14 | TARGET | 1834.36 |
| SELL | 2024-11-26 11:15:00 | 14599.25 | 2024-11-27 09:15:00 | 13372.03 | TARGET | 1227.22 |
| BUY | 2025-04-11 09:15:00 | 12455.10 | 2025-05-02 11:15:00 | 12183.00 | EXIT_EMA400 | -272.10 |
| SELL | 2025-10-16 15:15:00 | 12976.00 | 2025-11-27 10:15:00 | 13050.00 | EXIT_EMA400 | -74.00 |
| BUY | 2026-01-30 11:15:00 | 15093.00 | 2026-03-02 10:15:00 | 14828.00 | EXIT_EMA400 | -265.00 |
| SELL | 2026-04-06 09:15:00 | 13800.00 | 2026-04-08 09:15:00 | 14542.00 | EXIT_EMA400 | -742.00 |
