# EMA 200/400 — Loss Analysis (Fyers Nifty 500)

_Source: Fyers production data, 504 NSE Nifty 500 symbols, 720 days, 1H bars._

## Trade-level breakdown

| Metric | Count | % |
|--------|------|---|
| Total closed trades | 2695 | 100% |
| Winners | 852 | 31.6% |
| **LOSERS** | **1843** | **68.4%** |
| Total realized P&L per unit | +9429.09 | — |
| Avg P&L per trade | +3.50 | — |

## Loss source

| Exit Reason | Count | Winners | Losers | Sum P&L |
|-------------|-------|---------|--------|---------|
| TARGET hit (1:3 RR closed) | 758 | 758 | 0 | +120899.73 |
| EMA400 close-exit | 1937 | 94 | 1843 | -111470.64 |

**EMA400-exit accounts for 100% of losses** — strategy never loses on
a target hit. All losses come from 1H closing on the wrong side of EMA400
after entry.

## BUY vs SELL losses

| Direction | Losing trades |
|-----------|---------------|
| BUY  | 817 |
| SELL | 1026 |

## Extremes

- Biggest single-trade win : `mrf` +11353.98
- Biggest single-trade loss: `mrf` -4261.30

## Risk model recap
- Stop loss: 1H close past EMA400 (not intra-bar — closer-to-EMA stops not enforced)
- Target: 1:3 RR (equity), 5000 absolute pts (index)
- Win rate 31.6% but average winner > average loser → net P&L positive
- Multiple entries (Entry1 + Entry2) close together on EXIT signal → can compound losses
