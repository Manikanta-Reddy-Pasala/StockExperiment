# All Trading Models — Combined SUMMARY

3-year backtest window: **2023-05-15 → 2026-05-12 (≈3.00 years)** · ₹10L per model

6 models. All ≥ 80% CAGR except midcap-ex-ANGELONE (below threshold, see note). Only `momentum_n100_top5_max1` is LIVE-deployed.

## Headline — deploy ₹10L per model

| # | Model | Stock list filter | Rebalance | Final NAV | CAGR | Max DD | LIVE |
|--:|---|---|---|---:|---:|---:|:-:|
| 1 | `momentum_n100_top5_max1` | Real NSE Nifty 100 (104 stocks) | Monthly | ₹58.69 L | **+80.38%** | 29.71% | ✅ |
| 2 | `momentum_pseudo_n100_adv` | Top-100 ADV from N500 (yearly PIT) | Monthly | ₹1.32 Cr | **+136.39%** | 16.15% | ❌ |
| 3 | `midcap_narrow_60d_breakout` ¹ | N500 skip-30 ADV + next-100 (mid-cap proxy) | Event-driven (breakout) | ₹47.92 L | **+68.60%** ² | 17.83% | ❌ |
| 4 | `finnifty_ic_otm4_w300_lots5` | FINNIFTY options (no equity) | Monthly Iron Condor | ₹1.11 Cr (scaled) | **+123%** | 13.88% | ❌ |
| 5 | `n20_daily_30d_mc1_uptrend` (v1) | Top-20 ADV + uptrend filter | Daily | ₹1.70 Cr | **+157.27%** | 50.61% | ❌ |
| 6 | `n20_daily_v2_large_only` (v2) | Top-20 ADV + uptrend + **NSE Nifty 100** filter | Daily | ₹1.40 Cr | **+140.78%** | **26.92%** | ❌ |

¹ **ANGELONE excluded** (corp-action data anomaly: entry ₹316 → exit ₹2856 = 9x in 2 months, likely unadjusted bonus/split). With ANGELONE = +337.62% CAGR / ₹8.38 Cr / 13 trades / 6.76% DD but inflated by that one trade.

² Below user's 80% CAGR threshold. Strategy fragility exposed without anomaly.

## Unique stock-filtering approach per model

| Model | Filter mechanism | Why unique |
|---|---|---|
| `momentum_n100_top5_max1` | NSE official index list (no derived filter) | Only model using REAL NSE Nifty 100 constituents — large-cap pure |
| `momentum_pseudo_n100_adv` | ADV ranking from N500 → top 100 | Liquidity-based selection (vs market-cap); catches retail-volume mid-caps NSE excludes |
| `midcap_narrow_60d_breakout` | ADV rank 31-130 + 40d breakout + vol>2× + close>200d SMA | Only event-driven model (calendar-blind); 3-stage filter before entry |
| `finnifty_ic_otm4_w300_lots5` | Spot-derived strikes (4 legs at ±4% OTM + ±300pt wings) | No equity universe — options chain auto-built per Monday from FINNIFTY spot |
| `n20_daily_30d_mc1_uptrend` (v1) | Top-20 ADV + close>200d SMA gate | Smallest universe (20 vs 100); strictest uptrend filter; daily PIT rebuild |
| `n20_daily_v2_large_only` (v2) | v1 universe + NSE Nifty 100 membership filter | Halves Max DD vs v1 (50→27%) with -16pp CAGR cost. Calmar 5.23 vs 3.10. |

## Returns by NSE cap segment (across all equity models)

NSE classification (current snapshot): **Large** = in Nifty 100 (104 stocks). **Mid** = in Nifty Midcap 150 (150). **Small** = in Nifty Smallcap 250 (250). **Outside** = not in Nifty 500.

### momentum_n100_top5_max1

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Large** | 31 | 23 | 8 | 74% | +4,791,235 |

### momentum_pseudo_n100_adv

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Large** | 15 | 14 | 1 | 93% | +8,756,351 |
| **Mid** | 12 | 10 | 2 | 83% | +3,505,564 |
| **Small** | 3 | 2 | 1 | 67% | -226,439 |

### midcap_narrow_60d_breakout

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Large** | 4 | 2 | 2 | 50% | +1,699,565 |
| **Mid** | 5 | 5 | 0 | 100% | +2,198,826 |
| **Small** | 3 | 2 | 1 | 67% | -314,111 |

### n20_daily_30d_mc1_uptrend

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Large** | 50 | 27 | 23 | 54% | +10,263,100 |
| **Mid** | 46 | 25 | 21 | 54% | +5,698,998 |
| **Small** | 38 | 12 | 26 | 32% | -143,536 |

### n20_daily_v2_large_only

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Large** | 139 | 59 | 78 | 43% | +12,959,936 |

All trades Large-cap by construction.

## Composite ranking — risk-adjusted Calmar (CAGR / Max DD)

| Rank | Model | CAGR | MaxDD | Calmar | Notes |
|--:|---|---:|---:|---:|---|
| 1 | momentum_pseudo_n100_adv | +136.39% | 16.15% | **8.44** | Lookahead bias |
| 2 | **n20_daily_v2_large_only** | **+140.78%** | **26.92%** | **5.23** | **NSE Nifty 100 filter** |
| 2 | finnifty_ic_otm4_w300_lots5 | +123.30% | 13.88% | **8.88** | Honest, defined-risk |
| 3 | midcap_narrow (ex-ANGELONE) | +68.60% | 17.83% | **3.85** | Below 80% threshold |
| 4 | n20_daily_30d_mc1_uptrend | +157.27% | 50.61% | **3.11** | High vol, daily rotation |
| 5 | momentum_n100_top5_max1 | +80.38% | 29.71% | **2.71** | LIVE deployable |

## Deployment recommendation

| Goal | Use |
|---|---|
| **Live equity (real, deployable)** | `momentum_n100_top5_max1` (LIVE) |
| **Defined-risk income (options)** | `finnifty_ic_otm4_w300_lots5` |
| **Upper-bound research / aggressive sim** | `momentum_pseudo_n100_adv` |
| **Backtest exploration / breakout style** | `midcap_narrow_60d_breakout` (ANGELONE caveat) |
| **Highest raw return, accept volatility** | `n20_daily_30d_mc1_uptrend` (v1) |
| **Daily rotation with halved DD** | `n20_daily_v2_large_only` (v2, Calmar 5.23) |

## Cross-cutting caveats

1. **NSE cap classification** uses current Nifty 100/Midcap 150/Smallcap 250 snapshots (refreshed 2026-05-17). Index membership shifts over time — a stock classified Mid today may have been Large in 2023 or vice-versa. ~5-8%/yr index turnover.
2. **Lookahead universe** is biggest hidden lever — Models 2 + 3 + 5 rely on knowing future high-ADV stocks. Honest counterparts: real NSE Nifty 100 (+80%, Model 1) and real Nifty Midcap 150 (-18% on Model 3 strategy).
3. **Single-trade concentration** in Model 3: ANGELONE alone = 80% of returns. Excluded for honest result.
4. **Slippage modeled** only in Models 3 + 4. Rotation models (1, 2, 5) ignore slippage (~10-30 bps/round-trip drag).
5. **Costs**: STT + brokerage ~1-2%/yr drag for monthly models, ~3-5%/yr for daily (Model 5).
6. **Survivorship**: stocks delisted from N500 mid-window missing in all equity models.

## Files

```
exports/models/
├── SUMMARY.md                            ← this file (5-model combined)
├── momentum_n100_top5_max1/
│   ├── SUMMARY.md                        Stock pick logic + cap segments + per-trade table
│   └── TRADE_LEDGER.md
├── momentum_pseudo_n100_adv/
│   ├── SUMMARY.md
│   └── TRADE_LEDGER.md
├── midcap_narrow_60d_breakout/
│   ├── SUMMARY.md                        ANGELONE-excluded version
│   └── TRADE_LEDGER.md                   full V1 with ANGELONE included
├── finnifty_ic_otm4_w300_lots5/
│   ├── SUMMARY.md                        Per-leg option prices + amounts
│   ├── MONTHLY_INVESTED.md
│   ├── trades.csv
│   └── monthly.csv
├── n20_daily_30d_mc1_uptrend/
│   ├── SUMMARY.md                        Top winners + losses tables (v1, +157% CAGR)
│   └── TRADE_LEDGER.md
└── n20_daily_v2_large_only/
    ├── SUMMARY.md                        v2 with NSE Nifty 100 filter (+141% CAGR / 27% DD)
    └── TRADE_LEDGER.md
```

Per-model code + strategy + universe construction: `tools/models/<model>/README.md` and `backtest.py`.