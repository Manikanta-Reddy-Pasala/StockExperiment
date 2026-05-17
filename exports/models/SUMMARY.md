# All Trading Models — Combined SUMMARY

3-year backtest window: **2023-05-15 → 2026-05-12** · ₹10L per model · Fyers data (4-yr re-pull 2026-05-17)

5 models (4 equity + 1 options). Each has ONE canonical version.

## Headline — deploy ₹10L per model

| # | Model | Universe | Rebalance | Final NAV | CAGR | Max DD | LIVE |
|--:|---|---|---|---:|---:|---:|:-:|
| 1 | `momentum_n100_top5_max1` | Real NSE Nifty 100 | Monthly | ₹4,424,405 | **+64.17%** | 37.30% | ✅ |
| 2 | `momentum_pseudo_n100_adv` | Top-100 ADV from N500 MINUS Small | Monthly | ₹9,221,004 | **+109.70%** | 36.44% | ❌ |
| 3 | `midcap_narrow_60d_breakout` | Top-100 ADV from N500 MINUS Large | Event-driven | ₹13,456,535 | **+137.85%** | **8.12%** | ❌ |
| 4 | `finnifty_ic_otm4_w300_lots5` | FINNIFTY options (4 strikes per Mon) | Monthly IC | ₹22,25,673 | **+123.27%** compound | 13.88% | ❌ |
| 5 | `n20_daily_large_only` | Top-20 ADV + uptrend + NSE Nifty 100 | Daily | ₹11,813,452 | **+127.75%** | **24.74%** | ❌ |

## Unique stock-filtering approach per model

| Model | Filter mechanism | Why unique |
|---|---|---|
| `momentum_n100_top5_max1` | NSE Nifty 100 official list (no derived filter) | Only model using REAL NSE constituents — large-cap pure |
| `momentum_pseudo_n100_adv` | Top-100 ADV from N500, drop Small | Liquidity-based; excludes high-volume small caps |
| `midcap_narrow_60d_breakout` | Top-100 ADV from N500, drop Large | Highest-liquidity mid/small breakouts; event-driven |
| `finnifty_ic_otm4_w300_lots5` | Spot-derived 4 strikes (±4% OTM + ±300pt wings) | No equity universe — options chain auto-built per Monday |
| `n20_daily_large_only` | Top-20 ADV + close>200d SMA + NSE Nifty 100 | Smallest universe (20); strictest gate; daily rebuild |

## Composite ranking — risk-adjusted Calmar (CAGR / Max DD)

| Rank | Model | CAGR | MaxDD | Calmar |
|--:|---|---:|---:|---:|
| 1 | midcap_narrow_60d_breakout | +137.85% | 8.12% | **16.98** |
| 2 | finnifty_ic_otm4_w300_lots5 | +123.27% | 13.88% | **8.88** |
| 3 | n20_daily_large_only | +127.75% | 24.74% | **5.16** |
| 4 | momentum_pseudo_n100_adv | +109.70% | 36.44% | **3.01** |
| 5 | momentum_n100_top5_max1 | +64.17% | 37.30% | **1.72** |

## Deployment recommendation

| Goal | Use |
|---|---|
| **Live equity (real universe, deployable)** | `momentum_n100_top5_max1` (LIVE) |
| **Defined-risk income (options)** | `finnifty_ic_otm4_w300_lots5` |
| **Best risk-adjusted swing returns** | `midcap_narrow_60d_breakout` (Calmar 17+) |
| **Daily rotation (Nifty 100 only)** | `n20_daily_large_only` |
| **Aggressive pseudo-N100 momentum** | `momentum_pseudo_n100_adv` |

## Data integrity

- All data from Fyers production source (cont_flag=1 split-adjusted)
- 498/504 N500 stocks covered (4 DUMMYVEDL placeholders + 2 symbol mismatch = legitimately missing)
- 1 known anomaly: ABFRL 2025-05-22 real demerger (-67%, not data bug)
- yfinance NEVER used (project rule)

## Files

```
exports/models/
├── SUMMARY.md                          ← this file
├── momentum_n100_top5_max1/
│   ├── SUMMARY.md
│   └── TRADE_LEDGER.md
├── momentum_pseudo_n100_adv/
│   ├── SUMMARY.md
│   └── TRADE_LEDGER.md
├── midcap_narrow_60d_breakout/
│   ├── SUMMARY.md
│   └── TRADE_LEDGER.md
├── finnifty_ic_otm4_w300_lots5/
│   ├── SUMMARY.md + MONTHLY_INVESTED.md + trades.csv + monthly.csv
└── n20_daily_large_only/
    ├── SUMMARY.md
    └── TRADE_LEDGER.md
```