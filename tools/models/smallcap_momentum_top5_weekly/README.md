# smallcap_momentum_top5_weekly

## Status: ❌ NOT WIRED (underperforms Model 3)

Backtest result was worse than the deployed `momentum_n100_top5_max1`.
Kept in repo as research artifact + reusable scaffold for next attempt.

## Strategy

- Universe: 200 stocks from N500 by ADV ranking, skipping top-50 (large caps)
- Signal: 30-day return
- Pick top-5, hold max-3 equal-weight
- Weekly rebalance (Monday)
- Realistic costs: 10bps slip + ₹20 brokerage + 0.10% STT

## 3-year backtest result (2023-05 → 2026-05)

| Metric | Value |
|---|---:|
| Final equity | ₹3,28,538 (from ₹2,00,000) |
| Total return | **+64.3%** over 3 yrs |
| Avg/mo | +2.02% |
| Avg/yr | +17.23% |
| Best mo | +17.92% |
| Worst mo | -14.39% |
| Months ≥ 20% | 0 / 37 |
| Months ≥ 30% | 0 / 37 |
| Max DD | -46.57% |
| Trades | 501 |
| Fees | ₹1,87,937 (94% drag) |

## Why it loses to Model 3

| Aspect | Model 3 | This model |
|---|---|---|
| Rebalance | monthly | weekly |
| Universe | top-100 (large+mid) | 200 (mid+small, lower liquidity) |
| Fees | ~₹46k / 3yr | ~₹188k / 3yr |
| Yearly | +56.8% | +17.2% |

**Weekly rotation in small/midcap universe = too much whipsaw + too many fees.**
Best smallcap variant (top=5 max=1 lb=20) hit +47% best month BUT -78% max DD —
unusable.

## 20%/mo target verdict

❌ NOT achievable on Indian equity systematic strategies.

Web research (Capitalmind, Nifty momentum-quality indices, PEAD studies)
shows realistic top-decile equity strategies cap at 30-50%/yr (~3-4%/mo).
Sources:
- Capitalmind Momentum Portfolio (15-25 stocks, weekly, ~25-35% CAGR)
- Nifty Smallcap250 Momentum Quality 100 (semi-annual rebalance, ~28% CAGR)
- India PEAD anomaly (10-25% annual)

## Files (research scaffold, reusable)

```
tools/models/smallcap_momentum_top5_weekly/
├── README.md              (this file)
├── build_universe.py      smallcap selector (ADV rank rows 51-250)
├── backtest.py            weekly-rebalance backtest with real costs
└── sweep.py               variant sweep (top/max/lb combos)
```

Universe build:
```bash
python tools/models/smallcap_momentum_top5_weekly/build_universe.py \
    --skip-large 50 --top 200 \
    --out /app/logs/momrot/universes/smallcap_current.json
```

Run backtest:
```bash
python tools/models/smallcap_momentum_top5_weekly/backtest.py \
    --universe-file /app/logs/momrot/universes/smallcap_current.json \
    --from 2023-05-15 --to 2026-05-15 \
    --top 5 --max-conc 3 --lookback 30 --capital 200000 \
    --out exports/models/smallcap_momentum_top5_weekly/run_$(date +%F).md
```

## Honest path forward for higher equity returns

1. **Stay with Model 3** (+56.8%/yr, -41% DD, ~5%/mo avg) — it's the realistic top
2. **Add Nifty Fut leverage 1.5×** on Model 3 picks → boost to ~8%/mo but DD scales too
3. **Stack Model 3 (₹1.5L equity) + FinNifty IC (₹0.5L margin)** → combined ~6%/mo
4. **Accept that 20%/mo on equities is a fantasy number** — no public strategy hits it sustained
