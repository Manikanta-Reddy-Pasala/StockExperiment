# Wayback-Machine index snapshots (supplementary PIT membership)

Harvested from web.archive.org captures of niftyindices.com `IndexConstituent/*.csv`
(`tools/analysis/harvest_wayback_snapshots.py`). These supplement the official NSE
factsheet zips (indices_dataMar2021-2026) where those have gaps.

Filename = `{index}_{YYYYMMDD}.csv` (capture date). The membership builder
(`build_membership_from_nse_factsheets.py --wayback-dir`) ingests `n100_*.csv` /
`n500_*.csv` here as extra dated snapshots.

KEY USE: `n100_20230808.csv` is a REAL Nifty-100 snapshot inside the Sep2022–Mar2024
Next-50 gap (the 4 partial factsheet zips lacked Next-50), so it replaces a
carried-forward guess with actual membership at Aug-2023. `smallcap250_20240708.csv`
is the only historical Smallcap-250 capture (kept for reference; smallcap has no
dedicated PIT membership CSV yet). The open web has NO clean 2021–2026 semi-annual
coverage — captures are sparse + arbitrary-dated; the factsheets + verified N500
xlsx remain the authoritative base.
