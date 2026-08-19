import json, requests

payload = {
  "sector": "Semiconductors",
  "company_name": "Broadcom Inc.",
  "current_price": 427.76,
  "fair_value_low": 350.0,
  "fair_value_mid": 495.0,
  "fair_value_high": 570.0,
  "price_target": 495.0,
  "rating": "Buy",
  "methodology": {
    "multiples": [
      {"label": "Fwd P/E", "company": 21.95, "peer_avg": 23.85},
      {"label": "EV/EBITDA", "company": 49.43, "peer_avg": 34.07}
    ],
    "dcf": {
      "revenue_growth_5y": "24% declining to 12% (avg ~17%)",
      "terminal_growth": "3.5%",
      "discount_rate": "9.5%",
      "implied_value": 218.5
    }
  },
  "thesis": "Broadcom's custom AI ASIC franchise (est. 60-70% co-design share vs Marvell) now has multi-year contracted visibility via the Google TPU deal through 2031 and the Anthropic 3.5GW compute agreement starting 2027, with management guiding FY26 AI semiconductor revenue to $56B and FY27 to over $100B - a growth durability few semis peers can match. VMware is a second engine, with infrastructure software margins expanding to 77% (from 67% a year ago) as subscription conversion completes. Forward P/E (21.95x) sits modestly below the peer average (23.85x) despite superior growth visibility, while the premium EV/EBITDA multiple mostly reflects VMware-acquisition debt rather than operating weakness; a standard DCF undervalues the name because it cannot fully capture the contracted multi-year AI revenue ramp. Net: risk/reward favors accumulation with a price target near $495, below the ~$528 sell-side consensus average.",
  "catalysts": [
    "FY27 AI semiconductor revenue guided >$100B (vs $56B FY26) as Google TPU v7/v8 and Anthropic 3.5GW deployments ramp starting 2027",
    "Q3 FY26 guide of $29.4B revenue (+84% YoY) with AI semis of $16B (+200%+ YoY) due to report; continued beat-and-raise pattern",
    "VMware infrastructure software margin expansion (77% op margin, up from 67% YoY) and >90% of top 10,000 customers now on VMware Cloud Foundation",
    "$6B in new AI orders booked from two additional (undisclosed) hyperscaler customers, expanding the custom silicon customer base beyond Google/Meta/OpenAI/Anthropic"
  ],
  "risks": [
    "Marvell is gaining design-win share in select hyperscaler custom silicon work (e.g., reported Google diversification in April 2026), pressuring AVGO's long-run ASIC share (Counterpoint sees AVGO share easing to ~60% by 2027 from higher levels)",
    "Elevated net debt (~$45B) from the VMware acquisition keeps EV/EBITDA well above peer average (49.4x vs ~34x), leaving less valuation cushion if AI capex growth decelerates",
    "Customer concentration: a small number of hyperscaler/AI-lab customers (Google, Meta, OpenAI, Anthropic) drive the bulk of incremental AI revenue - any pause in hyperscaler capex plans is a direct hit",
    "Valuation already prices in a large multi-year growth story; standard DCF (~$218/share) implies the market is paying for duration/optionality well beyond a conservative 5-year forecast, so any guidance miss could trigger a sharp multiple reset"
  ],
  "sources": [
    {"title": "Broadcom Inc. Announces Second Quarter Fiscal Year 2026 Financial Results", "url": "https://www.prnewswire.com/news-releases/broadcom-inc-announces-second-quarter-fiscal-year-2026-financial-results-and-quarterly-dividend-302790698.html"},
    {"title": "Broadcom Q2 2026 revenue up 48%, guides to $29.4B - AVGO Stock News", "url": "https://www.stocktitan.net/news/AVGO/broadcom-inc-announces-second-quarter-fiscal-year-2026-financial-if4yrbje8hq6.html"},
    {"title": "Broadcom's Custom AI Chips Deal With Google and Anthropic Explained", "url": "https://techwireasia.com/2026/04/broadcom-custom-ai-chips-google-anthropic-deal/"},
    {"title": "Broadcom Google TPU Deal 2026: The $46B AI Contract", "url": "https://oplexa.com/broadcom-google-tpu-deal-2026/"},
    {"title": "Broadcom expects AI to power VMware revenue growth - CIO Dive", "url": "https://www.ciodive.com/news/broadcom-ceo-expects-vmware-revenue-boost-ai/813972/"},
    {"title": "Will Broadcom's VMware strategy keep paying big dividends? - Network World", "url": "https://www.networkworld.com/article/4180754/will-broadcoms-vmware-strategy-keep-paying-big-dividends.html"},
    {"title": "Broadcom Set To Dominate Custom AI Chip Market With 60% Share By 2027, Counterpoint Says", "url": "https://finviz.com/news/288744/broadcom-set-to-dominate-custom-ai-chip-market-with-60-share-by-2027-counterpoint-says"},
    {"title": "Broadcom vs. Marvell: A Valuation Showdown for the Custom AI Chip Trade - The Motley Fool", "url": "https://www.fool.com/investing/2026/07/12/broadcom-vs-marvell-a-valuation-showdown-for-the-c/"},
    {"title": "Broadcom (AVGO) Stock Forecast & Analyst Price Targets - stockanalysis.com", "url": "https://stockanalysis.com/stocks/avgo/forecast/"},
    {"title": "Broader Analyst Sentiment on Broadcom Inc. (AVGO) Remains Bullish Amid Growing Demand for Google's TPUs", "url": "https://finviz.com/news/303806/broader-analyst-sentiment-on-broadcom-inc-avgo-remains-bullish-amid-growing-demand-for-googles-tpus"}
  ],
  "blockers": [
    "TSM's yfinance enterpriseToEbitda field (4.70x) is inconsistent with its market cap/EV figures (likely a currency/unit mismatch, TWD vs USD) - excluded from the EV/EBITDA peer average, which uses NVDA/AMD/QCOM/TXN/INTC/MU only (avg 34.07x)",
    "MU's yfinance revenueGrowth field (345.7% YoY) and null marketCap look like data artifacts (likely trough-base-year comparison effect from the prior memory downcycle) - MU excluded from market-cap-based comparisons, forward P/E and EV/EBITDA figures used as-is with caution"
  ]
}

with open("avgo_report_body.json", "w") as f:
    json.dump(payload, f)

resp = requests.post("http://localhost:8000/research/reports/AVGO", json=payload, timeout=30)
print("STATUS", resp.status_code)
print(resp.text[:2000])
