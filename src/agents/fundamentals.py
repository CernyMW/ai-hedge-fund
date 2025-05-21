from langchain_core.messages import HumanMessage
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
import json
import datetime # Added for date manipulation in dividend analysis

from src.tools.api import get_financial_metrics, get_dividend_history, get_prices, search_line_items # Updated imports
from src.data.models import Dividend # Added import


##### Fundamental Agent #####
def fundamentals_agent(state: AgentState):
    """Analyzes fundamental data and generates trading signals for multiple tickers."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Initialize fundamental analysis for each ticker
    fundamental_analysis = {}

    for ticker in tickers:
        progress.update_status("fundamentals_agent", ticker, "Fetching financial metrics")

        # Get the financial metrics
        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=10,
        )

        if not financial_metrics:
            progress.update_status("fundamentals_agent", ticker, "Failed: No financial metrics found")
            continue

        # Pull the most recent financial metrics
        metrics = financial_metrics[0]

        progress.update_status("fundamentals_agent", ticker, "Fetching dividend history")
        # Fetch recent dividend history (e.g., 5 years of quarterly data)
        dividend_history: list[Dividend] = get_dividend_history(ticker, limit=20)

        progress.update_status("fundamentals_agent", ticker, "Fetching current price for yield")
        current_price_data = get_prices(ticker, end_date, end_date) # Fetches for the 'end_date'
        current_price = current_price_data[0].close if current_price_data else None

        progress.update_status("fundamentals_agent", ticker, "Fetching annual EPS for payout")
        annual_eps_items = search_line_items(ticker, ["earnings_per_share"], end_date, period="annual", limit=1)
        latest_annual_eps = annual_eps_items[0].earnings_per_share if annual_eps_items and hasattr(annual_eps_items[0], 'earnings_per_share') else None

        # Initialize signals list for different fundamental aspects
        signals = []
        reasoning = {}

        progress.update_status("fundamentals_agent", ticker, "Analyzing profitability")
        # 1. Profitability Analysis
        return_on_equity = metrics.return_on_equity
        net_margin = metrics.net_margin
        operating_margin = metrics.operating_margin

        thresholds = [
            (return_on_equity, 0.15),  # Strong ROE above 15%
            (net_margin, 0.20),  # Healthy profit margins
            (operating_margin, 0.15),  # Strong operating efficiency
        ]
        profitability_score = sum(metric is not None and metric > threshold for metric, threshold in thresholds)

        signals.append("bullish" if profitability_score >= 2 else "bearish" if profitability_score == 0 else "neutral")
        reasoning["profitability_signal"] = {
            "signal": signals[0],
            "details": (f"ROE: {return_on_equity:.2%}" if return_on_equity else "ROE: N/A") + ", " + (f"Net Margin: {net_margin:.2%}" if net_margin else "Net Margin: N/A") + ", " + (f"Op Margin: {operating_margin:.2%}" if operating_margin else "Op Margin: N/A"),
        }

        progress.update_status("fundamentals_agent", ticker, "Analyzing growth")
        # 2. Growth Analysis
        revenue_growth = metrics.revenue_growth
        earnings_growth = metrics.earnings_growth
        book_value_growth = metrics.book_value_growth

        thresholds = [
            (revenue_growth, 0.10),  # 10% revenue growth
            (earnings_growth, 0.10),  # 10% earnings growth
            (book_value_growth, 0.10),  # 10% book value growth
        ]
        growth_score = sum(metric is not None and metric > threshold for metric, threshold in thresholds)

        signals.append("bullish" if growth_score >= 2 else "bearish" if growth_score == 0 else "neutral")
        reasoning["growth_signal"] = {
            "signal": signals[1],
            "details": (f"Revenue Growth: {revenue_growth:.2%}" if revenue_growth else "Revenue Growth: N/A") + ", " + (f"Earnings Growth: {earnings_growth:.2%}" if earnings_growth else "Earnings Growth: N/A"),
        }

        progress.update_status("fundamentals_agent", ticker, "Analyzing financial health")
        # 3. Financial Health
        current_ratio = metrics.current_ratio
        debt_to_equity = metrics.debt_to_equity
        free_cash_flow_per_share = metrics.free_cash_flow_per_share
        earnings_per_share = metrics.earnings_per_share

        health_score = 0
        if current_ratio and current_ratio > 1.5:  # Strong liquidity
            health_score += 1
        if debt_to_equity and debt_to_equity < 0.5:  # Conservative debt levels
            health_score += 1
        if free_cash_flow_per_share and earnings_per_share and free_cash_flow_per_share > earnings_per_share * 0.8:  # Strong FCF conversion
            health_score += 1

        signals.append("bullish" if health_score >= 2 else "bearish" if health_score == 0 else "neutral")
        reasoning["financial_health_signal"] = {
            "signal": signals[2],
            "details": (f"Current Ratio: {current_ratio:.2f}" if current_ratio else "Current Ratio: N/A") + ", " + (f"D/E: {debt_to_equity:.2f}" if debt_to_equity else "D/E: N/A"),
        }

        progress.update_status("fundamentals_agent", ticker, "Analyzing valuation ratios")
        # 4. Price to X ratios
        pe_ratio = metrics.price_to_earnings_ratio
        pb_ratio = metrics.price_to_book_ratio
        ps_ratio = metrics.price_to_sales_ratio

        thresholds = [
            (pe_ratio, 25),  # Reasonable P/E ratio
            (pb_ratio, 3),  # Reasonable P/B ratio
            (ps_ratio, 5),  # Reasonable P/S ratio
        ]
        price_ratio_score = sum(metric is not None and metric > threshold for metric, threshold in thresholds)

        signals.append("bearish" if price_ratio_score >= 2 else "bullish" if price_ratio_score == 0 else "neutral")
        reasoning["price_ratios_signal"] = {
            "signal": signals[3],
            "details": (f"P/E: {pe_ratio:.2f}" if pe_ratio else "P/E: N/A") + ", " + (f"P/B: {pb_ratio:.2f}" if pb_ratio else "P/B: N/A") + ", " + (f"P/S: {ps_ratio:.2f}" if ps_ratio else "P/S: N/A"),
        }

        progress.update_status("fundamentals_agent", ticker, "Analyzing dividends")
        # 5. Dividend Analysis
        dividend_score = 0
        latest_full_year_dividend_total = 0.0
        years_with_dividends = set()
        payout_ratio = None
        dividend_yield = None
        dividend_details = "Dividend Analysis: "

        if dividend_history:
            current_year_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            current_year_str = str(current_year_dt.year)
            
            # Group dividends by year
            dividends_by_year = {}
            for div in dividend_history:
                if div.dividend_type == 'CD' and div.cash_amount > 0.0 and div.ex_dividend_date:
                    try:
                        year = div.ex_dividend_date[:4]
                        years_with_dividends.add(year) # Track all years with any cash dividend
                        if year not in dividends_by_year:
                            dividends_by_year[year] = 0.0
                        dividends_by_year[year] += div.cash_amount
                    except Exception:
                        continue # Skip malformed dividend data
            
            if dividends_by_year:
                # Get the most recent year with dividends that is not the current partial year
                sorted_div_years = sorted(dividends_by_year.keys(), reverse=True)
                latest_full_div_year_str = None
                for year_s in sorted_div_years:
                    if year_s < current_year_str: # Ensure it's a past, full year
                        latest_full_div_year_str = year_s
                        break
                if latest_full_div_year_str:
                    latest_full_year_dividend_total = dividends_by_year[latest_full_div_year_str]
                elif sorted_div_years: # if only current year dividends exist, use them as best estimate
                    latest_full_year_dividend_total = dividends_by_year[sorted_div_years[0]]


        if current_price and latest_full_year_dividend_total > 0:
            dividend_yield = latest_full_year_dividend_total / current_price
            if dividend_yield > 0.03: # Yield > 3%
                dividend_score += 1
            dividend_details += f"Yield: {dividend_yield:.2%}"
        else:
            dividend_details += "Yield: N/A (or 0.0%)"

        if latest_annual_eps and latest_annual_eps > 0 and latest_full_year_dividend_total > 0:
            payout_ratio = latest_full_year_dividend_total / latest_annual_eps
            if 0.25 <= payout_ratio <= 0.75: # Payout between 25% and 75%
                dividend_score += 1
            dividend_details += f", Payout: {payout_ratio:.1%}"
        else:
            dividend_details += ", Payout: N/A"
            
        # Consistency (e.g., paid in at least 3 of the last 5 years covered by dividend_history limit)
        # years_with_dividends is already populated with years from up to limit=20 (5 years of quarterly)
        # The number of distinct years is a good proxy for consistency within that period.
        if len(years_with_dividends) >= 3:
             dividend_score +=1
        dividend_details += f", Paid in {len(years_with_dividends)} distinct years (recent history)."

        signals.append("bullish" if dividend_score >= 2 else "bearish" if dividend_score == 0 and latest_full_year_dividend_total == 0 else "neutral")
        reasoning["dividend_signal"] = {
            "signal": signals[-1], # Use the last added signal
            "details": dividend_details
        }
        
        progress.update_status("fundamentals_agent", ticker, "Calculating final signal")
        # Determine overall signal
        bullish_signals = signals.count("bullish")
        bearish_signals = signals.count("bearish")

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level
        total_signals = len(signals)
        confidence = round(max(bullish_signals, bearish_signals) / total_signals, 2) * 100

        fundamental_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status("fundamentals_agent", ticker, "Done")

    # Create the fundamental analysis message
    message = HumanMessage(
        content=json.dumps(fundamental_analysis),
        name="fundamentals_agent",
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(fundamental_analysis, "Fundamental Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["fundamentals_agent"] = fundamental_analysis

    progress.update_status("fundamentals_agent", None, "Done")
    
    return {
        "messages": [message],
        "data": data,
    }
