from datetime import date, datetime
from importlib.metadata import PackageNotFoundError, version
import os
from typing import Any, Literal

from dotenv import load_dotenv
from massive import RESTClient
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from .formatters import json_to_csv

# Load environment variables from .env file if it exists
load_dotenv()

MASSIVE_API_KEY = os.environ.get("MASSIVE_API_KEY", "")
if not MASSIVE_API_KEY:
    print("Warning: MASSIVE_API_KEY environment variable not set.")
    print("Please set it in your environment or create a .env file with MASSIVE_API_KEY=your_key")

version_number = "MCP-Massive/unknown"
try:
    version_number = f"MCP-Massive/{version('mcp_massive')}"
except PackageNotFoundError:
    pass

massive_client = RESTClient(MASSIVE_API_KEY)
massive_client.headers["User-Agent"] += f" {version_number}"

poly_mcp = FastMCP("Massive")


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_aggs(
    ticker: str,
    multiplier: int,
    timespan: str,
    from_: str | int | datetime | date,
    to: str | int | datetime | date,
    adjusted: bool | None = None,
    sort: str | None = None,
    limit: int | None = 10,
    params: dict[str, Any] | None = None,
) -> str:
    """
    List aggregate bars for a ticker over a given date range in custom time window sizes.
    """
    try:
        results = massive_client.get_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=from_,
            to=to,
            adjusted=adjusted,
            sort=sort,
            limit=limit,
            params=params,
            raw=True,
        )

        # Parse the binary data to string and then to JSON
        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_aggs(
    ticker: str,
    multiplier: int,
    timespan: str,
    from_: str | int | datetime | date,
    to: str | int | datetime | date,
    adjusted: bool | None = None,
    sort: str | None = None,
    limit: int | None = 10,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Iterate through aggregate bars for a ticker over a given date range.
    """
    try:
        results = massive_client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=from_,
            to=to,
            adjusted=adjusted,
            sort=sort,
            limit=limit,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_grouped_daily_aggs(
    date: str,
    adjusted: bool | None = None,
    include_otc: bool | None = None,
    locale: str | None = None,
    market_type: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get grouped daily bars for entire market for a specific date.
    """
    try:
        results = massive_client.get_grouped_daily_aggs(
            date=date,
            adjusted=adjusted,
            include_otc=include_otc,
            locale=locale,
            market_type=market_type,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_daily_open_close_agg(
    ticker: str,
    date: str,
    adjusted: bool | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get daily open, close, high, and low for a specific ticker and date.
    """
    try:
        results = massive_client.get_daily_open_close_agg(
            ticker=ticker, date=date, adjusted=adjusted, params=params, raw=True
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_previous_close_agg(
    ticker: str,
    adjusted: bool | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get previous day's open, close, high, and low for a specific ticker.
    """
    try:
        results = massive_client.get_previous_close_agg(
            ticker=ticker, adjusted=adjusted, params=params, raw=True
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_trades(
    ticker: str,
    timestamp: str | int | datetime | date | None = None,
    timestamp_lt: str | int | datetime | date | None = None,
    timestamp_lte: str | int | datetime | date | None = None,
    timestamp_gt: str | int | datetime | date | None = None,
    timestamp_gte: str | int | datetime | date | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    order: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get trades for a ticker symbol.
    """
    try:
        results = massive_client.list_trades(
            ticker=ticker,
            timestamp=timestamp,
            timestamp_lt=timestamp_lt,
            timestamp_lte=timestamp_lte,
            timestamp_gt=timestamp_gt,
            timestamp_gte=timestamp_gte,
            limit=limit,
            sort=sort,
            order=order,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_last_trade(
    ticker: str,
) -> str:
    """
    Get the most recent trade for a ticker symbol.
    """
    try:
        results = massive_client.get_last_trade(ticker=ticker, raw=True)
        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_last_crypto_trade(
    from_: str,
    to: str,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get the most recent trade for a crypto pair.
    """
    try:
        results = massive_client.get_last_crypto_trade(from_=from_, to=to, params=params, raw=True)

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_quotes(
    ticker: str,
    timestamp: str | int | datetime | date | None = None,
    timestamp_lt: str | int | datetime | date | None = None,
    timestamp_lte: str | int | datetime | date | None = None,
    timestamp_gt: str | int | datetime | date | None = None,
    timestamp_gte: str | int | datetime | date | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    order: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get quotes for a ticker symbol.
    """
    try:
        results = massive_client.list_quotes(
            ticker=ticker,
            timestamp=timestamp,
            timestamp_lt=timestamp_lt,
            timestamp_lte=timestamp_lte,
            timestamp_gt=timestamp_gt,
            timestamp_gte=timestamp_gte,
            limit=limit,
            sort=sort,
            order=order,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_last_quote(
    ticker: str,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get the most recent quote for a ticker symbol.
    """
    try:
        results = massive_client.get_last_quote(ticker=ticker, params=params, raw=True)

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_last_forex_quote(
    from_: str,
    to: str,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get the most recent forex quote.
    """
    try:
        results = massive_client.get_last_forex_quote(from_=from_, to=to, params=params, raw=True)

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_real_time_currency_conversion(
    from_: str,
    to: str,
    amount: float | None = None,
    precision: int | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get real-time currency conversion.
    """
    try:
        results = massive_client.get_real_time_currency_conversion(
            from_=from_,
            to=to,
            amount=amount,
            precision=precision,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_universal_snapshots(
    type: str,
    ticker_any_of: list[str] | None = None,
    order: str | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get universal snapshots for multiple assets of a specific type.
    """
    try:
        results = massive_client.list_universal_snapshots(
            type=type,
            ticker_any_of=ticker_any_of,
            order=order,
            limit=limit,
            sort=sort,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_snapshot_all(
    market_type: str,
    tickers: list[str] | None = None,
    include_otc: bool | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get a snapshot of all tickers in a market.
    """
    try:
        results = massive_client.get_snapshot_all(
            market_type=market_type,
            tickers=tickers,
            include_otc=include_otc,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_snapshot_direction(
    market_type: str,
    direction: str,
    include_otc: bool | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get gainers or losers for a market.
    """
    try:
        results = massive_client.get_snapshot_direction(
            market_type=market_type,
            direction=direction,
            include_otc=include_otc,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_snapshot_ticker(
    market_type: str,
    ticker: str,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get snapshot for a specific ticker.
    """
    try:
        results = massive_client.get_snapshot_ticker(
            market_type=market_type, ticker=ticker, params=params, raw=True
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_snapshot_option(
    underlying_asset: str,
    option_contract: str,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get snapshot for a specific option contract.
    """
    try:
        results = massive_client.get_snapshot_option(
            underlying_asset=underlying_asset,
            option_contract=option_contract,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_snapshot_crypto_book(
    ticker: str,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get snapshot for a crypto ticker's order book.
    """
    try:
        results = massive_client.get_snapshot_crypto_book(ticker=ticker, params=params, raw=True)

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_market_holidays(
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get upcoming market holidays and their open/close times.
    """
    try:
        results = massive_client.get_market_holidays(params=params, raw=True)

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_market_status(
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get current trading status of exchanges and financial markets.
    """
    try:
        results = massive_client.get_market_status(params=params, raw=True)

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_tickers(
    ticker: str | None = None,
    type: str | None = None,
    market: str | None = None,
    exchange: str | None = None,
    cusip: str | None = None,
    cik: str | None = None,
    date: str | datetime | date | None = None,
    search: str | None = None,
    active: bool | None = None,
    sort: str | None = None,
    order: str | None = None,
    limit: int | None = 10,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Query supported ticker symbols across stocks, indices, forex, and crypto.
    """
    try:
        results = massive_client.list_tickers(
            ticker=ticker,
            type=type,
            market=market,
            exchange=exchange,
            cusip=cusip,
            cik=cik,
            date=date,
            search=search,
            active=active,
            sort=sort,
            order=order,
            limit=limit,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_ticker_details(
    ticker: str,
    date: str | datetime | date | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get detailed information about a specific ticker.
    """
    try:
        results = massive_client.get_ticker_details(
            ticker=ticker, date=date, params=params, raw=True
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_ticker_news(
    ticker: str | None = None,
    published_utc: str | datetime | date | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    order: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get recent news articles for a stock ticker.
    """
    try:
        results = massive_client.list_ticker_news(
            ticker=ticker,
            published_utc=published_utc,
            limit=limit,
            sort=sort,
            order=order,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_ticker_types(
    asset_class: str | None = None,
    locale: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    List all ticker types supported by Massive.com.
    """
    try:
        results = massive_client.get_ticker_types(
            asset_class=asset_class, locale=locale, params=params, raw=True
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_splits(
    ticker: str | None = None,
    execution_date: str | datetime | date | None = None,
    reverse_split: bool | None = None,
    limit: int | None = 10,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get historical stock splits.
    """
    try:
        results = massive_client.list_splits(
            ticker=ticker,
            execution_date=execution_date,
            reverse_split=reverse_split,
            limit=limit,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_dividends(
    ticker: str | None = None,
    ex_dividend_date: str | datetime | date | None = None,
    frequency: int | None = None,
    dividend_type: str | None = None,
    limit: int | None = 10,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get historical cash dividends.
    """
    try:
        results = massive_client.list_dividends(
            ticker=ticker,
            ex_dividend_date=ex_dividend_date,
            frequency=frequency,
            dividend_type=dividend_type,
            limit=limit,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_conditions(
    asset_class: str | None = None,
    data_type: str | None = None,
    id: int | None = None,
    sip: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    List conditions used by Massive.com.
    """
    try:
        results = massive_client.list_conditions(
            asset_class=asset_class,
            data_type=data_type,
            id=id,
            sip=sip,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_exchanges(
    asset_class: str | None = None,
    locale: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    List exchanges known by Massive.com.
    """
    try:
        results = massive_client.get_exchanges(
            asset_class=asset_class, locale=locale, params=params, raw=True
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_stock_financials(
    ticker: str | None = None,
    cik: str | None = None,
    company_name: str | None = None,
    company_name_search: str | None = None,
    sic: str | None = None,
    filing_date: str | datetime | date | None = None,
    filing_date_lt: str | datetime | date | None = None,
    filing_date_lte: str | datetime | date | None = None,
    filing_date_gt: str | datetime | date | None = None,
    filing_date_gte: str | datetime | date | None = None,
    period_of_report_date: str | datetime | date | None = None,
    period_of_report_date_lt: str | datetime | date | None = None,
    period_of_report_date_lte: str | datetime | date | None = None,
    period_of_report_date_gt: str | datetime | date | None = None,
    period_of_report_date_gte: str | datetime | date | None = None,
    timeframe: str | None = None,
    include_sources: bool | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    order: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get fundamental financial data for companies.
    """
    try:
        results = massive_client.vx.list_stock_financials(
            ticker=ticker,
            cik=cik,
            company_name=company_name,
            company_name_search=company_name_search,
            sic=sic,
            filing_date=filing_date,
            filing_date_lt=filing_date_lt,
            filing_date_lte=filing_date_lte,
            filing_date_gt=filing_date_gt,
            filing_date_gte=filing_date_gte,
            period_of_report_date=period_of_report_date,
            period_of_report_date_lt=period_of_report_date_lt,
            period_of_report_date_lte=period_of_report_date_lte,
            period_of_report_date_gt=period_of_report_date_gt,
            period_of_report_date_gte=period_of_report_date_gte,
            timeframe=timeframe,
            include_sources=include_sources,
            limit=limit,
            sort=sort,
            order=order,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_ipos(
    ticker: str | None = None,
    listing_date: str | datetime | date | None = None,
    listing_date_lt: str | datetime | date | None = None,
    listing_date_lte: str | datetime | date | None = None,
    listing_date_gt: str | datetime | date | None = None,
    listing_date_gte: str | datetime | date | None = None,
    ipo_status: str | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    order: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Retrieve upcoming or historical IPOs.
    """
    try:
        results = massive_client.vx.list_ipos(
            ticker=ticker,
            listing_date=listing_date,
            listing_date_lt=listing_date_lt,
            listing_date_lte=listing_date_lte,
            listing_date_gt=listing_date_gt,
            listing_date_gte=listing_date_gte,
            ipo_status=ipo_status,
            limit=limit,
            sort=sort,
            order=order,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_short_interest(
    ticker: str | None = None,
    settlement_date: str | datetime | date | None = None,
    settlement_date_lt: str | datetime | date | None = None,
    settlement_date_lte: str | datetime | date | None = None,
    settlement_date_gt: str | datetime | date | None = None,
    settlement_date_gte: str | datetime | date | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    order: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Retrieve short interest data for stocks.
    """
    try:
        results = massive_client.list_short_interest(
            ticker=ticker,
            settlement_date=settlement_date,
            settlement_date_lt=settlement_date_lt,
            settlement_date_lte=settlement_date_lte,
            settlement_date_gt=settlement_date_gt,
            settlement_date_gte=settlement_date_gte,
            limit=limit,
            sort=sort,
            order=order,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_short_volume(
    ticker: str | None = None,
    date: str | datetime | date | None = None,
    date_lt: str | datetime | date | None = None,
    date_lte: str | datetime | date | None = None,
    date_gt: str | datetime | date | None = None,
    date_gte: str | datetime | date | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    order: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Retrieve short volume data for stocks.
    """
    try:
        results = massive_client.list_short_volume(
            ticker=ticker,
            date=date,
            date_lt=date_lt,
            date_lte=date_lte,
            date_gt=date_gt,
            date_gte=date_gte,
            limit=limit,
            sort=sort,
            order=order,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_treasury_yields(
    date: str | datetime | date | None = None,
    date_any_of: str | None = None,
    date_lt: str | datetime | date | None = None,
    date_lte: str | datetime | date | None = None,
    date_gt: str | datetime | date | None = None,
    date_gte: str | datetime | date | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    order: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Retrieve treasury yield data.
    """
    try:
        results = massive_client.list_treasury_yields(
            date=date,
            date_lt=date_lt,
            date_lte=date_lte,
            date_gt=date_gt,
            date_gte=date_gte,
            limit=limit,
            sort=sort,
            order=order,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_inflation(
    date: str | datetime | date | None = None,
    date_any_of: str | None = None,
    date_gt: str | datetime | date | None = None,
    date_gte: str | datetime | date | None = None,
    date_lt: str | datetime | date | None = None,
    date_lte: str | datetime | date | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get inflation data from the Federal Reserve.
    """
    try:
        results = massive_client.list_inflation(
            date=date,
            date_any_of=date_any_of,
            date_gt=date_gt,
            date_gte=date_gte,
            date_lt=date_lt,
            date_lte=date_lte,
            limit=limit,
            sort=sort,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_benzinga_analyst_insights(
    date: str | date | None = None,
    date_any_of: str | None = None,
    date_gt: str | date | None = None,
    date_gte: str | date | None = None,
    date_lt: str | date | None = None,
    date_lte: str | date | None = None,
    ticker: str | None = None,
    ticker_any_of: str | None = None,
    ticker_gt: str | None = None,
    ticker_gte: str | None = None,
    ticker_lt: str | None = None,
    ticker_lte: str | None = None,
    last_updated: str | None = None,
    last_updated_any_of: str | None = None,
    last_updated_gt: str | None = None,
    last_updated_gte: str | None = None,
    last_updated_lt: str | None = None,
    last_updated_lte: str | None = None,
    firm: str | None = None,
    firm_any_of: str | None = None,
    firm_gt: str | None = None,
    firm_gte: str | None = None,
    firm_lt: str | None = None,
    firm_lte: str | None = None,
    rating_action: str | None = None,
    rating_action_any_of: str | None = None,
    rating_action_gt: str | None = None,
    rating_action_gte: str | None = None,
    rating_action_lt: str | None = None,
    rating_action_lte: str | None = None,
    benzinga_firm_id: str | None = None,
    benzinga_firm_id_any_of: str | None = None,
    benzinga_firm_id_gt: str | None = None,
    benzinga_firm_id_gte: str | None = None,
    benzinga_firm_id_lt: str | None = None,
    benzinga_firm_id_lte: str | None = None,
    benzinga_rating_id: str | None = None,
    benzinga_rating_id_any_of: str | None = None,
    benzinga_rating_id_gt: str | None = None,
    benzinga_rating_id_gte: str | None = None,
    benzinga_rating_id_lt: str | None = None,
    benzinga_rating_id_lte: str | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    List Benzinga analyst insights.
    """
    try:
        results = massive_client.list_benzinga_analyst_insights(
            date=date,
            date_any_of=date_any_of,
            date_gt=date_gt,
            date_gte=date_gte,
            date_lt=date_lt,
            date_lte=date_lte,
            ticker=ticker,
            ticker_any_of=ticker_any_of,
            ticker_gt=ticker_gt,
            ticker_gte=ticker_gte,
            ticker_lt=ticker_lt,
            ticker_lte=ticker_lte,
            last_updated=last_updated,
            last_updated_any_of=last_updated_any_of,
            last_updated_gt=last_updated_gt,
            last_updated_gte=last_updated_gte,
            last_updated_lt=last_updated_lt,
            last_updated_lte=last_updated_lte,
            firm=firm,
            firm_any_of=firm_any_of,
            firm_gt=firm_gt,
            firm_gte=firm_gte,
            firm_lt=firm_lt,
            firm_lte=firm_lte,
            rating_action=rating_action,
            rating_action_any_of=rating_action_any_of,
            rating_action_gt=rating_action_gt,
            rating_action_gte=rating_action_gte,
            rating_action_lt=rating_action_lt,
            rating_action_lte=rating_action_lte,
            benzinga_firm_id=benzinga_firm_id,
            benzinga_firm_id_any_of=benzinga_firm_id_any_of,
            benzinga_firm_id_gt=benzinga_firm_id_gt,
            benzinga_firm_id_gte=benzinga_firm_id_gte,
            benzinga_firm_id_lt=benzinga_firm_id_lt,
            benzinga_firm_id_lte=benzinga_firm_id_lte,
            benzinga_rating_id=benzinga_rating_id,
            benzinga_rating_id_any_of=benzinga_rating_id_any_of,
            benzinga_rating_id_gt=benzinga_rating_id_gt,
            benzinga_rating_id_gte=benzinga_rating_id_gte,
            benzinga_rating_id_lt=benzinga_rating_id_lt,
            benzinga_rating_id_lte=benzinga_rating_id_lte,
            limit=limit,
            sort=sort,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_benzinga_analysts(
    benzinga_id: str | None = None,
    benzinga_id_any_of: str | None = None,
    benzinga_id_gt: str | None = None,
    benzinga_id_gte: str | None = None,
    benzinga_id_lt: str | None = None,
    benzinga_id_lte: str | None = None,
    benzinga_firm_id: str | None = None,
    benzinga_firm_id_any_of: str | None = None,
    benzinga_firm_id_gt: str | None = None,
    benzinga_firm_id_gte: str | None = None,
    benzinga_firm_id_lt: str | None = None,
    benzinga_firm_id_lte: str | None = None,
    firm_name: str | None = None,
    firm_name_any_of: str | None = None,
    firm_name_gt: str | None = None,
    firm_name_gte: str | None = None,
    firm_name_lt: str | None = None,
    firm_name_lte: str | None = None,
    full_name: str | None = None,
    full_name_any_of: str | None = None,
    full_name_gt: str | None = None,
    full_name_gte: str | None = None,
    full_name_lt: str | None = None,
    full_name_lte: str | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    List Benzinga analysts.
    """
    try:
        results = massive_client.list_benzinga_analysts(
            benzinga_id=benzinga_id,
            benzinga_id_any_of=benzinga_id_any_of,
            benzinga_id_gt=benzinga_id_gt,
            benzinga_id_gte=benzinga_id_gte,
            benzinga_id_lt=benzinga_id_lt,
            benzinga_id_lte=benzinga_id_lte,
            benzinga_firm_id=benzinga_firm_id,
            benzinga_firm_id_any_of=benzinga_firm_id_any_of,
            benzinga_firm_id_gt=benzinga_firm_id_gt,
            benzinga_firm_id_gte=benzinga_firm_id_gte,
            benzinga_firm_id_lt=benzinga_firm_id_lt,
            benzinga_firm_id_lte=benzinga_firm_id_lte,
            firm_name=firm_name,
            firm_name_any_of=firm_name_any_of,
            firm_name_gt=firm_name_gt,
            firm_name_gte=firm_name_gte,
            firm_name_lt=firm_name_lt,
            firm_name_lte=firm_name_lte,
            full_name=full_name,
            full_name_any_of=full_name_any_of,
            full_name_gt=full_name_gt,
            full_name_gte=full_name_gte,
            full_name_lt=full_name_lt,
            full_name_lte=full_name_lte,
            limit=limit,
            sort=sort,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_benzinga_consensus_ratings(
    ticker: str,
    date: str | date | None = None,
    date_gt: str | date | None = None,
    date_gte: str | date | None = None,
    date_lt: str | date | None = None,
    date_lte: str | date | None = None,
    limit: int | None = 10,
    params: dict[str, Any] | None = None,
) -> str:
    """
    List Benzinga consensus ratings for a ticker.
    """
    try:
        results = massive_client.list_benzinga_consensus_ratings(
            ticker=ticker,
            date=date,
            date_gt=date_gt,
            date_gte=date_gte,
            date_lt=date_lt,
            date_lte=date_lte,
            limit=limit,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_benzinga_earnings(
    date: str | date | None = None,
    date_any_of: str | None = None,
    date_gt: str | date | None = None,
    date_gte: str | date | None = None,
    date_lt: str | date | None = None,
    date_lte: str | date | None = None,
    ticker: str | None = None,
    ticker_any_of: str | None = None,
    ticker_gt: str | None = None,
    ticker_gte: str | None = None,
    ticker_lt: str | None = None,
    ticker_lte: str | None = None,
    importance: int | None = None,
    importance_any_of: str | None = None,
    importance_gt: int | None = None,
    importance_gte: int | None = None,
    importance_lt: int | None = None,
    importance_lte: int | None = None,
    last_updated: str | None = None,
    last_updated_any_of: str | None = None,
    last_updated_gt: str | None = None,
    last_updated_gte: str | None = None,
    last_updated_lt: str | None = None,
    last_updated_lte: str | None = None,
    date_status: str | None = None,
    date_status_any_of: str | None = None,
    date_status_gt: str | None = None,
    date_status_gte: str | None = None,
    date_status_lt: str | None = None,
    date_status_lte: str | None = None,
    eps_surprise_percent: float | None = None,
    eps_surprise_percent_any_of: str | None = None,
    eps_surprise_percent_gt: float | None = None,
    eps_surprise_percent_gte: float | None = None,
    eps_surprise_percent_lt: float | None = None,
    eps_surprise_percent_lte: float | None = None,
    revenue_surprise_percent: float | None = None,
    revenue_surprise_percent_any_of: str | None = None,
    revenue_surprise_percent_gt: float | None = None,
    revenue_surprise_percent_gte: float | None = None,
    revenue_surprise_percent_lt: float | None = None,
    revenue_surprise_percent_lte: float | None = None,
    fiscal_year: int | None = None,
    fiscal_year_any_of: str | None = None,
    fiscal_year_gt: int | None = None,
    fiscal_year_gte: int | None = None,
    fiscal_year_lt: int | None = None,
    fiscal_year_lte: int | None = None,
    fiscal_period: str | None = None,
    fiscal_period_any_of: str | None = None,
    fiscal_period_gt: str | None = None,
    fiscal_period_gte: str | None = None,
    fiscal_period_lt: str | None = None,
    fiscal_period_lte: str | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    List Benzinga earnings.
    """
    try:
        results = massive_client.list_benzinga_earnings(
            date=date,
            date_any_of=date_any_of,
            date_gt=date_gt,
            date_gte=date_gte,
            date_lt=date_lt,
            date_lte=date_lte,
            ticker=ticker,
            ticker_any_of=ticker_any_of,
            ticker_gt=ticker_gt,
            ticker_gte=ticker_gte,
            ticker_lt=ticker_lt,
            ticker_lte=ticker_lte,
            importance=importance,
            importance_any_of=importance_any_of,
            importance_gt=importance_gt,
            importance_gte=importance_gte,
            importance_lt=importance_lt,
            importance_lte=importance_lte,
            last_updated=last_updated,
            last_updated_any_of=last_updated_any_of,
            last_updated_gt=last_updated_gt,
            last_updated_gte=last_updated_gte,
            last_updated_lt=last_updated_lt,
            last_updated_lte=last_updated_lte,
            date_status=date_status,
            date_status_any_of=date_status_any_of,
            date_status_gt=date_status_gt,
            date_status_gte=date_status_gte,
            date_status_lt=date_status_lt,
            date_status_lte=date_status_lte,
            eps_surprise_percent=eps_surprise_percent,
            eps_surprise_percent_any_of=eps_surprise_percent_any_of,
            eps_surprise_percent_gt=eps_surprise_percent_gt,
            eps_surprise_percent_gte=eps_surprise_percent_gte,
            eps_surprise_percent_lt=eps_surprise_percent_lt,
            eps_surprise_percent_lte=eps_surprise_percent_lte,
            revenue_surprise_percent=revenue_surprise_percent,
            revenue_surprise_percent_any_of=revenue_surprise_percent_any_of,
            revenue_surprise_percent_gt=revenue_surprise_percent_gt,
            revenue_surprise_percent_gte=revenue_surprise_percent_gte,
            revenue_surprise_percent_lt=revenue_surprise_percent_lt,
            revenue_surprise_percent_lte=revenue_surprise_percent_lte,
            fiscal_year=fiscal_year,
            fiscal_year_any_of=fiscal_year_any_of,
            fiscal_year_gt=fiscal_year_gt,
            fiscal_year_gte=fiscal_year_gte,
            fiscal_year_lt=fiscal_year_lt,
            fiscal_year_lte=fiscal_year_lte,
            fiscal_period=fiscal_period,
            fiscal_period_any_of=fiscal_period_any_of,
            fiscal_period_gt=fiscal_period_gt,
            fiscal_period_gte=fiscal_period_gte,
            fiscal_period_lt=fiscal_period_lt,
            fiscal_period_lte=fiscal_period_lte,
            limit=limit,
            sort=sort,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_benzinga_firms(
    benzinga_id: str | None = None,
    benzinga_id_any_of: str | None = None,
    benzinga_id_gt: str | None = None,
    benzinga_id_gte: str | None = None,
    benzinga_id_lt: str | None = None,
    benzinga_id_lte: str | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    List Benzinga firms.
    """
    try:
        results = massive_client.list_benzinga_firms(
            benzinga_id=benzinga_id,
            benzinga_id_any_of=benzinga_id_any_of,
            benzinga_id_gt=benzinga_id_gt,
            benzinga_id_gte=benzinga_id_gte,
            benzinga_id_lt=benzinga_id_lt,
            benzinga_id_lte=benzinga_id_lte,
            limit=limit,
            sort=sort,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_benzinga_guidance(
    date: str | date | None = None,
    date_any_of: str | None = None,
    date_gt: str | date | None = None,
    date_gte: str | date | None = None,
    date_lt: str | date | None = None,
    date_lte: str | date | None = None,
    ticker: str | None = None,
    ticker_any_of: str | None = None,
    ticker_gt: str | None = None,
    ticker_gte: str | None = None,
    ticker_lt: str | None = None,
    ticker_lte: str | None = None,
    positioning: str | None = None,
    positioning_any_of: str | None = None,
    positioning_gt: str | None = None,
    positioning_gte: str | None = None,
    positioning_lt: str | None = None,
    positioning_lte: str | None = None,
    importance: int | None = None,
    importance_any_of: str | None = None,
    importance_gt: int | None = None,
    importance_gte: int | None = None,
    importance_lt: int | None = None,
    importance_lte: int | None = None,
    last_updated: str | None = None,
    last_updated_any_of: str | None = None,
    last_updated_gt: str | None = None,
    last_updated_gte: str | None = None,
    last_updated_lt: str | None = None,
    last_updated_lte: str | None = None,
    fiscal_year: int | None = None,
    fiscal_year_any_of: str | None = None,
    fiscal_year_gt: int | None = None,
    fiscal_year_gte: int | None = None,
    fiscal_year_lt: int | None = None,
    fiscal_year_lte: int | None = None,
    fiscal_period: str | None = None,
    fiscal_period_any_of: str | None = None,
    fiscal_period_gt: str | None = None,
    fiscal_period_gte: str | None = None,
    fiscal_period_lt: str | None = None,
    fiscal_period_lte: str | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    List Benzinga guidance.
    """
    try:
        results = massive_client.list_benzinga_guidance(
            date=date,
            date_any_of=date_any_of,
            date_gt=date_gt,
            date_gte=date_gte,
            date_lt=date_lt,
            date_lte=date_lte,
            ticker=ticker,
            ticker_any_of=ticker_any_of,
            ticker_gt=ticker_gt,
            ticker_gte=ticker_gte,
            ticker_lt=ticker_lt,
            ticker_lte=ticker_lte,
            positioning=positioning,
            positioning_any_of=positioning_any_of,
            positioning_gt=positioning_gt,
            positioning_gte=positioning_gte,
            positioning_lt=positioning_lt,
            positioning_lte=positioning_lte,
            importance=importance,
            importance_any_of=importance_any_of,
            importance_gt=importance_gt,
            importance_gte=importance_gte,
            importance_lt=importance_lt,
            importance_lte=importance_lte,
            last_updated=last_updated,
            last_updated_any_of=last_updated_any_of,
            last_updated_gt=last_updated_gt,
            last_updated_gte=last_updated_gte,
            last_updated_lt=last_updated_lt,
            last_updated_lte=last_updated_lte,
            fiscal_year=fiscal_year,
            fiscal_year_any_of=fiscal_year_any_of,
            fiscal_year_gt=fiscal_year_gt,
            fiscal_year_gte=fiscal_year_gte,
            fiscal_year_lt=fiscal_year_lt,
            fiscal_year_lte=fiscal_year_lte,
            fiscal_period=fiscal_period,
            fiscal_period_any_of=fiscal_period_any_of,
            fiscal_period_gt=fiscal_period_gt,
            fiscal_period_gte=fiscal_period_gte,
            fiscal_period_lt=fiscal_period_lt,
            fiscal_period_lte=fiscal_period_lte,
            limit=limit,
            sort=sort,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_benzinga_news(
    published: str | None = None,
    channels: str | None = None,
    tags: str | None = None,
    author: str | None = None,
    stocks: str | None = None,
    tickers: str | None = None,
    limit: int | None = 100,
    sort: str | None = None,
) -> str:
    """
    Retrieve real-time structured, timestamped news articles from Benzinga v2 API, including headlines,
    full-text content, tickers, categories, and more. Each article entry contains metadata such as author,
    publication time, and topic channels, as well as optional elements like teaser summaries, article body text,
    and images. Articles can be filtered by ticker and time, and are returned in a consistent format for easy
    parsing and integration. This endpoint is ideal for building alerting systems, autonomous risk analysis,
    and sentiment-driven trading strategies.

    Args:
        published: The timestamp (formatted as an ISO 8601 timestamp) when the news article was originally
                  published. Value must be an integer timestamp in seconds or formatted 'yyyy-mm-dd'.
        channels: Filter for arrays that contain the value (e.g., 'News', 'Price Target').
        tags: Filter for arrays that contain the value.
        author: The name of the journalist or entity that authored the news article.
        stocks: Filter for arrays that contain the value.
        tickers: Filter for arrays that contain the value.
        limit: Limit the maximum number of results returned. Defaults to 100 if not specified.
               The maximum allowed limit is 50000.
        sort: A comma separated list of sort columns. For each column, append '.asc' or '.desc' to specify
              the sort direction. The sort column defaults to 'published' if not specified.
              The sort order defaults to 'desc' if not specified.
    """
    try:
        # Use the v2-specific method from the massive client library
        # This calls the /benzinga/v2/news endpoint
        results = massive_client.list_benzinga_news_v2(
            published=published,
            channels=channels,
            tags=tags,
            author=author,
            stocks=stocks,
            tickers=tickers,
            limit=limit,
            sort=sort,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_benzinga_ratings(
    date: str | date | None = None,
    date_any_of: str | None = None,
    date_gt: str | date | None = None,
    date_gte: str | date | None = None,
    date_lt: str | date | None = None,
    date_lte: str | date | None = None,
    ticker: str | None = None,
    ticker_any_of: str | None = None,
    ticker_gt: str | None = None,
    ticker_gte: str | None = None,
    ticker_lt: str | None = None,
    ticker_lte: str | None = None,
    importance: int | None = None,
    importance_any_of: str | None = None,
    importance_gt: int | None = None,
    importance_gte: int | None = None,
    importance_lt: int | None = None,
    importance_lte: int | None = None,
    last_updated: str | None = None,
    last_updated_any_of: str | None = None,
    last_updated_gt: str | None = None,
    last_updated_gte: str | None = None,
    last_updated_lt: str | None = None,
    last_updated_lte: str | None = None,
    rating_action: str | None = None,
    rating_action_any_of: str | None = None,
    rating_action_gt: str | None = None,
    rating_action_gte: str | None = None,
    rating_action_lt: str | None = None,
    rating_action_lte: str | None = None,
    price_target_action: str | None = None,
    price_target_action_any_of: str | None = None,
    price_target_action_gt: str | None = None,
    price_target_action_gte: str | None = None,
    price_target_action_lt: str | None = None,
    price_target_action_lte: str | None = None,
    benzinga_id: str | None = None,
    benzinga_id_any_of: str | None = None,
    benzinga_id_gt: str | None = None,
    benzinga_id_gte: str | None = None,
    benzinga_id_lt: str | None = None,
    benzinga_id_lte: str | None = None,
    benzinga_analyst_id: str | None = None,
    benzinga_analyst_id_any_of: str | None = None,
    benzinga_analyst_id_gt: str | None = None,
    benzinga_analyst_id_gte: str | None = None,
    benzinga_analyst_id_lt: str | None = None,
    benzinga_analyst_id_lte: str | None = None,
    benzinga_firm_id: str | None = None,
    benzinga_firm_id_any_of: str | None = None,
    benzinga_firm_id_gt: str | None = None,
    benzinga_firm_id_gte: str | None = None,
    benzinga_firm_id_lt: str | None = None,
    benzinga_firm_id_lte: str | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    List Benzinga ratings.
    """
    try:
        results = massive_client.list_benzinga_ratings(
            date=date,
            date_any_of=date_any_of,
            date_gt=date_gt,
            date_gte=date_gte,
            date_lt=date_lt,
            date_lte=date_lte,
            ticker=ticker,
            ticker_any_of=ticker_any_of,
            ticker_gt=ticker_gt,
            ticker_gte=ticker_gte,
            ticker_lt=ticker_lt,
            ticker_lte=ticker_lte,
            importance=importance,
            importance_any_of=importance_any_of,
            importance_gt=importance_gt,
            importance_gte=importance_gte,
            importance_lt=importance_lt,
            importance_lte=importance_lte,
            last_updated=last_updated,
            last_updated_any_of=last_updated_any_of,
            last_updated_gt=last_updated_gt,
            last_updated_gte=last_updated_gte,
            last_updated_lt=last_updated_lt,
            last_updated_lte=last_updated_lte,
            rating_action=rating_action,
            rating_action_any_of=rating_action_any_of,
            rating_action_gt=rating_action_gt,
            rating_action_gte=rating_action_gte,
            rating_action_lt=rating_action_lt,
            rating_action_lte=rating_action_lte,
            price_target_action=price_target_action,
            price_target_action_any_of=price_target_action_any_of,
            price_target_action_gt=price_target_action_gt,
            price_target_action_gte=price_target_action_gte,
            price_target_action_lt=price_target_action_lt,
            price_target_action_lte=price_target_action_lte,
            benzinga_id=benzinga_id,
            benzinga_id_any_of=benzinga_id_any_of,
            benzinga_id_gt=benzinga_id_gt,
            benzinga_id_gte=benzinga_id_gte,
            benzinga_id_lt=benzinga_id_lt,
            benzinga_id_lte=benzinga_id_lte,
            benzinga_analyst_id=benzinga_analyst_id,
            benzinga_analyst_id_any_of=benzinga_analyst_id_any_of,
            benzinga_analyst_id_gt=benzinga_analyst_id_gt,
            benzinga_analyst_id_gte=benzinga_analyst_id_gte,
            benzinga_analyst_id_lt=benzinga_analyst_id_lt,
            benzinga_analyst_id_lte=benzinga_analyst_id_lte,
            benzinga_firm_id=benzinga_firm_id,
            benzinga_firm_id_any_of=benzinga_firm_id_any_of,
            benzinga_firm_id_gt=benzinga_firm_id_gt,
            benzinga_firm_id_gte=benzinga_firm_id_gte,
            benzinga_firm_id_lt=benzinga_firm_id_lt,
            benzinga_firm_id_lte=benzinga_firm_id_lte,
            limit=limit,
            sort=sort,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_futures_aggregates(
    ticker: str,
    resolution: str,
    window_start: str | None = None,
    window_start_lt: str | None = None,
    window_start_lte: str | None = None,
    window_start_gt: str | None = None,
    window_start_gte: str | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get aggregates for a futures contract in a given time range.
    """
    try:
        results = massive_client.list_futures_aggregates(
            ticker=ticker,
            resolution=resolution,
            window_start=window_start,
            window_start_lt=window_start_lt,
            window_start_lte=window_start_lte,
            window_start_gt=window_start_gt,
            window_start_gte=window_start_gte,
            limit=limit,
            sort=sort,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_futures_contracts(
    product_code: str | None = None,
    first_trade_date: str | date | None = None,
    last_trade_date: str | date | None = None,
    as_of: str | date | None = None,
    active: str | None = None,
    type: str | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get a paginated list of futures contracts.
    """
    try:
        results = massive_client.list_futures_contracts(
            product_code=product_code,
            first_trade_date=first_trade_date,
            last_trade_date=last_trade_date,
            as_of=as_of,
            active=active,
            type=type,
            limit=limit,
            sort=sort,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_futures_contract_details(
    ticker: str,
    as_of: str | date | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get details for a single futures contract at a specified point in time.
    """
    try:
        results = massive_client.get_futures_contract_details(
            ticker=ticker,
            as_of=as_of,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_futures_products(
    name: str | None = None,
    name_search: str | None = None,
    as_of: str | date | None = None,
    trading_venue: str | None = None,
    sector: str | None = None,
    sub_sector: str | None = None,
    asset_class: str | None = None,
    asset_sub_class: str | None = None,
    type: str | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get a list of futures products (including combos).
    """
    try:
        results = massive_client.list_futures_products(
            name=name,
            name_search=name_search,
            as_of=as_of,
            trading_venue=trading_venue,
            sector=sector,
            sub_sector=sub_sector,
            asset_class=asset_class,
            asset_sub_class=asset_sub_class,
            type=type,
            limit=limit,
            sort=sort,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_futures_product_details(
    product_code: str,
    type: str | None = None,
    as_of: str | date | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get details for a single futures product as it was at a specific day.
    """
    try:
        results = massive_client.get_futures_product_details(
            product_code=product_code,
            type=type,
            as_of=as_of,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_futures_quotes(
    ticker: str,
    timestamp: str | None = None,
    timestamp_lt: str | None = None,
    timestamp_lte: str | None = None,
    timestamp_gt: str | None = None,
    timestamp_gte: str | None = None,
    session_end_date: str | None = None,
    session_end_date_lt: str | None = None,
    session_end_date_lte: str | None = None,
    session_end_date_gt: str | None = None,
    session_end_date_gte: str | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get quotes for a futures contract in a given time range.
    """
    try:
        results = massive_client.list_futures_quotes(
            ticker=ticker,
            timestamp=timestamp,
            timestamp_lt=timestamp_lt,
            timestamp_lte=timestamp_lte,
            timestamp_gt=timestamp_gt,
            timestamp_gte=timestamp_gte,
            session_end_date=session_end_date,
            session_end_date_lt=session_end_date_lt,
            session_end_date_lte=session_end_date_lte,
            session_end_date_gt=session_end_date_gt,
            session_end_date_gte=session_end_date_gte,
            limit=limit,
            sort=sort,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_futures_trades(
    ticker: str,
    timestamp: str | None = None,
    timestamp_lt: str | None = None,
    timestamp_lte: str | None = None,
    timestamp_gt: str | None = None,
    timestamp_gte: str | None = None,
    session_end_date: str | None = None,
    session_end_date_lt: str | None = None,
    session_end_date_lte: str | None = None,
    session_end_date_gt: str | None = None,
    session_end_date_gte: str | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get trades for a futures contract in a given time range.
    """
    try:
        results = massive_client.list_futures_trades(
            ticker=ticker,
            timestamp=timestamp,
            timestamp_lt=timestamp_lt,
            timestamp_lte=timestamp_lte,
            timestamp_gt=timestamp_gt,
            timestamp_gte=timestamp_gte,
            session_end_date=session_end_date,
            session_end_date_lt=session_end_date_lt,
            session_end_date_lte=session_end_date_lte,
            session_end_date_gt=session_end_date_gt,
            session_end_date_gte=session_end_date_gte,
            limit=limit,
            sort=sort,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_futures_schedules(
    session_end_date: str | None = None,
    trading_venue: str | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get trading schedules for multiple futures products on a specific date.
    """
    try:
        results = massive_client.list_futures_schedules(
            session_end_date=session_end_date,
            trading_venue=trading_venue,
            limit=limit,
            sort=sort,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_futures_schedules_by_product_code(
    product_code: str,
    session_end_date: str | None = None,
    session_end_date_lt: str | None = None,
    session_end_date_lte: str | None = None,
    session_end_date_gt: str | None = None,
    session_end_date_gte: str | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get schedule data for a single futures product across many trading dates.
    """
    try:
        results = massive_client.list_futures_schedules_by_product_code(
            product_code=product_code,
            session_end_date=session_end_date,
            session_end_date_lt=session_end_date_lt,
            session_end_date_lte=session_end_date_lte,
            session_end_date_gt=session_end_date_gt,
            session_end_date_gte=session_end_date_gte,
            limit=limit,
            sort=sort,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_futures_market_statuses(
    product_code_any_of: str | None = None,
    product_code: str | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get market statuses for futures products.
    """
    try:
        results = massive_client.list_futures_market_statuses(
            product_code_any_of=product_code_any_of,
            product_code=product_code,
            limit=limit,
            sort=sort,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


@poly_mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_futures_snapshot(
    ticker: str | None = None,
    ticker_any_of: str | None = None,
    ticker_gt: str | None = None,
    ticker_gte: str | None = None,
    ticker_lt: str | None = None,
    ticker_lte: str | None = None,
    product_code: str | None = None,
    product_code_any_of: str | None = None,
    product_code_gt: str | None = None,
    product_code_gte: str | None = None,
    product_code_lt: str | None = None,
    product_code_lte: str | None = None,
    limit: int | None = 10,
    sort: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Get snapshots for futures contracts.
    """
    try:
        results = massive_client.get_futures_snapshot(
            ticker=ticker,
            ticker_any_of=ticker_any_of,
            ticker_gt=ticker_gt,
            ticker_gte=ticker_gte,
            ticker_lt=ticker_lt,
            ticker_lte=ticker_lte,
            product_code=product_code,
            product_code_any_of=product_code_any_of,
            product_code_gt=product_code_gt,
            product_code_gte=product_code_gte,
            product_code_lt=product_code_lt,
            product_code_lte=product_code_lte,
            limit=limit,
            sort=sort,
            params=params,
            raw=True,
        )

        return json_to_csv(results.data.decode("utf-8"))
    except Exception as e:
        return f"Error: {e}"


# Directly expose the MCP server object
# It will be run from entrypoint.py


def run(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Run the Massive MCP server."""
    poly_mcp.run(transport)
