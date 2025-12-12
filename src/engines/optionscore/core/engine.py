"""
OptionsCore Engine

Main orchestration layer for options pricing and Greeks calculation.
Coordinates data fetching, enrichment, and caching for the Ordinis platform.

Author: Ordinis Project
License: MIT
"""

from datetime import datetime, timedelta
from typing import Any

from ..data import OptionContract
from ..pricing.black_scholes import BlackScholesEngine
from ..pricing.greeks import GreeksCalculator
from .config import OptionsEngineConfig
from .enrichment import (
    ChainEnrichmentEngine,
    EnrichedOptionsChain,
    GreeksResult,
    PricingResult,
)


class OptionsCoreEngine:
    """
    Main orchestrator for options pricing and analytics.

    Coordinates:
    - Data fetching from market data providers
    - Pricing calculation using Black-Scholes
    - Greeks calculation for risk management
    - Caching with configurable TTL
    - Chain-level enrichment and analytics

    Usage:
        >>> from src.plugins.market_data.polygon import PolygonDataPlugin
        >>>
        >>> # Initialize dependencies
        >>> polygon_config = PluginConfig(name="polygon", api_key=API_KEY)
        >>> polygon = PolygonDataPlugin(polygon_config)
        >>> await polygon.initialize()
        >>>
        >>> # Initialize engine
        >>> engine_config = OptionsEngineConfig(
        ...     engine_id="optionscore_main",
        ...     cache_ttl_seconds=300,
        ...     default_risk_free_rate=0.05
        ... )
        >>> engine = OptionsCoreEngine(engine_config, polygon)
        >>> await engine.initialize()
        >>>
        >>> # Fetch enriched chain
        >>> chain = await engine.get_option_chain("AAPL")
        >>> print(f"Found {len(chain.contracts)} contracts")
        >>> print(f"ATM strike: {chain.summary['atm_strike']}")
    """

    def __init__(self, config: OptionsEngineConfig, polygon_plugin):
        """
        Initialize OptionsCore engine.

        Args:
            config: Engine configuration
            polygon_plugin: Polygon data plugin instance (PolygonDataPlugin)
        """
        self.config = config
        self.polygon = polygon_plugin

        # Initialize pricing components
        self.pricing_engine = BlackScholesEngine()
        self.greeks_calc = GreeksCalculator(self.pricing_engine)
        self.enrichment_engine = ChainEnrichmentEngine(self.pricing_engine, self.greeks_calc)

        # Initialize cache
        self.cache: dict[str, tuple[datetime, Any]] = {}
        self.cache_ttl = timedelta(seconds=config.cache_ttl_seconds)

        # Engine state
        self.initialized = False

    async def initialize(self) -> bool:
        """
        Initialize the engine.

        Returns:
            True if initialization successful

        Raises:
            RuntimeError: If initialization fails
        """
        if self.initialized:
            return True

        # Verify Polygon plugin is ready
        if not self.polygon or self.polygon.status.value not in ["ready", "running"]:
            raise RuntimeError("Polygon plugin not ready")

        self.initialized = True
        return True

    async def get_option_chain(
        self,
        symbol: str,
        expiration: str | None = None,
        strike_price: float | None = None,
        contract_type: str | None = None,
    ) -> EnrichedOptionsChain:
        """
        Get enriched options chain with pricing and Greeks.

        Args:
            symbol: Underlying ticker symbol
            expiration: Optional expiration date filter (YYYY-MM-DD)
            strike_price: Optional strike price filter
            contract_type: Optional contract type filter ('call' or 'put')

        Returns:
            Enriched options chain with all contracts priced

        Raises:
            ValueError: If symbol is invalid
            RuntimeError: If data fetching fails

        Example:
            >>> # Get full chain
            >>> chain = await engine.get_option_chain("AAPL")
            >>>
            >>> # Get specific expiration
            >>> chain = await engine.get_option_chain("AAPL", expiration="2025-01-17")
            >>>
            >>> # Get calls only
            >>> chain = await engine.get_option_chain("AAPL", contract_type="call")
        """
        if not self.initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        # Validate symbol
        if not symbol or not symbol.isalpha():
            raise ValueError(f"Invalid symbol: {symbol}")

        # Check cache
        cache_key = (
            f"chain:{symbol}:{expiration or 'all'}:{strike_price or 'all'}:{contract_type or 'all'}"
        )
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            # Fetch underlying price
            quote = await self.polygon.get_quote(symbol)
            underlying_price = quote.get("last", 0.0)

            if underlying_price <= 0:
                raise ValueError(f"Invalid underlying price for {symbol}: {underlying_price}")

            # Fetch options chain from Polygon
            raw_chain = await self.polygon.get_options_chain(
                symbol=symbol,
                expiration=expiration,
                strike_price=strike_price,
                contract_type=contract_type,
            )

            if not raw_chain or not raw_chain.get("contracts"):
                raise RuntimeError(f"No options data returned for {symbol}")

            # Enrich chain with pricing and Greeks
            enriched_chain = self.enrichment_engine.enrich_chain(
                chain_data=raw_chain,
                underlying_price=underlying_price,
                risk_free_rate=self.config.default_risk_free_rate,
                dividend_yield=self.config.default_dividend_yield,
            )

            # Cache result
            self._set_cached(cache_key, enriched_chain)

            return enriched_chain

        except Exception as e:
            raise RuntimeError(f"Failed to fetch options chain for {symbol}: {e}") from e

    async def price_contract(
        self,
        contract: OptionContract,
        underlying_price: float,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        assumed_volatility: float = 0.25,
    ) -> PricingResult:
        """
        Price a single option contract.

        Args:
            contract: Option contract to price
            underlying_price: Current underlying price
            risk_free_rate: Risk-free rate (uses default if None)
            dividend_yield: Dividend yield (uses default if None)
            assumed_volatility: Volatility assumption

        Returns:
            Pricing result with theoretical price

        Example:
            >>> contract = OptionContract(
            ...     symbol="O:AAPL250117C00150000",
            ...     underlying="AAPL",
            ...     option_type=OptionType.CALL,
            ...     strike=150.0,
            ...     expiration=date(2025, 1, 17)
            ... )
            >>> pricing = await engine.price_contract(contract, 145.0)
            >>> print(f"Theoretical price: ${pricing.theoretical_price:.2f}")
        """
        if not self.initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        # Use defaults if not provided
        r = risk_free_rate if risk_free_rate is not None else self.config.default_risk_free_rate
        q = dividend_yield if dividend_yield is not None else self.config.default_dividend_yield

        # Create contract dict for enrichment
        contract_data = {
            "ticker": contract.symbol,
            "underlying": contract.underlying,
            "contract_type": "call" if contract.option_type.value == "call" else "put",
            "strike_price": contract.strike,
            "expiration_date": contract.expiration,
            "shares_per_contract": contract.multiplier,
        }

        # Enrich contract
        enriched = self.enrichment_engine.enrich_contract(
            contract_data=contract_data,
            underlying_price=underlying_price,
            risk_free_rate=r,
            dividend_yield=q,
            assumed_volatility=assumed_volatility,
        )

        return enriched.pricing

    async def calculate_greeks(
        self,
        contract: OptionContract,
        underlying_price: float,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        assumed_volatility: float = 0.25,
    ) -> GreeksResult:
        """
        Calculate Greeks for a single option contract.

        Args:
            contract: Option contract
            underlying_price: Current underlying price
            risk_free_rate: Risk-free rate (uses default if None)
            dividend_yield: Dividend yield (uses default if None)
            assumed_volatility: Volatility assumption

        Returns:
            Greeks result with delta, gamma, theta, vega, rho

        Example:
            >>> greeks = await engine.calculate_greeks(contract, 145.0)
            >>> print(f"Delta: {greeks.delta:.4f}")
            >>> print(f"Gamma: {greeks.gamma:.4f}")
            >>> print(f"Theta: {greeks.theta:.4f} per day")
        """
        if not self.initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        # Use defaults if not provided
        r = risk_free_rate if risk_free_rate is not None else self.config.default_risk_free_rate
        q = dividend_yield if dividend_yield is not None else self.config.default_dividend_yield

        # Create contract dict for enrichment
        contract_data = {
            "ticker": contract.symbol,
            "underlying": contract.underlying,
            "contract_type": "call" if contract.option_type.value == "call" else "put",
            "strike_price": contract.strike,
            "expiration_date": contract.expiration,
            "shares_per_contract": contract.multiplier,
        }

        # Enrich contract
        enriched = self.enrichment_engine.enrich_contract(
            contract_data=contract_data,
            underlying_price=underlying_price,
            risk_free_rate=r,
            dividend_yield=q,
            assumed_volatility=assumed_volatility,
        )

        return enriched.greeks

    def _get_cached(self, key: str) -> Any | None:
        """
        Get value from cache if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key not in self.cache:
            return None

        cached_time, cached_value = self.cache[key]

        # Check if expired
        if datetime.utcnow() - cached_time < self.cache_ttl:
            return cached_value

        # Expired - remove from cache
        del self.cache[key]
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        """
        Set value in cache with current timestamp.

        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = (datetime.utcnow(), value)

    def clear_cache(self, symbol: str | None = None) -> None:
        """
        Clear cache entries.

        Args:
            symbol: If provided, clear only entries for this symbol.
                   If None, clear all cache.

        Example:
            >>> # Clear all cache
            >>> engine.clear_cache()
            >>>
            >>> # Clear cache for specific symbol
            >>> engine.clear_cache("AAPL")
        """
        if symbol is None:
            self.cache.clear()
        else:
            # Clear entries matching symbol
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(f"chain:{symbol}:")]
            for key in keys_to_remove:
                del self.cache[key]

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache metrics

        Example:
            >>> stats = engine.get_cache_stats()
            >>> print(f"Cached items: {stats['total_items']}")
            >>> print(f"Cache size: {stats['size_bytes']} bytes")
        """
        import sys

        total_items = len(self.cache)
        size_bytes = sys.getsizeof(self.cache)

        # Count expired entries
        now = datetime.utcnow()
        expired_count = sum(
            1 for cached_time, _ in self.cache.values() if now - cached_time >= self.cache_ttl
        )

        return {
            "total_items": total_items,
            "active_items": total_items - expired_count,
            "expired_items": expired_count,
            "size_bytes": size_bytes,
            "ttl_seconds": self.cache_ttl.total_seconds(),
        }
