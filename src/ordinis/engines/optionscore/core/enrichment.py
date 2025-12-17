"""
Options Chain Enrichment Engine

Transforms raw options chain data from providers into enriched analytics
with theoretical pricing and Greeks calculations.

Author: Ordinis Project
License: MIT
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from ..data import OptionContract
from ..data import OptionType as DataOptionType
from ..pricing.black_scholes import BlackScholesEngine, OptionType, PricingParameters
from ..pricing.greeks import GreeksCalculator


@dataclass
class PricingResult:
    """
    Result of options pricing calculation.

    Attributes:
        contract: The priced option contract
        theoretical_price: Calculated theoretical price
        model_used: Pricing model identifier
        calculation_timestamp: When calculation was performed
        parameters_used: Input parameters for pricing
        metadata: Additional pricing data
    """

    contract: OptionContract
    theoretical_price: float
    model_used: str
    calculation_timestamp: datetime
    parameters_used: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GreeksResult:
    """
    Greeks calculation result for an option.

    Attributes:
        contract: The option contract
        delta: Price sensitivity to underlying ($/$)
        gamma: Delta sensitivity to underlying ($/$ per $)
        theta: Time decay per day ($/day)
        vega: Volatility sensitivity per 1% IV change ($/%IV)
        rho: Interest rate sensitivity per 1% rate change ($/%)
        calculation_timestamp: When calculation was performed
        underlying_price: Underlying price used
        metadata: Additional Greeks data
    """

    contract: OptionContract
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    calculation_timestamp: datetime
    underlying_price: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrichedContract:
    """
    Option contract enriched with pricing and Greeks.

    Attributes:
        contract: Base option contract data
        pricing: Theoretical pricing result
        greeks: Greeks calculation result
        implied_volatility: Market-implied volatility (future)
    """

    contract: OptionContract
    pricing: PricingResult
    greeks: GreeksResult
    implied_volatility: float | None = None


@dataclass
class EnrichedOptionsChain:
    """
    Complete options chain with enriched contract data.

    Attributes:
        symbol: Underlying ticker symbol
        underlying_price: Current underlying price
        timestamp: Chain data timestamp
        contracts: List of enriched contracts
        expirations: Unique expiration dates
        strikes: Unique strike prices
        summary: Chain-level statistics
    """

    symbol: str
    underlying_price: float
    timestamp: datetime
    contracts: list[EnrichedContract]
    expirations: list[str]
    strikes: list[float]
    summary: dict[str, Any] = field(default_factory=dict)


class ChainEnrichmentEngine:
    """
    Engine for enriching options chain data with pricing and Greeks.

    Transforms raw provider data into comprehensive analytics
    using Black-Scholes pricing and Greeks calculations.

    Usage:
        >>> pricing_engine = BlackScholesEngine()
        >>> greeks_calc = GreeksCalculator(pricing_engine)
        >>> enrichment = ChainEnrichmentEngine(pricing_engine, greeks_calc)
        >>> enriched_chain = enrichment.enrich_chain(
        ...     chain_data=provider_data,
        ...     underlying_price=100.0,
        ...     risk_free_rate=0.05
        ... )
    """

    def __init__(self, pricing_engine: BlackScholesEngine, greeks_calc: GreeksCalculator):
        """
        Initialize enrichment engine.

        Args:
            pricing_engine: Black-Scholes pricing engine
            greeks_calc: Greeks calculator
        """
        self.pricing_engine = pricing_engine
        self.greeks_calc = greeks_calc

    def enrich_contract(
        self,
        contract_data: dict[str, Any],
        underlying_price: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        assumed_volatility: float = 0.25,
    ) -> EnrichedContract:
        """
        Enrich a single contract with pricing and Greeks.

        Args:
            contract_data: Raw contract data from provider
            underlying_price: Current underlying price
            risk_free_rate: Risk-free rate (annual, decimal)
            dividend_yield: Dividend yield (annual, decimal)
            assumed_volatility: Volatility assumption for pricing (if IV not available)

        Returns:
            Enriched contract with pricing and Greeks

        Example:
            >>> contract_data = {
            ...     "ticker": "O:AAPL250117C00150000",
            ...     "underlying": "AAPL",
            ...     "contract_type": "call",
            ...     "strike_price": 150.0,
            ...     "expiration_date": "2025-01-17"
            ... }
            >>> enriched = enrichment.enrich_contract(contract_data, 145.0, 0.05)
        """
        # Parse contract data
        symbol = contract_data.get("ticker", "")
        underlying = contract_data.get("underlying", "")
        contract_type_str = contract_data.get("contract_type", "call").lower()
        strike = float(contract_data.get("strike_price", 0))
        expiration_str = contract_data.get("expiration_date", "")

        # Convert contract type
        if contract_type_str == "call":
            data_option_type = DataOptionType.CALL
            pricing_option_type = OptionType.CALL
        else:
            data_option_type = DataOptionType.PUT
            pricing_option_type = OptionType.PUT

        # Parse expiration date
        if isinstance(expiration_str, str):
            expiration = datetime.strptime(expiration_str, "%Y-%m-%d").date()
        elif isinstance(expiration_str, date):
            expiration = expiration_str
        else:
            raise ValueError(f"Invalid expiration date format: {expiration_str}")

        # Create OptionContract
        contract = OptionContract(
            symbol=symbol,
            underlying=underlying,
            option_type=data_option_type,
            strike=strike,
            expiration=expiration,
            multiplier=contract_data.get("shares_per_contract", 100),
            exchange=contract_data.get("primary_exchange", ""),
        )

        # Calculate time to expiration
        time_to_expiration = contract.time_to_expiration

        # Prepare pricing parameters
        pricing_params = PricingParameters(
            S=underlying_price,
            K=strike,
            T=time_to_expiration,
            r=risk_free_rate,
            sigma=assumed_volatility,
            q=dividend_yield,
        )

        # Calculate theoretical price
        theoretical_price = self.pricing_engine.price(pricing_params, pricing_option_type)

        # Create pricing result
        pricing_result = PricingResult(
            contract=contract,
            theoretical_price=theoretical_price,
            model_used="black_scholes",
            calculation_timestamp=datetime.utcnow(),
            parameters_used={
                "underlying_price": underlying_price,
                "strike": strike,
                "time_to_expiration": time_to_expiration,
                "risk_free_rate": risk_free_rate,
                "volatility": assumed_volatility,
                "dividend_yield": dividend_yield,
            },
        )

        # Calculate Greeks
        greeks_dict = self.greeks_calc.all_greeks(pricing_params, pricing_option_type)

        greeks_result = GreeksResult(
            contract=contract,
            delta=greeks_dict["delta"],
            gamma=greeks_dict["gamma"],
            theta=greeks_dict["theta"],
            vega=greeks_dict["vega"],
            rho=greeks_dict["rho"],
            calculation_timestamp=datetime.utcnow(),
            underlying_price=underlying_price,
        )

        # Create enriched contract
        return EnrichedContract(contract=contract, pricing=pricing_result, greeks=greeks_result)

    def enrich_chain(
        self,
        chain_data: dict[str, Any],
        underlying_price: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        assumed_volatility: float = 0.25,
    ) -> EnrichedOptionsChain:
        """
        Enrich entire options chain with pricing and Greeks.

        Args:
            chain_data: Raw chain data from provider
            underlying_price: Current underlying price
            risk_free_rate: Risk-free rate (annual, decimal)
            dividend_yield: Dividend yield (annual, decimal)
            assumed_volatility: Volatility assumption for pricing

        Returns:
            Enriched options chain with all contracts priced

        Example:
            >>> chain_data = provider.get_options_chain("AAPL")
            >>> enriched = enrichment.enrich_chain(
            ...     chain_data,
            ...     underlying_price=145.0,
            ...     risk_free_rate=0.05
            ... )
        """
        symbol = chain_data.get("symbol", "")
        contracts_data = chain_data.get("contracts", [])

        # Enrich all contracts
        enriched_contracts = []
        for contract_data in contracts_data:
            try:
                enriched = self.enrich_contract(
                    contract_data,
                    underlying_price,
                    risk_free_rate,
                    dividend_yield,
                    assumed_volatility,
                )
                enriched_contracts.append(enriched)
            except Exception as e:
                # Log error but continue with other contracts
                # In production, use proper logging
                continue

        # Extract unique expirations and strikes
        expirations = sorted(
            {contract.contract.expiration.isoformat() for contract in enriched_contracts}
        )
        strikes = sorted({contract.contract.strike for contract in enriched_contracts})

        # Create enriched chain
        enriched_chain = EnrichedOptionsChain(
            symbol=symbol,
            underlying_price=underlying_price,
            timestamp=datetime.utcnow(),
            contracts=enriched_contracts,
            expirations=expirations,
            strikes=strikes,
        )

        # Calculate summary statistics
        enriched_chain.summary = self.calculate_summary_stats(enriched_chain)

        return enriched_chain

    def calculate_summary_stats(self, chain: EnrichedOptionsChain) -> dict[str, Any]:
        """
        Calculate chain-level summary statistics.

        Args:
            chain: Enriched options chain

        Returns:
            Dictionary of summary statistics

        Statistics include:
            - Total contracts count
            - Call/put counts
            - ATM strike identification
            - Average implied volatilities (if available)
        """
        total_contracts = len(chain.contracts)

        calls = [c for c in chain.contracts if c.contract.option_type == DataOptionType.CALL]
        puts = [c for c in chain.contracts if c.contract.option_type == DataOptionType.PUT]

        # Find ATM strike (closest to underlying price)
        atm_strike = min(chain.strikes, key=lambda s: abs(s - chain.underlying_price), default=0.0)

        summary = {
            "total_contracts": total_contracts,
            "total_calls": len(calls),
            "total_puts": len(puts),
            "unique_expirations": len(chain.expirations),
            "unique_strikes": len(chain.strikes),
            "atm_strike": atm_strike,
            "underlying_price": chain.underlying_price,
            "chain_timestamp": chain.timestamp.isoformat(),
        }

        return summary
