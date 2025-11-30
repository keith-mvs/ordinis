"""
Technical Indicator Charts.

Provides interactive visualizations for technical indicators used in trading strategies.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engines.signalcore.core.signal import Signal, SignalType
from engines.signalcore.features.technical import TechnicalIndicators


class IndicatorChart:
    """Interactive technical indicator charts using Plotly."""

    @staticmethod
    def plot_bollinger_bands(
        data: pd.DataFrame,
        bb_period: int = 20,
        bb_std: float = 2.0,
        title: str = "Bollinger Bands",
        show_volume: bool = True,
    ) -> go.Figure:
        """
        Plot candlestick chart with Bollinger Bands overlay.

        Args:
            data: DataFrame with OHLCV data (index should be datetime)
            bb_period: Bollinger Bands period
            bb_std: Number of standard deviations
            title: Chart title
            show_volume: Whether to show volume subplot

        Returns:
            Plotly Figure object
        """
        # Calculate Bollinger Bands
        middle, upper, lower = TechnicalIndicators.bollinger_bands(data["close"], bb_period, bb_std)

        # Create figure with subplots
        if show_volume:
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=(title, "Volume"),
            )
        else:
            fig = go.Figure()

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name="Price",
                increasing_line_color="green",
                decreasing_line_color="red",
            ),
            row=1,
            col=1,
        )

        # Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=upper,
                name="Upper BB",
                line={"color": "rgba(250, 128, 114, 0.7)", "width": 1, "dash": "dash"},
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=middle,
                name="Middle BB",
                line={"color": "rgba(135, 206, 250, 0.9)", "width": 2},
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=lower,
                name="Lower BB",
                line={"color": "rgba(250, 128, 114, 0.7)", "width": 1, "dash": "dash"},
                fill="tonexty",
                fillcolor="rgba(135, 206, 250, 0.1)",
            ),
            row=1,
            col=1,
        )

        # Volume
        if show_volume:
            colors = [
                "green" if data["close"].iloc[i] >= data["open"].iloc[i] else "red"
                for i in range(len(data))
            ]
            fig.add_trace(
                go.Bar(x=data.index, y=data["volume"], name="Volume", marker_color=colors),
                row=2,
                col=1,
            )

        # Layout
        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            height=600 if show_volume else 500,
            hovermode="x unified",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )

        fig.update_xaxes(title_text="Date", row=2 if show_volume else 1, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        if show_volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1)

        return fig

    @staticmethod
    def plot_macd(
        data: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        title: str = "MACD Indicator",
    ) -> go.Figure:
        """
        Plot price with MACD indicator.

        Args:
            data: DataFrame with OHLCV data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            title: Chart title

        Returns:
            Plotly Figure object
        """
        # Calculate MACD
        macd_line, signal_line, histogram = TechnicalIndicators.macd(
            data["close"], fast, slow, signal
        )

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.4],
            subplot_titles=(title, "MACD"),
        )

        # Price candlestick
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name="Price",
                increasing_line_color="green",
                decreasing_line_color="red",
            ),
            row=1,
            col=1,
        )

        # MACD line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=macd_line,
                name="MACD",
                line={"color": "blue", "width": 2},
            ),
            row=2,
            col=1,
        )

        # Signal line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=signal_line,
                name="Signal",
                line={"color": "orange", "width": 2},
            ),
            row=2,
            col=1,
        )

        # Histogram
        colors = ["green" if h > 0 else "red" for h in histogram]
        fig.add_trace(
            go.Bar(x=data.index, y=histogram, name="Histogram", marker_color=colors),
            row=2,
            col=1,
        )

        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        # Layout
        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            height=700,
            hovermode="x unified",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)

        return fig

    @staticmethod
    def plot_strategy_signals(
        data: pd.DataFrame,
        signals: list[Signal],
        title: str = "Strategy Signals",
        show_volume: bool = True,
    ) -> go.Figure:
        """
        Plot price with strategy entry/exit signals annotated.

        Args:
            data: DataFrame with OHLCV data
            signals: List of Signal objects
            title: Chart title
            show_volume: Whether to show volume subplot

        Returns:
            Plotly Figure object
        """
        # Create figure
        if show_volume:
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=(title, "Volume"),
            )
        else:
            fig = go.Figure()

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name="Price",
                increasing_line_color="green",
                decreasing_line_color="red",
            ),
            row=1,
            col=1,
        )

        # Add signal annotations
        for sig in signals:
            if sig.signal_type == SignalType.ENTRY:
                # Find price at signal timestamp
                try:
                    price = float(data.loc[sig.timestamp, "low"])  # type: ignore[index,arg-type]
                except KeyError:
                    # Timestamp not in index, skip
                    continue

                fig.add_annotation(
                    x=sig.timestamp,
                    y=price,
                    text="BUY",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor="green",
                    ax=0,
                    ay=40,
                    font={"size": 10, "color": "green"},
                    bgcolor="rgba(144, 238, 144, 0.8)",
                    bordercolor="green",
                    row=1,
                    col=1,
                )

            elif sig.signal_type == SignalType.EXIT:
                try:
                    price = float(data.loc[sig.timestamp, "high"])  # type: ignore[index,arg-type]
                except KeyError:
                    continue

                fig.add_annotation(
                    x=sig.timestamp,
                    y=price,
                    text="SELL",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor="red",
                    ax=0,
                    ay=-40,
                    font={"size": 10, "color": "red"},
                    bgcolor="rgba(255, 99, 71, 0.8)",
                    bordercolor="red",
                    row=1,
                    col=1,
                )

        # Volume
        if show_volume:
            colors = [
                "green" if data["close"].iloc[i] >= data["open"].iloc[i] else "red"
                for i in range(len(data))
            ]
            fig.add_trace(
                go.Bar(x=data.index, y=data["volume"], name="Volume", marker_color=colors),
                row=2,
                col=1,
            )

        # Layout
        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            height=600 if show_volume else 500,
            hovermode="x unified",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )

        fig.update_xaxes(title_text="Date", row=2 if show_volume else 1, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        if show_volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1)

        return fig

    @staticmethod
    def plot_rsi(
        data: pd.DataFrame,
        rsi_period: int = 14,
        oversold: int = 30,
        overbought: int = 70,
        title: str = "RSI Indicator",
    ) -> go.Figure:
        """
        Plot price with RSI indicator.

        Args:
            data: DataFrame with OHLCV data
            rsi_period: RSI calculation period
            oversold: Oversold threshold
            overbought: Overbought threshold
            title: Chart title

        Returns:
            Plotly Figure object
        """
        # Calculate RSI
        rsi = TechnicalIndicators.rsi(data["close"], rsi_period)

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.4],
            subplot_titles=(title, "RSI"),
        )

        # Price
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

        # RSI line
        fig.add_trace(
            go.Scatter(x=data.index, y=rsi, name="RSI", line={"color": "purple", "width": 2}),
            row=2,
            col=1,
        )

        # Overbought/oversold levels
        fig.add_hline(y=overbought, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=oversold, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)

        # Layout
        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            height=700,
            hovermode="x unified",
        )

        fig.update_yaxes(range=[0, 100], row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)

        return fig
