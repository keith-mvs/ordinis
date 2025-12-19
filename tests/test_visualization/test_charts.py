"""Tests for visualization.charts module."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go
import pytest

from ordinis.visualization.charts import ChartUtils


class TestChartUtilsThemeConstants:
    """Tests for ChartUtils theme constants."""

    def test_dark_theme_structure(self):
        """Test dark theme has required keys."""
        theme = ChartUtils.DARK_THEME

        assert "template" in theme
        assert "paper_bgcolor" in theme
        assert "plot_bgcolor" in theme
        assert "font_color" in theme
        assert "grid_color" in theme

    def test_dark_theme_values(self):
        """Test dark theme has expected values."""
        theme = ChartUtils.DARK_THEME

        assert theme["template"] == "plotly_dark"
        assert theme["paper_bgcolor"] == "#1e1e1e"
        assert theme["plot_bgcolor"] == "#1e1e1e"
        assert theme["font_color"] == "#e0e0e0"
        assert theme["grid_color"] == "#333333"

    def test_light_theme_structure(self):
        """Test light theme has required keys."""
        theme = ChartUtils.LIGHT_THEME

        assert "template" in theme
        assert "paper_bgcolor" in theme
        assert "plot_bgcolor" in theme
        assert "font_color" in theme
        assert "grid_color" in theme

    def test_light_theme_values(self):
        """Test light theme has expected values."""
        theme = ChartUtils.LIGHT_THEME

        assert theme["template"] == "plotly_white"
        assert theme["paper_bgcolor"] == "#ffffff"
        assert theme["plot_bgcolor"] == "#ffffff"
        assert theme["font_color"] == "#2a2a2a"
        assert theme["grid_color"] == "#e5e5e5"


class TestApplyTheme:
    """Tests for ChartUtils.apply_theme method."""

    def test_apply_dark_theme(self):
        """Test applying dark theme to figure."""
        fig = go.Figure()

        result = ChartUtils.apply_theme(fig, theme="dark")

        assert isinstance(result, go.Figure)
        assert result is fig
        assert result.layout.paper_bgcolor == "#1e1e1e"
        assert result.layout.plot_bgcolor == "#1e1e1e"
        assert result.layout.font.color == "#e0e0e0"

    def test_apply_light_theme(self):
        """Test applying light theme to figure."""
        fig = go.Figure()

        result = ChartUtils.apply_theme(fig, theme="light")

        assert isinstance(result, go.Figure)
        assert result is fig
        assert result.layout.paper_bgcolor == "#ffffff"
        assert result.layout.plot_bgcolor == "#ffffff"
        assert result.layout.font.color == "#2a2a2a"

    def test_apply_theme_default_is_dark(self):
        """Test apply_theme uses dark theme by default."""
        fig = go.Figure()

        result = ChartUtils.apply_theme(fig)

        assert result.layout.paper_bgcolor == "#1e1e1e"

    def test_apply_theme_with_data(self):
        """Test apply_theme preserves figure data."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name="test"))

        result = ChartUtils.apply_theme(fig, theme="dark")

        assert len(result.data) == 1
        assert result.data[0].name == "test"

    def test_apply_theme_updates_axes_grid(self):
        """Test apply_theme updates axes grid colors."""
        fig = go.Figure()

        ChartUtils.apply_theme(fig, theme="dark")

        assert fig.layout.xaxis.gridcolor == "#333333"
        assert fig.layout.yaxis.gridcolor == "#333333"


class TestExportChart:
    """Tests for ChartUtils.export_chart method."""

    def test_export_chart_html_format(self, tmp_path):
        """Test exporting chart to HTML format."""
        fig = go.Figure()
        output_file = tmp_path / "test_chart.html"

        ChartUtils.export_chart(fig, output_file, file_format="html")

        assert output_file.exists()
        content = output_file.read_text()
        assert "plotly" in content.lower()

    def test_export_chart_json_format(self, tmp_path):
        """Test exporting chart to JSON format."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
        output_file = tmp_path / "test_chart.json"

        ChartUtils.export_chart(fig, output_file, file_format="json")

        assert output_file.exists()

    def test_export_chart_accepts_path_string(self, tmp_path):
        """Test export_chart accepts string path."""
        fig = go.Figure()
        output_file = str(tmp_path / "test_chart.html")

        ChartUtils.export_chart(fig, output_file, file_format="html")

        assert Path(output_file).exists()

    def test_export_chart_accepts_path_object(self, tmp_path):
        """Test export_chart accepts Path object."""
        fig = go.Figure()
        output_file = tmp_path / "test_chart.html"

        ChartUtils.export_chart(fig, output_file, file_format="html")

        assert output_file.exists()

    @patch("plotly.graph_objects.Figure.write_image")
    def test_export_chart_png_format(self, mock_write_image, tmp_path):
        """Test exporting chart to PNG format."""
        fig = go.Figure()
        output_file = tmp_path / "test_chart.png"

        ChartUtils.export_chart(
            fig,
            output_file,
            file_format="png",
            width=800,
            height=600,
        )

        mock_write_image.assert_called_once_with(
            str(output_file),
            format="png",
            width=800,
            height=600,
        )

    @patch("plotly.graph_objects.Figure.write_image")
    def test_export_chart_jpg_format(self, mock_write_image, tmp_path):
        """Test exporting chart to JPG format."""
        fig = go.Figure()
        output_file = tmp_path / "test_chart.jpg"

        ChartUtils.export_chart(fig, output_file, file_format="jpg")

        mock_write_image.assert_called_once_with(
            str(output_file),
            format="jpg",
            width=None,
            height=None,
        )

    @patch("plotly.graph_objects.Figure.write_image")
    def test_export_chart_svg_format(self, mock_write_image, tmp_path):
        """Test exporting chart to SVG format."""
        fig = go.Figure()
        output_file = tmp_path / "test_chart.svg"

        ChartUtils.export_chart(fig, output_file, file_format="svg")

        mock_write_image.assert_called_once_with(
            str(output_file),
            format="svg",
            width=None,
            height=None,
        )

    def test_export_chart_unsupported_format(self, tmp_path):
        """Test export_chart raises error for unsupported format."""
        fig = go.Figure()
        output_file = tmp_path / "test_chart.xyz"

        with pytest.raises(ValueError, match="Unsupported export format"):
            ChartUtils.export_chart(fig, output_file, file_format="xyz")

    def test_export_chart_default_format_is_html(self, tmp_path):
        """Test export_chart uses HTML as default format."""
        fig = go.Figure()
        output_file = tmp_path / "test_chart.html"

        ChartUtils.export_chart(fig, output_file)

        assert output_file.exists()


class TestCreateComparisonChart:
    """Tests for ChartUtils.create_comparison_chart method."""

    def test_create_comparison_chart_basic(self):
        """Test basic comparison chart creation."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        strategies = {
            "Strategy A": pd.DataFrame(
                {"cumulative_return": [1.0 + i * 0.01 for i in range(10)]},
                index=dates,
            ),
            "Strategy B": pd.DataFrame(
                {"cumulative_return": [1.0 + i * 0.015 for i in range(10)]},
                index=dates,
            ),
        }

        fig = ChartUtils.create_comparison_chart(strategies)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
        assert fig.data[0].name == "Strategy A"
        assert fig.data[1].name == "Strategy B"

    def test_create_comparison_chart_custom_metric(self):
        """Test comparison chart with custom metric."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        strategies = {
            "Strategy A": pd.DataFrame(
                {"equity_curve": [10000 + i * 100 for i in range(10)]},
                index=dates,
            ),
        }

        fig = ChartUtils.create_comparison_chart(
            strategies,
            metric="equity_curve",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_create_comparison_chart_custom_title(self):
        """Test comparison chart with custom title."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        strategies = {
            "Strategy A": pd.DataFrame(
                {"cumulative_return": [1.0 + i * 0.01 for i in range(10)]},
                index=dates,
            ),
        }

        fig = ChartUtils.create_comparison_chart(
            strategies,
            title="Custom Title",
        )

        assert fig.layout.title.text == "Custom Title"

    def test_create_comparison_chart_multiple_strategies(self):
        """Test comparison chart with many strategies."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        strategies = {}
        for i in range(10):
            strategies[f"Strategy {i}"] = pd.DataFrame(
                {"cumulative_return": [1.0 + j * 0.01 * (i + 1) for j in range(10)]},
                index=dates,
            )

        fig = ChartUtils.create_comparison_chart(strategies)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 10

    def test_create_comparison_chart_empty_strategies(self):
        """Test comparison chart with empty strategies dict."""
        fig = ChartUtils.create_comparison_chart({})

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_create_comparison_chart_color_palette(self):
        """Test comparison chart uses color palette."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        strategies = {
            "Strategy A": pd.DataFrame(
                {"cumulative_return": [1.0 + i * 0.01 for i in range(10)]},
                index=dates,
            ),
            "Strategy B": pd.DataFrame(
                {"cumulative_return": [1.0 + i * 0.015 for i in range(10)]},
                index=dates,
            ),
        }

        fig = ChartUtils.create_comparison_chart(strategies)

        assert fig.data[0].line.color == "#1f77b4"
        assert fig.data[1].line.color == "#ff7f0e"

    def test_create_comparison_chart_layout(self):
        """Test comparison chart layout configuration."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        strategies = {
            "Strategy A": pd.DataFrame(
                {"cumulative_return": [1.0 + i * 0.01 for i in range(10)]},
                index=dates,
            ),
        }

        fig = ChartUtils.create_comparison_chart(strategies)

        assert fig.layout.height == 600
        assert fig.layout.hovermode == "x unified"
        assert fig.layout.xaxis.title.text == "Date"
        assert "Cumulative Return" in fig.layout.yaxis.title.text


class TestCreateEquityCurve:
    """Tests for ChartUtils.create_equity_curve method."""

    def test_create_equity_curve_basic(self):
        """Test basic equity curve creation."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(10)], index=dates)

        fig = ChartUtils.create_equity_curve(equity)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].name == "Equity"

    def test_create_equity_curve_with_trades(self):
        """Test equity curve with trade markers."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(10)], index=dates)

        trades = pd.DataFrame(
            {
                "timestamp": [dates[2], dates[5], dates[7]],
                "type": ["entry", "exit", "entry"],
                "equity": [10200, 10500, 10700],
            }
        )

        fig = ChartUtils.create_equity_curve(equity, trades=trades)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3

    def test_create_equity_curve_with_entry_trades_only(self):
        """Test equity curve with only entry trades."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(10)], index=dates)

        trades = pd.DataFrame(
            {
                "timestamp": [dates[2], dates[5]],
                "type": ["entry", "entry"],
                "equity": [10200, 10500],
            }
        )

        fig = ChartUtils.create_equity_curve(equity, trades=trades)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_create_equity_curve_with_exit_trades_only(self):
        """Test equity curve with only exit trades."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(10)], index=dates)

        trades = pd.DataFrame(
            {
                "timestamp": [dates[3], dates[6]],
                "type": ["exit", "exit"],
                "equity": [10300, 10600],
            }
        )

        fig = ChartUtils.create_equity_curve(equity, trades=trades)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_create_equity_curve_empty_trades(self):
        """Test equity curve with empty trades DataFrame."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(10)], index=dates)

        trades = pd.DataFrame(columns=["timestamp", "type", "equity"])

        fig = ChartUtils.create_equity_curve(equity, trades=trades)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_create_equity_curve_no_trades(self):
        """Test equity curve without trades parameter."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(10)], index=dates)

        fig = ChartUtils.create_equity_curve(equity, trades=None)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_create_equity_curve_custom_title(self):
        """Test equity curve with custom title."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(10)], index=dates)

        fig = ChartUtils.create_equity_curve(equity, title="Custom Equity")

        assert fig.layout.title.text == "Custom Equity"

    def test_create_equity_curve_layout(self):
        """Test equity curve layout configuration."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(10)], index=dates)

        fig = ChartUtils.create_equity_curve(equity)

        assert fig.layout.height == 500
        assert fig.layout.hovermode == "x unified"
        assert fig.layout.xaxis.title.text == "Date"
        assert fig.layout.yaxis.title.text == "Equity ($)"

    def test_create_equity_curve_fill_area(self):
        """Test equity curve has fill area."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(10)], index=dates)

        fig = ChartUtils.create_equity_curve(equity)

        assert fig.data[0].fill == "tozeroy"
        assert fig.data[0].fillcolor == "rgba(0, 100, 255, 0.1)"


class TestCreateDrawdownChart:
    """Tests for ChartUtils.create_drawdown_chart method."""

    def test_create_drawdown_chart_basic(self):
        """Test basic drawdown chart creation."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        equity = pd.Series(
            [10000, 10100, 10200, 9900, 9800, 10000, 10300, 10200, 10400, 10500], index=dates
        )

        fig = ChartUtils.create_drawdown_chart(equity)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_create_drawdown_chart_calculation(self):
        """Test drawdown chart calculates drawdowns correctly."""
        dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
        equity = pd.Series([10000, 10000, 9000, 9000, 10000], index=dates)

        fig = ChartUtils.create_drawdown_chart(equity)

        drawdown_values = fig.data[0].y
        assert drawdown_values[0] == 0.0
        assert drawdown_values[1] == 0.0
        assert abs(drawdown_values[2] - (-10.0)) < 0.01
        assert abs(drawdown_values[3] - (-10.0)) < 0.01
        assert drawdown_values[4] == 0.0

    def test_create_drawdown_chart_custom_title(self):
        """Test drawdown chart with custom title."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(10)], index=dates)

        fig = ChartUtils.create_drawdown_chart(equity, title="Custom Drawdown")

        assert fig.layout.title.text == "Custom Drawdown"

    def test_create_drawdown_chart_layout(self):
        """Test drawdown chart layout configuration."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(10)], index=dates)

        fig = ChartUtils.create_drawdown_chart(equity)

        assert fig.layout.height == 400
        assert fig.layout.hovermode == "x unified"
        assert fig.layout.xaxis.title.text == "Date"
        assert fig.layout.yaxis.title.text == "Drawdown (%)"

    def test_create_drawdown_chart_fill_area(self):
        """Test drawdown chart has fill area."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(10)], index=dates)

        fig = ChartUtils.create_drawdown_chart(equity)

        assert fig.data[0].fill == "tozeroy"
        assert fig.data[0].fillcolor == "rgba(255, 0, 0, 0.3)"

    def test_create_drawdown_chart_zero_line(self):
        """Test drawdown chart has zero reference line."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(10)], index=dates)

        fig = ChartUtils.create_drawdown_chart(equity)

        assert len(fig.layout.shapes) > 0

    def test_create_drawdown_chart_increasing_equity(self):
        """Test drawdown chart with continuously increasing equity."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(10)], index=dates)

        fig = ChartUtils.create_drawdown_chart(equity)

        drawdown_values = fig.data[0].y
        assert all(v == 0.0 for v in drawdown_values)


class TestCreateReturnsDistribution:
    """Tests for ChartUtils.create_returns_distribution method."""

    def test_create_returns_distribution_basic(self):
        """Test basic returns distribution chart creation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.01, 0.02, 0.01])

        fig = ChartUtils.create_returns_distribution(returns)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Histogram)

    def test_create_returns_distribution_custom_title(self):
        """Test returns distribution with custom title."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])

        fig = ChartUtils.create_returns_distribution(
            returns,
            title="Custom Returns",
        )

        assert fig.layout.title.text == "Custom Returns"

    def test_create_returns_distribution_custom_bins(self):
        """Test returns distribution with custom bin count."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])

        fig = ChartUtils.create_returns_distribution(returns, bins=100)

        assert isinstance(fig, go.Figure)
        assert fig.data[0].nbinsx == 100

    def test_create_returns_distribution_default_bins(self):
        """Test returns distribution uses default bin count."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])

        fig = ChartUtils.create_returns_distribution(returns)

        assert fig.data[0].nbinsx == 50

    def test_create_returns_distribution_layout(self):
        """Test returns distribution layout configuration."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])

        fig = ChartUtils.create_returns_distribution(returns)

        assert fig.layout.height == 400
        assert fig.layout.showlegend is False
        assert fig.layout.xaxis.title.text == "Return (%)"
        assert fig.layout.yaxis.title.text == "Frequency"

    def test_create_returns_distribution_mean_line(self):
        """Test returns distribution has mean reference line."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])

        fig = ChartUtils.create_returns_distribution(returns)

        assert len(fig.layout.shapes) > 0

    def test_create_returns_distribution_histogram_properties(self):
        """Test returns distribution histogram properties."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])

        fig = ChartUtils.create_returns_distribution(returns)

        assert fig.data[0].marker.color == "blue"
        assert fig.data[0].opacity == 0.7


class TestChartUtilsIntegration:
    """Integration tests for ChartUtils methods."""

    def test_apply_theme_to_comparison_chart(self):
        """Test applying theme to comparison chart."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        strategies = {
            "Strategy A": pd.DataFrame(
                {"cumulative_return": [1.0 + i * 0.01 for i in range(10)]},
                index=dates,
            ),
        }

        fig = ChartUtils.create_comparison_chart(strategies)
        fig = ChartUtils.apply_theme(fig, theme="dark")

        assert fig.layout.paper_bgcolor == "#1e1e1e"

    def test_apply_theme_to_equity_curve(self):
        """Test applying theme to equity curve."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(10)], index=dates)

        fig = ChartUtils.create_equity_curve(equity)
        fig = ChartUtils.apply_theme(fig, theme="light")

        assert fig.layout.paper_bgcolor == "#ffffff"

    def test_apply_theme_to_drawdown_chart(self):
        """Test applying theme to drawdown chart."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(10)], index=dates)

        fig = ChartUtils.create_drawdown_chart(equity)
        fig = ChartUtils.apply_theme(fig, theme="dark")

        assert fig.layout.paper_bgcolor == "#1e1e1e"

    def test_apply_theme_to_returns_distribution(self):
        """Test applying theme to returns distribution."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])

        fig = ChartUtils.create_returns_distribution(returns)
        fig = ChartUtils.apply_theme(fig, theme="light")

        assert fig.layout.paper_bgcolor == "#ffffff"

    def test_export_comparison_chart(self, tmp_path):
        """Test exporting comparison chart."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        strategies = {
            "Strategy A": pd.DataFrame(
                {"cumulative_return": [1.0 + i * 0.01 for i in range(10)]},
                index=dates,
            ),
        }

        fig = ChartUtils.create_comparison_chart(strategies)
        output_file = tmp_path / "comparison.html"

        ChartUtils.export_chart(fig, output_file)

        assert output_file.exists()

    def test_export_equity_curve(self, tmp_path):
        """Test exporting equity curve."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(10)], index=dates)

        fig = ChartUtils.create_equity_curve(equity)
        output_file = tmp_path / "equity.json"

        ChartUtils.export_chart(fig, output_file, file_format="json")

        assert output_file.exists()

    def test_full_workflow_comparison_chart(self, tmp_path):
        """Test full workflow: create, theme, export comparison chart."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        strategies = {
            "Strategy A": pd.DataFrame(
                {"cumulative_return": [1.0 + i * 0.01 for i in range(10)]},
                index=dates,
            ),
            "Strategy B": pd.DataFrame(
                {"cumulative_return": [1.0 + i * 0.015 for i in range(10)]},
                index=dates,
            ),
        }

        fig = ChartUtils.create_comparison_chart(
            strategies,
            title="Performance Comparison",
        )
        fig = ChartUtils.apply_theme(fig, theme="dark")
        output_file = tmp_path / "comparison_full.html"

        ChartUtils.export_chart(fig, output_file)

        assert output_file.exists()
        assert len(fig.data) == 2

    def test_full_workflow_equity_curve(self, tmp_path):
        """Test full workflow: create, theme, export equity curve."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(10)], index=dates)

        trades = pd.DataFrame(
            {
                "timestamp": [dates[2], dates[7]],
                "type": ["entry", "exit"],
                "equity": [10200, 10700],
            }
        )

        fig = ChartUtils.create_equity_curve(equity, trades=trades, title="My Strategy")
        fig = ChartUtils.apply_theme(fig, theme="light")
        output_file = tmp_path / "equity_full.html"

        ChartUtils.export_chart(fig, output_file)

        assert output_file.exists()
        assert len(fig.data) == 3
