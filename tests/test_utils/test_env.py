"""Tests for utils.env module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from ordinis.utils.env import _get_user_env, get_alpaca_credentials


class TestGetUserEnv:
    """Tests for _get_user_env function."""

    @patch("ordinis.utils.env.subprocess.run")
    def test_get_user_env_success(self, mock_run):
        """Test successful user env retrieval."""
        mock_run.return_value = MagicMock(stdout="test_value\n", returncode=0)
        result = _get_user_env("TEST_VAR")
        assert result == "test_value"
        mock_run.assert_called_once()

    @patch("ordinis.utils.env.subprocess.run")
    def test_get_user_env_empty(self, mock_run):
        """Test empty user env value."""
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        result = _get_user_env("EMPTY_VAR")
        assert result == ""

    @patch("ordinis.utils.env.subprocess.run")
    def test_get_user_env_exception(self, mock_run):
        """Test exception handling."""
        mock_run.side_effect = Exception("PowerShell error")
        result = _get_user_env("ERROR_VAR")
        assert result == ""


class TestGetAlpacaCredentials:
    """Tests for get_alpaca_credentials function."""

    @patch("ordinis.utils.env._get_user_env")
    def test_get_credentials_from_user_env(self, mock_get_user_env):
        """Test getting credentials from Windows User env."""
        mock_get_user_env.side_effect = lambda name: {
            "APCA_API_KEY_ID": "user_key",
            "APCA_API_SECRET_KEY": "user_secret",
        }.get(name, "")

        api_key, api_secret = get_alpaca_credentials()
        assert api_key == "user_key"
        assert api_secret == "user_secret"

    @patch("ordinis.utils.env._get_user_env")
    def test_get_credentials_fallback_to_process_env(self, mock_get_user_env):
        """Test fallback to process environment."""
        mock_get_user_env.return_value = ""

        with patch.dict(
            os.environ,
            {
                "APCA_API_KEY_ID": "process_key",
                "APCA_API_SECRET_KEY": "process_secret",
            },
        ):
            api_key, api_secret = get_alpaca_credentials()
            assert api_key == "process_key"
            assert api_secret == "process_secret"

    @patch("ordinis.utils.env._get_user_env")
    def test_get_credentials_fallback_to_alpaca_env(self, mock_get_user_env):
        """Test fallback to ALPACA_* environment variables."""
        mock_get_user_env.return_value = ""

        with patch.dict(
            os.environ,
            {
                "ALPACA_API_KEY": "alpaca_key",
                "ALPACA_API_SECRET": "alpaca_secret",
            },
            clear=True,
        ):
            api_key, api_secret = get_alpaca_credentials()
            assert api_key == "alpaca_key"
            assert api_secret == "alpaca_secret"

    @patch("ordinis.utils.env._get_user_env")
    def test_get_credentials_empty(self, mock_get_user_env):
        """Test when no credentials found."""
        mock_get_user_env.return_value = ""

        with patch.dict(os.environ, {}, clear=True):
            api_key, api_secret = get_alpaca_credentials()
            assert api_key == ""
            assert api_secret == ""

    @patch("ordinis.utils.env._get_user_env")
    def test_get_credentials_partial_user_env(self, mock_get_user_env):
        """Test partial user env with process fallback."""
        mock_get_user_env.side_effect = lambda name: {
            "APCA_API_KEY_ID": "user_key",
            "APCA_API_SECRET_KEY": "",  # Empty secret in user env
        }.get(name, "")

        with patch.dict(
            os.environ,
            {"APCA_API_SECRET_KEY": "process_secret"},
        ):
            api_key, api_secret = get_alpaca_credentials()
            assert api_key == "user_key"
            assert api_secret == "process_secret"
