"""Tests for LoopConfig and config.yaml loading."""

from __future__ import annotations

from pathlib import Path

import yaml

from agent.loop import _DEFAULT_CONFIG_PATH, LoopConfig, _load_config_dict


class TestLoadConfigDict:
    """Tests for _load_config_dict function."""

    def test_load_config_returns_dict(self) -> None:
        """Test that _load_config_dict returns a dict."""
        result = _load_config_dict()
        assert isinstance(result, dict)

    def test_load_config_contains_agent_section(self) -> None:
        """Test that loaded config contains agent section."""
        result = _load_config_dict()
        # Should have agent-level keys
        expected_keys = ["model", "max_iterations", "max_context_tokens", "temperature", "timezone", "workspace_path"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


class TestLoopConfigDefaults:
    """Tests for LoopConfig default values."""

    def test_config_loads_defaults(self) -> None:
        """Test that LoopConfig() loads defaults from config.yaml."""
        config = LoopConfig()
        assert config.model == "claude-sonnet-4-20250514"
        assert config.max_iterations == 10
        assert config.max_context_tokens == 100000
        assert config.temperature == 0.7
        assert config.timezone == "Asia/Shanghai"
        assert config.workspace_path == "."

    def test_config_override_values(self) -> None:
        """Test that explicit values override defaults."""
        config = LoopConfig(
            model="gpt-4",
            max_iterations=5,
            max_context_tokens=50000,
            temperature=0.5,
            timezone="UTC",
            workspace_path="/tmp",
        )
        assert config.model == "gpt-4"
        assert config.max_iterations == 5
        assert config.max_context_tokens == 50000
        assert config.temperature == 0.5
        assert config.timezone == "UTC"
        assert config.workspace_path == "/tmp"


class TestLoopConfigPartialOverride:
    """Tests for partial override of config values."""

    def test_partial_override_only_overrides_specified(self) -> None:
        """Test that only specified fields are overridden."""
        config = LoopConfig(model="custom-model")
        # model should be overridden
        assert config.model == "custom-model"
        # others should use defaults
        assert config.max_iterations == 10
        assert config.max_context_tokens == 100000
        assert config.temperature == 0.7
        assert config.timezone == "Asia/Shanghai"
        assert config.workspace_path == "."

    def test_partial_override_multiple_fields(self) -> None:
        """Test partial override with multiple fields."""
        config = LoopConfig(max_iterations=20, temperature=1.0)
        assert config.max_iterations == 20
        assert config.temperature == 1.0
        assert config.model == "claude-sonnet-4-20250514"
        assert config.timezone == "Asia/Shanghai"


class TestConfigYamlExists:
    """Tests for config.yaml file presence."""

    def test_config_yaml_exists(self) -> None:
        """Test that config.yaml exists at expected path."""
        assert _DEFAULT_CONFIG_PATH.exists(), f"config.yaml not found at {_DEFAULT_CONFIG_PATH}"

    def test_config_yaml_valid_yaml(self) -> None:
        """Test that config.yaml is valid YAML."""
        with open(_DEFAULT_CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        assert data is not None
        assert "agent" in data

    def test_config_yaml_has_required_keys(self) -> None:
        """Test that config.yaml has all required agent keys."""
        with open(_DEFAULT_CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        agent_config = data["agent"]
        required_keys = ["model", "max_iterations", "max_context_tokens", "temperature", "timezone", "workspace_path"]
        for key in required_keys:
            assert key in agent_config, f"Missing required key in config.yaml: {key}"


class TestConfigYamlValues:
    """Tests for config.yaml values."""

    def test_model_is_string(self) -> None:
        """Test that model is a string."""
        config = LoopConfig()
        assert isinstance(config.model, str)

    def test_max_iterations_is_positive_int(self) -> None:
        """Test that max_iterations is a positive integer."""
        config = LoopConfig()
        assert isinstance(config.max_iterations, int)
        assert config.max_iterations > 0

    def test_max_context_tokens_is_positive_int(self) -> None:
        """Test that max_context_tokens is a positive integer."""
        config = LoopConfig()
        assert isinstance(config.max_context_tokens, int)
        assert config.max_context_tokens > 0

    def test_temperature_is_valid_range(self) -> None:
        """Test that temperature is in valid range."""
        config = LoopConfig()
        assert isinstance(config.temperature, (int, float))
        assert 0.0 <= config.temperature <= 2.0

    def test_timezone_is_string(self) -> None:
        """Test that timezone is a string."""
        config = LoopConfig()
        assert isinstance(config.timezone, str)

    def test_workspace_path_is_string_or_path(self) -> None:
        """Test that workspace_path is a string or Path."""
        config = LoopConfig()
        assert isinstance(config.workspace_path, (str, Path))
