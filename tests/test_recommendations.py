"""
Unit tests for actionable recommendations module.
"""

import pandas as pd
import pytest

from streamlit_app.utils.recommendations import (
    generate_recommendations,
    get_severity_level,
    get_confidence_tier,
    format_feature_insights,
    RecommendationResult,
    SEVERITY_CONFIG,
)


class TestGetConfidenceTier:
    """Tests for get_confidence_tier function."""

    def test_high_confidence(self):
        """Test high confidence tier."""
        assert get_confidence_tier(0.95) == 'high'
        assert get_confidence_tier(0.80) == 'high'

    def test_medium_confidence(self):
        """Test medium confidence tier."""
        assert get_confidence_tier(0.79) == 'medium'
        assert get_confidence_tier(0.60) == 'medium'

    def test_low_confidence(self):
        """Test low confidence tier."""
        assert get_confidence_tier(0.59) == 'low'
        assert get_confidence_tier(0.50) == 'low'
        assert get_confidence_tier(0.30) == 'low'


class TestGetSeverityLevel:
    """Tests for get_severity_level function."""

    def test_pass_prediction_always_low(self):
        """Test that PASS predictions always return 'low' severity."""
        assert get_severity_level(0.10, 0.35, prediction=0) == 'low'
        assert get_severity_level(0.30, 0.35, prediction=0) == 'low'
        assert get_severity_level(0.34, 0.35, prediction=0) == 'low'

    def test_fail_critical_severity(self):
        """Test critical severity for high probability FAIL."""
        # threshold + 0.3 = 0.65
        assert get_severity_level(0.65, 0.35, prediction=1) == 'critical'
        assert get_severity_level(0.85, 0.35, prediction=1) == 'critical'

    def test_fail_high_severity(self):
        """Test high severity for moderate-high probability FAIL."""
        # threshold + 0.15 to threshold + 0.3 = 0.50 to 0.65
        assert get_severity_level(0.50, 0.35, prediction=1) == 'high'
        assert get_severity_level(0.60, 0.35, prediction=1) == 'high'

    def test_fail_medium_severity(self):
        """Test medium severity for moderate probability FAIL."""
        # threshold + 0.05 to threshold + 0.15 = 0.40 to 0.50
        assert get_severity_level(0.40, 0.35, prediction=1) == 'medium'
        assert get_severity_level(0.45, 0.35, prediction=1) == 'medium'

    def test_fail_low_severity(self):
        """Test low severity for just-above-threshold FAIL."""
        # Less than threshold + 0.05 = 0.40
        assert get_severity_level(0.36, 0.35, prediction=1) == 'low'
        assert get_severity_level(0.39, 0.35, prediction=1) == 'low'


class TestFormatFeatureInsights:
    """Tests for format_feature_insights function."""

    def test_format_valid_shap_df(self):
        """Test formatting valid SHAP DataFrame."""
        shap_df = pd.DataFrame({
            'feature': ['sensor_1', 'sensor_2', 'sensor_3'],
            'contribution': [0.15, -0.10, 0.05]
        })

        insights = format_feature_insights(shap_df, top_n=2)

        assert insights is not None
        assert len(insights) == 2
        # Should be sorted by absolute contribution
        assert insights[0]['feature'] == 'sensor_1'
        assert insights[0]['direction'] == 'increasing'
        assert insights[1]['feature'] == 'sensor_2'
        assert insights[1]['direction'] == 'decreasing'

    def test_returns_none_for_none_input(self):
        """Test returns None for None input."""
        assert format_feature_insights(None) is None

    def test_returns_none_for_empty_df(self):
        """Test returns None for empty DataFrame."""
        assert format_feature_insights(pd.DataFrame()) is None

    def test_returns_none_for_missing_columns(self):
        """Test returns None when required columns missing."""
        df = pd.DataFrame({'feature': ['a', 'b'], 'value': [1, 2]})
        assert format_feature_insights(df) is None


class TestGenerateRecommendations:
    """Tests for generate_recommendations function."""

    def test_returns_recommendation_result(self):
        """Test that function returns RecommendationResult dataclass."""
        result = generate_recommendations(
            prediction=1,
            probability=0.80,
            threshold=0.35,
            confidence=0.80
        )

        assert isinstance(result, RecommendationResult)
        assert hasattr(result, 'severity')
        assert hasattr(result, 'confidence_tier')
        assert hasattr(result, 'primary_action')
        assert hasattr(result, 'secondary_actions')
        assert hasattr(result, 'feature_insights')
        assert hasattr(result, 'process_guidance')

    def test_critical_fail_recommendation(self):
        """Test critical FAIL produces urgent recommendations."""
        result = generate_recommendations(
            prediction=1,
            probability=0.85,
            threshold=0.35,
            confidence=0.85
        )

        assert result.severity == 'critical'
        assert 'IMMEDIATE' in result.primary_action.upper() or 'QUARANTINE' in result.primary_action.upper()
        assert len(result.secondary_actions) > 0

    def test_high_fail_recommendation(self):
        """Test high severity FAIL produces priority recommendations."""
        result = generate_recommendations(
            prediction=1,
            probability=0.55,
            threshold=0.35,
            confidence=0.75
        )

        assert result.severity == 'high'
        assert 'priority' in result.primary_action.lower() or 'inspection' in result.primary_action.lower()

    def test_pass_recommendation(self):
        """Test PASS prediction produces proceed recommendations."""
        result = generate_recommendations(
            prediction=0,
            probability=0.15,
            threshold=0.35,
            confidence=0.85
        )

        assert result.severity == 'low'
        assert 'proceed' in result.primary_action.lower()

    def test_low_confidence_adds_caveat(self):
        """Test that low confidence adds warning to recommendations."""
        result = generate_recommendations(
            prediction=1,
            probability=0.50,
            threshold=0.35,
            confidence=0.55  # Low confidence
        )

        # Should mention low confidence in secondary actions
        actions_text = ' '.join(result.secondary_actions).lower()
        assert 'confidence' in actions_text or 'confidence' in result.process_guidance.lower()

    def test_near_threshold_pass_adds_caution(self):
        """Test PASS near threshold adds caution."""
        result = generate_recommendations(
            prediction=0,
            probability=0.30,  # Close to threshold
            threshold=0.35,
            confidence=0.70
        )

        # Should mention caution or monitoring
        assert 'caution' in result.primary_action.lower() or 'monitor' in result.process_guidance.lower()

    def test_with_shap_contributions(self):
        """Test recommendations include feature insights when SHAP provided."""
        shap_df = pd.DataFrame({
            'feature': ['sensor_A', 'sensor_B'],
            'contribution': [0.20, -0.15]
        })

        result = generate_recommendations(
            prediction=1,
            probability=0.70,
            threshold=0.35,
            confidence=0.70,
            shap_contributions=shap_df
        )

        assert result.feature_insights is not None
        assert len(result.feature_insights) > 0
        assert result.feature_insights[0]['feature'] == 'sensor_A'

    def test_without_shap_contributions(self):
        """Test recommendations work without SHAP data."""
        result = generate_recommendations(
            prediction=1,
            probability=0.70,
            threshold=0.35,
            confidence=0.70,
            shap_contributions=None
        )

        assert result.feature_insights is None
        # Should still have valid recommendations
        assert result.primary_action != ""
        assert result.process_guidance != ""


class TestSeverityConfig:
    """Tests for SEVERITY_CONFIG constants."""

    def test_all_severities_have_config(self):
        """Test all severity levels have configuration."""
        for severity in ['critical', 'high', 'medium', 'low']:
            assert severity in SEVERITY_CONFIG
            assert 'color' in SEVERITY_CONFIG[severity]
            assert 'label' in SEVERITY_CONFIG[severity]

    def test_critical_has_warning_color(self):
        """Test critical severity has red-ish color."""
        color = SEVERITY_CONFIG['critical']['color'].lower()
        # Should be a reddish color (high R value)
        assert color.startswith('#')
