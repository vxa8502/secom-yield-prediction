"""
Actionable recommendations for SECOM yield predictions.

Generates context-aware, tiered recommendations based on:
- Prediction outcome (PASS/FAIL)
- Probability distance from threshold
- Confidence level
- SHAP feature contributions (when available)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


SeverityLevel = Literal['low', 'medium', 'high', 'critical']
ConfidenceTier = Literal['low', 'medium', 'high']

# Severity thresholds: distance from classification threshold
# These define how far above threshold triggers each severity level
SEVERITY_THRESHOLDS = {
    'critical': 0.30,  # probability >= threshold + 0.30
    'high': 0.15,      # probability >= threshold + 0.15
    'medium': 0.05,    # probability >= threshold + 0.05
}

# Confidence tier thresholds
CONFIDENCE_THRESHOLDS = {
    'high': 0.80,      # confidence >= 0.80
    'medium': 0.60,    # confidence >= 0.60
}

# Pass prediction margins
PASS_MARGIN_STRONG = 0.20   # High confidence pass
PASS_MARGIN_CAUTION = 0.10  # Near-threshold caution


@dataclass
class RecommendationResult:
    """Structured recommendation output."""
    severity: SeverityLevel
    confidence_tier: ConfidenceTier
    primary_action: str
    secondary_actions: list[str] = field(default_factory=list)
    feature_insights: list[dict] | None = None
    process_guidance: str = ""


def get_confidence_tier(confidence: float) -> ConfidenceTier:
    """
    Classify prediction confidence into tiers.

    Args:
        confidence: Model confidence (max probability)

    Returns:
        Confidence tier: 'low', 'medium', or 'high'
    """
    if confidence >= CONFIDENCE_THRESHOLDS['high']:
        return 'high'
    elif confidence >= CONFIDENCE_THRESHOLDS['medium']:
        return 'medium'
    else:
        return 'low'


def get_severity_level(
    probability: float,
    threshold: float,
    prediction: int
) -> SeverityLevel:
    """
    Assess severity based on probability distance from threshold.

    For FAIL predictions, severity increases with distance above threshold.
    For PASS predictions, always returns 'low' (no action urgency).

    Args:
        probability: Failure probability
        threshold: Classification threshold
        prediction: Binary prediction (0=PASS, 1=FAIL)

    Returns:
        Severity level
    """
    if prediction == 0:  # PASS
        return 'low'

    # FAIL prediction - assess severity based on distance from threshold
    distance = probability - threshold

    if distance >= SEVERITY_THRESHOLDS['critical']:
        return 'critical'
    elif distance >= SEVERITY_THRESHOLDS['high']:
        return 'high'
    elif distance >= SEVERITY_THRESHOLDS['medium']:
        return 'medium'
    else:
        return 'low'


def format_feature_insights(
    shap_df: pd.DataFrame | None,
    top_n: int = 3
) -> list[dict] | None:
    """
    Format top SHAP contributors into human-readable insights.

    Args:
        shap_df: DataFrame with 'feature' and 'contribution' columns
        top_n: Number of top features to include

    Returns:
        List of insight dicts with 'feature', 'contribution', 'direction'
        or None if no SHAP data available
    """
    if shap_df is None or shap_df.empty:
        return None

    if 'feature' not in shap_df.columns or 'contribution' not in shap_df.columns:
        return None

    insights = []
    # Sort by absolute contribution
    df = shap_df.copy()
    df['abs_contribution'] = df['contribution'].abs()
    df = df.sort_values('abs_contribution', ascending=False).head(top_n)

    for _, row in df.iterrows():
        contribution = row['contribution']
        direction = 'increasing' if contribution > 0 else 'decreasing'
        insights.append({
            'feature': row['feature'],
            'contribution': abs(contribution),
            'direction': direction,
            'description': f"{row['feature']} is {direction} failure risk"
        })

    return insights if insights else None


def _get_fail_recommendations(
    severity: SeverityLevel,
    confidence_tier: ConfidenceTier,
    feature_insights: list[dict] | None
) -> tuple[str, list[str], str]:
    """Get recommendations for FAIL predictions."""
    if severity == 'critical':
        primary = "IMMEDIATE ACTION REQUIRED: Quarantine wafer and halt batch processing"
        secondary = [
            "Conduct full inspection of production line",
            "Review all upstream process parameters",
            "Document anomaly for root cause analysis"
        ]
        guidance = (
            "This wafer shows strong failure indicators. "
            "Proceeding without intervention has high defect risk."
        )
    elif severity == 'high':
        primary = "Flag for priority inspection before proceeding"
        secondary = [
            "Review recent process parameter drift",
            "Compare with known-good batch parameters",
            "Consider sampling additional wafers from batch"
        ]
        guidance = (
            "Failure probability is significantly above threshold. "
            "Manual review recommended before continuing."
        )
    elif severity == 'medium':
        primary = "Schedule for detailed quality inspection"
        secondary = [
            "Monitor subsequent wafers in batch",
            "Log for trend analysis"
        ]
        guidance = (
            "Moderate failure risk detected. "
            "Standard inspection protocol applies."
        )
    else:  # low
        primary = "Standard inspection recommended"
        secondary = [
            "Continue normal processing with monitoring"
        ]
        guidance = (
            "Marginally above threshold. "
            "May benefit from additional sampling."
        )

    # Add confidence-based caveats
    if confidence_tier == 'low':
        secondary.append("NOTE: Low model confidence - consider re-measurement")
        guidance += " Low confidence suggests ambiguous sensor readings."

    # Add feature-specific guidance if available
    if feature_insights and len(feature_insights) > 0:
        top_feature = feature_insights[0]
        secondary.append(
            f"Key driver: {top_feature['feature']} ({top_feature['direction']} risk)"
        )

    return primary, secondary, guidance


def _get_pass_recommendations(
    confidence_tier: ConfidenceTier,
    probability: float,
    threshold: float
) -> tuple[str, list[str], str]:
    """Get recommendations for PASS predictions."""
    margin = threshold - probability

    if confidence_tier == 'high' and margin > PASS_MARGIN_STRONG:
        primary = "Proceed with standard processing"
        secondary = [
            "No additional inspection required"
        ]
        guidance = "Strong pass indicators. Normal production flow."
    elif margin < PASS_MARGIN_CAUTION:
        primary = "Proceed with caution - near threshold"
        secondary = [
            "Consider spot-check inspection",
            "Monitor subsequent wafers closely"
        ]
        guidance = (
            "Probability is close to threshold. "
            "Small process variations could push similar wafers to fail."
        )
    else:
        primary = "Proceed to next manufacturing stage"
        secondary = [
            "Continue standard monitoring"
        ]
        guidance = "Within expected pass range."

    if confidence_tier == 'low':
        secondary.append("NOTE: Low confidence - verify sensor calibration")

    return primary, secondary, guidance


def generate_recommendations(
    prediction: int,
    probability: float,
    threshold: float,
    confidence: float,
    shap_contributions: pd.DataFrame | None = None
) -> RecommendationResult:
    """
    Generate tiered, actionable recommendations based on prediction.

    Args:
        prediction: Binary prediction (0=PASS, 1=FAIL)
        probability: Failure probability from model
        threshold: Classification threshold used
        confidence: Model confidence (max probability)
        shap_contributions: Optional DataFrame with SHAP values
                           Must have 'feature' and 'contribution' columns

    Returns:
        RecommendationResult with severity, actions, and guidance

    Example:
        >>> result = generate_recommendations(
        ...     prediction=1,
        ...     probability=0.85,
        ...     threshold=0.35,
        ...     confidence=0.85
        ... )
        >>> print(result.severity)  # 'critical'
        >>> print(result.primary_action)  # 'IMMEDIATE ACTION...'
    """
    severity = get_severity_level(probability, threshold, prediction)
    confidence_tier = get_confidence_tier(confidence)
    feature_insights = format_feature_insights(shap_contributions)

    if prediction == 1:  # FAIL
        primary, secondary, guidance = _get_fail_recommendations(
            severity, confidence_tier, feature_insights
        )
    else:  # PASS
        primary, secondary, guidance = _get_pass_recommendations(
            confidence_tier, probability, threshold
        )

    return RecommendationResult(
        severity=severity,
        confidence_tier=confidence_tier,
        primary_action=primary,
        secondary_actions=secondary,
        feature_insights=feature_insights,
        process_guidance=guidance
    )


# Severity display configuration
SEVERITY_CONFIG = {
    'critical': {'color': '#DC2626', 'icon': '!!!', 'label': 'CRITICAL'},
    'high': {'color': '#EA580C', 'icon': '!!', 'label': 'HIGH'},
    'medium': {'color': '#F59E0B', 'icon': '!', 'label': 'MEDIUM'},
    'low': {'color': '#10B981', 'icon': '', 'label': 'LOW'},
}
