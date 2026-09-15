"""Evidence-aware investigation agent for ThreatLens AI.

This module does not run detection or calculate a new risk score. It retrieves
facts produced by the ML pipeline and turns them into a structured incident
report. Missing evidence is represented explicitly instead of inferred.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


_EVENTS: Dict[str, Dict[str, Any]] = {}
_TRAFFIC: Dict[str, Dict[str, Any]] = {}


def register_event(event: Dict[str, Any], flow: Dict[str, Any]) -> None:
    """Register the actual predictor output and raw input for later evidence retrieval."""
    event_id = event["event_id"]
    _EVENTS[event_id] = dict(event)
    _TRAFFIC[event_id] = dict(flow)


def get_alert(event_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve one stored security event by its real event ID."""
    return _EVENTS.get(event_id)


def get_traffic_details(event_id: str) -> Dict[str, Any]:
    """Retrieve the raw flow submitted to the predictor for this event."""
    flow = _TRAFFIC.get(event_id)
    if flow is None:
        return {"available": False, "message": "Traffic details are unavailable for this event."}
    return {"available": True, "event_id": event_id, "flow": dict(flow)}


def get_ip_history(source_ip: str) -> Dict[str, Any]:
    """Retrieve previously registered events for the same source IP."""
    events = [event for event in _EVENTS.values() if event.get("source_ip") == source_ip]
    return {
        "available": bool(events),
        "source_ip": source_ip,
        "event_count": len(events),
        "events": [
            {
                "event_id": event["event_id"],
                "timestamp": event["timestamp"],
                "destination_ip": event["destination_ip"],
                "attack_type": event["attack_type"],
                "risk_score": event["risk_score"],
                "severity": event["severity"],
            }
            for event in events
        ],
        "message": None if events else "IP history is unavailable for this source IP.",
    }


def get_attack_evidence(event_id: str) -> Dict[str, Any]:
    """Retrieve only attack evidence present in the ML security event."""
    event = get_alert(event_id)
    if event is None:
        return {"available": False, "message": "Attack evidence is unavailable for this event."}
    evidence = [
        f"Binary classifier suspicious probability: {event['suspicious_probability']:.1%}.",
        f"Attack classifier predicted {event['attack_type']} with {event['attack_probability']:.1%} probability.",
        f"Anomaly detector score: {event['anomaly_score']:.1%} ({'anomaly flagged' if event['is_anomaly'] else 'not flagged'}).",
    ]
    return {"available": True, "event_id": event_id, "evidence": evidence}


def get_shap_explanation(event_id: str) -> Dict[str, Any]:
    """Retrieve the predictor's stored explanation features without generating new ones."""
    event = get_alert(event_id)
    if event is None:
        return {"available": False, "message": "SHAP/explanation data is unavailable for this event."}
    features = event.get("top_features") or []
    if not features:
        return {"available": False, "message": "SHAP/explanation features are unavailable for this event."}
    return {
        "available": True,
        "event_id": event_id,
        "method": event.get("explanation_method") or "predictor-provided feature explanation",
        "top_features": features,
    }


def calculate_risk(event_id: str) -> Dict[str, Any]:
    """Return the deterministic risk result already produced by the ML pipeline."""
    event = get_alert(event_id)
    if event is None:
        return {"available": False, "message": "Risk assessment is unavailable for this event."}
    return {
        "available": True,
        "event_id": event_id,
        "risk_score": event["risk_score"],
        "severity": event["severity"],
        "source": "src.predictor -> src.risk_engine",
    }


def _recommendations(event: Dict[str, Any]) -> List[str]:
    actions = []
    if event["severity"] in {"CRITICAL", "HIGH"}:
        actions.append(f"Isolate or block source IP {event['source_ip']} pending analyst review.")
        actions.append(f"Monitor destination IP {event['destination_ip']} for follow-on activity.")
        actions.append("Escalate this incident for immediate SOC review.")
    elif event["severity"] == "MEDIUM":
        actions.append(f"Monitor source IP {event['source_ip']} and enrich with nearby traffic.")
        actions.append(f"Review subsequent traffic to {event['destination_ip']}.")
        actions.append("Escalate if suspicious activity persists or risk increases.")
    else:
        actions.append(f"Keep source IP {event['source_ip']} under observation.")
        actions.append("Review subsequent traffic if the anomaly or suspicion changes.")
        actions.append("No containment action is indicated by the current ML result.")
    return actions


def investigate_alert(event_id: str) -> Dict[str, Any]:
    """Run the single lightweight investigation workflow for a stored event."""
    event = get_alert(event_id)
    if event is None:
        raise KeyError(event_id)

    attack_evidence = get_attack_evidence(event_id)
    explanation = get_shap_explanation(event_id)
    ip_history = get_ip_history(event["source_ip"])
    risk = calculate_risk(event_id)

    evidence = []
    if attack_evidence["available"]:
        evidence.extend(attack_evidence["evidence"])
    else:
        evidence.append(attack_evidence["message"])
    if explanation["available"]:
        for feature in explanation["top_features"][:3]:
            evidence.append(
                f"Feature {feature['feature']} {feature['direction'].replace('_', ' ')} "
                f"(importance {feature['importance']:.4f})."
            )
    else:
        evidence.append(explanation["message"])
    if ip_history["available"]:
        evidence.append(
            f"Recorded source history contains {ip_history['event_count']} event(s) for {event['source_ip']}."
        )
    else:
        evidence.append(ip_history["message"])

    reasoning = [
        f"The ML binary classifier marked this flow as {'suspicious' if event['is_suspicious'] else 'not suspicious'}.",
        f"The ML taxonomy identifies {event['attack_type']}; the agent does not replace that classification.",
    ]
    if event["is_anomaly"]:
        reasoning.append("The anomaly detector flagged this flow as behaviorally unusual.")
    else:
        reasoning.append("The anomaly detector did not flag this flow, so anomaly evidence is limited to its score.")

    return {
        "event_id": event_id,
        "summary": f"{event['attack_type']} activity detected from {event['source_ip']} to {event['destination_ip']}.",
        "attack_type": event["attack_type"],
        "risk_score": risk["risk_score"],
        "severity": risk["severity"],
        "confidence": round(max(event["suspicious_probability"], event["attack_probability"]), 4),
        "evidence": evidence,
        "reasoning": reasoning,
        "recommended_actions": _recommendations(event),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
