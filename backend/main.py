from collections import Counter
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from backend.investigation_agent import investigate_alert, register_event
from src.predictor import predict_threat


class TrafficRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    flow: Optional[Dict[str, Any]] = Field(default=None, description="Optional raw flow wrapper")

    def to_flow(self) -> Dict[str, Any]:
        if self.flow is not None:
            return self.flow
        return self.model_dump(exclude_none=True, exclude={"flow"})


class InvestigationRequest(BaseModel):
    event_id: Optional[str] = Field(default=None, description="Previously detected event ID")
    flow: Optional[Dict[str, Any]] = Field(default=None, description="Raw flow to detect and investigate")


class IncidentReport(BaseModel):
    event_id: str
    summary: str
    attack_type: str
    risk_score: int = Field(ge=0, le=100)
    severity: str
    confidence: float = Field(ge=0, le=1)
    evidence: List[str]
    reasoning: List[str]
    recommended_actions: List[str]
    generated_at: str


class TopFeature(BaseModel):
    feature: str
    importance: float
    direction: str


class SecurityEvent(BaseModel):
    event_id: str
    timestamp: str
    source_ip: str
    destination_ip: str
    source_port: Optional[int] = None
    destination_port: Optional[int] = None
    protocol: str
    is_suspicious: bool
    suspicious_probability: float = Field(ge=0, le=1)
    is_anomaly: bool
    anomaly_score: float = Field(ge=0, le=1)
    attack_type: str
    attack_probability: float = Field(ge=0, le=1)
    risk_score: int = Field(ge=0, le=100)
    severity: str
    explanation_method: Optional[str] = None
    top_features: List[TopFeature] = Field(default_factory=list)


class Investigation(BaseModel):
    summary: str
    confidence: float = Field(ge=0, le=1)
    evidence: List[str]
    recommendation: str
    priority: str
    generated_at: str


class InvestigationResult(BaseModel):
    event: SecurityEvent
    investigation: Investigation


class AlertList(BaseModel):
    alerts: List[SecurityEvent]
    count: int


class DashboardStats(BaseModel):
    total_traffic: int
    normal_count: int
    suspicious_count: int
    critical_count: int
    average_risk_score: float
    attack_distribution: Dict[str, int]
    recent_alerts: List[SecurityEvent]


app = FastAPI(
    title="ThreatLens AI API",
    description="Evidence-aware network security prediction and investigation API",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_events: List[SecurityEvent] = []
_investigations: Dict[str, Investigation] = {}
_MAX_EVENTS = 100


def _port(flow: Dict[str, Any], *names: str) -> Optional[int]:
    for name in names:
        value = flow.get(name)
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _standardize_event(flow: Dict[str, Any]) -> SecurityEvent:
    try:
        prediction = predict_threat(flow)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    explanation = prediction.get("explanation") or {}
    event = SecurityEvent(
        event_id=prediction["event_id"],
        timestamp=prediction["timestamp"],
        source_ip=prediction["source_ip"],
        destination_ip=prediction["destination_ip"],
        source_port=_port(flow, "source_port", "src_port", "sport"),
        destination_port=_port(flow, "destination_port", "dst_port", "dport"),
        protocol=str(prediction["protocol"]).upper(),
        is_suspicious=prediction["is_suspicious"],
        suspicious_probability=prediction["suspicious_probability"],
        is_anomaly=prediction["is_anomaly"],
        anomaly_score=prediction["anomaly_score"],
        attack_type=prediction["attack_type"],
        attack_probability=prediction["attack_probability"],
        risk_score=prediction["risk_score"],
        severity=prediction["severity"],
        explanation_method=explanation.get("method"),
        top_features=explanation.get("top_features", []),
    )
    register_event(event.model_dump(), flow)
    _events.insert(0, event)
    del _events[_MAX_EVENTS:]
    return event


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "ThreatLens AI"}


@app.post("/api/predict", response_model=SecurityEvent)
def predict(request: TrafficRequest) -> SecurityEvent:
    return _standardize_event(request.to_flow())


@app.post("/api/investigate", response_model=IncidentReport)
def investigate(request: InvestigationRequest) -> IncidentReport:
    if request.event_id:
        try:
            return IncidentReport(**investigate_alert(request.event_id))
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Security event {request.event_id} is unavailable",
            ) from exc
    if request.flow:
        event = _standardize_event(request.flow)
        return IncidentReport(**investigate_alert(event.event_id))
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Provide either event_id or flow for investigation",
    )


@app.get("/api/alerts", response_model=AlertList)
def alerts() -> AlertList:
    suspicious = [event for event in _events if event.is_suspicious]
    return AlertList(alerts=suspicious[:25], count=len(suspicious[:25]))


@app.get("/api/alerts/{event_id}", response_model=SecurityEvent)
def alert(event_id: str) -> SecurityEvent:
    for event in _events:
        if event.event_id == event_id and event.is_suspicious:
            return event
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Suspicious event not found")


@app.get("/api/dashboard/stats", response_model=DashboardStats)
def dashboard_stats() -> DashboardStats:
    suspicious = [event for event in _events if event.is_suspicious]
    distribution = Counter(event.attack_type for event in suspicious)
    return DashboardStats(
        total_traffic=len(_events),
        normal_count=sum(not event.is_suspicious for event in _events),
        suspicious_count=len(suspicious),
        critical_count=sum(event.severity == "CRITICAL" for event in _events),
        average_risk_score=round(
            sum(event.risk_score for event in _events) / len(_events), 1
        ) if _events else 0.0,
        attack_distribution=dict(sorted(distribution.items())),
        recent_alerts=suspicious[:10],
    )


DEMO_FLOW = {
    "dur": 0.000009, "proto": "udp", "service": "dns", "state": "INT",
    "spkts": 2, "dpkts": 0, "sbytes": 114, "dbytes": 0, "rate": 111111.1072,
    "sttl": 254, "dttl": 0, "sload": 50666664.0, "dload": 0.0, "sloss": 0,
    "dloss": 0, "sinpkt": 0.009, "dinpkt": 0.0, "sjit": 0.0, "djit": 0.0,
    "swin": 0, "stcpb": 0, "dtcpb": 0, "dwin": 0, "tcprtt": 0.0,
    "synack": 0.0, "ackdat": 0.0, "smean": 57, "dmean": 0, "trans_depth": 0,
    "response_body_len": 0, "ct_srv_src": 2, "ct_state_ttl": 2, "ct_dst_ltm": 1,
    "ct_src_dport_ltm": 1, "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 2,
    "is_ftp_login": 0, "ct_ftp_cmd": 0, "ct_flw_http_mthd": 0, "ct_src_ltm": 1,
    "ct_srv_dst": 2, "is_sm_ips_ports": 0,
}


@app.get("/api/sample", response_model=InvestigationResult)
def sample_event() -> InvestigationResult:
    event = _standardize_event(DEMO_FLOW)
    report = IncidentReport(**investigate_alert(event.event_id))
    investigation = Investigation(
        summary=report.summary,
        confidence=report.confidence,
        evidence=report.evidence,
        recommendation=report.recommended_actions[0],
        priority="immediate" if report.severity in {"HIGH", "CRITICAL"} else "monitor",
        generated_at=report.generated_at,
    )
    return InvestigationResult(event=event, investigation=investigation)
