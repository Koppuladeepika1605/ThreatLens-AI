import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const API = 'http://127.0.0.1:8000/api';

function App() {
  const [stats, setStats] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [selected, setSelected] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [investigating, setInvestigating] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [apiError, setApiError] = useState('');
  const [toast, setToast] = useState('');
  const [reviewed, setReviewed] = useState(new Set());

  async function loadDashboard() {
    setLoading(true);
    try {
      const [statsResponse, alertsResponse] = await Promise.all([fetch(`${API}/dashboard/stats`), fetch(`${API}/alerts`)]);
      if (!statsResponse.ok || !alertsResponse.ok) throw new Error('API unavailable');
      setStats(await statsResponse.json());
      setAlerts((await alertsResponse.json()).alerts || []);
      setDemoMode(false);
      setApiError('');
    } catch {
      setStats(null);
      setAlerts([]);
      setDemoMode(true);
      setApiError('Live API unavailable. Live metrics and alerts are currently unavailable.');
    } finally { setLoading(false); }
  }

  useEffect(() => { loadDashboard(); }, []);

  async function selectAlert(alert) {
    setSelected(alert); setReport(null); setInvestigating(true);
    try {
      const response = await fetch(`${API}/investigate`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ event_id: alert.event_id }) });
      if (!response.ok) throw new Error('Investigation unavailable');
      setReport(await response.json());
      setApiError('');
    } catch {
      setApiError('Live investigation unavailable. Showing only the event evidence already in the dashboard.');
      setReport(null);
    } finally { setInvestigating(false); }
  }

  async function loadDemoEvent() {
    try {
      const response = await fetch(`${API}/sample`);
      if (!response.ok) throw new Error();
      const data = await response.json();
      const event = data.event;
      await loadDashboard();
      await selectAlert(event);
      setToast('Live model event loaded');
    } catch { setToast('Live demo event unavailable'); setDemoMode(true); setStats(null); setAlerts([]); setApiError('Live demo event unavailable. Start the FastAPI backend and retry.'); }
  }

  function action(message) { setToast(message); window.setTimeout(() => setToast(''), 2600); }
  const averageRisk = stats?.average_risk_score ?? 0;
  const attackDistribution = stats?.attack_distribution || {};
  const maxAttack = Math.max(...Object.values(attackDistribution), 1);

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark">TL</span><div><strong>THREATLENS</strong><small>AI SECURITY OPERATIONS</small></div></div>
      <nav><a className="active" href="#overview"><span>01</span> Overview</a><a href="#alerts"><span>02</span> Live alerts</a><a href="#analytics"><span>03</span> Attack analytics</a><a href="#investigation"><span>04</span> Investigation</a></nav>
      <div className="sidebar-foot"><div className="system-state"><i /> Pipeline online</div><small>UNSW-NB15 / MODEL V1.0</small><small>LAST SYNC {new Date().toLocaleTimeString()}</small></div>
    </aside>
    <main className="main-content">
      <header className="topbar"><div><span className="eyebrow">SECURITY OPERATIONS CENTER / 24H VIEW</span><h1>ThreatLens AI</h1><p className="title-subtitle">Evidence-Aware Autonomous Network Security Analyst</p><p className="tagline">Detect. Investigate. Explain. Respond.</p></div><div className="header-actions"><span className={`data-badge ${demoMode ? 'demo' : ''}`}><i /> {demoMode ? 'DEMO DATA' : 'LIVE TELEMETRY'}</span><button className="icon-button" onClick={loadDashboard} aria-label="Refresh dashboard">↻</button><button className="demo-button" onClick={loadDemoEvent}>Load model event <span>+</span></button></div></header>
      {apiError && <div className="api-notice" role="status"><span>!</span>{apiError}<button onClick={loadDashboard}>Retry API</button></div>}
      {toast && <div className="toast">{toast}</div>}
      <section id="overview" className="section overview-section"><SectionTitle number="01" title="Overview" meta="TRAFFIC POSTURE / NOW" /><div className="stat-grid"><StatCard label="Total traffic" value={stats?.total_traffic} suffix="flows" accent="lime" loading={loading} /><StatCard label="Normal traffic" value={stats?.normal_count} suffix="flows" accent="slate" loading={loading} /><StatCard label="Suspicious traffic" value={stats?.suspicious_count} suffix="flows" accent="amber" loading={loading} /><StatCard label="Critical alerts" value={stats?.critical_count} suffix="requires action" accent="red" loading={loading} /><StatCard label="Average risk score" value={averageRisk} suffix="/ 100" accent="lime" loading={loading} /></div></section>
      <section id="analytics" className="section analytics-section"><SectionTitle number="02" title="Attack analytics" meta="CLASSIFICATION / RISK / TREND" /><div className="analytics-grid"><div className="card distribution-card"><CardTitle title="Attack type distribution" caption="SUSPICIOUS FLOWS" /><div className="bars">{Object.entries(attackDistribution).slice(0, 6).map(([name, value]) => <div className="bar-row" key={name}><div><span>{name}</span><b>{value}</b></div><div className="track"><i style={{ width: `${(value / maxAttack) * 100}%` }} /></div></div>)}{!Object.keys(attackDistribution).length && <Empty label="No attack classifications yet" />}</div></div><div className="card risk-card"><CardTitle title="Risk distribution" caption="CURRENT ALERTS" /><RiskDistribution alerts={alerts} /></div><div className="card trend-card"><CardTitle title="Recent threat trend" caption="LAST 60 MINUTES" /><TrendChart alerts={alerts} /></div></div></section>
      <section id="alerts" className="section alerts-section"><SectionTitle number="03" title="Live / recent alerts" meta={`${alerts.length} SUSPICIOUS EVENTS`} /><div className="alert-table-wrap"><table><thead><tr><th>Event ID</th><th>Time</th><th>Source IP</th><th>Destination IP</th><th>Attack type</th><th>Risk</th><th>Severity</th><th>Status</th><th>Action</th></tr></thead><tbody>{alerts.map((alert) => <tr key={alert.event_id} className={selected?.event_id === alert.event_id ? 'selected-row' : ''} onClick={() => selectAlert(alert)}><td className="event-id">{alert.event_id}</td><td className="time">{formatTime(alert.timestamp)}</td><td>{alert.source_ip}</td><td>{alert.destination_ip}</td><td><span className="attack-name"><i className="attack-dot" />{alert.attack_type}</span></td><td><strong className="risk-number">{alert.risk_score}</strong></td><td><Severity level={alert.severity} /></td><td><span className={`status ${reviewed.has(alert.event_id) ? 'reviewed' : 'open'}`}>{reviewed.has(alert.event_id) ? 'Reviewed' : 'Open'}</span></td><td><button className="row-action" onClick={(event) => { event.stopPropagation(); selectAlert(alert); }}>View →</button></td></tr>)}</tbody></table>{!alerts.length && <Empty label="No suspicious alerts in the current window" />}</div></section>
      <section id="investigation" className="section investigation-section"><SectionTitle number="04" title="Incident investigation" meta={selected ? `INCIDENT #${selected.event_id}` : 'SELECT AN ALERT TO BEGIN'} />{selected ? <InvestigationPanel alert={selected} report={report} loading={investigating} reviewed={reviewed.has(selected.event_id)} onReviewed={() => { setReviewed((current) => new Set(current).add(selected.event_id)); action('Alert marked as reviewed'); }} onInvestigate={() => selectAlert(selected)} onAction={action} /> : <div className="investigation-empty"><div className="crosshair">+</div><h3>Investigation workspace ready</h3><p>Select an alert above to inspect evidence, reasoning, and response guidance.</p></div>}</section>
      <footer><span>THREATLENS AI</span><span>Evidence-aware autonomous network security analyst</span><span>API / {demoMode ? 'FALLBACK' : 'CONNECTED'}</span></footer>
    </main>
  </div>;
}

function SectionTitle({ number, title, meta }) { return <div className="section-title"><div><span className="section-number">{number}</span><h2>{title}</h2></div><span className="section-meta">{meta}</span></div>; }
function CardTitle({ title, caption }) { return <div className="card-title"><div><h3>{title}</h3><span>{caption}</span></div><span className="card-menu">···</span></div>; }
function StatCard({ label, value, suffix, accent, loading }) { const displayValue = value === null || value === undefined ? 'N/A' : value.toLocaleString(); return <div className={`stat-card ${accent}`}><span className="stat-label">{label}</span><div className="stat-value">{loading ? <span className="skeleton" /> : displayValue} <small>{suffix}</small></div><div className="stat-line" /></div>; }
function Severity({ level }) { return <span className={`severity ${String(level).toLowerCase()}`}><i />{level}</span>; }
function Empty({ label }) { return <div className="empty-state"><span>--</span>{label}</div>; }
function RiskDistribution({ alerts }) { const bins = [{ label: 'LOW', min: 0, max: 24 }, { label: 'MEDIUM', min: 25, max: 49 }, { label: 'HIGH', min: 50, max: 74 }, { label: 'CRITICAL', min: 75, max: 100 }]; const max = Math.max(...bins.map((bin) => alerts.filter((alert) => alert.risk_score >= bin.min && alert.risk_score <= bin.max).length), 1); return <div className="risk-distribution">{bins.map((bin) => { const count = alerts.filter((alert) => alert.risk_score >= bin.min && alert.risk_score <= bin.max).length; return <div className="risk-bin" key={bin.label}><div className="risk-bar"><i className={bin.label.toLowerCase()} style={{ height: `${Math.max((count / max) * 100, count ? 12 : 3)}%` }} /></div><span>{bin.label}</span><b>{count}</b></div>; })}</div>; }
function TrendChart({ alerts }) { const points = Array.from({ length: 12 }, (_, index) => { const cutoff = Date.now() - (11 - index) * 5 * 60000; return alerts.filter((alert) => new Date(alert.timestamp).getTime() <= cutoff).length; }); const max = Math.max(...points, 1); const path = points.map((point, index) => `${index * 30},${92 - (point / max) * 66}`).join(' '); return <div className="trend"><svg viewBox="0 0 330 110" preserveAspectRatio="none"><polyline points={path} fill="none" stroke="#b8e986" strokeWidth="2" /><polyline points={`0,110 ${path} 330,110`} fill="url(#area)" stroke="none" /><defs><linearGradient id="area" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#b8e986" stopOpacity=".2" /><stop offset="1" stopColor="#b8e986" stopOpacity="0" /></linearGradient></defs></svg><div className="trend-labels"><span>-60m</span><span>-30m</span><span>now</span></div><div className="trend-readout"><strong>{alerts.length}</strong><span>threats observed</span></div></div>; }
function InvestigationPanel({ alert, report, loading, reviewed, onReviewed, onInvestigate, onAction }) { return <div className="investigation-grid"><div className="investigation-main"><div className="incident-head"><div><span className="incident-code">INCIDENT / {alert.event_id}</span><h3>{alert.attack_type} activity</h3><p>{alert.source_ip} <span>→</span> {alert.destination_ip} <b>·</b> {alert.protocol}</p></div><Severity level={alert.severity} /></div><div className="incident-metrics"><MetricBox label="Attack type" value={alert.attack_type} /><MetricBox label="Risk score" value={`${alert.risk_score} / 100`} tone="lime" /><MetricBox label="Confidence" value={report ? `${(report.confidence * 100).toFixed(1)}%` : '--'} /><MetricBox label="Severity" value={alert.severity} tone={alert.severity.toLowerCase()} /></div><div className="detail-block"><div className="block-heading"><span>WHY WAS THIS FLAGGED?</span><small>MODEL EVIDENCE</small></div>{loading ? <LoadingLine /> : <ul className="evidence-list">{(report?.evidence || ['Evidence unavailable']).map((item, index) => <li key={index}><span>{String(index + 1).padStart(2, '0')}</span>{item}</li>)}</ul>}</div><div className="detail-block"><div className="block-heading"><span>AI INVESTIGATION</span><small>AGENT SYNTHESIS</small></div>{loading ? <LoadingLine /> : <><p className="agent-summary">{report?.summary || 'Investigation unavailable.'}</p><ul className="reasoning-list">{(report?.reasoning || []).map((item, index) => <li key={index}>{item}</li>)}</ul></>}</div></div><aside className="investigation-side"><div className="side-block"><div className="block-heading"><span>EVIDENCE</span><small>TELEMETRY SNAPSHOT</small></div><EvidenceMeters alert={alert} /></div><div className="side-block"><div className="block-heading"><span>TOP CONTRIBUTING FEATURES</span><small>EXPLAINABILITY</small></div><div className="feature-list">{(alert.top_features || []).slice(0, 5).map((feature) => <div key={feature.feature}><span>{feature.feature}</span><b>{feature.importance.toFixed(3)}</b></div>)}{!alert.top_features?.length && <p className="unavailable">Feature explanation unavailable.</p>}</div></div><div className="side-block traffic-details"><div className="block-heading"><span>TRAFFIC DETAILS</span><small>FLOW CONTEXT</small></div><div><span>Protocol</span><b>{alert.protocol || 'Unavailable'}</b></div><div><span>Source</span><b>{alert.source_ip}</b></div><div><span>Destination</span><b>{alert.destination_ip}</b></div></div></aside><div className="response-block"><div className="block-heading"><span>RECOMMENDED RESPONSE</span><small>CONTROLLED ACTIONS</small></div><div className="recommendations">{(report?.recommended_actions || ['Recommendations unavailable.']).map((item, index) => <div key={index}><span>{index + 1}</span>{item}</div>)}</div><div className="action-row"><button className="action-primary" onClick={onInvestigate} disabled={loading}>Investigate with AI <span>↗</span></button><button className="action-secondary" onClick={onReviewed} disabled={reviewed}>{reviewed ? 'Reviewed' : 'Mark reviewed'}</button><button className="action-secondary" onClick={() => onAction(`Incident ${alert.event_id} created`)}>Create incident <span>+</span></button><button className="action-danger" onClick={() => onAction(`Simulated block for ${alert.source_ip}`)}>Simulate block IP <span>⊘</span></button></div></div></div>; }
function MetricBox({ label, value, tone }) { return <div className={`metric-box ${tone || ''}`}><span>{label}</span><strong>{value}</strong></div>; }
function EvidenceMeters({ alert }) { return <div className="evidence-meters"><Meter label="Suspicious probability" value={alert.suspicious_probability} /><Meter label="Anomaly score" value={alert.anomaly_score} /><Meter label="Attack probability" value={alert.attack_probability} /></div>; }
function Meter({ label, value }) { const actual = typeof value === 'number' ? value : 0; return <div className="meter"><div><span>{label}</span><b>{percent(actual)}</b></div><div className="meter-track"><i style={{ width: `${actual * 100}%` }} /></div></div>; }
function LoadingLine() { return <div className="loading-line"><i /><i /><i /></div>; }
function percent(value) { return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : 'Unavailable'; }
function formatTime(timestamp) { if (!timestamp) return '--:--'; return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); }

createRoot(document.getElementById('root')).render(<App />);
