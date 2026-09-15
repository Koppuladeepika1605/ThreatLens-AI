import React, { useEffect, useMemo, useState } from 'react';
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
  const [apiError, setApiError] = useState('');
  const [toast, setToast] = useState('');
  const [query, setQuery] = useState('');
  const [reviewed, setReviewed] = useState(new Set());

  async function loadDashboard() {
    setLoading(true);
    try {
      const [statsResponse, alertsResponse] = await Promise.all([
        fetch(`${API}/dashboard/stats`),
        fetch(`${API}/alerts`),
      ]);
      if (!statsResponse.ok || !alertsResponse.ok) throw new Error('API unavailable');
      setStats(await statsResponse.json());
      setAlerts((await alertsResponse.json()).alerts || []);
      setApiError('');
    } catch {
      setStats(null);
      setAlerts([]);
      setApiError('Live API unavailable. Metrics and alerts are currently unavailable.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { loadDashboard(); }, []);

  async function selectAlert(alert) {
    setSelected(alert);
    setReport(null);
    setInvestigating(true);
    try {
      const response = await fetch(`${API}/investigate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ event_id: alert.event_id }),
      });
      if (!response.ok) throw new Error('Investigation unavailable');
      setReport(await response.json());
      setApiError('');
    } catch {
      setApiError('Live investigation unavailable. Evidence is unavailable until the API reconnects.');
      setReport(null);
    } finally {
      setInvestigating(false);
    }
  }

  async function loadModelEvent() {
    try {
      const response = await fetch(`${API}/sample`);
      if (!response.ok) throw new Error('Demo event unavailable');
      const data = await response.json();
      await loadDashboard();
      await selectAlert(data.event);
      notify('Live model event loaded');
    } catch {
      setApiError('Live model event unavailable. Start FastAPI and retry.');
    }
  }

  function notify(message) {
    setToast(message);
    window.setTimeout(() => setToast(''), 2600);
  }

  const filteredAlerts = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return alerts;
    return alerts.filter((alert) => [
      alert.event_id, alert.source_ip, alert.destination_ip, alert.attack_type, alert.severity,
    ].some((value) => String(value || '').toLowerCase().includes(normalized)));
  }, [alerts, query]);

  const distribution = stats?.attack_distribution || {};
  const maxDistribution = Math.max(...Object.values(distribution), 1);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <div className="brand-mark">TL</div>
          <div><strong>ThreatLens AI</strong><small>SOC / INTELLIGENCE PLATFORM</small></div>
        </div>
        <nav className="primary-nav" aria-label="Primary navigation">
          <NavItem href="#overview" icon="◈" number="01" label="Overview" active />
          <NavItem href="#alerts" icon="◉" number="02" label="Live Alerts" />
          <NavItem href="#analytics" icon="⌁" number="03" label="Attack Analytics" />
          <NavItem href="#investigation" icon="⊙" number="04" label="Investigation" />
          <NavItem href="#reports" icon="▤" number="05" label="Reports" />
        </nav>
        <div className="system-panel">
          <span className="eyebrow">SYSTEM STATUS</span>
          <StatusLine label="ML Models" />
          <StatusLine label="Agent System" />
          <StatusLine label="Database" />
        </div>
        <div className="sidebar-footer"><span>THREATLENS / MVP</span><span>UNSW-NB15 TELEMETRY</span></div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="topbar-title"><span className="eyebrow">SECURITY OPERATIONS CENTER / 24H VIEW</span><h1>ThreatLens AI</h1><p>Evidence-Aware Autonomous Network Security Analyst</p></div>
          <div className="topbar-tools">
            <label className="search-box"><span>⌕</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search IP, Event ID, attack type..." aria-label="Search alerts" /></label>
            <div className="monitoring-status"><i /> <span>SYSTEM ONLINE</span><small>Monitoring active</small></div>
            <button className="notification" aria-label={`${alerts.length} alerts`} onClick={() => document.getElementById('alerts')?.scrollIntoView({ behavior: 'smooth' })}>◌<b>{alerts.length}</b></button>
            <div className="profile"><span>AK</span><div><strong>Analyst</strong><small>Security team</small></div></div>
          </div>
        </header>
        <div className="header-meta"><span className="tagline">Detect. Investigate. Explain. Respond.</span><span>Last updated {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span></div>
        {apiError && <div className="api-notice" role="status"><span>!</span>{apiError}<button onClick={loadDashboard}>Retry connection</button></div>}
        {toast && <div className="toast">{toast}</div>}

        <section id="overview" className="section overview-section">
          <SectionHeader number="01" title="Overview" meta="NETWORK POSTURE / LIVE" />
          <p className="section-lead">Detect. Investigate. Explain. Respond.</p>
          <div className="kpi-grid">
            <Kpi icon="⌁" label="Total flows" value={stats?.total_traffic} description="Observed network telemetry" tone="cyan" loading={loading} />
            <Kpi icon="✓" label="Normal traffic" value={stats?.normal_count} description="Baseline behavior" tone="green" loading={loading} />
            <Kpi icon="!" label="Threats detected" value={stats?.suspicious_count} description="Suspicious classifications" tone="amber" loading={loading} />
            <Kpi icon="◈" label="Critical alerts" value={stats?.critical_count} description="Immediate review required" tone="red" loading={loading} />
            <Kpi icon="◎" label="Average risk" value={stats?.average_risk_score} suffix="/100" description="Across processed flows" tone="blue" loading={loading} />
          </div>
        </section>

        <section id="analytics" className="section">
          <SectionHeader number="02" title="Threat activity" meta="MODEL OUTPUT / CURRENT WINDOW" />
          <div className="analytics-grid">
            <div className="surface distribution-card"><CardHeading title="Attack distribution" subtitle="SUSPICIOUS FLOWS" /><AttackDistribution values={distribution} max={maxDistribution} /></div>
            <div className="surface trend-card"><CardHeading title="Risk score trend" subtitle="RECENT RETURNED ALERTS" /><RiskTrend alerts={alerts} /></div>
            <div className="surface feed-card"><CardHeading title="Recent alerts" subtitle="LIVE FEED" /><RecentFeed alerts={alerts.slice(0, 5)} onSelect={selectAlert} /></div>
          </div>
        </section>

        <section id="alerts" className="section">
          <SectionHeader number="03" title="Live security alerts" meta={`${filteredAlerts.length} MATCHING EVENTS`} />
          <div className="surface table-surface">
            <div className="table-toolbar"><div><strong>Threat queue</strong><span>Click a row to open the investigation workspace</span></div><button className="refresh-button" onClick={loadDashboard}>↻ Refresh</button></div>
            <div className="table-scroll"><table><thead><tr><th>Status</th><th>Event</th><th>Attack type</th><th>Risk</th><th>Severity</th><th>Source</th><th>Time</th><th>Action</th></tr></thead><tbody>{filteredAlerts.map((alert) => <AlertRow key={alert.event_id} alert={alert} selected={selected?.event_id === alert.event_id} reviewed={reviewed.has(alert.event_id)} onSelect={selectAlert} />)}</tbody></table></div>
            {!filteredAlerts.length && <EmptyState label={query ? 'No alerts match this search' : 'No suspicious alerts returned by the API'} />}
          </div>
        </section>

        <section id="investigation" className="section">
          <SectionHeader number="04" title="Incident investigation" meta={selected ? `INCIDENT / ${selected.event_id}` : 'SELECT AN ALERT'} />
          {selected ? <InvestigationPanel alert={selected} report={report} loading={investigating} reviewed={reviewed.has(selected.event_id)} onReviewed={() => { setReviewed((current) => new Set(current).add(selected.event_id)); notify('Alert marked as reviewed'); }} onInvestigate={() => selectAlert(selected)} onAction={notify} /> : <EmptyInvestigation />}
        </section>

        <section id="reports" className="section reports-section"><SectionHeader number="05" title="Reports" meta="API AVAILABILITY" /><div className="report-strip"><span className="report-icon">▤</span><div><strong>Reports are not available from the current API</strong><p>Use the live alert queue and investigation workspace for this MVP demo.</p></div><span className="na-badge">N/A</span></div></section>
        <footer><strong>THREATLENS AI</strong><span>Evidence-aware autonomous network security analyst</span><span>API / {apiError ? 'ATTENTION REQUIRED' : 'CONNECTED'}</span></footer>
      </main>
    </div>
  );
}

function NavItem({ href, icon, number, label, active }) { return <a className={active ? 'nav-item active' : 'nav-item'} href={href}><span className="nav-icon">{icon}</span><span className="nav-number">{number}</span><span>{label}</span></a>; }
function StatusLine({ label }) { return <div className="status-line"><i />{label}<b>Online</b></div>; }
function SectionHeader({ number, title, meta }) { return <div className="section-header"><div><span className="section-number">{number}</span><h2>{title}</h2></div><span className="section-meta">{meta}</span></div>; }
function CardHeading({ title, subtitle }) { return <div className="card-heading"><div><h3>{title}</h3><span>{subtitle}</span></div><span className="card-menu">···</span></div>; }
function Kpi({ icon, label, value, suffix, description, tone, loading }) { return <article className={`kpi-card ${tone}`}><div className="kpi-top"><span className="kpi-icon">{icon}</span><span className="kpi-label">{label}</span><span className="kpi-signal">●</span></div><div className="kpi-value">{loading ? <span className="skeleton" /> : value == null ? 'N/A' : value.toLocaleString()}<small>{suffix}</small></div><p>{description}</p></article>; }
function AttackDistribution({ values, max }) { const entries = Object.entries(values).slice(0, 9); return entries.length ? <div className="distribution-list">{entries.map(([name, value]) => <div className="distribution-row" key={name}><div><span>{name}</span><b>{value}</b></div><div className="distribution-track"><i style={{ width: `${(value / max) * 100}%` }} /></div></div>)} </div> : <EmptyState label="No attack distribution returned" />; }
function RiskTrend({ alerts }) { const points = alerts.slice(0, 8).reverse(); if (!points.length) return <EmptyState label="No recent risk values returned" />; const max = 100; const coords = points.map((alert, index) => `${(index / Math.max(points.length - 1, 1)) * 100},${100 - ((alert.risk_score || 0) / max) * 82}`).join(' '); return <div className="risk-trend"><svg viewBox="0 0 100 100" preserveAspectRatio="none"><polyline points={`0,100 ${coords} 100,100`} fill="url(#trendFill)" stroke="none" /><polyline points={coords} fill="none" stroke="#22d3ee" strokeWidth="1.8" vectorEffect="non-scaling-stroke" /><defs><linearGradient id="trendFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#22d3ee" stopOpacity=".25" /><stop offset="1" stopColor="#22d3ee" stopOpacity="0" /></linearGradient></defs></svg><div className="axis-labels"><span>0</span><span>50</span><span>100 risk</span></div><div className="trend-note">Derived from {alerts.length} returned alert{alerts.length === 1 ? '' : 's'}</div></div>; }
function RecentFeed({ alerts, onSelect }) { return alerts.length ? <div className="recent-feed">{alerts.map((alert) => <button key={alert.event_id} onClick={() => onSelect(alert)}><Severity level={alert.severity} /><span><strong>{alert.attack_type}</strong><small>{alert.source_ip} → {alert.destination_ip}</small></span><b>{alert.risk_score}</b></button>)}</div> : <EmptyState label="No recent alerts returned" />; }
function AlertRow({ alert, selected, reviewed, onSelect }) { return <tr className={`${selected ? 'selected-row ' : ''}${alert.severity === 'CRITICAL' ? 'critical-row' : ''}`} onClick={() => onSelect(alert)}><td><span className={`status-dot ${reviewed ? 'reviewed' : 'open'}`} />{reviewed ? 'Reviewed' : 'New'}</td><td className="mono event-id">{alert.event_id}</td><td><span className="attack-name"><i />{alert.attack_type || 'N/A'}</span></td><td><span className={`risk-pill ${riskTone(alert.risk_score)}`}>{alert.risk_score ?? 'N/A'}</span></td><td><Severity level={alert.severity} /></td><td className="mono">{alert.source_ip || 'N/A'}<small> → {alert.destination_ip || 'N/A'}</small></td><td className="mono">{formatTime(alert.timestamp)}</td><td><button className="investigate-button" onClick={(event) => { event.stopPropagation(); onSelect(alert); }}>Investigate <span>→</span></button></td></tr>; }
function InvestigationPanel({ alert, report, loading, reviewed, onReviewed, onInvestigate, onAction }) { return <div className="investigation-layout"><div className="investigation-main surface"><div className="incident-header"><div><span className="eyebrow">INCIDENT OVERVIEW</span><h3>{alert.attack_type || 'N/A'} activity</h3><p className="mono">{alert.event_id} <span>·</span> {alert.source_ip || 'N/A'} → {alert.destination_ip || 'N/A'}</p></div><Severity level={alert.severity} /></div><div className="incident-metrics"><MetricBox label="Attack type" value={alert.attack_type || 'N/A'} /><MetricBox label="Risk score" value={alert.risk_score == null ? 'N/A' : `${alert.risk_score} / 100`} tone="cyan" /><MetricBox label="Confidence" value={report ? `${(report.confidence * 100).toFixed(1)}%` : 'N/A'} /><MetricBox label="Timestamp" value={formatTime(alert.timestamp)} /></div><DetailBlock title="WHY WAS THIS FLAGGED?" eyebrow="ACTUAL EVIDENCE">{loading ? <LoadingLine /> : <ul className="evidence-list">{(report?.evidence || ['Evidence unavailable']).map((item, index) => <li key={index}><span>{String(index + 1).padStart(2, '0')}</span>{item}</li>)}</ul>}</DetailBlock><DetailBlock title="AI INVESTIGATION" eyebrow="AGENT ANALYST LAYER" ai>{loading ? <LoadingLine /> : report ? <><p className="agent-summary">{report.summary}</p><div className="report-columns"><ReportList title="Evidence" values={report.evidence} /><ReportList title="Reasoning" values={report.reasoning} /></div></> : <p className="unavailable">Investigation unavailable. Reconnect the API to retrieve the agent report.</p>}</DetailBlock></div><aside className="investigation-side"><div className="surface side-card"><CardHeading title="Model evidence" subtitle="PREDICTOR OUTPUT" /><EvidenceMeters alert={alert} /></div><div className="surface side-card"><CardHeading title="Top features" subtitle={alert.explanation_method || 'EXPLANATION OUTPUT'} /><div className="feature-list">{(alert.top_features || []).slice(0, 5).map((feature) => <div key={feature.feature}><span className="mono">{feature.feature}</span><b>{feature.importance?.toFixed(3) || 'N/A'}</b><small>{feature.direction === 'increases_risk' ? '↑ increases risk' : feature.direction === 'decreases_risk' ? '↓ decreases risk' : 'N/A'}</small></div>)}{!alert.top_features?.length && <p className="unavailable">Top features unavailable.</p>}</div></div><div className="surface side-card traffic-card"><CardHeading title="Traffic context" subtitle="SECURITY EVENT" /><InfoLine label="Protocol" value={alert.protocol} /><InfoLine label="Source" value={alert.source_ip} /><InfoLine label="Destination" value={alert.destination_ip} /></div></aside><div className="response-panel"><div><span className="eyebrow">RECOMMENDED RESPONSE</span><h3>Controlled actions for this event</h3><div className="recommendations">{(report?.recommended_actions || ['N/A']).map((action, index) => <div key={index}><span>{index + 1}</span>{action}</div>)}</div></div><div className="action-row"><button className="action-primary" onClick={onInvestigate} disabled={loading}>✦ Investigate with AI</button><button className="action-secondary" onClick={onReviewed} disabled={reviewed}>{reviewed ? 'Reviewed' : 'Mark reviewed'}</button><button className="action-secondary" onClick={() => onAction(`Incident ${alert.event_id} created`)}>Create incident</button><button className="action-danger" onClick={() => onAction(`Simulated block for ${alert.source_ip || 'N/A'}`)}>Simulate block IP</button></div></div></div>; }
function DetailBlock({ title, eyebrow, children, ai }) { return <div className={`detail-block ${ai ? 'ai-block' : ''}`}><div className="block-heading"><span>{title}</span><small>{eyebrow}</small></div>{children}</div>; }
function ReportList({ title, values }) { return <div className="report-list"><strong>{title}</strong><ul>{(values || ['N/A']).map((value, index) => <li key={index}>{value}</li>)}</ul></div>; }
function MetricBox({ label, value, tone }) { return <div className={`metric-box ${tone || ''}`}><span>{label}</span><strong>{value}</strong></div>; }
function EvidenceMeters({ alert }) { return <div className="evidence-meters"><Meter label="Suspicious probability" value={alert.suspicious_probability} /><Meter label="Anomaly score" value={alert.anomaly_score} /><Meter label="Attack probability" value={alert.attack_probability} /></div>; }
function Meter({ label, value }) { const actual = typeof value === 'number' ? value : null; return <div className="meter"><div><span>{label}</span><b>{actual == null ? 'N/A' : `${(actual * 100).toFixed(1)}%`}</b></div><div className="meter-track"><i style={{ width: `${actual == null ? 0 : actual * 100}%` }} /></div></div>; }
function InfoLine({ label, value }) { return <div className="info-line"><span>{label}</span><b className="mono">{value || 'N/A'}</b></div>; }
function Severity({ level }) { const safeLevel = level || 'N/A'; return <span className={`severity ${String(safeLevel).toLowerCase()}`}><i />{safeLevel}</span>; }
function EmptyState({ label }) { return <div className="empty-state"><span>—</span>{label}</div>; }
function EmptyInvestigation() { return <div className="empty-investigation"><div className="empty-crosshair">⊙</div><h3>Investigation workspace ready</h3><p>Select a returned alert to inspect evidence, reasoning, and response guidance.</p></div>; }
function LoadingLine() { return <div className="loading-line"><i /><i /><i /></div>; }
function riskTone(score) { if (score >= 75) return 'critical'; if (score >= 50) return 'high'; if (score >= 25) return 'medium'; return 'low'; }
function formatTime(timestamp) { if (!timestamp) return 'N/A'; return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); }

createRoot(document.getElementById('root')).render(<App />);
