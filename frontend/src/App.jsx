import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const API_BASE = '/api';

const starterQuestions = [
  'How does the self-improving feedback loop reduce future token usage?',
  'Compare baseline RAG and GraphRAG for multi-hop reasoning.',
  'Which system components update after a high-confidence CRAG grade?',
];

function formatNumber(value, type = 'number') {
  const n = Number(value || 0);
  if (type === 'percent') return `${n.toFixed(1)}%`;
  if (type === 'ms') return `${n.toFixed(0)}ms`;
  if (type === 'cost') return `$${n.toFixed(4)}`;
  if (type === 'score') return n.toFixed(2);
  if (type === 'ratio') return `${(n * 100).toFixed(1)}%`;
  return n.toLocaleString();
}

function cx(...classes) {
  return classes.filter(Boolean).join(' ');
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <div className="tooltip-label">Batch {label}</div>
      {payload.map((entry) => (
        <div key={entry.dataKey} className="tooltip-row" style={{ color: entry.color }}>
          <span>{entry.name}</span>
          <strong>{typeof entry.value === 'number' ? entry.value.toFixed(3) : entry.value}</strong>
        </div>
      ))}
    </div>
  );
}

function shortText(value, limit = 86) {
  const text = String(value || '');
  return text.length > limit ? `${text.slice(0, limit - 3)}...` : text;
}

function StatusPill({ label, status, detail }) {
  const normalized = String(status || 'unknown').toLowerCase();
  const tone = normalized.includes('connected') || normalized.includes('healthy')
    ? 'good'
    : normalized.includes('memory') || normalized.includes('degraded')
      ? 'warn'
      : 'bad';
  return (
    <div className={cx('status-pill', tone)}>
      <span className="status-dot" />
      <span>{label}</span>
      <strong>{status || 'unknown'}</strong>
      {detail && <small>{shortText(detail)}</small>}
    </div>
  );
}

function StatCard({ label, value, type, accent = 'cyan', sublabel }) {
  return (
    <div className="stat-card">
      <div className={cx('stat-value', accent)}>{formatNumber(value, type)}</div>
      <div className="stat-label">{label}</div>
      {sublabel && <div className="stat-sublabel">{sublabel}</div>}
    </div>
  );
}

function PipelinePanel({ title, subtitle, result, loading, tone }) {
  const metrics = result?.metrics || {};
  return (
    <section className={cx('panel pipeline-panel', tone)}>
      <div className="panel-heading">
        <div>
          <p className="eyebrow">{subtitle}</p>
          <h2>{title}</h2>
        </div>
        {loading && <span className="spinner" />}
      </div>

      <div className="mini-grid">
        <div>
          <span>Tokens</span>
          <strong>{formatNumber(metrics.tokens_used)}</strong>
        </div>
        <div>
          <span>Latency</span>
          <strong>{formatNumber(metrics.response_time_ms, 'ms')}</strong>
        </div>
        <div>
          <span>Cost</span>
          <strong>{formatNumber(metrics.cost_usd, 'cost')}</strong>
        </div>
        <div>
          <span>{tone === 'graph' ? 'CRAG' : 'Method'}</span>
          <strong>{tone === 'graph' ? formatNumber(metrics.crag_grade, 'score') : metrics.retrieval_method || 'llm'}</strong>
        </div>
      </div>

      {tone === 'graph' && (
        <div className="graph-badges">
          <span>{metrics.cache_hit ? 'Cache hit' : 'Cache miss'}</span>
          <span>{metrics.entities_resolved || 0} entities</span>
          <span>{metrics.relationships_traversed || 0} edges</span>
          <span>{metrics.hops_used || 0} hops</span>
        </div>
      )}

      <div className="answer-box">
        {result?.answer ? (
          <div>{result.answer}</div>
        ) : loading ? (
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <span className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }} />
            <span>Waiting for model response...</span>
          </div>
        ) : (
          <span style={{ color: 'var(--muted)' }}>Run a comparison to see the generated answer.</span>
        )}
      </div>
    </section>
  );
}

function Flywheel({ stats }) {
  const steps = [
    ['Graph', `${stats.graph_edges_updated || 0} edge updates`],
    ['Cache', `${formatNumber(stats.cache_hit_rate || 0, 'ratio')} hit rate`],
    ['Prompts', `${stats.prompt_versions_evolved || 0} refinements`],
    ['Routing', `${stats.query_patterns_learned || 0} learned patterns`],
  ];

  return (
    <section className="panel flywheel-panel">
      <div className="panel-heading compact">
        <div>
          <p className="eyebrow">Self improvement loop</p>
          <h2>Every answer feeds the next retrieval</h2>
        </div>
      </div>
      <div className="flywheel">
        {steps.map(([label, value], index) => (
          <div className="flywheel-step" key={label}>
            <span className="step-index">{index + 1}</span>
            <strong>{label}</strong>
            <small>{value}</small>
          </div>
        ))}
      </div>
    </section>
  );
}

function ReadinessPanel({ health }) {
  const deps = health?.dependencies || {};
  const items = [
    {
      name: 'Local LLM',
      status: deps.llm?.status || 'checking',
      detail: deps.llm?.model ? `${deps.llm.provider} / ${deps.llm.model}` : 'Ollama model health check',
      command: 'ollama list',
    },
    {
      name: 'Graph layer',
      status: deps.tigergraph?.status || 'checking',
      detail: deps.tigergraph?.status === 'connected'
        ? `${deps.tigergraph.total_vertices || 0} vertices, ${deps.tigergraph.total_edges || 0} edges`
        : 'Fallback retrieval is active until TigerGraph is running.',
      command: 'docker compose up tigergraph',
    },
    {
      name: 'Semantic cache',
      status: deps.redis?.status || 'checking',
      detail: deps.redis?.status === 'connected'
        ? `${deps.redis.entries || 0} cached entries`
        : 'Memory cache is active; Redis enables persistence and Redis Insight.',
      command: 'docker compose up redis',
    },
    {
      name: 'Metrics store',
      status: deps.metrics_db?.status || 'checking',
      detail: `${deps.metrics_db?.total_comparisons || 0} saved comparisons`,
      command: 'sqlite3 data/apex_metrics.db',
    },
  ];

  const handleCopy = (text) => {
    navigator.clipboard.writeText(text);
    // Visual feedback could be added here if we had a toast system
  };

  return (
    <section className="panel readiness-panel">
      <div className="panel-heading compact">
        <div>
          <p className="eyebrow">Production readiness</p>
          <h2>Dependency checklist</h2>
        </div>
      </div>
      <div className="readiness-list">
        {items.map((item) => {
          const normalized = String(item.status).toLowerCase();
          const tone = normalized.includes('connected') || normalized.includes('healthy')
            ? 'good'
            : normalized.includes('memory') || normalized.includes('checking')
              ? 'warn'
              : 'bad';
          return (
            <div className={cx('readiness-row', tone)} key={item.name}>
              <span className="readiness-light" />
              <div>
                <strong>{item.name}</strong>
                <small>{item.detail}</small>
              </div>
              <code 
                onClick={() => handleCopy(item.command)} 
                title="Click to copy" 
                style={{ cursor: 'pointer' }}
              >
                {item.command}
              </code>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function ArchitecturePanel() {
  const layers = [
    ['Layer 1', 'Graph', 'TigerGraph + local fallback retrieval'],
    ['Layer 2', 'Orchestration', 'Router, CRAG, decomposer, semantic cache'],
    ['Layer 3', 'LLM', 'Ollama llama3.2:latest with zero API cost'],
    ['Layer 4', 'Evaluation', 'Savings, latency, cache, CRAG, query history'],
  ];

  return (
    <section className="panel architecture-panel">
      <div className="panel-heading compact">
        <div>
          <p className="eyebrow">AI factory model</p>
          <h2>Four-layer execution map</h2>
        </div>
      </div>
      <div className="layer-map">
        {layers.map(([index, name, detail]) => (
          <div className="layer-card" key={name}>
            <span>{index}</span>
            <strong>{name}</strong>
            <small>{detail}</small>
          </div>
        ))}
      </div>
    </section>
  );
}

export default function App() {
  const [query, setQuery] = useState(starterQuestions[0]);
  const [isLoading, setIsLoading] = useState(false);
  const [baselineResult, setBaselineResult] = useState(null);
  const [graphragResult, setGraphragResult] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [summary, setSummary] = useState(null);
  const [improvementCurve, setImprovementCurve] = useState([]);
  const [latest, setLatest] = useState([]);
  const [health, setHealth] = useState(null);
  const [eventLog, setEventLog] = useState([]);
  const [errorMsg, setErrorMsg] = useState(null);

  const addEvent = useCallback((text) => {
    setEventLog((items) => [{ text, time: new Date().toLocaleTimeString() }, ...items].slice(0, 8));
  }, []);

  const fetchDashboard = useCallback(async () => {
    const endpoints = [
      fetch(`${API_BASE}/metrics/summary`),
      fetch(`${API_BASE}/metrics/improvement-curve`),
      fetch(`${API_BASE}/metrics/latest`),
      fetch(`${API_BASE}/health`),
    ];

    try {
      const [summaryRes, curveRes, latestRes, healthRes] = await Promise.all(endpoints);
      if (summaryRes.ok) setSummary(await summaryRes.json());
      if (curveRes.ok) setImprovementCurve(await curveRes.json());
      if (latestRes.ok) setLatest(await latestRes.json());
      if (healthRes.ok) setHealth(await healthRes.json());
      else setHealth(prev => ({ ...prev, status: 'disconnected' }));
    } catch (error) {
      addEvent(`Dashboard refresh skipped: ${error.message}`);
      setHealth(prev => ({ ...prev, status: 'disconnected' }));
    }
  }, [addEvent]);

  useEffect(() => {
    fetchDashboard();
    const id = window.setInterval(fetchDashboard, 15000);
    return () => window.clearInterval(id);
  }, [fetchDashboard]);

  const runQuery = async () => {
    const cleanQuery = query.trim();
    if (!cleanQuery || isLoading) return;

    setIsLoading(true);
    setErrorMsg(null);
    setBaselineResult(null);
    setGraphragResult(null);
    setComparison(null);
    addEvent('Comparison started');

    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: cleanQuery, mode: 'comparison' }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      if (response.headers.get('content-type')?.includes('text/event-stream')) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const chunks = buffer.split('\n\n');
          buffer = chunks.pop() || '';

          for (const chunk of chunks) {
            const event = chunk.match(/^event: (.+)$/m)?.[1];
            const data = chunk.match(/^data: (.+)$/m)?.[1];
            if (!event || !data) continue;

            const parsed = JSON.parse(data);
            if (event === 'baseline_start') addEvent('Baseline pipeline running');
            if (event === 'graphrag_start') addEvent('GraphRAG pipeline running');
            if (event === 'baseline_complete') setBaselineResult(parsed);
            if (event === 'graphrag_complete') setGraphragResult(parsed);
            if (event === 'comparison') {
              setComparison(parsed);
              addEvent('Savings computed');
            }
          }
        }
      } else {
        const data = await response.json();
        setGraphragResult(data);
      }

      await fetchDashboard();
    } catch (error) {
      addEvent(`Query failed: ${error.message}`);
      setErrorMsg(`Failed to run comparison: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setQuery('');
    setBaselineResult(null);
    setGraphragResult(null);
    setComparison(null);
    setErrorMsg(null);
  };

  const savings = useMemo(() => ({
    tokens: comparison?.token_savings_pct ?? summary?.savings?.tokens_pct ?? 0,
    speed: comparison?.speed_improvement_pct ?? summary?.savings?.latency_pct ?? 0,
    cost: comparison?.cost_savings_pct ?? summary?.savings?.cost_pct ?? 0,
  }), [comparison, summary]);

  const selfImprovement = summary?.self_improvement || {};
  const deps = health?.dependencies || {};
  const llm = deps.llm || {};

  return (
    <main className="app-shell">
      <section className="hero">
        <div>
          <p className="eyebrow">APEX GraphRAG control room</p>
          <h1>Self-improving RAG with a local Ollama LLM.</h1>
          <p className="hero-copy">
            Compare baseline generation against graph retrieval, CRAG grading,
            feedback learning, and semantic caching in one live workspace.
          </p>
        </div>
        <div className="status-stack">
          <StatusPill label="API" status={health?.status || 'checking'} detail="FastAPI query, metrics, and health endpoints" />
          <StatusPill label="LLM" status={llm.status || 'checking'} detail={llm.model || 'Ollama health check'} />
          <StatusPill label="Graph" status={deps.tigergraph?.status || 'checking'} detail={deps.tigergraph?.error || 'TigerGraph graph layer'} />
          <StatusPill label="Cache" status={deps.redis?.status || 'checking'} detail={deps.redis?.error || 'Redis semantic cache'} />
          <div className="model-card">
            <span>Active model</span>
            <strong>{llm.model || 'llama3.2:latest'}</strong>
            <small>{llm.provider || 'ollama'} at {formatNumber(llm.latency_ms, 'ms')}</small>
          </div>
        </div>
      </section>

      <section className="query-console">
        <textarea
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') runQuery();
          }}
          placeholder="Ask a benchmark question..."
          disabled={isLoading}
        />
        <div className="console-actions">
          <div className="prompt-chips">
            {starterQuestions.map((item) => (
              <button key={item} type="button" onClick={() => setQuery(item)}>
                {item}
              </button>
            ))}
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button 
              className="run-button" 
              type="button" 
              onClick={handleReset} 
              disabled={isLoading || (!query && !baselineResult)}
              style={{ background: 'transparent', border: '1px solid var(--line)', color: '#c2cfdd', minWidth: '100px' }}
            >
              Reset
            </button>
            <button className="run-button" type="button" onClick={runQuery} disabled={isLoading || !query.trim()}>
              {isLoading ? 'Running...' : 'Run comparison'}
            </button>
          </div>
        </div>
        {errorMsg && (
          <div style={{ marginTop: '14px', color: 'var(--red)', fontSize: '0.9rem', padding: '12px', background: 'rgba(251, 113, 133, 0.1)', borderRadius: '8px', border: '1px solid rgba(251, 113, 133, 0.2)' }}>
            {errorMsg}
          </div>
        )}
      </section>

      <section className="stat-grid">
        <StatCard label="Token savings" value={savings.tokens} type="percent" accent="green" />
        <StatCard label="Speed lift" value={savings.speed} type="percent" accent="blue" />
        <StatCard label="Cost savings" value={savings.cost} type="percent" accent="amber" />
        <StatCard label="Total runs" value={summary?.total_queries || 0} accent="violet" />
        <StatCard label="Avg CRAG grade" value={selfImprovement.avg_crag_grade || 0} type="score" accent="cyan" />
        <StatCard label="Cache hit rate" value={selfImprovement.cache_hit_rate || 0} type="ratio" accent="green" />
      </section>

      <section className="pipelines">
        <PipelinePanel
          title="Baseline LLM"
          subtitle="Control path"
          tone="baseline"
          result={baselineResult}
          loading={isLoading && !baselineResult}
        />
        <PipelinePanel
          title="Self-improving GraphRAG"
          subtitle="Graph path"
          tone="graph"
          result={graphragResult}
          loading={isLoading && !graphragResult}
        />
      </section>

      <section className="analytics-grid">
        <div className="panel chart-panel wide">
          <div className="panel-heading compact">
            <div>
              <p className="eyebrow">Improvement curve</p>
              <h2>Tokens fall as cache and graph feedback accumulate</h2>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={improvementCurve}>
              <defs>
                <linearGradient id="tokens" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#2dd4bf" stopOpacity={0.34} />
                  <stop offset="95%" stopColor="#2dd4bf" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(148,163,184,0.12)" vertical={false} />
              <XAxis dataKey="batch_number" tick={{ fill: '#8ea0b8', fontSize: 12 }} />
              <YAxis tick={{ fill: '#8ea0b8', fontSize: 12 }} />
              <Tooltip content={<ChartTooltip />} />
              <Area type="monotone" dataKey="avg_tokens" name="Avg tokens" stroke="#2dd4bf" fill="url(#tokens)" strokeWidth={2} />
              <Line type="monotone" dataKey="avg_crag_grade" name="CRAG grade" stroke="#f59e0b" strokeWidth={2} yAxisId={0} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <Flywheel stats={selfImprovement} />

        <div className="panel chart-panel">
          <div className="panel-heading compact">
            <div>
              <p className="eyebrow">Efficiency split</p>
              <h2>Current averages</h2>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={[
              { name: 'Tokens', Baseline: summary?.baseline_avg?.tokens_used || 0, GraphRAG: summary?.graphrag_avg?.tokens_used || 0 },
              { name: 'Latency', Baseline: summary?.baseline_avg?.response_time_ms || 0, GraphRAG: summary?.graphrag_avg?.response_time_ms || 0 },
            ]}>
              <CartesianGrid stroke="rgba(148,163,184,0.12)" vertical={false} />
              <XAxis dataKey="name" tick={{ fill: '#8ea0b8', fontSize: 12 }} />
              <YAxis tick={{ fill: '#8ea0b8', fontSize: 12 }} />
              <Tooltip content={<ChartTooltip />} />
              <Legend />
              <Bar dataKey="Baseline" fill="#fb7185" radius={[6, 6, 0, 0]} />
              <Bar dataKey="GraphRAG" fill="#2dd4bf" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="panel chart-panel">
          <div className="panel-heading compact">
            <div>
              <p className="eyebrow">Cache curve</p>
              <h2>Zero-token path growth</h2>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={improvementCurve}>
              <CartesianGrid stroke="rgba(148,163,184,0.12)" vertical={false} />
              <XAxis dataKey="batch_number" tick={{ fill: '#8ea0b8', fontSize: 12 }} />
              <YAxis tick={{ fill: '#8ea0b8', fontSize: 12 }} domain={[0, 1]} />
              <Tooltip content={<ChartTooltip />} />
              <Line type="monotone" dataKey="cache_hit_rate" name="Cache hit rate" stroke="#60a5fa" strokeWidth={3} dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="ops-grid">
        <ReadinessPanel health={health} />
        <ArchitecturePanel />
      </section>

      <section className="bottom-grid">
        <div className="panel">
          <div className="panel-heading compact">
            <div>
              <p className="eyebrow">Recent comparisons</p>
              <h2>Benchmark trail</h2>
            </div>
          </div>
          <div className="runs-table">
            {latest.length === 0 && <div className="empty-row">No saved comparisons yet.</div>}
            {latest.map((item, index) => (
              <div className="run-row" key={`${item.query}-${index}`}>
                <span>{item.query}</span>
                <strong>{formatNumber(item.token_savings_pct, 'percent')} token savings</strong>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="panel-heading compact">
            <div>
              <p className="eyebrow">Live events</p>
              <h2>Pipeline timeline</h2>
            </div>
          </div>
          <div className="event-log">
            {eventLog.length === 0 && <div className="empty-row">Events will appear during a run.</div>}
            {eventLog.map((event, index) => (
              <div className="event-row" key={`${event.time}-${index}`}>
                <span>{event.time}</span>
                <strong>{event.text}</strong>
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
