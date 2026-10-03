import { ArrowRight, Binary, ChartNoAxesCombined, FileCheck2, Radar, SearchCheck } from 'lucide-react'
import { PageHeader, Panel } from '../components/Shared'

const capabilities = [
  { icon: SearchCheck, title: 'Grounded Q&A', copy: 'Answers built from ranked filing evidence, not general model memory.' },
  { icon: ChartNoAxesCombined, title: 'Temporal analysis', copy: 'Deterministic year-over-year financial comparison with provenance.' },
  { icon: Radar, title: 'Risk Radar', copy: 'Structured disclosure changes without invented severity scores.' },
  { icon: FileCheck2, title: 'Claim verification', copy: 'Issuer statements assessed against attributable filing evidence.' },
  { icon: Binary, title: 'Citation verification', copy: 'A separate NLI model checks whether cited evidence supports claims.' },
]

export function Overview({ navigate }: { navigate: (page: string) => void }) {
  return <div className="page"><PageHeader eyebrow="Research workspace" title="Evidence-grounded financial intelligence" description="A local-first due diligence system that combines hybrid retrieval, deterministic financial reasoning, grounded generation, and semantic citation verification." />
    <div className="hero-grid"><Panel className="hero-panel"><div className="hero-mark">DD</div><div><span className="kicker">AI Due Diligence Copilot</span><h2>Move from filing to defensible insight.</h2><p>Every result keeps the document, reporting period, page, and verification state visible.</p><button className="primary-button" onClick={() => navigate('qa')}>Start grounded analysis <ArrowRight size={16} /></button></div></Panel>
      <Panel className="dataset-panel"><div className="panel-label">Current validated dataset</div><div className="metric"><strong>3</strong><span>source documents</span></div><div className="metric"><strong>639</strong><span>embedded chunks</span></div><ul><li>Apple FY2024 10-K</li><li>Apple FY2025 10-K</li><li>Apple FY2025 Q4 earnings release</li></ul><small>Single-company validation scope</small></Panel></div>
    <div className="capability-grid">{capabilities.map(({ icon: Icon, title, copy }) => <Panel className="capability-card" key={title}><Icon size={20} /><h3>{title}</h3><p>{copy}</p></Panel>)}</div>
    <Panel className="architecture"><div className="panel-label">System architecture</div><div className="pipeline">{['Filing', 'BGE + PostgreSQL FTS', 'RRF fusion', 'MiniLM reranking', 'Domain analysis', 'Qwen3 generation', 'Citation validation', 'DeBERTa NLI'].map((step, index) => <div key={step} className="pipeline-step"><span>{String(index + 1).padStart(2, '0')}</span><strong>{step}</strong>{index < 7 && <ArrowRight size={14} />}</div>)}</div></Panel>
  </div>
}
