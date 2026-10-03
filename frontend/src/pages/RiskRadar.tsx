import { useState } from 'react'
import { Radar } from 'lucide-react'
import { api } from '../api/client'
import { ErrorState, EvidenceCard, LoadingState, PageHeader, Panel, RequestMeta, StatusBadge } from '../components/Shared'
import { useAnalysis } from '../hooks/useAnalysis'
import type { RiskResponse } from '../types/api'

const topics = [
  ['supply_chain_manufacturing', 'Supply chain and manufacturing'],
  ['competition_innovation', 'Competition and innovation'],
  ['cybersecurity_information_systems', 'Cybersecurity and information systems'],
  ['legal_regulatory', 'Legal and regulatory'],
] as const

export function RiskRadarPage() {
  const [topic, setTopic] = useState<string>(topics[0][0])
  const analysis = useAnalysis<RiskResponse>()
  const run = () => void analysis.run(() => api.risks({ company: 'Apple', older_year: 2024, newer_year: 2025, document_type: '10-K', topic_keys: [topic], top_k: 2 }))
  return <div className="page"><PageHeader eyebrow="Risk Radar" title="Track disclosure change, not invented severity" description="Compare complete Item 1A risk evidence across filings. “Changed” describes disclosure wording or content—not business severity." />
    <Panel><div className="topic-grid">{topics.map(([key, label]) => <button className={topic === key ? 'topic active' : 'topic'} onClick={() => setTopic(key)} key={key}>{label}</button>)}</div><button className="primary-button" onClick={run} disabled={analysis.loading}><Radar size={16} /> Compare disclosures</button></Panel>
    {analysis.loading && <LoadingState label="Comparing risk disclosures" />}{analysis.error && <ErrorState error={analysis.error} />}{analysis.result && <><div className="risk-grid">{analysis.result.data.comparisons.map((comparison) => <Panel className="risk-card" key={comparison.topic_key}><div className="risk-heading"><div><span className="kicker">{comparison.topic_label}</span><h2>Disclosure comparison</h2></div><StatusBadge state={comparison.temporal_state} /></div><p className="risk-note">Changed indicates a change in disclosure wording/content, not necessarily increased business severity.</p><div className="period-grid">{[comparison.older, comparison.newer].map((signal) => <div key={signal.fiscal_year}><div className="period-title"><strong>FY{signal.fiscal_year}</strong><StatusBadge state={signal.presence} /></div><small>{signal.qualifying_passage_count} qualifying passage{signal.qualifying_passage_count === 1 ? '' : 's'} · {signal.complete_item_1a_scan ? 'Complete Item 1A scan' : 'Partial scan'}</small>{signal.evidence.map((item, index) => <EvidenceCard key={index} evidence={item} />)}</div>)}</div>{comparison.error && <p className="inline-warning">{comparison.error}</p>}</Panel>)}</div><RequestMeta requestId={analysis.result.requestId} /></>}
  </div>
}
