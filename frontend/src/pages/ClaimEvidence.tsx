import { ArrowDown, Scale } from 'lucide-react'
import { api } from '../api/client'
import { ErrorState, EvidenceList, LoadingState, PageHeader, Panel, RequestMeta, StatusBadge } from '../components/Shared'
import { useAnalysis } from '../hooks/useAnalysis'
import type { ClaimResponse } from '../types/api'

function amount(value: string | null) { return value == null ? 'Not applicable' : `$${(Number(value) / 1000).toFixed(3)} billion` }

export function ClaimEvidence() {
  const analysis = useAnalysis<ClaimResponse>()
  const run = () => void analysis.run(() => api.claim({ company: 'Apple', fiscal_year: 2025, claim_id: 'fy2025_revenue_amount', older_year: 2024, newer_year: 2025, top_k: 10 }))
  return <div className="page"><PageHeader eyebrow="Management Claim vs Evidence" title="Separate the statement from the proof" description="Assess attributable issuer statements against filing evidence—without inferring management intent or honesty." />
    <Panel className="claim-launch"><div><span className="kicker">Demo claim</span><blockquote>“Revenue reaching $416 billion”</blockquote><small>Apple FY2025 Q4 earnings release · official SEC Exhibit 99.1</small></div><button className="primary-button" onClick={run} disabled={analysis.loading}><Scale size={16} /> Assess alignment</button></Panel>
    {analysis.loading && <LoadingState label="Aligning claim and filing evidence" />}{analysis.error && <ErrorState error={analysis.error} />}{analysis.result && <><div className="claim-flow"><Panel><div className="flow-label">01 · Management statement</div><blockquote>“{analysis.result.data.source.quote}”</blockquote><p>{analysis.result.data.source.speaker_name} · {analysis.result.data.source.speaker_role}</p><small>{analysis.result.data.source.source_title} · {analysis.result.data.source.source_date}</small></Panel><ArrowDown className="flow-arrow" /><Panel><div className="flow-label">02 · Numeric alignment</div><div className="numeric-alignment"><div><span>Claimed</span><strong>{amount(analysis.result.data.claimed_amount_millions)}</strong></div><div><span>Filing</span><strong>{amount(analysis.result.data.filing_amount_millions)}</strong></div></div></Panel><ArrowDown className="flow-arrow" /><Panel className="assessment-panel"><div className="flow-label">03 · Evidence assessment</div><StatusBadge state={analysis.result.data.state} /><h2>{analysis.result.data.state.replaceAll('_', ' ')}</h2><p>{analysis.result.data.rationale}</p><small>This is evidence alignment, not a judgment of intent or credibility.</small></Panel></div><EvidenceList title="Filing evidence" evidence={analysis.result.data.filing_evidence} /><RequestMeta requestId={analysis.result.requestId} /></>}
  </div>
}
