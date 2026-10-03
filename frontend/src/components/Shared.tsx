import { useEffect, useState, type ReactNode } from 'react'
import { AlertTriangle, CheckCircle2, ChevronDown, Clock3, Database, FileText, ShieldCheck } from 'lucide-react'
import type { Evidence, VerificationState } from '../types/api'
import { ApiError } from '../types/api'

export function PageHeader({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) {
  return <header className="page-header"><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></header>
}

export function Panel({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <section className={`panel ${className}`}>{children}</section>
}

export function StatusBadge({ state }: { state: string }) {
  const normalized = state.toLowerCase()
  const good = ['verified', 'supported', 'ready', 'ok', 'increased', 'disclosed'].includes(normalized)
  const caution = ['flagged', 'ambiguous', 'partially_supported', 'recurring_ambiguous'].includes(normalized)
  return <span className={`status-badge ${good ? 'positive' : caution ? 'caution' : 'neutral'}`}>{good && <CheckCircle2 size={13} />}{state.replaceAll('_', ' ')}</span>
}

function score(value: number | null) { return value == null ? '—' : value.toFixed(3) }

export function EvidenceCard({ evidence, defaultOpen = false }: { evidence: Evidence; defaultOpen?: boolean }) {
  return <details className="evidence-card" id={evidence.evidence_id == null ? undefined : `evidence-${evidence.evidence_id}`} open={defaultOpen}>
    <summary>
      <span className="evidence-id">[{evidence.evidence_id ?? '—'}]</span>
      <span><strong>{evidence.company} · {evidence.document_type} · FY{evidence.fiscal_year}</strong><small>Page {evidence.page_number} · Chunk {evidence.chunk_id}</small></span>
      <ChevronDown size={16} className="chevron" />
    </summary>
    <div className="evidence-body">
      <p>{evidence.excerpt}</p>
      <div className="score-row" aria-label="Retrieval scores">
        <span>Vector <b>{score(evidence.vector_similarity)}</b></span>
        <span>Lexical <b>{score(evidence.lexical_score)}</b></span>
        <span>Fusion <b>{score(evidence.fusion_score)}</b></span>
        <span>Reranker <b>{score(evidence.rerank_score)}</b></span>
      </div>
      <small>Document {evidence.document_id}</small>
    </div>
  </details>
}

export function EvidenceList({ evidence, title = 'Filing evidence' }: { evidence: Evidence[]; title?: string }) {
  return <div className="evidence-list"><div className="section-heading"><FileText size={16} /><h3>{title}</h3><span>{evidence.length} source{evidence.length === 1 ? '' : 's'}</span></div>{evidence.map((item, index) => <EvidenceCard key={`${item.document_id}-${item.chunk_id}-${index}`} evidence={item} defaultOpen={index === 0} />)}</div>
}

export function LoadingState({ label = 'Analysis in progress' }: { label?: string }) {
  const [seconds, setSeconds] = useState(0)
  useEffect(() => { const timer = window.setInterval(() => setSeconds((value) => value + 1), 1000); return () => window.clearInterval(timer) }, [])
  return <div className="loading-state" role="status"><div className="pulse-ring"><Clock3 size={20} /></div><div><strong>{label}</strong><p>Retrieving, analyzing, and verifying local evidence. First-run inference may take up to two minutes.</p><small>{seconds}s elapsed · the API confirms completion only when the result returns</small></div></div>
}

export function ErrorState({ error }: { error: unknown }) {
  const apiError = error instanceof ApiError ? error : new ApiError('Unexpected interface error.', 0, 'ui_error', null)
  const busy = apiError.status === 429
  return <div className="error-state" role="alert"><AlertTriangle size={20} /><div><strong>{busy ? 'Local analysis is busy' : 'Analysis unavailable'}</strong><p>{apiError.message}</p><details><summary>Technical details</summary><code>{apiError.code} · HTTP {apiError.status || 'network'}{apiError.requestId ? ` · Request ${apiError.requestId}` : ''}</code></details></div></div>
}

export function RequestMeta({ requestId }: { requestId: string | null }) {
  if (!requestId) return null
  return <details className="request-meta"><summary>Technical details</summary><code>Request ID: {requestId}</code></details>
}

export function VerificationBanner({ state, citations }: { state: VerificationState; citations: number[] }) {
  const insufficient = state === 'insufficient_evidence'
  return <div className={`verification-banner ${state}`}><div>{insufficient ? <Database size={18} /> : <ShieldCheck size={18} />}<div><strong>{insufficient ? 'Insufficient evidence' : state === 'verified' ? 'Citation verified' : 'Verification flagged'}</strong><small>{insufficient ? 'The supplied filing evidence did not support a safe answer.' : `Validated citation IDs: ${citations.map((id) => `[${id}]`).join(', ') || 'none'}`}</small></div></div><StatusBadge state={state} /></div>
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return <label className="field"><span>{label}</span>{children}{hint && <small>{hint}</small>}</label>
}

export function CitationText({ text }: { text: string }) {
  const parts = text.split(/(\[\d+\])/g)
  return <>{parts.map((part, index) => /^\[\d+\]$/.test(part) ? <a key={index} className="inline-citation" href={`#evidence-${part.slice(1, -1)}`}>{part}</a> : part)}</>
}
