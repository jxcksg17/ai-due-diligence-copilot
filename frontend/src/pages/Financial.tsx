import { ArrowRight, Calculator } from 'lucide-react'
import { api } from '../api/client'
import { ErrorState, EvidenceList, Field, LoadingState, PageHeader, Panel, RequestMeta, StatusBadge } from '../components/Shared'
import { useAnalysis } from '../hooks/useAnalysis'
import type { RevenueResponse } from '../types/api'

function billions(amount: string, unit: string) { const value = Number(amount); return unit.toLowerCase().includes('million') ? `$${(value / 1000).toFixed(3)}B` : `$${value.toLocaleString()} ${unit}` }

export function Financial() {
  const analysis = useAnalysis<RevenueResponse>()
  const run = () => void analysis.run(() => api.revenue({ company: 'Apple', older_year: 2024, newer_year: 2025, document_type: '10-K', top_k: 10, rounding_places: 1 }))
  return <div className="page"><PageHeader eyebrow="Financial comparison" title="Authoritative arithmetic, traceable inputs" description="Source values come from filing evidence. Percentage change is computed deterministically with Python Decimal—not by the language model." />
    <Panel><div className="compact-controls"><Field label="Company"><input value="Apple" readOnly /></Field><Field label="Period"><input value="FY2024 → FY2025" readOnly /></Field><button className="primary-button" onClick={run} disabled={analysis.loading}><Calculator size={16} /> Compare revenue</button></div></Panel>
    {analysis.loading && <LoadingState label="Calculating from filing evidence" />}{analysis.error && <ErrorState error={analysis.error} />}{analysis.result && <><Panel className="financial-result"><div className="deterministic-label"><Calculator size={15} /> Deterministic calculation</div><div className="comparison-strip"><div><span>FY{analysis.result.data.older.fiscal_year}</span><strong>{billions(analysis.result.data.older.amount, analysis.result.data.older.unit)}</strong><small>Reported net sales</small></div><div className="change-block"><ArrowRight size={20} /><strong>{Number(analysis.result.data.absolute_change) >= 0 ? '+' : ''}{billions(analysis.result.data.absolute_change, analysis.result.data.older.unit)}</strong><b>{analysis.result.data.percentage_change}%</b><StatusBadge state={analysis.result.data.state} /></div><div><span>FY{analysis.result.data.newer.fiscal_year}</span><strong>{billions(analysis.result.data.newer.amount, analysis.result.data.newer.unit)}</strong><small>Reported net sales</small></div></div><div className="formula"><code>{analysis.result.data.formula}</code><span>Rounded to {analysis.result.data.rounding_places} decimal place</span></div><RequestMeta requestId={analysis.result.requestId} /></Panel><EvidenceList title="Source-value provenance" evidence={[analysis.result.data.older.evidence, analysis.result.data.newer.evidence]} /></>}
  </div>
}
