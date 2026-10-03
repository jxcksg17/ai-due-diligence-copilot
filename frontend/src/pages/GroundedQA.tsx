import { useState } from 'react'
import { Send } from 'lucide-react'
import { api } from '../api/client'
import { CitationText, ErrorState, EvidenceList, Field, LoadingState, PageHeader, Panel, RequestMeta, VerificationBanner } from '../components/Shared'
import { useAnalysis } from '../hooks/useAnalysis'
import type { QueryResponse } from '../types/api'

const examples = [
  "What were Apple's total net sales in fiscal 2025?",
  'What supply-chain risks did Apple identify?',
  'What did Apple spend on R&D in 2025?',
  "What was Apple's operating cash flow?",
  "What was Apple's employee satisfaction score in 2025?",
]

export function GroundedQA() {
  const [question, setQuestion] = useState(examples[0])
  const [company, setCompany] = useState('Apple')
  const [documentType, setDocumentType] = useState('10-K')
  const [year, setYear] = useState(2025)
  const [topK, setTopK] = useState(5)
  const analysis = useAnalysis<QueryResponse>()
  const submit = (event: React.FormEvent) => { event.preventDefault(); void analysis.run(() => api.query({ question, company, document_type: documentType, fiscal_year: year, top_k: topK })) }
  return <div className="page"><PageHeader eyebrow="Grounded Q&A" title="Ask the filing, inspect the proof" description="Hybrid retrieval finds evidence, Qwen3 answers from that evidence, and DeBERTa checks citation support." />
    <div className="workspace-grid"><Panel><form onSubmit={submit} className="analysis-form"><Field label="Question"><textarea value={question} onChange={(e) => setQuestion(e.target.value)} rows={5} /></Field><div className="form-row"><Field label="Company"><input value={company} onChange={(e) => setCompany(e.target.value)} /></Field><Field label="Document"><select value={documentType} onChange={(e) => setDocumentType(e.target.value)}><option>10-K</option></select></Field><Field label="Fiscal year"><input type="number" value={year} onChange={(e) => setYear(Number(e.target.value))} /></Field><Field label="Top evidence"><input type="number" min="1" max="10" value={topK} onChange={(e) => setTopK(Number(e.target.value))} /></Field></div><button className="primary-button" disabled={analysis.loading}><Send size={16} /> Analyze filing</button></form><div className="example-list"><span>Demo prompts</span>{examples.map((example) => <button key={example} onClick={() => setQuestion(example)}>{example}</button>)}</div></Panel>
      <div className="result-column">{analysis.loading && <LoadingState />}{analysis.error && <ErrorState error={analysis.error} />}{!analysis.loading && !analysis.error && !analysis.result && <Panel className="empty-state"><span className="empty-index">01</span><h3>Ready for a grounded question</h3><p>Choose a prompt or ask a question about Apple’s validated filings.</p></Panel>}{analysis.result && <><Panel className="answer-panel"><VerificationBanner state={analysis.result.data.verification_state} citations={analysis.result.data.citation_ids} /><div className="answer-copy"><CitationText text={analysis.result.data.answer} /></div>{analysis.result.data.claim_verifications.length > 0 && <div className="claim-checks">{analysis.result.data.claim_verifications.map((claim, index) => <div key={index}><span>{claim.status}</span><p>{claim.claim_text}</p></div>)}</div>}<RequestMeta requestId={analysis.result.requestId} /></Panel><EvidenceList evidence={analysis.result.data.evidence} /></>}</div></div>
  </div>
}
