export type VerificationState = 'verified' | 'flagged' | 'insufficient_evidence'

export interface Evidence {
  evidence_id: number | null
  chunk_id: number
  document_id: number
  company: string
  document_type: string
  fiscal_year: number
  page_number: number
  excerpt: string
  vector_similarity: number | null
  lexical_score: number | null
  fusion_score: number | null
  rerank_score: number | null
}

export interface QueryResponse {
  answer: string
  citation_ids: number[]
  insufficient_evidence: boolean
  verification_state: VerificationState
  evidence: Evidence[]
  claim_verifications: Array<{
    claim_text: string
    citation_ids: number[]
    status: 'supported' | 'unsupported' | 'ambiguous' | 'error'
  }>
}

export interface RevenueResponse {
  metric: string
  older: { fiscal_year: number; amount: string; unit: string; evidence: Evidence }
  newer: { fiscal_year: number; amount: string; unit: string; evidence: Evidence }
  absolute_change: string
  percentage_change: string | null
  state: 'increased' | 'declined' | 'unchanged'
  formula: string
  rounding_places: number
}

export interface RiskComparison {
  topic_key: string
  topic_label: string
  temporal_state: string
  older: RiskSignal
  newer: RiskSignal
  error: string | null
}

export interface RiskSignal {
  fiscal_year: number
  presence: 'disclosed' | 'not_found'
  qualifying_passage_count: number
  complete_item_1a_scan: boolean
  evidence: Evidence[]
}

export interface RiskResponse {
  company: string
  document_type: string
  older_year: number
  newer_year: number
  comparisons: RiskComparison[]
}

export type ClaimState = 'supported' | 'partially_supported' | 'contradicted' | 'ambiguous' | 'insufficient_evidence' | 'not_objectively_verifiable'

export interface ClaimResponse {
  claim_id: string
  normalized_claim: string
  state: ClaimState
  rationale: string
  claimed_amount_millions: string | null
  filing_amount_millions: string | null
  rounding_tolerance_millions: string | null
  source: {
    speaker_name: string
    speaker_role: string
    source_title: string
    source_date: string
    exhibit_label: string
    source_url: string
    quote: string
    evidence: Evidence
  }
  filing_evidence: Evidence[]
}

export interface ReadinessResponse {
  status: 'ready' | 'not_ready'
  checks: Record<string, { status: 'ok' | 'error'; detail: string | null }>
}

export interface ApiResult<T> { data: T; requestId: string | null }

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code: string,
    public requestId: string | null,
    public details: unknown = null,
  ) { super(message) }
}
