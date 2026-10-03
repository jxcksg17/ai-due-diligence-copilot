import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { App } from './App'

const evidence = { evidence_id: 1, chunk_id: 77, document_id: 1, company: 'Apple', document_type: '10-K', fiscal_year: 2025, page_number: 29, excerpt: 'Apple reported total net sales of $416.161 billion.', vector_similarity: .81, lexical_score: .72, fusion_score: .03, rerank_score: 8.42 }

function response(body: unknown, status = 200, requestId = 'ui-test-1') {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json', 'X-Request-ID': requestId } }))
}

beforeEach(() => { vi.restoreAllMocks() })

describe('portfolio interface', () => {
  it('renders the evidence-first overview', () => {
    render(<App />)
    expect(screen.getByText('Evidence-grounded financial intelligence')).toBeInTheDocument()
    expect(screen.getByText('639')).toBeInTheDocument()
    expect(screen.getByText('DeBERTa NLI')).toBeInTheDocument()
  })

  it('renders a verified answer with filing evidence', async () => {
    vi.stubGlobal('fetch', vi.fn(() => response({ answer: 'Net sales were $416.161 billion [1].', citation_ids: [1], insufficient_evidence: false, verification_state: 'verified', evidence: [evidence], claim_verifications: [{ claim_text: 'Net sales were $416.161 billion.', citation_ids: [1], status: 'supported' }] })))
    const user = userEvent.setup(); render(<App />)
    await user.click(screen.getByRole('button', { name: 'Grounded Q&A' })); await user.click(screen.getByRole('button', { name: /Analyze filing/ }))
    expect(await screen.findByText('Citation verified')).toBeInTheDocument()
    expect(screen.getByText(/Apple reported total net sales/)).toBeInTheDocument()
    expect(screen.getByText(/Request ID: ui-test-1/)).toBeInTheDocument()
  })

  it('treats insufficient evidence as a safe outcome', async () => {
    vi.stubGlobal('fetch', vi.fn(() => response({ answer: 'The evidence is insufficient to answer this question.', citation_ids: [], insufficient_evidence: true, verification_state: 'insufficient_evidence', evidence: [], claim_verifications: [] })))
    const user = userEvent.setup(); render(<App />)
    await user.click(screen.getByRole('button', { name: 'Grounded Q&A' })); await user.click(screen.getByText(/employee satisfaction score/)); await user.click(screen.getByRole('button', { name: /Analyze filing/ }))
    expect(await screen.findByText('Insufficient evidence')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('renders structured API errors and request IDs', async () => {
    vi.stubGlobal('fetch', vi.fn(() => response({ error: { code: 'model_unavailable', message: 'Local model is unavailable.', request_id: 'error-42' } }, 503, 'error-42')))
    const user = userEvent.setup(); render(<App />)
    await user.click(screen.getByRole('button', { name: 'Grounded Q&A' })); await user.click(screen.getByRole('button', { name: /Analyze filing/ }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Local model is unavailable.')
    await user.click(screen.getByText('Technical details'))
    expect(screen.getByText(/error-42/)).toBeInTheDocument()
  })

  it('renders deterministic financial comparison and provenance', async () => {
    vi.stubGlobal('fetch', vi.fn(() => response({ metric: 'total_net_sales', older: { fiscal_year: 2024, amount: '391035', unit: 'millions USD', evidence }, newer: { fiscal_year: 2025, amount: '416161', unit: 'millions USD', evidence: { ...evidence, chunk_id: 78 } }, absolute_change: '25126', percentage_change: '6.4', state: 'increased', formula: '((newer - older) / older) * 100', rounding_places: 1 })))
    const user = userEvent.setup(); render(<App />)
    await user.click(screen.getByRole('button', { name: 'Financial comparison' })); await user.click(screen.getByRole('button', { name: /Compare revenue/ }))
    expect(await screen.findByText('$391.035B')).toBeInTheDocument()
    expect(screen.getByText('$416.161B')).toBeInTheDocument()
    expect(screen.getByText('Deterministic calculation')).toBeInTheDocument()
  })

  it('renders neutral Risk Radar state and claim alignment', async () => {
    const fetchMock = vi.fn()
      .mockImplementationOnce(() => response({ company: 'Apple', document_type: '10-K', older_year: 2024, newer_year: 2025, comparisons: [{ topic_key: 'supply_chain_manufacturing', topic_label: 'Supply chain and manufacturing', temporal_state: 'recurring_changed', older: { fiscal_year: 2024, presence: 'disclosed', qualifying_passage_count: 2, complete_item_1a_scan: true, evidence: [evidence] }, newer: { fiscal_year: 2025, presence: 'disclosed', qualifying_passage_count: 2, complete_item_1a_scan: true, evidence: [evidence] }, error: null }] }))
      .mockImplementationOnce(() => response({ claim_id: 'fy2025_revenue_amount', normalized_claim: 'Revenue reached approximately $416 billion.', state: 'supported', rationale: 'The rounded claim agrees with the filing amount.', claimed_amount_millions: '416000', filing_amount_millions: '416161', rounding_tolerance_millions: '500', source: { speaker_name: 'Tim Cook', speaker_role: 'CEO', source_title: 'FY2025 Q4 earnings release', source_date: '2025-10-30', exhibit_label: 'Exhibit 99.1', source_url: 'https://example.test', quote: 'Revenue reached $416 billion.', evidence }, filing_evidence: [evidence] }))
    vi.stubGlobal('fetch', fetchMock); const user = userEvent.setup(); render(<App />)
    await user.click(screen.getByRole('button', { name: 'Risk Radar' })); await user.click(screen.getByRole('button', { name: /Compare disclosures/ }))
    expect(await screen.findByText('recurring changed')).toBeInTheDocument()
    expect(screen.getByText(/not necessarily increased business severity/)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Claim vs Evidence' })); await user.click(screen.getByRole('button', { name: /Assess alignment/ }))
    expect(await screen.findByText('$416.161 billion')).toBeInTheDocument()
    expect(screen.getByText(/evidence alignment, not a judgment/i)).toBeInTheDocument()
  })

  it('shows loading and readiness checks', async () => {
    let resolve!: (value: Response) => void
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>((done) => { resolve = done })))
    const user = userEvent.setup(); render(<App />)
    await user.click(screen.getByRole('button', { name: 'Grounded Q&A' })); await user.click(screen.getByRole('button', { name: /Analyze filing/ }))
    expect(screen.getByRole('status')).toHaveTextContent('Analysis in progress')
    resolve(await response({ answer: 'Done [1].', citation_ids: [1], insufficient_evidence: false, verification_state: 'verified', evidence: [evidence], claim_verifications: [] }))
    await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument())

    vi.stubGlobal('fetch', vi.fn(() => response({ status: 'ready', checks: { configuration: { status: 'ok', detail: null }, database: { status: 'ok', detail: null }, migrations: { status: 'ok', detail: null }, ollama: { status: 'ok', detail: null } } })))
    await user.click(screen.getByRole('button', { name: 'System status' }))
    expect(await screen.findByText('All required systems operational')).toBeInTheDocument()
    expect(screen.getByText('database')).toBeInTheDocument()
  })
})
