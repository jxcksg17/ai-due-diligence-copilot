import type { ApiResult, ClaimResponse, QueryResponse, ReadinessResponse, RevenueResponse, RiskResponse } from '../types/api'
import { ApiError } from '../types/api'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000').replace(/\/$/, '')
const DEFAULT_TIMEOUT_MS = 210_000

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<ApiResult<T>> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init.headers },
      signal: controller.signal,
    })
    const requestId = response.headers.get('x-request-id')
    const body = await response.json().catch(() => null)
    if (!response.ok) {
      const error = body?.error
      throw new ApiError(error?.message ?? `Request failed (${response.status})`, response.status, error?.code ?? 'unknown_error', error?.request_id ?? requestId, error?.details)
    }
    return { data: body as T, requestId }
  } catch (error) {
    if (error instanceof ApiError) throw error
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError('Local analysis exceeded the client timeout.', 504, 'client_timeout', null)
    }
    throw new ApiError('Unable to reach the local API.', 0, 'network_error', null)
  } finally {
    window.clearTimeout(timer)
  }
}

const post = <T>(path: string, payload: unknown) => request<T>(path, { method: 'POST', body: JSON.stringify(payload) })

export const api = {
  query: (payload: unknown) => post<QueryResponse>('/api/v1/query', payload),
  revenue: (payload: unknown) => post<RevenueResponse>('/api/v1/temporal/revenue', payload),
  risks: (payload: unknown) => post<RiskResponse>('/api/v1/risk-radar/compare', payload),
  claim: (payload: unknown) => post<ClaimResponse>('/api/v1/claim-evidence/assess', payload),
  live: () => request<{ status: string; app_name: string; environment: string }>('/health/live', {}, 8_000),
  ready: () => request<ReadinessResponse>('/health/ready', {}, 8_000),
}
