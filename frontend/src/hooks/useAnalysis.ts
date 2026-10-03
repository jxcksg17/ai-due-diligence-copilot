import { useState } from 'react'
import { ApiError, type ApiResult } from '../types/api'

export function useAnalysis<T>() {
  const [result, setResult] = useState<ApiResult<T> | null>(null)
  const [error, setError] = useState<ApiError | false>(false)
  const [loading, setLoading] = useState(false)
  const run = async (operation: () => Promise<ApiResult<T>>) => {
    setLoading(true); setError(false); setResult(null)
    try { setResult(await operation()) } catch (caught) { setError(caught instanceof ApiError ? caught : new ApiError('Unexpected interface error.', 0, 'ui_error', null)) } finally { setLoading(false) }
  }
  return { result, error, loading, run }
}
