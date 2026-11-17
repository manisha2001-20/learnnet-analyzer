import { useCallback, useEffect, useState } from 'react'

const BASE = import.meta.env.VITE_API_URL || ''

export async function api(path, options = {}) {
  let res
  try {
    res = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    })
  } catch {
    throw new Error('Cannot reach the API. Is the backend running on port 8000?')
  }
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join('; ') : body.detail
    throw new Error(detail || `Request failed (${res.status})`)
  }
  return body
}

/** Small data-fetching hook: { data, error, loading, reload } */
export function useApi(path) {
  const [state, setState] = useState({ data: null, error: null, loading: true })
  const load = useCallback(() => {
    if (!path) return
    setState((s) => ({ ...s, loading: true, error: null }))
    api(path)
      .then((data) => setState({ data, error: null, loading: false }))
      .catch((error) => setState({ data: null, error: error.message, loading: false }))
  }, [path])
  useEffect(() => { load() }, [load])
  return { ...state, reload: load }
}
