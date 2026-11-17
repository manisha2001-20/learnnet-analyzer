export const PALETTE = ['#4f46e5', '#0ea5e9', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#64748b', '#84cc16']

export const LEVEL_COLORS = { beginner: '#10b981', intermediate: '#f59e0b', advanced: '#ef4444' }

export function colorFor(index) {
  const i = Number.isFinite(index) ? Math.abs(Math.trunc(index)) : 0
  return PALETTE[i % PALETTE.length]
}

export function levelOf(difficulty) {
  if (difficulty < 0.34) return 'beginner'
  if (difficulty < 0.67) return 'intermediate'
  return 'advanced'
}

export function pct(x, digits = 0) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—'
  return `${(x * 100).toFixed(digits)}%`
}

export function formatNumber(n) {
  if (n === null || n === undefined) return '—'
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`
  return String(n)
}

export function wordCount(text) {
  return (text || '').trim().split(/\s+/).filter(Boolean).length
}
