import { describe, expect, it } from 'vitest'
import { colorFor, formatNumber, levelOf, pct, wordCount, PALETTE } from './utils.js'

describe('utils', () => {
  it('maps difficulty to levels', () => {
    expect(levelOf(0.1)).toBe('beginner')
    expect(levelOf(0.5)).toBe('intermediate')
    expect(levelOf(0.9)).toBe('advanced')
  })
  it('formats percentages and numbers', () => {
    expect(pct(0.256, 1)).toBe('25.6%')
    expect(pct(null)).toBe('—')
    expect(formatNumber(15300)).toBe('15.3k')
    expect(formatNumber(2500000)).toBe('2.5M')
    expect(formatNumber(42)).toBe('42')
  })
  it('cycles colours safely', () => {
    expect(colorFor(PALETTE.length)).toBe(PALETTE[0])
    expect(colorFor(undefined)).toBe(PALETTE[0])
  })
  it('counts words', () => {
    expect(wordCount('  a  b c ')).toBe(3)
    expect(wordCount('')).toBe(0)
  })
})
