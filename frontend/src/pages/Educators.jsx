import { Bar, BarChart, CartesianGrid, Legend, PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useApi } from '../api.js'
import { Card, ErrorBox, Loading, PageHeader } from '../components/ui.jsx'
import { colorFor, formatNumber } from '../utils.js'

const LABELS = { coverage: 'Coverage', foundational: 'Foundational', precedence: 'Precedence', breadth: 'Breadth', reach: 'Reach' }

export default function Educators() {
  const ed = useApi('/api/educators')
  if (ed.loading) return <Loading />
  if (ed.error) return <ErrorBox error={ed.error} onRetry={ed.reload} />
  const rows = ed.data.educators
  const comps = Object.keys(rows[0]?.components || {})
  const radar = comps.map((c) => Object.fromEntries([['component', LABELS[c] || c], ...rows.map((r) => [r.educator, r.components[c]])]))
  const tipStyle = { background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8 }

  return (
    <>
      <PageHeader title="Pedagogical influence">
        A transparent score (0–100) that combines how much of the curriculum an educator covers, how central those
        concepts are, whether they taught concepts before other educators (precedence network PageRank), topic
        breadth and audience reach. It compares educators within this corpus only.
      </PageHeader>
      <div className="grid two">
        <Card title="Influence score">
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={rows} layout="vertical" margin={{ left: 30, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--grid)" />
              <XAxis type="number" domain={[0, 100]} tick={{ fill: 'var(--muted)' }} />
              <YAxis type="category" dataKey="educator" width={130} tick={{ fill: 'var(--text)', fontSize: 13 }} />
              <Tooltip contentStyle={tipStyle} />
              <Bar dataKey="influence" fill="var(--primary)" radius={[0, 6, 6, 0]} label={{ position: 'right', fill: 'var(--text)', fontSize: 12 }} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Score components" subtitle="Each scaled 0–1 across educators">
          <ResponsiveContainer width="100%" height={340}>
            <RadarChart data={radar} outerRadius="72%">
              <PolarGrid stroke="var(--grid)" />
              <PolarAngleAxis dataKey="component" tick={{ fill: 'var(--text)', fontSize: 12 }} />
              <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
              {rows.map((r, i) => <Radar key={r.educator} name={r.educator} dataKey={r.educator} stroke={colorFor(i)} fill={colorFor(i)} fillOpacity={0.12} />)}
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Tooltip contentStyle={tipStyle} />
            </RadarChart>
          </ResponsiveContainer>
        </Card>
      </div>
      <Card title="Details">
        <div className="table-wrap">
          <table className="table">
            <thead><tr><th>#</th><th>Educator</th><th className="num">Influence</th><th className="num">Videos</th><th className="num">Concepts</th><th className="num">Views</th>{comps.map((c) => <th key={c} className="num">{LABELS[c]}</th>)}</tr></thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={r.educator}>
                  <td>{i + 1}</td><td className="strong"><span className="dot" style={{ background: colorFor(i) }} />{r.educator}</td>
                  <td className="num strong">{r.influence}</td><td className="num">{r.videos}</td><td className="num">{r.concepts_taught}</td>
                  <td className="num">{formatNumber(r.total_views)}</td>
                  {comps.map((c) => <td key={c} className="num">{r.components[c].toFixed(2)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted small">Precedence links (who learned from whom, by publish date): {ed.data.precedence_links.map((l) => `${l.source} → ${l.target} (${l.weight})`).join(' · ') || 'none'}</p>
      </Card>
    </>
  )
}
