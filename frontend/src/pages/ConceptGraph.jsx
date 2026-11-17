import { useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { useApi } from '../api.js'
import { Card, ErrorBox, LevelBadge, Loading, PageHeader } from '../components/ui.jsx'
import { colorFor, levelOf, LEVEL_COLORS } from '../utils.js'

function useSize(ref) {
  const [w, setW] = useState(800)
  useEffect(() => {
    if (!ref.current) return
    const ro = new ResizeObserver(([e]) => setW(Math.max(300, e.contentRect.width)))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [ref])
  return w
}

export default function ConceptGraph() {
  const [kind, setKind] = useState('concept')
  const [colorBy, setColorBy] = useState('community')
  const [selected, setSelected] = useState(null)
  const graph = useApi(`/api/graph?kind=${kind}`)
  const concepts = useApi('/api/concepts')
  const box = useRef(null)
  const fg = useRef(null)
  const width = useSize(box)
  const dark = window.matchMedia?.('(prefers-color-scheme: dark)').matches

  const data = useMemo(() => graph.data && {
    nodes: graph.data.nodes.map((n) => ({ ...n })),
    links: graph.data.links.map((l) => ({ ...l })),
  }, [graph.data])
  const info = selected && concepts.data?.find((c) => c.concept === selected)

  return (
    <>
      <PageHeader title="Concept relationship graph">
        Nodes are concepts sized by PageRank. In the <b>concept</b> view, edges join concepts that co-occur or are
        semantically similar. In the <b>prerequisite</b> view, arrows point from a prerequisite to what it unlocks.
      </PageHeader>
      <div className="toolbar">
        <div className="segmented">
          {['concept', 'prerequisite'].map((k) => (
            <button key={k} className={kind === k ? 'on' : ''} onClick={() => setKind(k)}>{k}</button>
          ))}
        </div>
        <label className="inline">Colour by
          <select value={colorBy} onChange={(e) => setColorBy(e.target.value)}>
            <option value="community">community</option>
            <option value="difficulty">difficulty</option>
          </select>
        </label>
      </div>
      <div className="grid graph-layout">
        <Card className="graph-card">
          <div ref={box} className="graph-box">
            {graph.loading && <Loading />}
            {graph.error && <ErrorBox error={graph.error} onRetry={graph.reload} />}
            {data && (
              <ForceGraph2D
                ref={fg}
                graphData={data}
                onEngineStop={() => fg.current?.zoomToFit(400, 30)}
                width={width}
                height={560}
                backgroundColor="rgba(0,0,0,0)"
                nodeRelSize={4}
                nodeVal={(n) => 2 + n.pagerank * 400}
                nodeColor={(n) => colorBy === 'community' ? colorFor(n.community) : LEVEL_COLORS[levelOf(n.difficulty)]}
                linkColor={() => (dark ? 'rgba(148,163,184,0.25)' : 'rgba(100,116,139,0.25)')}
                linkWidth={(l) => Math.min(4, 0.4 + (l.weight || 1) * (kind === 'concept' ? 0.25 : 1.5))}
                linkDirectionalArrowLength={kind === 'prerequisite' ? 5 : 0}
                linkDirectionalArrowRelPos={1}
                cooldownTicks={120}
                onNodeClick={(n) => setSelected(n.id)}
                nodeCanvasObjectMode={() => 'after'}
                nodeCanvasObject={(node, ctx, scale) => {
                  if (scale < 1.2 && node.pagerank < 0.02 && node.id !== selected) return
                  ctx.font = `${Math.max(10 / scale, 3)}px Inter, sans-serif`
                  ctx.fillStyle = dark ? '#e2e8f0' : '#0f172a'
                  ctx.textAlign = 'center'
                  ctx.fillText(node.id, node.x, node.y + 10 / scale + 4)
                }}
              />
            )}
          </div>
        </Card>
        <Card title={info ? info.concept : 'Concept details'} subtitle={info ? null : 'Click a node to inspect it'}>
          {info && (
            <div className="details">
              <p><LevelBadge level={levelOf(info.difficulty)} /> difficulty {info.difficulty.toFixed(2)}</p>
              <p className="muted small">Appears in {info.doc_freq} videos · PageRank {info.pagerank.toFixed(4)} · community {info.community}</p>
              <h4>Prerequisites</h4>
              <div className="chips">{info.prerequisites.length ? info.prerequisites.map((p) => <button key={p} className="chip" onClick={() => setSelected(p)}>{p}</button>) : <span className="muted small">none detected</span>}</div>
              <h4>Unlocks</h4>
              <div className="chips">{info.unlocks.length ? info.unlocks.map((p) => <button key={p} className="chip" onClick={() => setSelected(p)}>{p}</button>) : <span className="muted small">none detected</span>}</div>
              <a className="btn primary small" href={`#/path`} onClick={() => sessionStorage.setItem('ln-goal', info.concept)}>Learning path to this concept →</a>
            </div>
          )}
        </Card>
      </div>
    </>
  )
}
