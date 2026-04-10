import {useCallback, useEffect, useRef, useState} from 'react'
import AuroralMap from './components/AuroralMap'
import TimelineChart from './components/TimelineChart'
import ModelSelector from './components/ModelSelector'
import StatsPanel from './components/StatsPanel'
import Controls from './components/Controls'
import ForecastTab from './components/ForecastTab'
import {useModels, usePredictions, useSnapshot} from './hooks/usePredictions'
import type {ModelKey} from './utils'

const styles = {
    root: {
        display: 'flex', flexDirection: 'column', height: '100vh',
        background: 'var(--bg)',
    } as const,

    container: {
        maxWidth: 1400, width: '100%', margin: '0 auto',
        display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0,
        borderInline: '1px solid var(--border)',
    } as const,

    header: {
        padding: '9px 16px',
        borderBottom: '1px solid var(--border)',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        background: 'var(--surface)',
        gap: 16, flexShrink: 0,
    } as const,

    brand: {
        display: 'flex', alignItems: 'center', gap: 9,
        fontFamily: 'var(--font-mono)',
        fontSize: 12, fontWeight: 500,
        letterSpacing: '0.07em',
        color: 'var(--text)',
        textTransform: 'uppercase',
        flexShrink: 0,
    } as const,

    brandIcon: {
        fontSize: 15, color: 'var(--accent)', lineHeight: 1,
    } as const,

    statusBar: {
        display: 'flex', alignItems: 'center', gap: 10,
        fontFamily: 'var(--font-mono)',
        fontSize: 11, color: 'var(--text-2)',
    } as const,

    tabBar: {
        display: 'flex',
        borderBottom: '1px solid var(--border)',
        background: 'var(--surface)',
        flexShrink: 0,
    } as const,

    controlsBar: {
        padding: '6px 14px',
        borderBottom: '1px solid var(--border)',
        display: 'flex', gap: 10, alignItems: 'center',
        background: 'var(--panel)',
        flexShrink: 0,
    } as const,

    main: {display: 'flex', flex: 1, minHeight: 0},

    mapArea: {
        flex: 3, minWidth: 0, overflow: 'hidden',
        borderRight: '1px solid var(--border)',
    } as const,

    sidebar: {
        flex: 1, minWidth: 260, maxWidth: 320,
        overflow: 'hidden',
        background: 'var(--panel)',
    } as const,

    timelineBar: {
        padding: '8px 14px 10px',
        borderTop: '1px solid var(--border)',
        background: 'var(--surface)',
        flexShrink: 0,
    } as const,

    tlRow: {
        display: 'flex', alignItems: 'center',
        marginBottom: 6, gap: 8,
    } as const,

    tlDt: {
        fontFamily: 'var(--font-mono)',
        fontSize: 12, color: 'var(--text)',
    } as const,

    tlIdx: {
        marginLeft: 'auto',
        fontFamily: 'var(--font-mono)',
        fontSize: 11, color: 'var(--text-3)',
    } as const,

    markersWrap: {position: 'relative' as const, height: 4, marginBottom: 3},

    marker: (pct: number) => ({
        position: 'absolute' as const, left: `${pct}%`,
        width: 1, height: 4,
        background: 'rgba(249,115,22,0.7)',
    }),
}

function buildMarkers(cl?: string[]): number[] {
    if (!cl) return []
    return cl
        .map((c, i) => ({c, pct: (i / (cl.length - 1)) * 100}))
        // Only moderate-or-above steps produce a visible marker.
        .filter(({c}) => c === 'm')
        .map(({pct}) => pct)
}

export default function App() {
    const [activeTab, setActiveTab] = useState<'history' | 'forecast'>('history')
    const [activeModel, setActiveModel] = useState<ModelKey>('rf')
    const [currentIdx, setCurrentIdx] = useState(0)
    const [view, setView] = useState<'north' | 'south' | 'rect'>('north')
    const [playing, setPlaying] = useState(false)
    const [speed, setSpeed] = useState(1)
    const [markers, setMarkers] = useState<number[]>([])

    const {models} = useModels()
    const {data, loading} = usePredictions(activeModel)
    const snapshot = useSnapshot(currentIdx)

    const playRef = useRef<number | null>(null)

    const n = data?.n ?? 0
    const dt = data?.dt?.[currentIdx] ?? '—'

    // Globe reflects observed conditions, not predictions.
    const globeSSI = snapshot?.true ?? data?.true?.[currentIdx] ?? 0
    const globeAuroralLatitudeDeg = snapshot?.lat ?? data?.lat?.[currentIdx] ?? 63

    const markersInitialised = useRef(false)
    useEffect(() => {
        if (data && !markersInitialised.current) {
            markersInitialised.current = true
            setMarkers(buildMarkers(data.cl))
        }
    }, [data])

    const tick = useCallback(() => {
        setCurrentIdx(prev => {
            const next = prev + 1
            if (!data || next >= data.n) { setPlaying(false); return prev }
            return next
        })
    }, [data])

    useEffect(() => {
        if (!playing) { playRef.current && clearTimeout(playRef.current); return }
        const ms = Math.max(30, 300 / speed)
        const loop = () => { tick(); playRef.current = setTimeout(loop, ms) }
        playRef.current = setTimeout(loop, ms)
        return () => { playRef.current && clearTimeout(playRef.current) }
    }, [playing, speed, tick])

    const handleSeek = useCallback((idx: number) => { setPlaying(false); setCurrentIdx(idx) }, [])

    const handleTabChange = useCallback((tab: 'history' | 'forecast') => {
        setPlaying(false)
        setActiveTab(tab)
    }, [])

    return (
        <div style={styles.root}>
            <div style={styles.container}>

                <header style={styles.header}>
                    <div style={styles.brand}>
                        <span style={styles.brandIcon}>⊕</span>
                        Geomagnetic Storm Dashboard
                    </div>
                    <div style={styles.statusBar}>
                        <span className="live-dot"/>
                        {activeTab === 'history'
                            ? loading
                                ? 'Loading…'
                                : !data
                                    ? <span style={{color: '#ef4444'}}>API unavailable</span>
                                    : <><span style={{color: 'var(--text-3)'}}>
                                        Test set · Dec 2022 – Present · {n} steps · 6 hr
                                      </span><span style={{color: 'var(--text)'}}>{dt}</span></>
                            : <span style={{color: 'var(--text-3)'}}>7-day forecast · DSCOVR real-time · 6-hour steps</span>
                        }
                    </div>
                </header>

                <div style={styles.tabBar}>
                    <button className="tab-btn" disabled={activeTab === 'history'}
                            onClick={() => handleTabChange('history')}>
                        History
                    </button>
                    <button className="tab-btn" disabled={activeTab === 'forecast'}
                            onClick={() => handleTabChange('forecast')}>
                        Forecast
                    </button>
                </div>

                {activeTab === 'forecast' && <ForecastTab/>}

                {activeTab === 'history' && <>

                    <div style={styles.controlsBar}>
                        <ModelSelector activeModel={activeModel} models={models} onSelect={setActiveModel}/>
                        <div className="vdivider"/>
                        <Controls
                            playing={playing} speed={speed} view={view}
                            onTogglePlay={() => data && setPlaying(p => !p)}
                            onSpeed={setSpeed}
                            onView={setView}
                        />
                    </div>

                    <div style={styles.main}>
                        <div style={styles.mapArea}>
                            {loading
                                ? <p style={{padding: 12, color: 'var(--text-2)'}}>Loading predictions…</p>
                                : <AuroralMap ssi={globeSSI} auroralLatitudeDeg={globeAuroralLatitudeDeg} view={view}/>
                            }
                        </div>
                        <div style={styles.sidebar}>
                            <StatsPanel snapshot={snapshot} activeModel={activeModel} onSelectModel={setActiveModel}/>
                        </div>
                    </div>

                    <div style={styles.timelineBar}>
                        <div style={styles.tlRow}>
                            <span style={styles.tlDt}>{dt}</span>
                            <span style={styles.tlIdx}>{n > 0 ? `${currentIdx + 1} / ${n}` : ''}</span>
                        </div>

                        <div style={styles.markersWrap}>
                            {markers.map((pct, i) => <div key={i} style={styles.marker(pct)}/>)}
                        </div>

                        <input
                            type="range" min={0} max={Math.max(0, n - 1)} value={currentIdx}
                            onChange={e => handleSeek(Number(e.target.value))}
                            style={{marginBottom: 4}}
                        />

                        {data && (
                            <TimelineChart
                                data={data} modelKey={activeModel}
                                currentIdx={currentIdx} onSeek={handleSeek}
                            />
                        )}
                    </div>
                </>}

            </div>
        </div>
    )
}
