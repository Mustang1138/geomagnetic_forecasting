import {useCallback, useEffect, useRef, useState} from 'react'
import AuroralMap from './components/AuroralMap'
import TimelineChart from './components/TimelineChart'
import ModelSelector from './components/ModelSelector'
import StatsPanel from './components/StatsPanel'
import Controls from './components/Controls'
import ForecastTab from './components/ForecastTab'
import {useModels, usePredictions, useSnapshot} from './hooks/usePredictions'
import type {ModelKey} from './utils'

// Styles

const s = {
    root: {
        display: 'flex', flexDirection: 'column', height: '100vh',
    } as const,

    container: {
        maxWidth: 1400, width: '100%', margin: '0 auto',
        display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0,
    } as const,

    header: {
        padding: '8px 12px', borderBottom: '1px solid #ccc',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    },

    tabBar: {
        display: 'flex',
        borderBottom: '1px solid #ccc',
    },

    tab: (active: boolean) => ({
        padding: '6px 20px',
        cursor: 'pointer',
        border: 'none',
        borderBottom: active ? '2px solid #38bdf8' : '2px solid transparent',
        background: 'transparent',
        color: active ? '#38bdf8' : 'inherit',
        fontWeight: active ? 'bold' as const : 'normal' as const,
        fontSize: 13,
    }),

    controlsBar: {
        padding: '6px 12px', borderBottom: '1px solid #ccc',
        display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' as const,
    },

    main: {display: 'flex', flex: 1, minHeight: 0, gap: 8, padding: 8},

    mapArea: {
        flex: 3, minWidth: 0,
        border: '1px solid #ccc', overflow: 'hidden',
    },

    sidebar: {
        flex: 1, minWidth: 260, maxWidth: 340,
        border: '1px solid #ccc', overflow: 'hidden',
    },

    timelineBar: {
        padding: '8px 12px 12px', borderTop: '1px solid #ccc',
    },

    tlRow: {display: 'flex', alignItems: 'center', marginBottom: 4},
    tlIdx: {marginLeft: 'auto'},
    markersWrap: {position: 'relative' as const, height: 4, marginBottom: 4},

    marker: (pct: number) => ({
        position: 'absolute' as const, left: `${pct}%`,
        width: 1, height: 4, background: '#999',
    }),
} as const

// Helpers

/**
 * Converts storm-class entries to percentage positions for timeline tick marks.
 * Only steps classified as moderate (``'m'``) receive a visible marker.
 */
function buildMarkers(cl?: string[]): number[] {
    if (!cl) return []
    return cl
        .map((c, i) => ({c, pct: (i / (cl.length - 1)) * 100}))
        .filter(({c}) => c === 'm')
        .map(({pct}) => pct)
}

// Component

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
    const currentSSI = data?.pred[currentIdx] ?? 0
    const currentLat = data?.lat?.[currentIdx] ?? 63

    // Initialise markers once when prediction data first arrives.
    // A ref guards against re-initialising on subsequent data identity changes.
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
            if (!data || next >= data.n) {
                setPlaying(false);
                return prev
            }
            return next
        })
    }, [data])

    useEffect(() => {
        if (!playing) {
            playRef.current && clearTimeout(playRef.current);
            return
        }
        const ms = Math.max(30, 300 / speed)
        const loop = () => {
            tick();
            playRef.current = setTimeout(loop, ms)
        }
        playRef.current = setTimeout(loop, ms)
        return () => {
            playRef.current && clearTimeout(playRef.current)
        }
    }, [playing, speed, tick])

    const handleSeek = useCallback((idx: number) => {
        setPlaying(false)
        setCurrentIdx(idx)
    }, [])

    // Stop playback when switching tabs
    const handleTabChange = useCallback((tab: 'history' | 'forecast') => {
        setPlaying(false)
        setActiveTab(tab)
    }, [])

    return (
        <div style={s.root}>
            <div style={s.container}>

                {/* Header */}
                <header style={s.header}>
                    <strong>AURORA/CAST</strong>
                    <small>
                        {activeTab === 'history'
                            ? loading
                                ? 'Loading predictions…'
                                : !data
                                    ? 'API unavailable'
                                    : <>Test set: Feb 2021 – Feb 2026 · {n} steps (6-hr)<br/>{dt}</>
                            : '7-day forecast · Real-time DSCOVR · 6-hour steps'
                        }
                    </small>
                </header>

                {/* Tab bar */}
                <div style={s.tabBar}>
                    <button
                        style={s.tab(activeTab === 'history')}
                        onClick={() => handleTabChange('history')}
                    >
                        History
                    </button>
                    <button
                        style={s.tab(activeTab === 'forecast')}
                        onClick={() => handleTabChange('forecast')}
                    >
                        Forecast
                    </button>
                </div>

                {/* Forecast tab */}
                {activeTab === 'forecast' && <ForecastTab/>}

                {/* History tab */}
                {activeTab === 'history' && <>

                    {/* Controls bar */}
                    <div style={s.controlsBar}>
                        <ModelSelector activeModel={activeModel} models={models} onSelect={setActiveModel}/>
                        <Controls
                            playing={playing} speed={speed} view={view}
                            onTogglePlay={() => data && setPlaying(p => !p)}
                            onSpeed={setSpeed}
                            onView={setView}
                        />
                    </div>

                    {/* Main */}
                    <div style={s.main}>
                        <div style={s.mapArea}>
                            {loading
                                ? <p style={{padding: 8}}>Loading predictions…</p>
                                : <AuroralMap ssi={currentSSI} aLat={currentLat} view={view}/>
                            }
                        </div>
                        <div style={s.sidebar}>
                            <StatsPanel snapshot={snapshot} activeModel={activeModel} onSelectModel={setActiveModel}/>
                        </div>
                    </div>

                    {/* Timeline */}
                    <div style={s.timelineBar}>
                        <div style={s.tlRow}>
                            <span>{dt}</span>
                            <span style={s.tlIdx}>{n > 0 ? `${currentIdx + 1} / ${n}` : ''}</span>
                        </div>

                        <div style={s.markersWrap}>
                            {markers.map((pct, i) => <div key={i} style={s.marker(pct)}/>)}
                        </div>

                        <input
                            type="range" min={0} max={Math.max(0, n - 1)} value={currentIdx}
                            onChange={e => handleSeek(Number(e.target.value))}
                            style={{width: '100%'}}
                        />

                        {data && (
                            <TimelineChart
                                data={data}
                                modelKey={activeModel}
                                currentIdx={currentIdx}
                                onSeek={handleSeek}
                            />
                        )}
                    </div>
                </>}

            </div>
        </div>
    )
}