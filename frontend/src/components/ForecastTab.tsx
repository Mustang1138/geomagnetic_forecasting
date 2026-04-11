import {useCallback, useState} from 'react'
import AuroralMap from './AuroralMap'
import ForecastChart from './ForecastChart'
import ModelSelector from './ModelSelector'
import Controls from './Controls'
import StatsPanel from './StatsPanel'
import {useForecast} from '../hooks/useForecast'
import {useModels} from '../hooks/usePredictions'
import type {ModelKey, Snapshot} from '../utils'
import {MODEL_META, ssiColor, ssiLabel} from '../utils'

const styles = {
    root: {display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0} as const,

    notice: {
        padding: '5px 16px',
        background: 'var(--surface)',
        borderBottom: '1px solid var(--border)',
        fontSize: 11,
        color: 'var(--text-3)',
        fontFamily: 'var(--font-mono)',
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
        marginBottom: 6, gap: 10,
    } as const,

    ssiPill: (ssi: number) => ({
        fontFamily: 'var(--font-mono)',
        padding: '1px 8px',
        background: ssiColor(ssi, 0.12),
        color: ssiColor(ssi),
        border: `1px solid ${ssiColor(ssi, 0.35)}`,
        fontSize: 11,
    }),

    tlIdx: {
        marginLeft: 'auto',
        fontFamily: 'var(--font-mono)',
        fontSize: 11, color: 'var(--text-3)',
    } as const,

    legend: {
        display: 'flex', gap: 12, flexWrap: 'wrap' as const,
        marginTop: 5, fontSize: 11,
        fontFamily: 'var(--font-mono)',
        color: 'var(--text-3)',
    } as const,
}

function buildSnapshot(
    data: ReturnType<typeof useForecast>['data'],
    activeModel: ModelKey,
    idx: number,
): Snapshot | null {
    if (!data) return null
    return {
        true: data.models[activeModel]?.ssi[idx] ?? 0,
        lat:  data.models[activeModel]?.auroral_lat[idx] ?? 63,
        models: Object.fromEntries(
            MODEL_META.map(({key}) => [key, data.models[key]?.ssi[idx] ?? 0])
        ) as Record<ModelKey, number>,
    }
}

function ForecastLoading() {
    return (
        <div style={{padding: 24, flex: 1, color: 'var(--text-2)'}}>
            <p>Fetching real-time DSCOVR solar wind data…</p>
            <p style={{color: 'var(--text-3)', fontSize: 12, marginTop: 6}}>
                Building seed window and running forecast models. This may take a moment.
            </p>
        </div>
    )
}

function ForecastError({message}: {message: string}) {
    return (
        <div style={{padding: 24, flex: 1}}>
            <p style={{color: '#ef4444'}}>Forecast unavailable</p>
            <p style={{color: 'var(--text-3)', fontSize: 12, marginTop: 6}}>
                {message}<br/>Please try refreshing the page.
            </p>
        </div>
    )
}

/** Real-time 7-day forecast view driven by live DSCOVR solar wind observations. */
export default function ForecastTab() {
    const {data, loading, error} = useForecast()
    const {models} = useModels()

    const [activeModel, setActiveModel] = useState<ModelKey>('rf')
    const [currentIdx, setCurrentIdx] = useState(0)
    const [view, setView] = useState<'north' | 'south' | 'rect'>('north')

    const handleSeek = useCallback((idx: number) => setCurrentIdx(idx), [])

    if (loading) return <ForecastLoading/>
    if (error || !data) return <ForecastError
        message={error ?? 'Could not retrieve real-time DSCOVR solar wind data from NOAA SWPC.'}/>

    const currentSSI = data.models[activeModel]?.ssi[currentIdx] ?? 0
    const currentDt  = data.timestamps[currentIdx] ?? '—'
    const snapshot   = buildSnapshot(data, activeModel, currentIdx)

    return (
        <div style={styles.root}>

            <div style={styles.notice}>
                Driven by real DSCOVR solar wind observations (past 7 days, 6-hour averages).
                {' '}Generated {data.generated_at} · {data.steps} steps · {data.step_hours}h cadence
            </div>

            <div style={styles.controlsBar}>
                <ModelSelector activeModel={activeModel} models={models} onSelect={setActiveModel}/>
                <div className="vdivider"/>
                <Controls
                    playing={false} speed={1} view={view}
                    onTogglePlay={() => {}} onSpeed={() => {}} onView={setView}
                />
            </div>

            <div style={styles.main}>
                <div style={styles.mapArea}>
                    <AuroralMap
                        ssi={currentSSI}
                        auroralLatitudeDeg={data.models[activeModel]?.auroral_lat[currentIdx] ?? 63}
                        view={view}
                    />
                </div>
                <div style={styles.sidebar}>
                    <StatsPanel snapshot={snapshot} activeModel={activeModel} onSelectModel={setActiveModel}/>
                </div>
            </div>

            <div style={styles.timelineBar}>
                <div style={styles.tlRow}>
                    <span style={{fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text)'}}>{currentDt}</span>
                    <span style={styles.ssiPill(currentSSI)}>
                        {ssiLabel(currentSSI)} · {currentSSI.toFixed(4)}
                    </span>
                    <span style={styles.tlIdx}>{currentIdx + 1} / {data.steps}</span>
                </div>

                <input
                    type="range" min={0} max={data.steps - 1} value={currentIdx}
                    onChange={e => handleSeek(Number(e.target.value))}
                    style={{marginBottom: 4}}
                />

                <ForecastChart data={data} currentIdx={currentIdx} onSeek={handleSeek}/>

                <div style={styles.legend}>
                    {MODEL_META.map(({key, label, color}) => (
                        <span key={key} style={{color}}>
                            — {label}: {data.models[key]?.ssi[currentIdx]?.toFixed(4)}
                        </span>
                    ))}
                </div>
            </div>

        </div>
    )
}
