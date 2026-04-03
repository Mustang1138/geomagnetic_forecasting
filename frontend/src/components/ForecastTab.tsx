import {useCallback, useState} from 'react'
import AuroralMap from './AuroralMap'
import ForecastChart from './ForecastChart'
import ModelSelector from './ModelSelector'
import Controls from './Controls'
import StatsPanel from './StatsPanel'
import {useForecast} from '../hooks/useForecast'
import {useModels} from '../hooks/usePredictions'
import type {ModelKey} from '../utils'
import {MODEL_META, ssiColor, ssiLabel} from '../utils'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Snapshot shape expected by StatsPanel — one predicted SSI per model. */
interface ForecastSnapshot {
    true: number
    lat: number
    models: Record<ModelKey, number>
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const s = {
    root: {display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0} as const,

    notice: {
        padding: '4px 12px',
        background: '#1a1a2e',
        borderBottom: '1px solid #333',
        fontSize: 11,
        color: '#aaa',
    },

    controlsBar: {
        padding: '6px 12px',
        borderBottom: '1px solid #ccc',
        display: 'flex',
        gap: 12,
        alignItems: 'center',
        flexWrap: 'wrap' as const,
    },

    main: {display: 'flex', flex: 1, minHeight: 0, gap: 8, padding: 8},

    mapArea: {
        flex: 3,
        minWidth: 0,
        border: '1px solid #ccc',
        overflow: 'hidden',
    },

    sidebar: {
        flex: 1,
        minWidth: 260,
        maxWidth: 340,
        border: '1px solid #ccc',
        overflow: 'hidden',
    },

    timelineBar: {padding: '8px 12px 12px', borderTop: '1px solid #ccc'},

    tlRow: {display: 'flex', alignItems: 'center', marginBottom: 4},

    tlIdx: {marginLeft: 'auto'},

    ssiPill: (ssi: number) => ({
        display: 'inline-block',
        padding: '1px 8px',
        borderRadius: 4,
        background: ssiColor(ssi, 0.2),
        color: ssiColor(ssi),
        border: `1px solid ${ssiColor(ssi, 0.5)}`,
        fontWeight: 'bold' as const,
        fontSize: 12,
        marginLeft: 8,
    }),

    legend: {
        display: 'flex',
        gap: 12,
        flexWrap: 'wrap' as const,
        marginTop: 4,
        fontSize: 11,
        color: '#999',
    },
} as const

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Builds a StatsPanel-compatible snapshot from forecast data at a given step.
 *
 * No ground truth is available for future steps, so the active model's SSI
 * is used as a proxy for the ``true`` field.
 */
function buildSnapshot(
    data: ReturnType<typeof useForecast>['data'],
    activeModel: ModelKey,
    idx: number,
): ForecastSnapshot | null {
    if (!data) return null
    return {
        true: data.models[activeModel]?.ssi[idx] ?? 0,
        lat: data.models[activeModel]?.auroral_lat[idx] ?? 63,
        models: Object.fromEntries(
            MODEL_META.map(({key}) => [key, data.models[key]?.ssi[idx] ?? 0])
        ) as Record<ModelKey, number>,
    }
}

// ---------------------------------------------------------------------------
// Loading / error states
// ---------------------------------------------------------------------------

function ForecastLoading() {
    return (
        <div style={{padding: 24, flex: 1}}>
            <p>Fetching real-time DSCOVR solar wind data…</p>
            <p style={{color: '#999', fontSize: 13}}>
                Building seed window and running forecast models. This may take a moment.
            </p>
        </div>
    )
}

function ForecastError({message}: { message: string }) {
    return (
        <div style={{padding: 24, flex: 1}}>
            <p style={{color: '#ef4444'}}>Forecast unavailable</p>
            <p style={{color: '#999', fontSize: 13}}>
                {message}
                <br/>Please try refreshing the page.
            </p>
        </div>
    )
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * Forecast tab displaying a 7-day, 6-hourly geomagnetic storm severity
 * forecast for all five models.
 *
 * Reuses AuroralMap, StatsPanel, ModelSelector, and Controls unchanged.
 * ForecastChart replaces the history TimelineChart with a multi-model
 * canvas chart over the 28 forecast steps.
 */
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
    const currentDt = data.timestamps[currentIdx] ?? '—'
    const snapshot = buildSnapshot(data, activeModel, currentIdx)

    return (
        <div style={s.root}>

            {/* Methodology notice */}
            <div style={s.notice}>
                Forecast driven by real DSCOVR solar wind observations from the past 7 days (6-hour averages).
                Solar wind conditions beyond the observation window are extrapolated using the same 7-day pattern.
                Generated at {data.generated_at} · {data.steps} steps · {data.step_hours}h cadence
            </div>

            {/* Controls bar */}
            <div style={s.controlsBar}>
                <ModelSelector activeModel={activeModel} models={models} onSelect={setActiveModel}/>
                <Controls
                    playing={false}
                    speed={1}
                    view={view}
                    onTogglePlay={() => {
                    }}
                    onSpeed={() => {
                    }}
                    onView={setView}
                />
            </div>

            {/* Main area */}
            <div style={s.main}>
                <div style={s.mapArea}>
                    <AuroralMap ssi={currentSSI} aLat={data.models[activeModel]?.auroral_lat[currentIdx] ?? 63}
                                view={view}/>
                </div>
                <div style={s.sidebar}>
                    <StatsPanel snapshot={snapshot} activeModel={activeModel} onSelectModel={setActiveModel}/>
                </div>
            </div>

            {/* Timeline */}
            <div style={s.timelineBar}>
                <div style={s.tlRow}>
                    <span>{currentDt}</span>
                    <span style={s.ssiPill(currentSSI)}>
                        {ssiLabel(currentSSI)} · SSI {currentSSI.toFixed(4)}
                    </span>
                    <span style={s.tlIdx}>{currentIdx + 1} / {data.steps}</span>
                </div>

                <input
                    type="range"
                    min={0}
                    max={data.steps - 1}
                    value={currentIdx}
                    onChange={e => handleSeek(Number(e.target.value))}
                    style={{width: '100%', marginBottom: 4}}
                />

                <ForecastChart data={data} currentIdx={currentIdx} onSeek={handleSeek}/>

                {/* Model colour legend */}
                <div style={s.legend}>
                    {MODEL_META.map(({key, label, color}) => (
                        <span key={key} style={{color}}>
                            ── {label}: {data.models[key]?.ssi[currentIdx]?.toFixed(4)}
                        </span>
                    ))}
                </div>
            </div>

        </div>
    )
}