import type {ModelKey} from '../utils'
import {MODEL_META, ssiColor, ssiLabel} from '../utils'

// Types

interface Snapshot {
    true: number
    lat: number
    models: Record<ModelKey, number>
}

interface Props {
    snapshot: Snapshot | null
    activeModel: ModelKey
    onSelectModel: (key: ModelKey) => void
}

// Data

const LEGEND = [
    {label: 'Quiet (< 0.05)', color: '#22c55e'},
    {label: 'Minor (0.05–0.10)', color: '#eab308'},
    {label: 'Moderate (0.10–0.20)', color: '#f97316'},
    {label: 'Severe (0.20–0.30)', color: '#ef4444'},
    {label: 'Extreme (> 0.30)', color: '#a855f7'},
]

// Component

export default function StatsPanel({snapshot, activeModel, onSelectModel}: Props) {
    if (!snapshot) return <p style={{padding: 8}}>Loading…</p>

    const pred = snapshot.models[activeModel] ?? 0

    return (
        <div style={{padding: '8px 12px', overflowY: 'auto', height: '100%', boxSizing: 'border-box'}}>

            {/* Predicted SSI */}
            <p style={{margin: '0 0 2px'}}>
                <small>Predicted SSI</small>
            </p>
            <p style={{margin: '0 0 2px', fontSize: 24, color: ssiColor(pred)}}>
                <strong>{pred.toFixed(4)}</strong>
            </p>
            <p style={{margin: '0 0 2px'}}>
                <small>observed: {snapshot.true.toFixed(4)}</small>
            </p>
            <p style={{margin: '0 0 12px'}}>
                <span style={{color: ssiColor(pred)}}>[{ssiLabel(pred)}]</span>
            </p>

            <hr/>

            {/* Auroral boundary latitude */}
            <p style={{margin: '8px 0 2px'}}>
                <small>Auroral boundary</small>
            </p>
            <p style={{margin: '0 0 2px', fontSize: 18}}>
                {snapshot.lat.toFixed(1)}°
            </p>
            <p style={{margin: '0 0 12px'}}>
                <small>geomagnetic latitude</small>
            </p>

            <hr/>

            {/* Per-model comparison – clicking a row switches the active model */}
            <p style={{margin: '8px 0 4px'}}><small>Models</small></p>
            {MODEL_META.map(m => {
                const val = snapshot.models[m.key] ?? 0
                const active = m.key === activeModel
                return (
                    <div
                        key={m.key}
                        onClick={() => onSelectModel(m.key)}
                        style={{display: 'flex', cursor: 'pointer', opacity: active ? 1 : 0.6, marginBottom: 2}}
                    >
                        {/* Name column grows to fill remaining space */}
                        <span style={{
                            color: m.color,
                            fontWeight: active ? 'bold' : 'normal',
                            flex: 1,
                            minWidth: 0
                        }}>{m.label}</span>
                        {/* Value and error columns are fixed-width and right-aligned so they never wrap */}
                        <span style={{width: 56, textAlign: 'right', flexShrink: 0}}>{val.toFixed(4)}</span>
                        <small style={{
                            width: 64,
                            textAlign: 'right',
                            flexShrink: 0
                        }}>±{Math.abs(val - snapshot.true).toFixed(4)}</small>
                    </div>
                )
            })}

            <hr/>

            {/* Severity legend */}
            <p style={{margin: '8px 0 4px'}}><small>Severity</small></p>
            {LEGEND.map(l => (
                <div key={l.label} style={{display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2}}>
                    <span style={{
                        display: 'inline-block',
                        width: 10,
                        height: 10,
                        borderRadius: '50%',
                        background: l.color
                    }}/>
                    <small>{l.label}</small>
                </div>
            ))}

            <hr/>

            {/* Dataset */}
            <p style={{margin: '8px 0 4px'}}><small>Dataset</small></p>
            <small>
                Test set: Feb 2021 – Feb 2026<br/>
                6-hour steps · OMNI2 hourly data<br/>
                LSTM / GRU · Persistence baseline
            </small>

        </div>
    )
}