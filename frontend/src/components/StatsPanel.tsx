import type {ModelKey, Snapshot} from '../utils'
import {MODEL_META, ssiColor, ssiLabel} from '../utils'

interface Props {
    snapshot: Snapshot | null
    activeModel: ModelKey
    onSelectModel: (key: ModelKey) => void
}

const SSI_LEGEND = [
    {label: 'Quiet',    range: '0.00 – 0.15', color: '#22c55e', description: 'No significant activity'},
    {label: 'Minor',    range: '0.15 – 0.30', color: '#eab308', description: 'Faint aurora at high latitudes'},
    {label: 'Moderate', range: '0.30 – 0.50', color: '#f97316', description: 'Aurora visible to ~55°N/S'},
    {label: 'Severe',   range: '0.50 – 0.75', color: '#ef4444', description: 'Aurora to ~50°N/S, radio disruption possible'},
    {label: 'Extreme',  range: '> 0.75',       color: '#a855f7', description: 'Aurora to mid-latitudes, infrastructure risk'},
]

const VISIBILITY_LEGEND = [
    {label: 'Rare',       color: '#22c55e'},
    {label: 'Occasional', color: '#eab308'},
    {label: 'Moderate',   color: '#f97316'},
    {label: 'High',       color: '#ef4444'},
]

function auroralLatContext(lat: number): string {
    if (lat >= 66) return 'Aurora confined to polar regions'
    if (lat >= 60) return 'Visible from Iceland, Alaska, northern Scandinavia'
    if (lat >= 55) return 'Visible from Scotland, southern Scandinavia, Canada'
    if (lat >= 50) return 'Visible from northern England, Germany, northern US'
    if (lat >= 45) return 'Visible from central Europe and most of Canada'
    return 'Exceptional storm — aurora visible at very low latitudes'
}

/** Sidebar showing SSI, auroral boundary, per-model predictions, and reference legends. */
export default function StatsPanel({snapshot, activeModel, onSelectModel}: Props) {
    if (!snapshot) return <p style={{padding: 12, color: 'var(--text-2)'}}>Loading…</p>

    const pred = snapshot.models[activeModel] ?? 0
    const ssiPct = Math.min(100, snapshot.true * 100)
    const activeLabel = MODEL_META.find(m => m.key === activeModel)?.label ?? activeModel

    return (
        <div style={{padding: '4px 12px 16px', overflowY: 'auto', height: '100%', boxSizing: 'border-box'}}>

            <div className="section-hd">Storm Severity Index</div>

            <div style={{
                fontFamily: 'var(--font-mono)', fontSize: 34,
                lineHeight: 1, fontWeight: 500,
                color: ssiColor(snapshot.true),
            }}>
                {snapshot.true.toFixed(4)}
            </div>
            <div style={{fontSize: 13, color: ssiColor(snapshot.true), marginTop: 3}}>
                {ssiLabel(snapshot.true)}
            </div>

            <div style={{
                height: 4, background: 'var(--border)',
                borderRadius: 2, overflow: 'hidden',
                margin: '8px 0 6px',
            }}>
                <div style={{
                    height: '100%', width: `${ssiPct}%`,
                    background: ssiColor(snapshot.true), borderRadius: 2,
                    transition: 'width 0.25s ease',
                }}/>
            </div>

            <div style={{fontSize: 11, color: 'var(--text-3)'}}>
                {activeLabel} prediction:{' '}
                <span style={{fontFamily: 'var(--font-mono)', color: 'var(--text-2)'}}>{pred.toFixed(4)}</span>
            </div>

            <div className="section-hd">Auroral Oval Boundary</div>

            <div style={{
                fontFamily: 'var(--font-mono)', fontSize: 28,
                lineHeight: 1, color: 'var(--text)', fontWeight: 400,
            }}>
                {snapshot.lat.toFixed(1)}
                <span style={{fontSize: 14, marginLeft: 3, color: 'var(--text-2)'}}>°N/S</span>
            </div>
            <div style={{fontSize: 12, color: 'var(--text-2)', marginTop: 5}}>
                {auroralLatContext(snapshot.lat)}
            </div>

            <div className="section-hd">Model Predictions</div>

            <div style={{
                display: 'flex', marginBottom: 4,
                fontSize: 10, color: 'var(--text-3)',
                textTransform: 'uppercase', letterSpacing: '0.07em',
            }}>
                <span style={{flex: 1}}>Model</span>
                <span style={{width: 58, textAlign: 'right', flexShrink: 0}}>SSI</span>
                <span style={{width: 62, textAlign: 'right', flexShrink: 0}}>Error</span>
            </div>

            {MODEL_META.map(m => {
                const val = snapshot.models[m.key] ?? 0
                const active = m.key === activeModel
                const err = Math.abs(val - snapshot.true)
                return (
                    <div
                        key={m.key}
                        onClick={() => onSelectModel(m.key)}
                        style={{
                            display: 'flex', cursor: 'pointer',
                            opacity: active ? 1 : 0.5,
                            marginBottom: 3,
                            padding: '3px 6px',
                            borderRadius: 3,
                            background: active ? `${m.color}12` : 'transparent',
                            border: `1px solid ${active ? m.color + '40' : 'transparent'}`,
                            transition: 'opacity 0.12s',
                        }}
                    >
                        <span style={{
                            color: m.color, fontWeight: active ? 500 : 400,
                            flex: 1, minWidth: 0, fontSize: 12,
                        }}>
                            {m.label}
                        </span>
                        <span style={{
                            width: 58, textAlign: 'right', flexShrink: 0,
                            fontFamily: 'var(--font-mono)', fontSize: 11,
                            color: active ? 'var(--text)' : 'var(--text-2)',
                        }}>
                            {val.toFixed(4)}
                        </span>
                        <span style={{
                            width: 62, textAlign: 'right', flexShrink: 0,
                            fontFamily: 'var(--font-mono)', fontSize: 11,
                            color: err < 0.01 ? '#22c55e' : err < 0.05 ? '#eab308' : '#ef4444',
                        }}>
                            ±{err.toFixed(4)}
                        </span>
                    </div>
                )
            })}

            <div className="section-hd">Severity Scale</div>

            {SSI_LEGEND.map(l => (
                <div key={l.label} style={{display: 'flex', alignItems: 'flex-start', gap: 7, marginBottom: 6}}>
                    <span style={{
                        display: 'inline-block', width: 8, height: 8,
                        borderRadius: '50%', background: l.color,
                        flexShrink: 0, marginTop: 4,
                    }}/>
                    <div>
                        <span style={{color: l.color, fontSize: 12, fontWeight: 500}}>{l.label}</span>
                        <span style={{color: 'var(--text-3)', fontSize: 11}}> · {l.range}</span>
                        <br/>
                        <span style={{color: 'var(--text-2)', fontSize: 11}}>{l.description}</span>
                    </div>
                </div>
            ))}

            <div className="section-hd">Country Visibility</div>

            <div style={{fontSize: 11, color: 'var(--text-2)', marginBottom: 7, lineHeight: 1.55}}>
                Border colours show the historical likelihood of aurora being visible from each
                country at the current SSI level, derived from 25 years of OMNI2 data.
            </div>
            <div style={{display: 'flex', flexWrap: 'wrap', gap: '4px 14px', marginBottom: 5}}>
                {VISIBILITY_LEGEND.map(l => (
                    <div key={l.label} style={{display: 'flex', alignItems: 'center', gap: 5}}>
                        <span style={{
                            display: 'inline-block', width: 18, height: 3,
                            background: l.color, borderRadius: 2, flexShrink: 0,
                        }}/>
                        <span style={{color: 'var(--text-2)', fontSize: 11}}>{l.label}</span>
                    </div>
                ))}
            </div>
            <div style={{fontSize: 10, color: 'var(--text-3)'}}>
                Grey borders = not historically recorded at this SSI level.
            </div>

            <div className="section-hd">Dataset</div>
            <div style={{fontSize: 11, color: 'var(--text-2)', lineHeight: 1.7}}>
                Test set: Feb 2021 – Feb 2026<br/>
                6-hour steps · OMNI2 hourly (NASA SPDF)<br/>
                RF · LR · LSTM · GRU · Persistence
            </div>

        </div>
    )
}
