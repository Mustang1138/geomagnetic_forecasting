import type {ModelKey, ModelMetrics} from '../utils'
import {MODEL_META} from '../utils'

interface Props {
    activeModel: ModelKey
    models: ModelMetrics[]
    onSelect: (key: ModelKey) => void
}

/** Button group for selecting the active forecast model, with inline RMSE and R² metrics. */
export default function ModelSelector({activeModel, models, onSelect}: Props) {
    const meta = models.find(m => m.key === activeModel)

    return (
        <div style={{display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap'}}>
            <span style={{fontSize: 11, color: 'var(--text-3)', marginRight: 2}}>Model</span>

            {MODEL_META.map(m => (
                <button
                    key={m.key}
                    onClick={() => onSelect(m.key)}
                    disabled={m.key === activeModel}
                    style={m.key === activeModel ? {
                        borderColor: m.color,
                        color: m.color,
                        background: `${m.color}18`,
                    } : undefined}
                >
                    {m.label}
                </button>
            ))}

            {meta && (
                <span style={{
                    marginLeft: 4,
                    fontFamily: 'var(--font-mono)',
                    fontSize: 11,
                    color: 'var(--text-3)',
                }}>
                    RMSE <span style={{color: meta.color}}>{meta.rmse.toFixed(5)}</span>
                    {' · '}
                    MAE <span style={{color: meta.color}}>{meta.mae.toFixed(5)}</span>
                    {' · '}
                    R² <span style={{color: meta.color}}>{meta.r2.toFixed(4)}</span>
                </span>
            )}
        </div>
    )
}
