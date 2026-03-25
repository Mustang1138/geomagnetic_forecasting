import type {ModelKey} from '../utils'
import {MODEL_META} from '../utils'

interface Props {
    activeModel: ModelKey
    models: any[]   // live metrics from /api/models
    onSelect: (key: ModelKey) => void
}

// Component

export default function ModelSelector({activeModel, models, onSelect}: Props) {
    const meta = models.find(m => m.key === activeModel)

    return (
        <div style={{display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap'}}>
            {MODEL_META.map(m => (
                // Active model is disabled (visually pressed) and shown in bold
                <button
                    key={m.key}
                    onClick={() => onSelect(m.key)}
                    disabled={m.key === activeModel}
                    style={{fontWeight: m.key === activeModel ? 'bold' : 'normal'}}
                >
                    {m.label}
                </button>
            ))}

            {/* Show live metrics for the active model once the API responds */}
            {meta && (
                <small style={{marginLeft: 8}}>
                    RMSE <span style={{color: meta.color}}>{meta.rmse.toFixed(5)}</span>
                    {' · '}
                    MAE <span style={{color: meta.color}}>{meta.mae.toFixed(5)}</span>
                    {' · '}
                    R² <span style={{color: meta.color}}>{meta.r2.toFixed(4)}</span>
                </small>
            )}
        </div>
    )
}