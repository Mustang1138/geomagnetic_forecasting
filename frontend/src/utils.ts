export const MODEL_META = [
    {key: 'rf', label: 'Random Forest', color: '#60a5fa'},
    {key: 'lr', label: 'Linear Regression', color: '#a78bfa'},
    {key: 'ls', label: 'LSTM', color: '#34d399'},
    {key: 'gr', label: 'GRU', color: '#fb923c'},
    {key: 'pe', label: 'Persistence', color: '#94a3b8'},
] as const

export type ModelKey = typeof MODEL_META[number]['key']

/** Evaluation metrics returned by the /api/models endpoint for each model. */
export interface ModelMetrics {
    key: ModelKey
    label: string
    rmse: number
    mae: number
    r2: number
    color: string
}

/** Single-timestep snapshot returned by /api/snapshot. */
export interface Snapshot {
    true: number
    lat: number
    models: Record<ModelKey, number>
}

/** Minimal GeoJSON structure for the country border overlay. */
export interface GeoJSONData {
    features: {
        properties: Record<string, string | undefined> | null
        geometry: {
            type: 'Polygon' | 'MultiPolygon'
            coordinates: number[][][] | number[][][][]
        }
    }[]
}

/** Prediction time series returned by /api/predictions for a given model. */
export interface TimelineData {
    n: number
    true: number[]
    pred: number[]
    cl?: string[]
    dt?: string[]
    lat?: number[]
}

export function ssiColor(ssiValue: number, alpha: number = 1): string {
    let r: number, g: number, b: number
    if (ssiValue < 0.15) {
        r = 34; g = 197; b = 94
    } else if (ssiValue < 0.30) {
        r = 234; g = 179; b = 8
    } else if (ssiValue < 0.50) {
        r = 249; g = 115; b = 22
    } else if (ssiValue < 0.75) {
        r = 239; g = 68; b = 68
    } else {
        r = 168; g = 85; b = 247
    }
    return `rgba(${r},${g},${b},${alpha})`
}

export function ssiLabel(ssiValue: number): string {
    if (ssiValue < 0.15) return 'Quiet'
    if (ssiValue < 0.30) return 'Minor'
    if (ssiValue < 0.50) return 'Moderate'
    if (ssiValue < 0.75) return 'Severe'
    return 'Extreme'
}
