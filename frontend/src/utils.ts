// frontend/src/utils.ts
export const MODEL_META = [
    {key: 'rf', label: 'Random Forest', color: '#38bdf8'},
    {key: 'lr', label: 'Linear Regression', color: '#a78bfa'},
    {key: 'ls', label: 'LSTM', color: '#34d399'},
    {key: 'gr', label: 'GRU', color: '#fb923c'},
    {key: 'pe', label: 'Persistence', color: '#94a3b8'},
] as const

export type ModelKey = typeof MODEL_META[number]['key']

export function ssiColor(v: number, alpha: number = 1): string {
    let r: number, g: number, b: number
    if (v < 0.05) {
        r = 34;
        g = 197;
        b = 94
    } else if (v < 0.10) {
        r = 234;
        g = 179;
        b = 8
    } else if (v < 0.20) {
        r = 249;
        g = 115;
        b = 22
    } else if (v < 0.30) {
        r = 239;
        g = 68;
        b = 68
    } else {
        r = 168;
        g = 85;
        b = 247
    }
    return `rgba(${r},${g},${b},${alpha})`
}

export function ssiLabel(v: number): string {
    if (v < 0.05) return 'Quiet'
    if (v < 0.10) return 'Minor'
    if (v < 0.20) return 'Moderate'
    if (v < 0.30) return 'Severe'
    return 'Extreme'
}

export interface TimelineData {
    n: number
    true: number[]
    pred: number[]
    cl?: string[]
    dt?: string[]
    lat?: number[]
}