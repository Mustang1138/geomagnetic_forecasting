// CSS variables are not accessible from a 2D canvas context, so design tokens are duplicated here.
export const CANVAS_COLOURS = {
    bg:      '#0e0e0e',
    surface: '#131313',
    border:  '#282828',
    text:    '#ffffff',
    text2:   '#e0e0e0',
    text3:   '#b0b0b0',
    accent:  '#e0e0e0',
} as const

export const SEVERITY_BANDS = [
    {min: 0,    max: 0.15, color: 'rgba(34,  197, 94,  0.05)'},
    {min: 0.15, max: 0.30, color: 'rgba(234, 179, 8,   0.06)'},
    {min: 0.30, max: 0.50, color: 'rgba(249, 115, 22,  0.06)'},
    {min: 0.50, max: 0.75, color: 'rgba(239, 68,  68,  0.07)'},
    {min: 0.75, max: 1.00, color: 'rgba(168, 85,  247, 0.09)'},
] as const
