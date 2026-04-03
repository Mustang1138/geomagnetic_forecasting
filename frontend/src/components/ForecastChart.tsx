import {useCallback, useEffect, useRef} from 'react'
import {MODEL_META, ssiColor} from '../utils'
import type {ForecastData} from '../hooks/useForecast'

interface Props {
    data: ForecastData
    currentIdx: number
    onSeek: (idx: number) => void
}

// Drawing helpers

/** Returns an x-pixel coordinate for a given step index. */
function xAt(i: number, n: number, pad: number, innerWidth: number): number {
    return pad + (i / (n - 1)) * innerWidth
}

/** Returns a y-pixel coordinate for a given SSI value. */
function yAt(v: number, maxV: number, pad: number, height: number): number {
    return height - pad - (v / maxV) * (height - pad * 2)
}

/** Computes the shared vertical scale ceiling across all model SSI arrays. */
function computeMaxSSI(data: ForecastData): number {
    const allSSI = MODEL_META.flatMap(m => data.models[m.key]?.ssi ?? [])
    return Math.max(...allSSI, 0.05) * 1.1
}

// Component

/**
 * Canvas chart displaying all 5 model SSI forecasts over 28 steps.
 *
 * Renders each model line in its own colour (from MODEL_META), a dashed
 * playhead at the current step, and a dot on each model line at the playhead
 * position. Clicking or dragging seeks to that step.
 */
export default function ForecastChart({data, currentIdx, onSeek}: Props) {
    const canvasRef = useRef<HTMLCanvasElement>(null)

    const draw = useCallback(() => {
        const canvas = canvasRef.current
        if (!canvas || !data) return

        const W = canvas.offsetWidth
        const H = canvas.height
        canvas.width = W

        const ctx = canvas.getContext('2d')
        if (!ctx) return

        const n = data.steps
        const pad = 8
        const iW = W - pad * 2
        const maxV = computeMaxSSI(data)

        const xi = (i: number) => xAt(i, n, pad, iW)
        const yi = (v: number) => yAt(v, maxV, pad, H)

        ctx.clearRect(0, 0, W, H)

        // Model SSI lines
        MODEL_META.forEach(({key, color}) => {
            const ssi = data.models[key]?.ssi
            if (!ssi) return

            ctx.beginPath()
            ssi.forEach((v, i) => i === 0 ? ctx.moveTo(xi(i), yi(v)) : ctx.lineTo(xi(i), yi(v)))
            ctx.strokeStyle = color
            ctx.lineWidth = 1.5
            ctx.setLineDash([])
            ctx.stroke()
        })

        // Dashed playhead
        const xPlay = xi(currentIdx)
        ctx.beginPath()
        ctx.moveTo(xPlay, 0)
        ctx.lineTo(xPlay, H)
        ctx.strokeStyle = 'rgba(255,255,255,0.4)'
        ctx.setLineDash([2, 3])
        ctx.lineWidth = 1
        ctx.stroke()
        ctx.setLineDash([])

        // Dot on each model line at the playhead position
        MODEL_META.forEach(({key}) => {
            const ssi = data.models[key]?.ssi
            if (!ssi) return
            ctx.beginPath()
            ctx.arc(xPlay, yi(ssi[currentIdx]), 3, 0, Math.PI * 2)
            ctx.fillStyle = ssiColor(ssi[currentIdx])
            ctx.fill()
        })
    }, [data, currentIdx])

    useEffect(() => {
        draw()
    }, [draw])

    useEffect(() => {
        const obs = new ResizeObserver(draw)
        if (canvasRef.current) obs.observe(canvasRef.current)
        return () => obs.disconnect()
    }, [draw])

    const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
        const canvas = canvasRef.current
        if (!canvas || !data) return
        const frac = (e.clientX - canvas.getBoundingClientRect().left) / canvas.offsetWidth
        onSeek(Math.round(Math.max(0, Math.min(1, frac)) * (data.steps - 1)))
    }, [data, onSeek])

    return (
        <canvas
            ref={canvasRef}
            height={70}
            onClick={handleClick}
            style={{width: '100%', display: 'block', cursor: 'crosshair', background: '#070b16'}}
        />
    )
}