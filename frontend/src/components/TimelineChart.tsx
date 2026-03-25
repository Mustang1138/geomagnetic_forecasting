import {useCallback, useEffect, useRef} from 'react'
import {MODEL_META, ssiColor} from '../utils'

interface TimelineData {
    n: number
    true: number[]
    pred: number[]
}

interface Props {
    data: TimelineData
    modelKey: string
    currentIdx: number
    onSeek: (idx: number) => void
}

// Component

export default function TimelineChart({data, modelKey, currentIdx, onSeek}: Props) {
    const canvasRef = useRef<HTMLCanvasElement>(null)

    const draw = useCallback(() => {
        const canvas = canvasRef.current
        if (!canvas || !data) return

        const W = canvas.offsetWidth
        const H = canvas.height
        canvas.width = W   // re-assign to clear and match layout width

        const ctx = canvas.getContext('2d')
        if (!ctx) return

        const {n, true: truth, pred} = data
        const pad = 8
        const iW = W - pad * 2
        const maxV = Math.max(...truth) * 1.05   // headroom so the peak doesn't clip

        // Map data coordinates to canvas pixels
        const xi = (i: number) => pad + (i / (n - 1)) * iW
        const yi = (v: number) => H - pad - (v / maxV) * (H - pad * 2)

        ctx.clearRect(0, 0, W, H)

        /* Observed (ground-truth) line */
        ctx.beginPath()
        truth.forEach((v, i) => i === 0 ? ctx.moveTo(xi(i), yi(v)) : ctx.lineTo(xi(i), yi(v)))
        ctx.strokeStyle = 'rgba(0,0,0,0.35)'
        ctx.lineWidth = 1
        ctx.stroke()

        /* Model prediction line */
        const color = MODEL_META.find(m => m.key === modelKey)?.color ?? '#2563eb'
        ctx.beginPath()
        pred.forEach((v, i) => i === 0 ? ctx.moveTo(xi(i), yi(v)) : ctx.lineTo(xi(i), yi(v)))
        ctx.strokeStyle = color
        ctx.lineWidth = 1.5
        ctx.stroke()

        /* Playhead – dashed vertical line at the current index */
        const xPlay = xi(currentIdx)
        ctx.beginPath()
        ctx.moveTo(xPlay, 0)
        ctx.lineTo(xPlay, H)
        ctx.strokeStyle = 'rgba(0,0,0,0.25)'
        ctx.setLineDash([2, 3])
        ctx.lineWidth = 1
        ctx.stroke()
        ctx.setLineDash([])

        // Dot on the truth line at the playhead position
        ctx.beginPath()
        ctx.arc(xPlay, yi(truth[currentIdx]), 3, 0, Math.PI * 2)
        ctx.fillStyle = ssiColor(truth[currentIdx])
        ctx.fill()
    }, [data, modelKey, currentIdx])

    useEffect(() => {
        draw()
    }, [draw])

    // Redraw on resize so the canvas fills its container correctly
    useEffect(() => {
        const obs = new ResizeObserver(draw)
        if (canvasRef.current) obs.observe(canvasRef.current)
        return () => obs.disconnect()
    }, [draw])

    /* Convert click X to a timestep index and notify the parent. */
    const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
        const canvas = canvasRef.current
        if (!canvas || !data) return
        const frac = (e.clientX - canvas.getBoundingClientRect().left) / canvas.offsetWidth
        onSeek(Math.round(Math.max(0, Math.min(1, frac)) * (data.n - 1)))
    }, [data, onSeek])

    return (
        <canvas
            ref={canvasRef}
            height={70}
            onClick={handleClick}
            style={{width: '100%', display: 'block', cursor: 'crosshair'}}
        />
    )
}
