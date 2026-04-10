import {useCallback, useEffect, useRef} from 'react'
import {MODEL_META, ssiColor} from '../utils'
import type {TimelineData} from '../utils'
import {xAt, yAt} from '../canvas/chartUtils'
import {C, SEVERITY_BANDS} from '../theme'

interface Props {
    data: TimelineData
    modelKey: string
    currentIdx: number
    onSeek: (idx: number) => void
}

/** Scrubable canvas chart showing observed SSI and model predictions over the test set. */
export default function TimelineChart({data, modelKey, currentIdx, onSeek}: Props) {
    const canvasRef = useRef<HTMLCanvasElement>(null)

    const draw = useCallback(() => {
        const canvas = canvasRef.current
        if (!canvas || !data) return

        const W = canvas.offsetWidth
        const H = canvas.height
        canvas.width = W

        const ctx = canvas.getContext('2d')
        if (!ctx) return

        const {n, true: truth, pred} = data
        const pad   = 8
        const iW    = W - pad * 2
        const maxV  = Math.max(...truth) * 1.08

        const xi = (i: number) => xAt(i, n, pad, iW)
        const yi = (v: number) => yAt(v, maxV, pad, H)

        ctx.fillStyle = C.surface
        ctx.fillRect(0, 0, W, H)

        for (const {min, max, color} of SEVERITY_BANDS) {
            if (min >= maxV) break
            const effectiveMax = Math.min(max, maxV)
            ctx.fillStyle = color
            ctx.fillRect(pad, yi(effectiveMax), iW, yi(min) - yi(effectiveMax))
        }

        // Observed ground-truth line — lower weight to distinguish from model line.
        ctx.beginPath()
        truth.forEach((v, i) => i === 0 ? ctx.moveTo(xi(i), yi(v)) : ctx.lineTo(xi(i), yi(v)))
        ctx.strokeStyle = 'rgba(255,255,255,0.18)'
        ctx.lineWidth = 1
        ctx.setLineDash([])
        ctx.stroke()

        const color = MODEL_META.find(m => m.key === modelKey)?.color ?? C.accent
        ctx.beginPath()
        pred.forEach((v, i) => i === 0 ? ctx.moveTo(xi(i), yi(v)) : ctx.lineTo(xi(i), yi(v)))
        ctx.strokeStyle = color
        ctx.lineWidth = 1.5
        ctx.stroke()

        const xPlay = xi(currentIdx)
        ctx.beginPath()
        ctx.moveTo(xPlay, 0)
        ctx.lineTo(xPlay, H)
        ctx.strokeStyle = 'rgba(255,255,255,0.35)'
        ctx.setLineDash([2, 3])
        ctx.lineWidth = 1
        ctx.stroke()
        ctx.setLineDash([])

        ctx.beginPath()
        ctx.arc(xPlay, yi(truth[currentIdx]), 3, 0, Math.PI * 2)
        ctx.fillStyle = ssiColor(truth[currentIdx])
        ctx.fill()
    }, [data, modelKey, currentIdx])

    useEffect(() => { draw() }, [draw])

    useEffect(() => {
        const obs = new ResizeObserver(draw)
        if (canvasRef.current) obs.observe(canvasRef.current)
        return () => obs.disconnect()
    }, [draw])

    const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
        const canvas = canvasRef.current
        if (!canvas || !data) return
        const frac = (e.clientX - canvas.getBoundingClientRect().left) / canvas.offsetWidth
        onSeek(Math.round(Math.max(0, Math.min(1, frac)) * (data.n - 1)))
    }, [data, onSeek])

    return (
        <canvas
            ref={canvasRef}
            height={72}
            onClick={handleClick}
            style={{width: '100%', display: 'block', cursor: 'crosshair', borderRadius: 3}}
        />
    )
}
