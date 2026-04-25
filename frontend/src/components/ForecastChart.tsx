import {useCallback, useEffect, useRef} from 'react'
import {MODEL_META, ssiColor} from '../utils'
import type {ForecastData} from '../hooks/useForecast'
import {xAt, yAt} from '../canvas/chartUtils'
import {CANVAS_COLOURS, SEVERITY_BANDS} from '../theme'

interface Props {
    data: ForecastData
    currentIdx: number
    onSeek: (idx: number) => void
}

function computeMaxSSI(data: ForecastData): number {
    const allSSI = MODEL_META.flatMap(m => data.models[m.key]?.ssi ?? [])
    const rfUpper = data.models.rf?.ssi_upper ?? []
    return Math.max(...allSSI, ...rfUpper, 0.05) * 1.1
}

/** Canvas chart displaying all 5 model SSI forecasts over 28 steps. */
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

        const n    = data.steps
        const pad  = 8
        const iW   = W - pad * 2
        const maxV = computeMaxSSI(data)

        const xi = (i: number) => xAt(i, n, pad, iW)
        const yi = (v: number) => yAt(v, maxV, pad, H)

        ctx.fillStyle = CANVAS_COLOURS.surface
        ctx.fillRect(0, 0, W, H)

        for (const {min, max, color} of SEVERITY_BANDS) {
            if (min >= maxV) break
            const effectiveMax = Math.min(max, maxV)
            ctx.fillStyle = color
            ctx.fillRect(pad, yi(effectiveMax), iW, yi(min) - yi(effectiveMax))
        }

        // Random Forest confidence-interval band, drawn beneath the model lines.
        const rfForecast = data.models.rf
        const rfLower = rfForecast?.ssi_lower
        const rfUpper = rfForecast?.ssi_upper
        if (rfLower && rfUpper && rfLower.length === rfUpper.length && rfLower.length > 0) {
            const rfMeta = MODEL_META.find(m => m.key === 'rf')
            const rfColor = rfMeta?.color ?? 'rgb(232,118,33)'
            ctx.beginPath()
            rfUpper.forEach((v, i) => i === 0 ? ctx.moveTo(xi(i), yi(v)) : ctx.lineTo(xi(i), yi(v)))
            for (let i = rfLower.length - 1; i >= 0; i--) ctx.lineTo(xi(i), yi(rfLower[i]))
            ctx.closePath()
            ctx.fillStyle = rfColor.startsWith('#')
                ? `${rfColor}33`
                : rfColor.replace('rgb(', 'rgba(').replace(')', ',0.2)')
            ctx.fill()
        }

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

        const xPlay = xi(currentIdx)
        ctx.beginPath()
        ctx.moveTo(xPlay, 0)
        ctx.lineTo(xPlay, H)
        ctx.strokeStyle = 'rgba(255,255,255,0.35)'
        ctx.setLineDash([2, 3])
        ctx.lineWidth = 1
        ctx.stroke()
        ctx.setLineDash([])

        MODEL_META.forEach(({key}) => {
            const ssi = data.models[key]?.ssi
            if (!ssi) return
            ctx.beginPath()
            ctx.arc(xPlay, yi(ssi[currentIdx]), 3, 0, Math.PI * 2)
            ctx.fillStyle = ssiColor(ssi[currentIdx])
            ctx.fill()
        })
    }, [data, currentIdx])

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
        onSeek(Math.round(Math.max(0, Math.min(1, frac)) * (data.steps - 1)))
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
