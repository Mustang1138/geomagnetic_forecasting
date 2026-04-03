import {useEffect, useState} from 'react'
import type {ModelKey} from '../utils'

// Types

export interface ModelForecast {
    ssi: number[]
    auroral_lat: number[]
    storm_class: string[]
}

export interface ForecastData {
    generated_at: string
    steps: number
    step_hours: number
    dscovr_conditions_used: boolean
    timestamps: string[]
    models: Record<ModelKey, ModelForecast>
}

// Hook

/**
 * Fetches the 7-day forecast from /api/forecast.
 *
 * Unlike usePredictions, forecast data is not cached between renders — a
 * fresh fetch is performed each time the component mounts so that the
 * forecast always reflects the latest DSCOVR conditions.
 */
export function useForecast() {
    const [data, setData] = useState<ForecastData | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        setLoading(true)
        setError(null)

        fetch('/api/forecast')
            .then(r => {
                if (!r.ok) throw new Error(`Forecast API returned ${r.status}: ${r.statusText}`)
                return r.json() as Promise<ForecastData>
            })
            .then(payload => {
                setData(payload)
                setLoading(false)
            })
            .catch((e: Error) => {
                setError(e.message)
                setLoading(false)
            })
    }, [])

    return {data, loading, error}
}