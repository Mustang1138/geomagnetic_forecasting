import {useEffect, useState} from 'react'
import type {ModelKey} from '../utils'

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

/** Fetches the 7-day forecast from /api/forecast; performs a fresh fetch on each mount. */
export function useForecast() {
    const [data, setData] = useState<ForecastData | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        setLoading(true)
        setError(null)

        fetch('/api/forecast')
            .then(response => {
                if (!response.ok) throw new Error(`Forecast API returned ${response.status}: ${response.statusText}`)
                return response.json() as Promise<ForecastData>
            })
            .then(payload => {
                setData(payload)
                setLoading(false)
            })
            .catch((error: Error) => {
                setError(error.message)
                setLoading(false)
            })
    }, [])

    return {data, loading, error}
}
