import {useCallback, useEffect, useRef, useState} from 'react'
import type {ModelKey, ModelMetrics, Snapshot, TimelineData} from '../utils'

const API_BASE_URL = '/api'

// In-memory cache so switching models is instant on revisit.
const predictionCache: Record<string, TimelineData> = {}

/** Fetches the list of available models and their evaluation metrics. */
export function useModels() {
    const [models, setModels] = useState<ModelMetrics[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        fetch(`${API_BASE_URL}/models`)
            .then(response => {
                if (!response.ok) throw new Error(response.statusText)
                return response.json() as Promise<ModelMetrics[]>
            })
            .then(data => {
                setModels(data)
                setLoading(false)
            })
            .catch((error: Error) => {
                setError(error.message)
                setLoading(false)
            })
    }, [])

    return {models, loading, error}
}

/** Fetches the full prediction time series for a given model key. */
export function usePredictions(modelKey: ModelKey) {
    const [data, setData] = useState<TimelineData | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    const activeModelKey = useRef(modelKey)

    useEffect(() => {
        activeModelKey.current = modelKey

        if (predictionCache[modelKey]) {
            setData(predictionCache[modelKey])
            setLoading(false)
            return
        }

        setLoading(true)
        fetch(`${API_BASE_URL}/predictions?model=${modelKey}`)
            .then(response => {
                if (!response.ok) throw new Error(response.statusText)
                return response.json() as Promise<TimelineData>
            })
            .then(payload => {
                // Discard responses that arrived after the model key changed.
                if (activeModelKey.current !== modelKey) return
                predictionCache[modelKey] = payload
                setData(payload)
                setLoading(false)
            })
            .catch((error: Error) => {
                setError(error.message)
                setLoading(false)
            })
    }, [modelKey])

    return {data, loading, error}
}

/** Fetches a single-timestep snapshot of all model predictions. */
export function useSnapshot(idx: number) {
    const [snapshot, setSnapshot] = useState<Snapshot | null>(null)

    const fetchSnapshot = useCallback((i: number) => {
        fetch(`${API_BASE_URL}/snapshot?idx=${i}`)
            .then(response => response.json() as Promise<Snapshot>)
            .then(setSnapshot)
            .catch(() => {})
    }, [])

    useEffect(() => {
        if (idx !== null) fetchSnapshot(idx)
    }, [idx, fetchSnapshot])

    return snapshot
}
