import {useCallback, useEffect, useRef, useState} from 'react'
import type {ModelKey} from '../utils'

const BASE = '/api'

// In-memory cache so switching models is instant on revisit
const _cache: Record<string, any> = {}

// Hooks

/** Fetches the list of available models and their evaluation metrics. */
export function useModels() {
    const [models, setModels] = useState<any[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        fetch(`${BASE}/models`)
            .then(r => {
                if (!r.ok) throw new Error(r.statusText);
                return r.json()
            })
            .then(data => {
                setModels(data);
                setLoading(false)
            })
            .catch(e => {
                setError(e.message);
                setLoading(false)
            })
    }, [])

    return {models, loading, error}
}

/** Fetches the full prediction time series for a given model key. */
export function usePredictions(modelKey: ModelKey) {
    const [data, setData] = useState<any>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    // Track the in-flight key so responses from a previous model are discarded
    const activeKey = useRef(modelKey)

    useEffect(() => {
        activeKey.current = modelKey

        if (_cache[modelKey]) {
            setData(_cache[modelKey])
            setLoading(false)
            return
        }

        setLoading(true)
        fetch(`${BASE}/predictions?model=${modelKey}`)
            .then(r => {
                if (!r.ok) throw new Error(r.statusText);
                return r.json()
            })
            .then(payload => {
                if (activeKey.current !== modelKey) return   // stale response – discard
                _cache[modelKey] = payload
                setData(payload)
                setLoading(false)
            })
            .catch(e => {
                setError(e.message);
                setLoading(false)
            })
    }, [modelKey])

    return {data, loading, error}
}

/** Fetches a single-timestep snapshot of all model predictions. */
export function useSnapshot(idx: number) {
    const [snapshot, setSnapshot] = useState<any>(null)

    const fetch_ = useCallback((i: number) => {
        fetch(`${BASE}/snapshot?idx=${i}`)
            .then(r => r.json())
            .then(setSnapshot)
            .catch(() => {
            })
    }, [])

    useEffect(() => {
        if (idx !== null) fetch_(idx)
    }, [idx, fetch_])

    return snapshot
}
