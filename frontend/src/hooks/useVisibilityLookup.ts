import {useEffect, useState} from 'react'

export interface CountryMeta {
    label: string
    hemisphere: string
    geomag_lat: number
    visibility: Record<string, number>
}

export type VisibilityLookup = Record<string, CountryMeta>

/** Fetches the aurora visibility lookup table from /aurora_visibility.json; returns null until loaded. */
export function useVisibilityLookup(): VisibilityLookup | null {
    const [lookup, setLookup] = useState<VisibilityLookup | null>(null)

    useEffect(() => {
        fetch('/aurora_visibility.json')
            .then(response => response.ok ? response.json() as Promise<VisibilityLookup> : null)
            .then(data => {
                if (data) setLookup(data)
            })
            .catch(() => {})
    }, [])

    return lookup
}
