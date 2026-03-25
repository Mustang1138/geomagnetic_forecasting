import {useMemo} from 'react'
import {Line} from '@react-three/drei'
import * as THREE from 'three'

interface Props {
    geojson: any
    color: string
    aLat: number
    view: 'north' | 'south' | 'rect'
}

interface LineItem {
    key: string
    points: [number, number, number][]
    opacity: number
}

// Geometry

function latLngToVec3(lat: number, lng: number, r: number): THREE.Vector3 {
    const phi = THREE.MathUtils.degToRad(90 - lat)
    const theta = THREE.MathUtils.degToRad(lng + 180)
    return new THREE.Vector3(
        -r * Math.sin(phi) * Math.cos(theta),
        r * Math.cos(phi),
        r * Math.sin(phi) * Math.sin(theta),
    )
}

/**
 * Closes a GeoJSON ring if the first and last vertices don't already match,
 * then projects each coordinate onto the sphere at radius r.
 */
function ringToPoints(ring: number[][], r = 2.01): [number, number, number][] {
    const closed =
        ring[0][0] === ring.at(-1)![0] && ring[0][1] === ring.at(-1)![1]
            ? ring
            : [...ring, ring[0]]

    return closed.map(([lng, lat]) => {
        const v = latLngToVec3(lat, lng, r)
        return [v.x, v.y, v.z]
    })
}

// Shared render helper — keeps the two export components free of JSX duplication

function LineGroup({lines, color}: { lines: LineItem[]; color: string }) {
    return (
        <group>
            {lines.map(({key, points, opacity}) => (
                <Line key={key} points={points} color={color} lineWidth={2} transparent opacity={opacity}/>
            ))}
        </group>
    )
}

// Northern overlay

/**
 * Symmetric Gaussian falloff (s = 10 deg) centred on the auroral oval.
 * Intensity and culling are computed inside useMemo so that changes to
 * aLat, color, and view correctly invalidate the output during playback.
 */
export default function CountryOverlay({geojson, color, aLat, view}: Props) {
    const lines = useMemo((): LineItem[] => {
        if (!geojson) return []

        return geojson.features.flatMap((feature: any, fi: number) => {
            const polygons: any[] =
                feature.geometry.type === 'MultiPolygon'
                    ? feature.geometry.coordinates
                    : [feature.geometry.coordinates]

            return polygons.flatMap((polygon: any, pi: number) => {
                const ring: number[][] = polygon[0]

                const avgLat = ring.reduce((s, [, lat]) => s + lat, 0) / ring.length
                const sign = view === 'south' ? -1 : 1
                const dist = Math.abs(sign * avgLat - sign * aLat)
                const intensity = Math.exp(-(dist * dist) / 200)   // 200 = 2 * s^2 where s = 10 deg

                if (intensity <= 0.5) return []

                return [{key: `${fi}-${pi}`, points: ringToPoints(ring), opacity: 0.1 + intensity * 0.5}]
            })
        })
    }, [geojson, color, aLat, view])

    return <LineGroup lines={lines} color={color}/>
}

// Southern overlay

/**
 * Asymmetric equatorward-only falloff for the southern hemisphere (s = 15 deg).
 * Only countries equatorward of the oval are rendered, preventing the highlight
 * from bleeding into northern-hemisphere rendering.  Australia and New Zealand
 * (~35-46 S) become faintly visible when the oval descends to ~50-55 S during
 * stronger storms.
 */
export function CountryOverlaySouth({geojson, color, aLat}: Omit<Props, 'view'>) {
    const lines = useMemo((): LineItem[] => {
        if (!geojson) return []

        return geojson.features.flatMap((feature: any, fi: number) => {
            const polygons: any[] =
                feature.geometry.type === 'MultiPolygon'
                    ? feature.geometry.coordinates
                    : [feature.geometry.coordinates]

            return polygons.flatMap((polygon: any, pi: number) => {
                const ring: number[][] = polygon[0]

                const avgLat = ring.reduce((s, [, lat]) => s + lat, 0) / ring.length

                // Ignore northern-hemisphere polygons entirely
                if (avgLat >= 0) return []

                // offset > 0: country is equatorward of the oval (aurora potentially visible)
                // offset < 0: country is poleward / inside the oval -- cull it
                const offset = Math.abs(aLat) - Math.abs(avgLat)
                if (offset < 0 || offset > 40) return []

                const intensity = Math.exp(-(offset * offset) / 450)   // 450 = 2 * s^2 where s = 15 deg
                if (intensity <= 0.15) return []

                return [{key: `${fi}-${pi}`, points: ringToPoints(ring), opacity: 0.1 + intensity * 0.5}]
            })
        })
    }, [geojson, color, aLat])

    return <LineGroup lines={lines} color={color}/>
}