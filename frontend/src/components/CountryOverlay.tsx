import {useMemo} from 'react'
import {Line} from '@react-three/drei'
import * as THREE from 'three'
import type {GeoJSONData} from '../utils'
import {useVisibilityLookup} from '../hooks/useVisibilityLookup'

interface Props {
    geojson: GeoJSONData
    ssi: number
    hemisphere: 'north' | 'south' | 'both'
}

interface LineData {
    id: string
    points: [number, number, number][]
    color: string
    lineWidth: number
    opacity: number
}

const SSI_BIN_EDGES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
const SSI_BIN_LABELS = [
    '0.00-0.05', '0.05-0.10', '0.10-0.15', '0.15-0.20',
    '0.20-0.25', '0.25-0.30', '0.30-0.35', '0.35-0.40',
    '0.40-0.45', '0.45-0.50', '>0.50',
]

function getSsiBin(ssi: number): string {
    for (let i = 0; i < SSI_BIN_EDGES.length - 1; i++) {
        if (ssi >= SSI_BIN_EDGES[i] && ssi < SSI_BIN_EDGES[i + 1]) return SSI_BIN_LABELS[i]
    }
    return '>0.50'
}

function visibilityColour(fraction: number): string | null {
    if (fraction <= 0) return null
    if (fraction < 0.002) return '#22c55e'  // rare (<0.2%)
    if (fraction < 0.01) return '#eab308'   // occasional (<1%)
    if (fraction < 0.10) return '#f97316'   // moderate (<10%)
    return '#ef4444'                          // frequent (>=10%)
}

function visibilityOpacity(fraction: number): number {
    if (fraction <= 0) return 0
    if (fraction < 0.002) return 0.35
    if (fraction < 0.01) return 0.5
    if (fraction < 0.10) return 0.65
    return 0.85
}

function latLngToVec3(lat: number, lng: number, r: number): THREE.Vector3 {
    const phi = THREE.MathUtils.degToRad(90 - lat)
    const theta = THREE.MathUtils.degToRad(lng + 180)
    return new THREE.Vector3(
        -r * Math.sin(phi) * Math.cos(theta),
        r * Math.cos(phi),
        r * Math.sin(phi) * Math.sin(theta),
    )
}

function ringToPoints(ring: number[][], r = 2.02): [number, number, number][] {
    const closed =
        ring[0][0] === ring.at(-1)![0] && ring[0][1] === ring.at(-1)![1]
            ? ring : [...ring, ring[0]]
    return closed.map(([lng, lat]) => {
        const v = latLngToVec3(lat, lng, r)
        return [v.x, v.y, v.z]
    })
}

/** Country border overlay coloured by historical aurora visibility probability. */
export default function CountryOverlay({geojson, ssi, hemisphere}: Props) {
    const lookup = useVisibilityLookup()
    const ssiBin = getSsiBin(ssi)

    // Pure render-data computed outside JSX — safe to memoise in R3F.
    const lineData = useMemo((): LineData[] => {
        if (!geojson) return []
        const result: LineData[] = []

        geojson.features.forEach((feature, fi: number) => {
            const p = feature.properties ?? {}
            const lookupKey = p.iso_3166_2 || p.GU_A3 || ''
            const meta = lookup?.[lookupKey]

            if (meta) {
                const metaHemi = meta.hemisphere === 'N' ? 'north' : 'south'
                if (hemisphere !== 'both' && metaHemi !== hemisphere) return
            }

            const polygons: number[][][][] =
                feature.geometry.type === 'MultiPolygon'
                    ? (feature.geometry.coordinates as number[][][][])
                    : [(feature.geometry.coordinates as number[][][])]

            polygons.forEach((polygon: number[][][], pi: number) => {
                const ring: number[][] = polygon[0]
                const id = `${fi}-${pi}`

                if (meta) {
                    const fraction = meta.visibility[ssiBin] ?? 0
                    const colour = visibilityColour(fraction)
                    if (colour) {
                        result.push({
                            id, points: ringToPoints(ring),
                            color: colour, lineWidth: 2,
                            opacity: visibilityOpacity(fraction),
                        })
                    } else {
                        result.push({
                            id, points: ringToPoints(ring),
                            color: '#334155', lineWidth: 1, opacity: 0.25,
                        })
                    }
                } else {
                    result.push({
                        id, points: ringToPoints(ring),
                        color: '#334155', lineWidth: 1, opacity: 0.15,
                    })
                }
            })
        })

        return result
    }, [geojson, lookup, ssiBin, hemisphere])

    return (
        <group>
            {lineData.map(d => (
                <Line
                    key={d.id}
                    points={d.points}
                    color={d.color}
                    lineWidth={d.lineWidth}
                    transparent
                    opacity={d.opacity}
                />
            ))}
        </group>
    )
}
