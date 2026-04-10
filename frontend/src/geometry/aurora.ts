/** Pure geometry helpers for rendering the auroral oval on the 3D globe. */

import * as THREE from 'three'

/** IGRF-13 geomagnetic north pole, epoch 2025. */
export const GMAG_NORTH = {lat: 80.7, lon: -72.7}

/** Longitude of the SH geomagnetic pole in radians (136.6°E). */
export const GMAG_SOUTH_LON_RAD = THREE.MathUtils.degToRad(136.6)

/** Empirical amplitude (degrees) of the sinusoidal latitude shift for the SH auroral oval. */
export const SH_OFFSET_AMP = 8.0

/** Render parameters for a single ring in the glow band. */
export interface RingDef {
    opacity: number
    lineWidth: number
    /** Degrees offset from the oval centre ring. */
    offset: number
    radius: number
}

/** Convert geographic lat/lon (degrees) to a unit 3-D vector. */
export function geoToVec(latDeg: number, lonDeg: number): THREE.Vector3 {
    const lat = THREE.MathUtils.degToRad(latDeg)
    const lon = THREE.MathUtils.degToRad(lonDeg)
    return new THREE.Vector3(
        Math.cos(lat) * Math.cos(lon),
        Math.sin(lat),
        Math.cos(lat) * Math.sin(lon),
    )
}

/** Generate a ring of points at a given angular distance from a centre pole. */
export function ringAroundCentre(
    centre: THREE.Vector3,
    colatDeg: number,
    radius: number,
    segments = 256,
): [number, number, number][] {
    const referenceVector = Math.abs(centre.y) < 0.99
        ? new THREE.Vector3(0, 1, 0)
        : new THREE.Vector3(1, 0, 0)
    const tangentU = new THREE.Vector3().crossVectors(centre, referenceVector).normalize()
    const tangentV = new THREE.Vector3().crossVectors(centre, tangentU).normalize()
    const cosColatitude = Math.cos(THREE.MathUtils.degToRad(colatDeg))
    const sinColatitude = Math.sin(THREE.MathUtils.degToRad(colatDeg))

    return Array.from({length: segments + 1}, (_, i) => {
        const angleRad = (i / segments) * Math.PI * 2
        const point = new THREE.Vector3()
            .addScaledVector(tangentU, sinColatitude * Math.cos(angleRad))
            .addScaledVector(tangentV, sinColatitude * Math.sin(angleRad))
            .addScaledVector(centre, cosColatitude)
            .normalize()
            .multiplyScalar(radius)
        return [point.x, point.y, point.z] as [number, number, number]
    })
}

/** Generate a geographic-latitude ring with a sinusoidal longitude-dependent latitude shift. */
export function ringAtLatitude(
    latDeg: number,
    radius: number,
    segments = 256,
): [number, number, number][] {
    return Array.from({length: segments + 1}, (_, i) => {
        const lon = (i / segments) * Math.PI * 2

        // Sinusoidal latitude shift approximates the SH geomagnetic pole offset
        // (136.6°E): oval sits more equatorward on the opposite side of the globe.
        const shift = SH_OFFSET_AMP * Math.cos(lon - GMAG_SOUTH_LON_RAD)
        const lat = THREE.MathUtils.degToRad(latDeg - shift)

        return [
            radius * Math.cos(lat) * Math.cos(lon),
            radius * Math.sin(lat),
            radius * Math.cos(lat) * Math.sin(lon),
        ] as [number, number, number]
    })
}

/** Build the multi-ring glow band definition. */
export function buildGlowBand(ssi: number): RingDef[] {
    const polewardHalfWidthDeg = 1.5 + ssi * 3
    const equatorwardHalfWidthDeg = 3.0 + ssi * 9
    const peakOpacity = 0.50 + ssi * 0.45
    const steps = 5
    const rings: RingDef[] = []

    for (let i = -steps; i <= steps; i++) {
        const halfWidth = i < 0 ? polewardHalfWidthDeg : equatorwardHalfWidthDeg
        const offsetFraction = Math.abs(i) / steps
        const sigma = i < 0 ? 0.42 : 0.58
        const opacity = peakOpacity * Math.exp(-(offsetFraction * offsetFraction) / (2 * sigma * sigma))
        const offset = (i / steps) * halfWidth
        const radius = 2.02 + offsetFraction * 0.012
        const lineWidth = i === 0 ? 4 + ssi * 4 : 2 + ssi * 2 * (1 - offsetFraction * 0.5)
        rings.push({offset, opacity: Math.max(0.02, opacity), lineWidth, radius})
    }

    // Asymmetric band: equatorward skirt widens with SSI to simulate auroral
    // expansion toward lower latitudes during stronger storms.
    rings.push({
        offset: equatorwardHalfWidthDeg * 1.4,
        opacity: 0.05 + ssi * 0.07,
        lineWidth: 8 + ssi * 6,
        radius: 2.03,
    })

    return rings
}
