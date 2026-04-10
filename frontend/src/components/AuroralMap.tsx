import {Canvas} from '@react-three/fiber'
import {Line, OrbitControls, Stars} from '@react-three/drei'
import {useEffect, useMemo, useState} from 'react'
import * as THREE from 'three'
import {ssiColor} from '../utils'
import type {GeoJSONData} from '../utils'
import {
    buildGlowBand,
    geoToVec,
    GMAG_NORTH,
    ringAroundCentre,
    ringAtLatitude,
} from '../geometry/aurora'
import type {RingDef} from '../geometry/aurora'
import CountryOverlay from './CountryOverlay'

interface Props {
    ssi: number
    auroralLatitudeDeg: number
    view: 'north' | 'south' | 'rect'
}

function NorthOval({auroralLatitudeDeg, ssi, color}: { auroralLatitudeDeg: number; ssi: number; color: string }) {
    // NH oval centred on the geomagnetic north pole (~80.7°N, 72.7°W) — correctly
    // offset toward Canada rather than the geographic pole.
    const gmagCentre = useMemo(() => geoToVec(GMAG_NORTH.lat, GMAG_NORTH.lon), [])
    const glowBand = useMemo(() => buildGlowBand(ssi), [ssi])

    const rings = useMemo(() => {
        // Oval centre sits a few degrees poleward of the equatorward boundary.
        const ovalCentreLat = auroralLatitudeDeg + 3.0 + ssi * 2
        const ovalCentreColatitudeDeg = Math.max(2, GMAG_NORTH.lat - ovalCentreLat)
        return glowBand.map((ring: RingDef) => ({...ring, colatDeg: ovalCentreColatitudeDeg + ring.offset}))
    }, [auroralLatitudeDeg, ssi, glowBand])

    return (
        <>
            {rings.map((ring, i) => (
                <Line
                    key={i}
                    points={ringAroundCentre(gmagCentre, Math.max(1, ring.colatDeg), ring.radius)}
                    color={color}
                    lineWidth={ring.lineWidth}
                    transparent
                    opacity={ring.opacity}
                />
            ))}
        </>
    )
}

function SouthOval({auroralLatitudeDeg, ssi, color}: { auroralLatitudeDeg: number; ssi: number; color: string }) {
    // SH oval uses geographic-latitude rings; geomagnetic pole geometry
    // for the south is complex, and the geographic approach is visually reliable.
    const glowBand = useMemo(() => buildGlowBand(ssi), [ssi])

    const rings = useMemo(() => {
        // Negate auroralLatitudeDeg for southern latitudes.
        const ovalCentreLat = -(auroralLatitudeDeg + 3.0 + ssi * 2)
        return glowBand.map((ring: RingDef) => ({
            ...ring,
            // Poleward offsets decrease latitude (more negative), equatorward increase.
            latDeg: ovalCentreLat - ring.offset,
        }))
    }, [auroralLatitudeDeg, ssi, glowBand])

    return (
        <>
            {rings.map((ring, i) => (
                <Line
                    key={i}
                    points={ringAtLatitude(ring.latDeg, ring.radius)}
                    color={color}
                    lineWidth={ring.lineWidth}
                    transparent
                    opacity={ring.opacity}
                />
            ))}
        </>
    )
}

/** 3D Earth globe with dynamic auroral ovals driven by SSI and auroral latitude. */
export default function AuroralMap({ssi, auroralLatitudeDeg, view}: Props) {
    const [geoData, setGeoData] = useState<GeoJSONData | null>(null)

    useEffect(() => {
        fetch('/merged.geojson')
            .then(r => r.json() as Promise<GeoJSONData>)
            .then(setGeoData)
            .catch(() => {})
    }, [])

    const earthTexture = useMemo(
        () => new THREE.TextureLoader().load(
            'https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg'
        ), []
    )

    const color = ssiColor(ssi, 1)

    const overlayHemisphere: 'north' | 'south' | 'both' =
        view === 'north' ? 'north' : view === 'south' ? 'south' : 'both'

    return (
        <div style={{width: '100%', height: '100%', background: '#070b16'}}>
            <Canvas camera={{position: [3, 3, 3], fov: 50}}>
                <ambientLight intensity={1.4}/>
                <directionalLight position={[5, 3, 5]} intensity={1.8}/>

                <mesh>
                    <sphereGeometry args={[2, 64, 64]}/>
                    <meshPhongMaterial map={earthTexture} shininess={10}/>
                </mesh>

                {(view === 'north' || view === 'rect') && (
                    <NorthOval auroralLatitudeDeg={auroralLatitudeDeg} ssi={ssi} color={color}/>
                )}
                {(view === 'south' || view === 'rect') && (
                    <SouthOval auroralLatitudeDeg={auroralLatitudeDeg} ssi={ssi} color={color}/>
                )}

                {geoData && (
                    <CountryOverlay
                        geojson={geoData}
                        ssi={ssi}
                        hemisphere={overlayHemisphere}
                    />
                )}

                <Stars radius={300} depth={50} count={12000} factor={6} fade/>
                <OrbitControls enablePan={false} minDistance={2.4} maxDistance={9}/>
            </Canvas>
        </div>
    )
}
