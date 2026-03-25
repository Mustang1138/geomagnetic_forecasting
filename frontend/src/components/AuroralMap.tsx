import {Canvas} from '@react-three/fiber'
import {Line, OrbitControls, Stars} from '@react-three/drei'
import {useEffect, useMemo, useState} from 'react'
import * as THREE from 'three'
import {ssiColor} from '../utils'
import CountryOverlay, {CountryOverlaySouth} from './CountryOverlay'

interface Props {
    ssi: number
    aLat: number
    view: 'north' | 'south' | 'rect'
}

// Geometry

/** Generates a latitude ring as a closed array of 3-D points. */
function generateRing(latDeg: number, radius: number, segments = 180): [number, number, number][] {
    const lat = THREE.MathUtils.degToRad(latDeg)
    return Array.from({length: segments + 1}, (_, i) => {
        const a = (i / segments) * Math.PI * 2
        return [
            radius * Math.cos(lat) * Math.cos(a),
            radius * Math.sin(lat),
            radius * Math.cos(lat) * Math.sin(a),
        ] as [number, number, number]
    })
}

/** Returns inner/outer ring pair for a given signed latitude. */
function auroraRings(lat: number, ssi: number) {
    return {
        inner: generateRing(lat - (2 + ssi * 2), 2.05),
        outer: generateRing(lat + (2 + ssi * 5), 2.15),
    }
}

// Sub-component

interface AuroraOvalProps {
    lat: number
    ssi: number
    color: string
}

/* Renders one auroral oval (inner bright band + outer diffuse fringe). */
function AuroraOval({lat, ssi, color}: AuroraOvalProps) {
    const rings = useMemo(() => auroraRings(lat, ssi), [lat, ssi])
    return (
        <>
            <Line points={rings.inner} color={color} lineWidth={6} transparent opacity={0.8}/>
            <Line points={rings.outer} color={color} lineWidth={4} transparent opacity={0.3}/>
        </>
    )
}

// Component

export default function AuroralMap({ssi, aLat, view}: Props) {
    const [geoData, setGeoData] = useState<any>(null)

    useEffect(() => {
        fetch('/ne_50m_admin_0_map_units.json')
            .then(r => r.json())
            .then(setGeoData)
            .catch(console.error)
    }, [])

    const earthTexture = useMemo(
        () => new THREE.TextureLoader().load(
            'https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg'
        ), []
    )

    const color = ssiColor(ssi, 1)

    /*
     * north/south: single oval, latitude sign determined by view.
     * rect (global): both hemispheres shown simultaneously at ±aLat.
     * The CountryOverlay also receives the correct lat(s) for its
     * Gaussian highlight, so in global view we render two overlays.
     */
    const isGlobal = view === 'rect'
    const northLat = aLat
    const southLat = -aLat
    const singleLat = view === 'south' ? southLat : northLat

    return (
        <div style={{width: '100%', height: '100%', background: '#070b16'}}>
            <Canvas camera={{position: [3, 3, 3], fov: 50}}>
                <ambientLight intensity={1.4}/>
                <directionalLight position={[5, 3, 5]} intensity={1.8}/>

                {/* Earth sphere */}
                <mesh>
                    <sphereGeometry args={[2, 64, 64]}/>
                    <meshPhongMaterial map={earthTexture} shininess={10}/>
                </mesh>

                {isGlobal ? (
                    <>
                        <AuroraOval lat={northLat} ssi={ssi} color={color}/>
                        <AuroraOval lat={southLat} ssi={ssi} color={color}/>
                    </>
                ) : (
                    <AuroraOval lat={singleLat} ssi={ssi} color={color}/>
                )}

                {geoData && isGlobal ? (
                    <>
                        <CountryOverlay geojson={geoData} color={color} aLat={northLat} view="north"/>
                        <CountryOverlaySouth geojson={geoData} color={color} aLat={southLat}/>
                    </>
                ) : geoData && view === 'south' ? (
                    <CountryOverlaySouth geojson={geoData} color={color} aLat={singleLat}/>
                ) : geoData ? (
                    <CountryOverlay geojson={geoData} color={color} aLat={singleLat} view={view}/>
                ) : null}

                <Stars radius={300} depth={50} count={12000} factor={6} fade/>
                <OrbitControls enablePan={false} minDistance={2.4} maxDistance={9}/>
            </Canvas>
        </div>
    )
}