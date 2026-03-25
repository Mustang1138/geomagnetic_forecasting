// Types

type View = 'north' | 'south' | 'rect'

interface Props {
    playing: boolean
    speed: number
    view: View
    onTogglePlay: () => void
    onSpeed: (speed: number) => void
    onView: (view: View) => void
}

// Constants

const SPEEDS = [0.5, 1, 5] as const
const VIEWS: { key: View; label: string }[] = [
    {key: 'north', label: 'North'},
    {key: 'south', label: 'South'},
    {key: 'rect', label: 'Global'},
]

// Component

export default function Controls({playing, speed, view, onTogglePlay, onSpeed, onView}: Props) {
    return (
        <div style={{display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap'}}>
            {/* Play/pause gets a fixed width so the layout doesn't shift on label change */}
            <button style={{minWidth: 60}} onClick={onTogglePlay}>
                {playing ? 'Pause' : 'Play'}
            </button>

            <span>Speed:</span>
            {SPEEDS.map(sp => (
                <button key={sp} onClick={() => onSpeed(sp)} disabled={speed === sp}>
                    {sp}×
                </button>
            ))}

            <span style={{marginLeft: 8}}>View:</span>
            {VIEWS.map(v => (
                <button key={v.key} onClick={() => onView(v.key)} disabled={view === v.key}>
                    {v.label}
                </button>
            ))}
        </div>
    )
}