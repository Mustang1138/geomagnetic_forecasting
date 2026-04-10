type View = 'north' | 'south' | 'rect'

interface Props {
    playing: boolean
    speed: number
    view: View
    onTogglePlay: () => void
    onSpeed: (speed: number) => void
    onView: (view: View) => void
}

const SPEEDS = [0.5, 1, 5] as const
const VIEWS: { key: View; label: string }[] = [
    {key: 'north', label: 'North'},
    {key: 'south', label: 'South'},
    {key: 'rect', label: 'Global'},
]

/** Playback speed and globe-view controls. */
export default function Controls({playing, speed, view, onTogglePlay, onSpeed, onView}: Props) {
    return (
        <div style={{display: 'flex', alignItems: 'center', gap: 6}}>
            <button style={{minWidth: 58}} onClick={onTogglePlay}>
                {playing ? '⏸ Pause' : '▶ Play'}
            </button>

            <div className="vdivider"/>

            <span style={{fontSize: 11, color: 'var(--text-3)', marginRight: 2}}>Speed</span>
            {SPEEDS.map(sp => (
                <button key={sp} onClick={() => onSpeed(sp)} disabled={speed === sp}>
                    {sp}×
                </button>
            ))}

            <div className="vdivider"/>

            <span style={{fontSize: 11, color: 'var(--text-3)', marginRight: 2}}>View</span>
            {VIEWS.map(v => (
                <button key={v.key} onClick={() => onView(v.key)} disabled={view === v.key}>
                    {v.label}
                </button>
            ))}
        </div>
    )
}
