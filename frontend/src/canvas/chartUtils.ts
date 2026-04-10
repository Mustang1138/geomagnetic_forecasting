/** Returns the x-pixel coordinate for step index `i` in an n-step series. */
export function xAt(i: number, n: number, pad: number, innerWidth: number): number {
    return pad + (i / (n - 1)) * innerWidth
}

/** Returns the y-pixel coordinate for SSI value `v` given the vertical scale ceiling. */
export function yAt(v: number, maxV: number, pad: number, height: number): number {
    return height - pad - (v / maxV) * (height - pad * 2)
}
