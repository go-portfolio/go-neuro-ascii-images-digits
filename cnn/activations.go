package cnn

func ReLU(x [][][]float64) [][][]float64 {
    out := make([][][]float64, len(x))
    for c := range x {
        out[c] = make([][]float64, len(x[c]))
        for i := range x[c] {
            out[c][i] = make([]float64, len(x[c][i]))
            for j := range x[c][i] {
                v := x[c][i][j]
                if v > 0 {
                    out[c][i][j] = v
                } else {
                    out[c][i][j] = 0
                }
            }
        }
    }
    return out
}

func MaxPool2x2(x [][][]float64) [][][]float64 {
    c := len(x)
    h := len(x[0])
    w := len(x[0][0])

    outH := h / 2
    outW := w / 2

    out := make([][][]float64, c)
    for ch := 0; ch < c; ch++ {
        out[ch] = make([][]float64, outH)
        for i := 0; i < outH; i++ {
            out[ch][i] = make([]float64, outW)
            for j := 0; j < outW; j++ {
                a := x[ch][2*i][2*j]
                b := x[ch][2*i][2*j+1]
                c2 := x[ch][2*i+1][2*j]
                d := x[ch][2*i+1][2*j+1]
                out[ch][i][j] = max4(a, b, c2, d)
            }
        }
    }
    return out
}

func max4(a, b, c, d float64) float64 {
    m := a
    if b > m { m = b }
    if c > m { m = c }
    if d > m { m = d }
    return m
}
