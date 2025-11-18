package cnn

func Flatten(x [][][]float64) []float64 {
    c := len(x)
    h := len(x[0])
    w := len(x[0][0])
    out := make([]float64, c*h*w)

    idx := 0
    for ch := 0; ch < c; ch++ {
        for i := 0; i < h; i++ {
            for j := 0; j < w; j++ {
                out[idx] = x[ch][i][j]
                idx++
            }
        }
    }
    return out
}

func Dense(input []float64, W [][]float64, b []float64) []float64 {
    out := make([]float64, len(W))
    for o := 0; o < len(W); o++ {
        sum := b[o]
        for i := 0; i < len(input); i++ {
            sum += input[i] * W[o][i]
        }
        out[o] = sum
    }
    return out
}
