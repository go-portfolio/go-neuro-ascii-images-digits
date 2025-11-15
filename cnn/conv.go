package cnn

// Conv2D - простая свёртка без padding.
// inputs: H×W
// kernel: Kh×Kw
func Conv2D(input [][]float64, kernels [][][]float64) [][][]float64 {
    h := len(input)
    w := len(input[0])
    kCount := len(kernels)
    kH := len(kernels[0])
    kW := len(kernels[0][0])

    outH := h - kH + 1
    outW := w - kW + 1

    out := make([][][]float64, kCount)
    for k := 0; k < kCount; k++ {
        out[k] = make([][]float64, outH)
        for i := 0; i < outH; i++ {
            out[k][i] = make([]float64, outW)
            for j := 0; j < outW; j++ {
                sum := 0.0
                for ki := 0; ki < kH; ki++ {
                    for kj := 0; kj < kW; kj++ {
                        sum += input[i+ki][j+kj] * kernels[k][ki][kj]
                    }
                }
                out[k][i][j] = sum
            }
        }
    }
    return out
}
