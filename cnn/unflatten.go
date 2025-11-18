package cnn

// Unflatten превращает 1D-вектор обратно в 3D-тензор: C × H × W
func Unflatten(x []float64, C, H, W int) [][][]float64 {
	out := make([][][]float64, C)

	idx := 0
	for c := 0; c < C; c++ {
		out[c] = make([][]float64, H)
		for h := 0; h < H; h++ {
			out[c][h] = make([]float64, W)
			for w := 0; w < W; w++ {
				out[c][h][w] = x[idx]
				idx++
			}
		}
	}

	return out
}
