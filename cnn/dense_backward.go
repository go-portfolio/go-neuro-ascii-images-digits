package cnn

// DenseBackward вычисляет градиенты по W, b и входу x.
//   gradOut — dL/d(logits), shape = (out)
//   x       — вход dense слоя, shape = (in)
//   W       — матрица весов (out × in)
func DenseBackward(gradOut []float64, x []float64, W [][]float64) (dW [][]float64, db []float64, dx []float64) {
	outDim := len(W)
	inDim := len(W[0])

	// Градиент по W (такая же размерность)
	dW = make([][]float64, outDim)
	for i := range dW {
		dW[i] = make([]float64, inDim)
	}

	// Градиент по b
	db = make([]float64, outDim)

	// Градиент по входу (dx = W^T * gradOut)
	dx = make([]float64, inDim)

	// dW[o][i] = x[i] * gradOut[o]
	for o := 0; o < outDim; o++ {
		db[o] = gradOut[o]
		for i := 0; i < inDim; i++ {
			dW[o][i] = gradOut[o] * x[i]
			dx[i] += W[o][i] * gradOut[o]
		}
	}

	return dW, db, dx
}
