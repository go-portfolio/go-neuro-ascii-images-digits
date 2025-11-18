package cnn

// MaxPool2x2Backward восстанавливает градиент после maxpool.
// gradOut — градиент после pooling (размер H/2 × W/2)
// input   — активации перед pooling (для поиска максимума)
func MaxPool2x2Backward(gradOut [][][]float64, input [][][]float64) [][][]float64 {
	C := len(input)
	H := len(input[0])
	W := len(input[0][0])

	gradIn := make([][][]float64, C)
	for c := range gradIn {
		gradIn[c] = make([][]float64, H)
		for i := 0; i < H; i++ {
			gradIn[c][i] = make([]float64, W)
		}
	}

	for c := 0; c < C; c++ {
		for y := 0; y < H; y += 2 {
			for x := 0; x < W; x += 2 {

				// Найдём индекс максимума в блоке 2×2
				maxY, maxX := y, x
				maxVal := input[c][y][x]

				if input[c][y][x+1] > maxVal {
					maxVal = input[c][y][x+1]
					maxX = x + 1
				}
				if input[c][y+1][x] > maxVal {
					maxVal = input[c][y+1][x]
					maxY = y + 1
					maxX = x
				}
				if input[c][y+1][x+1] > maxVal {
					maxY = y + 1
					maxX = x + 1
				}

				// Передаём градиент только максимуму
				gy := y / 2
				gx := x / 2
				gradIn[c][maxY][maxX] += gradOut[c][gy][gx]
			}
		}
	}

	return gradIn
}
