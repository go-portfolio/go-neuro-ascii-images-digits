package data

import "math/rand"

// -------------------------------------------------------------
// Генерация псевдо-MNIST изображения 28×28 (штрихи и линии)
// -------------------------------------------------------------
func RandomImage28x28() [][]float64 {
	w, h := 28, 28

	// Создаем пустое изображение
	img := make([][]float64, h)
	for i := 0; i < h; i++ {
		img[i] = make([]float64, w)
	}

	// Лёгкий фоновый шум
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img[y][x] = rand.Float64() * 0.05
		}
	}

	// Случайные штрихи (имитация рукописных цифр)
	strokes := rand.Intn(6) + 3 // 3..8

	for s := 0; s < strokes; s++ {
		x := rand.Intn(w)
		y := rand.Intn(h)

		dx := rand.Intn(3) - 1
		dy := rand.Intn(3) - 1

		length := rand.Intn(6) + 3

		for i := 0; i < length; i++ {
			if x >= 0 && x < w && y >= 0 && y < h {
				img[y][x] += 0.5 + rand.Float64()*0.5
			}
			x += dx
			y += dy
		}
	}

	// Лёгкое размытие
	blur := func(a, b, c float64) float64 {
		return (a + b + c) / 3
	}

	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			img[y][x] = blur(img[y][x], img[y][x-1], img[y][x+1])
		}
	}

	// Нормализация 0..1
	maxVal := 0.0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if img[y][x] > maxVal {
				maxVal = img[y][x]
			}
		}
	}

	if maxVal > 0 {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				img[y][x] /= maxVal
			}
		}
	}

	return img
}
