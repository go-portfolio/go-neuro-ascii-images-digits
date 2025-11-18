package data

import (
	"math/rand"
)

// Генерация псевдо-MNIST изображения 28×28 с примерно узнаваемыми цифрами
func RandomImage28x28(label int) [][]float64 {
	w, h := 28, 28
	img := make([][]float64, h)
	for y := 0; y < h; y++ {
		img[y] = make([]float64, w)
		for x := 0; x < w; x++ {
			img[y][x] = rand.Float64() * 0.05 // фон
		}
	}

	// Простейшие шаблоны для цифр 0..9
	drawLine := func(x1, y1, x2, y2 int) {
		dx := x2 - x1
		dy := y2 - y1
		steps := abs(dx)
		if abs(dy) > steps {
			steps = abs(dy)
		}
		for i := 0; i <= steps; i++ {
			x := x1 + i*dx/steps
			y := y1 + i*dy/steps
			if x >= 0 && x < w && y >= 0 && y < h {
				img[y][x] += 0.5 + rand.Float64()*0.5
			}
		}
	}

	// Добавляем примитивные линии для каждого числа
	switch label {
	case 0:
		drawLine(8, 8, 8, 19)
		drawLine(19, 8, 19, 19)
		drawLine(8, 8, 19, 8)
		drawLine(8, 19, 19, 19)
	case 1:
		drawLine(14, 8, 14, 19)
	case 2:
		drawLine(8, 8, 19, 8)
		drawLine(19, 8, 19, 13)
		drawLine(8, 13, 19, 19)
		drawLine(8, 19, 19, 19)
	case 3:
		drawLine(8, 8, 19, 8)
		drawLine(19, 8, 19, 19)
		drawLine(8, 19, 19, 19)
	case 4:
		drawLine(8, 8, 8, 13)
		drawLine(8, 13, 19, 13)
		drawLine(19, 8, 19, 19)
	case 5:
		drawLine(19, 8, 8, 8)
		drawLine(8, 8, 8, 13)
		drawLine(8, 13, 19, 13)
		drawLine(19, 13, 19, 19)
		drawLine(19, 19, 8, 19)
	case 6:
		drawLine(19, 8, 8, 8)
		drawLine(8, 8, 8, 19)
		drawLine(8, 13, 19, 13)
		drawLine(19, 13, 19, 19)
		drawLine(19, 19, 8, 19)
	case 7:
		drawLine(8, 8, 19, 8)
		drawLine(19, 8, 14, 19)
	case 8:
		drawLine(8, 8, 19, 8)
		drawLine(8, 19, 19, 19)
		drawLine(8, 8, 8, 19)
		drawLine(19, 8, 19, 19)
		drawLine(8, 13, 19, 13)
	case 9:
		drawLine(8, 19, 19, 19)
		drawLine(19, 8, 19, 19)
		drawLine(8, 8, 19, 8)
		drawLine(8, 8, 8, 13)
		drawLine(8, 13, 19, 13)
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

	// Нормализация
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

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}
