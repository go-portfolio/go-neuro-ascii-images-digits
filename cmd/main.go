package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/go-portfolio/go-cnn/cnn"
	"github.com/go-portfolio/go-cnn/internal/data"
)

func main() {
	// Инициализация генератора случайных чисел — чтобы веса были случайными при каждом запуске
	rand.Seed(time.Now().UnixNano())
	// Создаём набор из 8 случайных свёрточных фильтров 3×3
	kernels := make([][][]float64, 8)
	for k := range kernels {
		kernels[k] = make([][]float64, 3)
		for i := 0; i < 3; i++ {
			kernels[k][i] = make([]float64, 3)
			for j := 0; j < 3; j++ {
				// Случайная инициализация весов маленькими значениями
				kernels[k][i][j] = rand.NormFloat64() * 0.01
			}
		}
	}

	// -------------------------------------------------------------
	// 2) Инициализация полносвязного (Dense) слоя
	// -------------------------------------------------------------
	// После свёртки + ReLU + MaxPool размер тензора становится 1352.
	// Значит, Dense принимает 1352 значений на вход.
	// Выход — 10 чисел (классы MNIST).
	W := make([][]float64, 10) // матрица весов 10×1352
	b := make([]float64, 10)   // 10 смещений (bias)

	// Заполняем веса маленькими случайными значениями
	for o := 0; o < 10; o++ {
		W[o] = make([]float64, 1352)
		for i := 0; i < 1352; i++ {
			W[o][i] = rand.NormFloat64() * 0.01
		}
		// bias по умолчанию = 0
		b[o] = 0
	}

	// Скорость обучения для градиентного спуска
	learningRate := 0.01

	// -------------------------------------------------------------
	// 3) Простой цикл обучения (10 эпох)
	// -------------------------------------------------------------
	for epoch := 0; epoch < 10; epoch++ {

		// Вместо реальных данных MNIST — создаём псевдослучайную картинку 28×28
		input := data.RandomImage28x28()

		// Случайная "метка" (класс) — тоже заглушка, замените на реальные метки
		label := rand.Intn(10)

		// --------------------
		// FORWARD PASS
		// --------------------

		// 3×3 свёртка 8-ю фильтрами
		conv := cnn.Conv2D(input, kernels)

		// ReLU активация
		act := cnn.ReLU(conv)

		// MaxPool 2×2 уменьшает размер карты признаков в 4 раза
		pooled := cnn.MaxPool2x2(act)

		// Превращаем 3D-тензор в 1D-вектор длиной 1352
		flat := cnn.Flatten(pooled)

		// Применяем Dense слой: logits = W*x + b
		logits := cnn.Dense(flat, W, b)

		// Softmax для получения вероятностей классов
		pred := cnn.Softmax(logits)

		// Функция потерь — кросс-энтропия
		loss := cnn.CrossEntropy(pred, label)

		fmt.Printf("Epoch %d  Loss=%f\n", epoch, loss)

		// --------------------
		// BACKPROP (только Dense слой)
		// --------------------
		// Градиент по выходу Softmax + CrossEntropy:
		// gradient = (p - one_hot(label))
		gradLogits := make([]float64, len(pred))
		for i := range pred {
			gradLogits[i] = pred[i] // p_i
		}
		gradLogits[label] -= 1 // p_i - 1 на правильном классе

		// Обновление весов Dense слоя:
		// W = W - lr * grad
		for o := 0; o < 10; o++ {
			for i := 0; i < len(flat); i++ {
				W[o][i] -= learningRate * gradLogits[o] * flat[i]
			}
			b[o] -= learningRate * gradLogits[o]
		}

		// Свёрточные веса не обучаются в этом примере.
		// Если нужно — могу написать полную реализацию backprop для Conv2D.
	}
}

// -------------------------------------------------------------
// Генерация псевдо-MNIST изображения 28×28 (штрихи и линии)
// -------------------------------------------------------------
func randomImage28x28() [][]float64 {
	w, h := 28, 28

	// Создаем пустое изображение
	img := make([][]float64, h)
	for i := 0; i < h; i++ {
		img[i] = make([]float64, w)
	}

	// Лёгкий фон (как "бумага")
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img[y][x] = rand.Float64() * 0.05 // слабый шум
		}
	}

	// Случайное число штрихов
	strokes := rand.Intn(6) + 3 // 3..8 штрихов

	for s := 0; s < strokes; s++ {
		x := rand.Intn(w)
		y := rand.Intn(h)

		// случайное направление
		dx := rand.Intn(3) - 1 // -1, 0, 1
		dy := rand.Intn(3) - 1

		// случайная длина линии
		length := rand.Intn(6) + 3

		for i := 0; i < length; i++ {
			if x >= 0 && x < w && y >= 0 && y < h {
				img[y][x] += 0.5 + rand.Float64()*0.5 // яркость штриха
			}
			x += dx
			y += dy
		}
	}

	// лёгкое размытие, чтобы линии не были пиксельными
	blur := func(a, b, c float64) float64 {
		return (a + b + c) / 3
	}

	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			img[y][x] = blur(img[y][x], img[y][x-1], img[y][x+1])
		}
	}

	// нормализация 0..1
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
