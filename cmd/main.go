package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/go-portfolio/go-cnn/cnn"

)

func main() {
	rand.Seed(time.Now().UnixNano())

	// 1) Генерируем случайные веса conv
	convKernels := make([][][][]float64, 1)

	// 8 фильтров 3×3
	kernels := make([][][]float64, 8)
	for k := range kernels {
		kernels[k] = make([][]float64, 3)
		for i := 0; i < 3; i++ {
			kernels[k][i] = make([]float64, 3)
			for j := 0; j < 3; j++ {
				kernels[k][i][j] = rand.NormFloat64() * 0.01
			}
		}
	}

	// Dense слой: 1352 -> 10
	W := make([][]float64, 10)
	b := make([]float64, 10)
	for o := 0; o < 10; o++ {
		W[o] = make([]float64, 1352)
		for i := 0; i < 1352; i++ {
			W[o][i] = rand.NormFloat64() * 0.01
		}
		b[o] = 0
	}

	learningRate := 0.01

	// ---------- Один учебный тренинг ----------
	for epoch := 0; epoch < 10; epoch++ {
		// Замените на реальные картинки MNIST
		input := randomImage28x28()
		label := rand.Intn(10)

		// Forward
		conv := cnn.Conv2D(input, kernels)
		act := cnn.ReLU(conv)
		pooled := cnn.MaxPool2x2(act)
		flat := cnn.Flatten(pooled)
		logits := cnn.Dense(flat, W, b)
		pred := cnn.Softmax(logits)
		loss := cnn.CrossEntropy(pred, label)

		fmt.Printf("Epoch %d  Loss=%f\n", epoch, loss)

		// Backprop (только dense для простоты):
		gradLogits := make([]float64, len(pred))
		for i := range pred {
			gradLogits[i] = pred[i]
		}
		gradLogits[label] -= 1 // softmax grad

		// Обновляем dense:
		for o := 0; o < 10; o++ {
			for i := 0; i < len(flat); i++ {
				W[o][i] -= learningRate * gradLogits[o] * flat[i]
			}
			b[o] -= learningRate * gradLogits[o]
		}

		// (свёртку можно тоже обучать — если нужно, напишу полный backprop)
	}
}

func randomImage28x28() [][]float64 {
	img := make([][]float64, 28)
	for i := range img {
		img[i] = make([]float64, 28)
		for j := range img[i] {
			img[i][j] = rand.Float64()
		}
	}
	return img
}
