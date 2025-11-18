package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/go-portfolio/go-cnn/cnn"
	"github.com/go-portfolio/go-cnn/internal/data"
)

func main() {

	rand.Seed(time.Now().UnixNano())

	// -------------------------------------------------------------
	// 1) Инициализация свёрточных фильтров 8×3×3
	// -------------------------------------------------------------
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

	// -------------------------------------------------------------
	// 2) Инициализация Dense весов: 1352 → 10
	// -------------------------------------------------------------
	W := make([][]float64, 10) // 10×1352
	b := make([]float64, 10)

	for o := 0; o < 10; o++ {
		W[o] = make([]float64, 1352)
		for i := 0; i < 1352; i++ {
			W[o][i] = rand.NormFloat64() * 0.01
		}
		b[o] = 0
	}

	learningRate := 0.01

	// -------------------------------------------------------------
	// 3) Тренировка
	// -------------------------------------------------------------
	for epoch := 0; epoch < 10; epoch++ {

		// Псевдо-MNIST картинка
		input := data.RandomImage28x28()

		label := rand.Intn(10)

		// -------------------------------
		// FORWARD
		// -------------------------------
		conv := cnn.Conv2D(input, kernels) // (8×26×26)

		act := cnn.ReLU(conv) // (8×26×26)

		pooled := cnn.MaxPool2x2(act) // (8×13×13)

		flat := cnn.Flatten(pooled) // 1352

		logits := cnn.Dense(flat, W, b) // 10

		pred := cnn.Softmax(logits)

		loss := cnn.CrossEntropy(pred, label)

		fmt.Printf("Epoch %d  Loss=%f\n", epoch, loss)

		// -------------------------------
		// BACKPROP — Softmax + CE
		// -------------------------------
		gradLogits := make([]float64, len(pred))
		for i := range pred {
			gradLogits[i] = pred[i]
		}
		gradLogits[label] -= 1

		// -------------------------------
		// BACKPROP — Dense
		// -------------------------------
		// BACKPROP Dense
		dW, db, dFlat := cnn.DenseBackward(gradLogits, flat, W)

		// Обновляем Dense веса
		for o := 0; o < 10; o++ {
			for i := 0; i < len(flat); i++ {
				W[o][i] -= learningRate * dW[o][i]
			}
			b[o] -= learningRate * db[o]
		}

		// Unflatten → dPool
		dPool := cnn.Unflatten(dFlat, len(pooled), len(pooled[0]), len(pooled[0][0]))

		// -------------------------------
		// BACKPROP — MaxPool
		// -------------------------------
		dAct := cnn.MaxPool2x2Backward(dPool, act)


		// -------------------------------
		// BACKPROP — Conv2D
		// -------------------------------
		dInput, dKernels := cnn.BackpropConv2D(input, kernels, dAct)

		_ = dInput // (вход не обновляется)

		// обновить фильтры
		for f := range kernels {
			for i := range kernels[f] {
				for j := range kernels[f][i] {
					kernels[f][i][j] -= learningRate * dKernels[f][i][j]
				}
			}
		}

		// Вывод некоторых градиентов для контроля
		fmt.Printf("   ConvGrad[0][0][0]=%.6f  ConvGrad[0][1][1]=%.6f  ConvGrad[1][0][0]=%.6f\n",
			dKernels[0][0][0], dKernels[0][1][1], dKernels[1][0][0])
	}
}
