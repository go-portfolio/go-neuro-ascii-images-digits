package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/go-portfolio/go-cnn/cnn"
	"github.com/go-portfolio/go-cnn/internal/data"
)

// Функция для печати изображения или карты признаков в ASCII
func PrintFeatureMap(map2D [][]float64, title string) {
	fmt.Println(title)
	for y := 0; y < len(map2D); y++ {
		for x := 0; x < len(map2D[0]); x++ {
			v := map2D[y][x]
			if v > 0.7 {
				fmt.Print("#")
			} else if v > 0.4 {
				fmt.Print("*")
			} else if v > 0.1 {
				fmt.Print("+")
			} else {
				fmt.Print(".")
			}
		}
		fmt.Println()
	}
	fmt.Println()
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Классы цифр
	classes := []string{"Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"}

	// Инициализация Conv фильтров 8x3x3
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

	// Инициализация Dense 1352->10
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

	for epoch := 0; epoch < 3; epoch++ { // сокращаем для наглядности
		label := rand.Intn(10)          // случайная цифра 0..9
		input := data.RandomImage28x28(label)


		fmt.Println("Original Image:")
		PrintFeatureMap(input, "")

		// --- FORWARD ---
		conv := cnn.Conv2D(input, kernels)
		act := cnn.ReLU(conv)
		pooled := cnn.MaxPool2x2(act)
		flat := cnn.Flatten(pooled)
		logits := cnn.Dense(flat, W, b)
		pred := cnn.Softmax(logits)

		// Печать карт признаков первых 2 фильтров
		for f := 0; f < 2; f++ {
			PrintFeatureMap(act[f], fmt.Sprintf("Feature Map Filter %d after ReLU", f))
			PrintFeatureMap(pooled[f], fmt.Sprintf("Pooled Feature Map Filter %d", f))
		}

		// Предсказанный класс
		maxProb := 0.0
		predClass := 0
		for i, p := range pred {
			if p > maxProb {
				maxProb = p
				predClass = i
			}
		}

		correct := "✗"
		if predClass == label {
			correct = "✓"
		}

		loss := cnn.CrossEntropy(pred, label)

		fmt.Printf("Epoch %d  Loss=%.6f  Label=%s  Pred=%s (%.2f%%) %s\n",
			epoch, loss, classes[label], classes[predClass], maxProb*100, correct)

		fmt.Print("Probabilities: [")
		for i, p := range pred {
			fmt.Printf("%s: %.2f%%", classes[i], p*100)
			if i < len(pred)-1 {
				fmt.Print(", ")
			}
		}
		fmt.Println("]")

		// --- BACKPROP ---
		gradLogits := make([]float64, len(pred))
		for i := range pred {
			gradLogits[i] = pred[i]
		}
		gradLogits[label] -= 1

		dW, db, dFlat := cnn.DenseBackward(gradLogits, flat, W)
		for o := 0; o < 10; o++ {
			for i := 0; i < len(flat); i++ {
				W[o][i] -= learningRate * dW[o][i]
			}
			b[o] -= learningRate * db[o]
		}

		dPool := cnn.Unflatten(dFlat, len(pooled), len(pooled[0]), len(pooled[0][0]))
		dAct := cnn.MaxPool2x2Backward(dPool, act)
		dInput, dKernels := cnn.BackpropConv2D(input, kernels, dAct)

		_ = dInput

		for f := range kernels {
			for i := range kernels[f] {
				for j := range kernels[f][i] {
					kernels[f][i][j] -= learningRate * dKernels[f][i][j]
				}
			}
		}

		fmt.Println("-------------------------------------------------------------")
	}
}
