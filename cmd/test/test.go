package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/go-portfolio/go-cnn/cnn"
	"github.com/go-portfolio/go-cnn/internal/data"
	"github.com/go-portfolio/go-cnn/internal/model"
)

// PrintFeatureMap выводит 2D массив в ASCII
func PrintFeatureMap(map2D [][]float64, title string) {
	fmt.Println(title)
	for y := 0; y < len(map2D); y++ {
		for x := 0; x < len(map2D[0]); x++ {
			v := map2D[y][x]
			switch {
			case v > 0.7:
				fmt.Print("#")
			case v > 0.4:
				fmt.Print("*")
			case v > 0.1:
				fmt.Print("+")
			default:
				fmt.Print(".")
			}
		}
		fmt.Println()
	}
	fmt.Println()
}

func test() {
	classes := []string{"Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"}

	kernels, W, b, err := model.LoadModel("cnn_model.json")
	if err != nil {
		fmt.Println("Error loading model:", err)
		return
	}

	rand.Seed(time.Now().UnixNano())
	fmt.Println("=== Testing on new images ===")

	total := 5
	correctCount := 0

	for i := 0; i < total; i++ {
		label := rand.Intn(10)
		input := data.RandomImage28x28(label)

		// Печатаем исходное изображение
		PrintFeatureMap(input, "Original Image:")

		// --- FORWARD ---
		conv := cnn.Conv2D(input, kernels)
		act := cnn.ReLU(conv)
		pooled := cnn.MaxPool2x2(act)
		flat := cnn.Flatten(pooled)
		logits := cnn.Dense(flat, W, b)
		pred := cnn.Softmax(logits)

		// Печатаем первые 2 карты признаков после ReLU и MaxPool
		for f := 0; f < 2 && f < len(act); f++ {
			PrintFeatureMap(act[f], fmt.Sprintf("Feature Map Filter %d after ReLU", f))
			PrintFeatureMap(pooled[f], fmt.Sprintf("Pooled Feature Map Filter %d", f))
		}

		// Определяем предсказанный класс
		maxProb := 0.0
		predClass := 0
		for j, p := range pred {
			if p > maxProb {
				maxProb = p
				predClass = j
			}
		}

		if predClass == label {
			correctCount++
		}

		fmt.Printf("Label=%s  Pred=%s (%.2f%%)\n", classes[label], classes[predClass], maxProb*100)
		fmt.Println("-------------------------------------------------------------")
	}

	fmt.Printf("Validation Accuracy: %.2f%%\n", float64(correctCount)/float64(total)*100)
}

func main() {
	test()
}
