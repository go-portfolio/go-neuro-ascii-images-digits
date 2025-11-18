package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/go-portfolio/go-cnn/cnn"
	"github.com/go-portfolio/go-cnn/internal/data"
	"github.com/go-portfolio/go-cnn/internal/model"
)

func test() {
	classes := []string{"Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"}

	kernels, W, b, err := model.LoadModel("cnn_model.json")
	if err != nil {
		fmt.Println("Error loading model:", err)
		return
	}

	rand.Seed(time.Now().UnixNano())
	fmt.Println("=== Testing on new images ===")
	total := 10
	correctCount := 0

	for i := 0; i < total; i++ {
		label := rand.Intn(10)
		input := data.RandomImage28x28(label)

		// FORWARD
		conv := cnn.Conv2D(input, kernels)
		act := cnn.ReLU(conv)
		pooled := cnn.MaxPool2x2(act)
		flat := cnn.Flatten(pooled)
		logits := cnn.Dense(flat, W, b)
		pred := cnn.Softmax(logits)

		// Определяем предсказанный класс
		maxProb := 0.0
		predClass := 0
		for i, p := range pred {
			if p > maxProb {
				maxProb = p
				predClass = i
			}
		}
		if predClass == label {
			correctCount++
		}

		fmt.Printf("Label=%s  Pred=%s (%.2f%%)\n", classes[label], classes[predClass], maxProb*100)
	}

	fmt.Printf("Validation Accuracy: %.2f%%\n", float64(correctCount)/float64(total)*100)
}

func main() {
	test()
}
