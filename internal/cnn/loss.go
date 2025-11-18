package cnn

import "math"

func Softmax(x []float64) []float64 {
    max := x[0]
    for _, v := range x {
        if v > max {
            max = v
        }
    }

    exp := make([]float64, len(x))
    sum := 0.0
    for i, v := range x {
        exp[i] = math.Exp(v - max)
        sum += exp[i]
    }

    for i := range exp {
        exp[i] /= sum
    }

    return exp
}

func CrossEntropy(pred []float64, label int) float64 {
    return -math.Log(pred[label] + 1e-12)
}
