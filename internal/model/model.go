package model

import (
	"encoding/json"
	"os"
)

// CNNModel хранит веса сети для сохранения/загрузки
type CNNModel struct {
	Kernels [][][]float64 `json:"kernels"`
	W       [][]float64   `json:"W"`
	B       []float64     `json:"b"`
}

// SaveModel сохраняет модель в JSON
func SaveModel(path string, kernels [][][]float64, W [][]float64, B []float64) error {
	model := CNNModel{Kernels: kernels, W: W, B: B}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	encoder := json.NewEncoder(f)
	return encoder.Encode(model)
}

// LoadModel загружает модель из JSON
func LoadModel(path string) ([][][]float64, [][]float64, []float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, nil, err
	}
	defer f.Close()
	var model CNNModel
	decoder := json.NewDecoder(f)
	if err := decoder.Decode(&model); err != nil {
		return nil, nil, nil, err
	}
	return model.Kernels, model.W, model.B, nil
}
