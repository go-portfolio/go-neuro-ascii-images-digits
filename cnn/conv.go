package cnn

// Conv2D - простая свёртка без padding.
// input: H×W — входное изображение размером H (высота) на W (ширина)
// kernel: Kh×Kw — свёрточное ядро размером Kh (высота) на Kw (ширина)
func Conv2D(input [][]float64, kernels [][][]float64) [][][]float64 {
    // Определение размеров входного изображения
    h := len(input)         // Высота входного изображения
    w := len(input[0])      // Ширина входного изображения

    // Определение количества фильтров (ядер) и их размеров
    kCount := len(kernels)  // Количество свёрточных фильтров
    kH := len(kernels[0])   // Высота первого фильтра (предполагается, что все фильтры одинаковые)
    kW := len(kernels[0][0]) // Ширина первого фильтра

    // Вычисление размеров выходного изображения (без padding)
    outH := h - kH + 1  // Высота выходного изображения после свёртки
    outW := w - kW + 1  // Ширина выходного изображения после свёртки

    // Создание выходного массива для хранения результатов свёртки
    out := make([][][]float64, kCount) // Массив для каждого фильтра

    // Перебор всех фильтров
    for k := 0; k < kCount; k++ {
        // Создание двумерного массива для каждого фильтра
        out[k] = make([][]float64, outH)
        
        // Перебор всех позиций в выходном изображении (по высоте)
        for i := 0; i < outH; i++ {
            // Создание строки для текущей позиции в выходном изображении
            out[k][i] = make([]float64, outW)
            
            // Перебор всех позиций в выходном изображении (по ширине)
            for j := 0; j < outW; j++ {
                sum := 0.0  // Переменная для накопления суммы произведений

                // Перебор всех элементов ядра (фильтра)
                for ki := 0; ki < kH; ki++ {
                    for kj := 0; kj < kW; kj++ {
                        // Вычисление суммы произведений соответствующих элементов
                        sum += input[i+ki][j+kj] * kernels[k][ki][kj]
                    }
                }

                // Записываем результат свёртки в выходное изображение для данного фильтра
                out[k][i][j] = sum
            }
        }
    }

    // Возвращаем результат работы свёртки (выходное изображение для каждого фильтра)
    return out
}

// BackpropConv2D вычисляет:
// 1) dInput  — градиент по входу
// 2) dKernels — градиент по фильтрам
func BackpropConv2D(input [][]float64, kernels [][][]float64, dOut [][][]float64) ([][]float64, [][][]float64) {

	h := len(input)
	w := len(input[0])
	kN := len(kernels)
	kH := len(kernels[0])
	kW := len(kernels[0][0])

	outH := h - kH + 1
	outW := w - kW + 1

	// dInput
	dInput := make([][]float64, h)
	for i := range dInput {
		dInput[i] = make([]float64, w)
	}

	// dKernels
	dKernels := make([][][]float64, kN)
	for k := range dKernels {
		dKernels[k] = make([][]float64, kH)
		for i := 0; i < kH; i++ {
			dKernels[k][i] = make([]float64, kW)
		}
	}

	// ---- backprop ----
	for f := 0; f < kN; f++ {
		for y := 0; y < outH; y++ {
			for x := 0; x < outW; x++ {

				grad := dOut[f][y][x]

				// градиент по весам
				for i := 0; i < kH; i++ {
					for j := 0; j < kW; j++ {
						dKernels[f][i][j] += input[y+i][x+j] * grad
					}
				}

				// градиент по входу
				for i := 0; i < kH; i++ {
					for j := 0; j < kW; j++ {
						dInput[y+i][x+j] += kernels[f][i][j] * grad
					}
				}

			}
		}
	}

	return dInput, dKernels
}

