package data

import "fmt"

// PrintFeatureMap печатает 2D массив как ASCII изображение
func PrintFeatureMap(map2D [][]float64, title string) {
	if title != "" {
		fmt.Println(title)
	}
	for y := 0; y < len(map2D); y++ {
		for x := 0; x < len(map2D[0]); x++ {
			v := map2D[y][x]
			switch {
			case v > 0.2:
				fmt.Print("#")
			case v > 0.1:
				fmt.Print("*")
			case v > 0.01:
				fmt.Print("+")
			default:
				fmt.Print(".")
			}

		}
		fmt.Println()
	}
	fmt.Println()
}
