package calc

import "fmt"

//
// this functions in multi.go perform functions across entire slices
//

func NormalizeSet(regrets []float64, sd float64) []float64 {

	nr := make([]float64, len(regrets))

	for i, r := range regrets {
		normal := NormalizeRegret(r, sd)
		nr[i] = normal
	}

	return nr
}

func NaturalLogSet(floats []float64) {
	for _, f := range floats {
		fmt.Println(Ln(f))
	}
}
