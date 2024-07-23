package calc

import (
	"fmt"
	"math"
	"math/rand"
	random "math/rand/v2"
)

// Equation (3)
func WeightedInference(weights []float64, inferences []float64) {

	n := len(weights)

	weightedInferences := make([]float64, n)

	for i, w := range weights {
		weightedInferences[i] = w * inferences[i]
	}

	sumOfWeights := sum(weights)

	fmt.Println("weightedInferences: ", weightedInferences)
	fmt.Println("sumOfWeights: ", sumOfWeights)
}

func sum(floats []float64) float64 {
	var sum float64
	for _, f := range floats {
		sum += f
	}
	return sum
}

func CreateRandomInts(length int) []int {
	set := make([]int, length)
	for i := 0; i < length; i++ {
		set[i] = rand.Intn(1000)
	}
	return set
}

func CreateRandomFloats(length int) []float64 {
	set := make([]float64, length)
	for i := 0; i < length; i++ {
		set[i] = random.Float64() * 1000
	}
	return set
}

// the power to which e would have to be raised to equal x
func Ln(x float64) float64 {
	return math.Log(x)
}

// the power to which 10 would have to be raised to equal x
func Log(x float64) float64 {
	return math.Log10(x)
}

func NaturalLogSet(floats []float64) {
	for _, f := range floats {
		fmt.Println(Ln(f))
	}
}
