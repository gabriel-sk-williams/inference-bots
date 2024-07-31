package calc

import (
	"math"
)

var (
	precision = 1000.0 // for rounding
)

func Sum(floats []float64) float64 {
	var sum float64
	for _, f := range floats {
		sum += f
	}
	return sum
}

// the power to which e would have to be raised to equal x
func Ln(x float64) float64 {
	return math.Log(x)
}

// the power to which 10 would have to be raised to equal x
func Log(x float64) float64 {
	return math.Log10(x)
}

// make a float negative
func neg(x float64) float64 {
	return x * -1
}

// takes the standard deviation
func Sd(set []float64) float64 {
	mean := Avg(set)
	total := 0.0
	for _, num := range set {
		total += math.Pow(num-mean, 2)
	}
	variance := total / float64(len(set)-1)
	return math.Sqrt(variance)
}

// takes the mean
func Avg(set []float64) float64 {
	elements := float64(len(set))
	total := Sum(set)
	return total / elements
}

func Round(num float64) float64 {
	return (math.Round(num*precision) / precision)
}

/*
func RoundCustom(num float64, magnitude float64) float64 {
	return (math.Round(num*magnitude) / magnitude)
}
*/
