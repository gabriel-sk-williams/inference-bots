package calc

import (
	"math"
)

// addend + addend = sum
// minuend - subtrahend = difference
// multiplicand * multiplier = product
// dividen / divisor = quotient

// i: The variable is associated with the network itself (‘topic coordinator’ in Figure 1).
// j: The variable is associated with a worker carrying out an inference task, in which it infers the topic’s target
// variable (‘worker: inference’ in Figure 1).
// k: The variable is associated with a worker carrying out a forecasting task, in which it forecasts the loss of
// another worker’s inference (‘worker: forecasting’ in Figure 1).
// l: The variable is associated with a worker carrying out either the inference or the forecasting task, and has
// been obtained by appending the arrays associated with each of these individual tasks.
// m: The variable is associated with a reputer,

// Equation (5)
// weight = potential_function(Regret[ijk])

var (
	p = 3.0
	c = 0.75
	e = math.E
	// epsilon sets the numerical precision at which
	// the network should distinguish differences
	// in the logarithm of the loss
	epsilon = 0.01
)

// Exp returns e**x, the base-e exponential of x.
// mega := math.Exp(2)
// fmt.Println(mega)

// Equation (6)
// Phi is a gradient descent function for defining weights
// ln [1 + e^p(x-c)]
func Phi(x float64) float64 {
	exp := p * (x - c)
	addend := math.Pow(e, exp)
	return Ln(1 + addend)
}

// Equation (7)
// a smoothly differentiable approximation of max (0, p(x − c))
func Gradient(x float64) float64 {
	exp := neg(p * (x - c))
	divisor := math.Pow(e, exp) + 1
	return p / divisor
}

// Equation (8)
func RegretGradient(regretes []float64)

func neg(x float64) float64 {
	return x * -1
}

// takes the standard deviation
func sd(numbers []float64, mean float64) float64 {
	total := 0.0
	for _, number := range numbers {
		total += math.Pow(number-mean, 2)
	}
	variance := total / float64(len(numbers)-1)
	return math.Sqrt(variance)
}

func mean(set []float64) float64 {
	elements := float64(len(set))
	total := sum(set)

	return total / elements
}
