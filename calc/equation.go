package calc

import (
	"math"
)

// addend + addend = sum
// minuend - subtrahend = difference
// multiplicand * multiplier = product
// dividend / divisor = quotient

// Exp returns e**x, the base-e exponential of x.
// mega := math.Exp(2)
// fmt.Println(mega)

// i: the network itself (‘topic coordinator’ in Figure 1).
// j: a worker carrying out an inference task, in which it infers the topic’s target
// variable (‘worker: inference’ in Figure 1).
// k: a worker carrying out a forecasting task, in which it forecasts the loss of
// another worker’s inference (‘worker: forecasting’ in Figure 1).
// l: a worker carrying out either the inference or the forecasting task, and has
// been obtained by appending the arrays associated with each of these individual tasks.
// m: a reputer, which calculates and reports the loss of an inference of the topic’s
// target variable (‘reputer’ in Figure 1).

var (
	p       = 3.0
	c       = 0.75
	e       = math.E
	epsilon = 0.01
	// epsilon sets the numerical precision at which
	// the network should distinguish differences
	// in the logarithm of the loss
)

// Equation (5)
// weight = potential_function(Regret[ijk])
// TODO this equation is still unclear but we'll figure it out
func Weight(x float64, regret float64) float64 {
	return Phi(x) * regret
}

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
// the standard deviation of the forecasted regrets
// restricts ˆRijk to values in relative proximity to zero.
func RegretGradient(inferences []float64, regret float64) float64 {

	// TODO this equation is still unclear but we'll figure it out
	// sd of inferences? j = worker
	// ...indicates take the standard deviation over all j ∈ {1,...,N(i)}
	// ∈ "is an element of" "belongs to"
	divisor := sd(inferences) * regret * epsilon

	return regret / divisor
}

// make a float negative
func neg(x float64) float64 {
	return x * -1
}

// takes the standard deviation
func sd(set []float64) float64 {
	mean := avg(set)
	total := 0.0
	for _, num := range set {
		total += math.Pow(num-mean, 2)
	}
	variance := total / float64(len(set)-1)
	return math.Sqrt(variance)
}

// takes the mean
func avg(set []float64) float64 {
	elements := float64(len(set))
	total := sum(set)
	return total / elements
}
