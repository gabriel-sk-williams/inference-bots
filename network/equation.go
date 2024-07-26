package network

import (
	"math"
)

// addend + addend = sum
// minuend - subtrahend = difference
// multiplicand * multiplier = product
// dividend / divisor = quotient

var (
	// phi is a sigmoidal function:
	p       = 3.0  // local maximum
	c       = 0.75 // y-intercept
	e       = math.E
	epsilon = 0.01
	// epsilon sets the numerical precision at which
	// the network should distinguish differences
	// in the logarithm of the loss
)

func Loss(x float64, hatx float64) float64 {
	loss := math.Abs(x - hatx)
	return Round(loss)
}

// Equation (4)
// Regret = log Loss[i-1] — log Loss[ijk]
func Regret(networkLoss float64, forecastedLosses float64) float64 {
	minuend := Log(networkLoss)
	subtrahend := Log(forecastedLosses)
	return minuend - subtrahend
}

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
// weight(ijk) ∝ L^-p(ijk)
func PhiPrime(x float64) float64 {
	exp := neg(p * (x - c))
	divisor := math.Pow(e, exp) + 1
	result := p / divisor
	return RoundCustom(result, 1000.0)
}

// Equation (8)
// the standard deviation of the forecasted regrets
// restricts ˆRijk to values in relative proximity to zero.
func RegretGradient(inferences []float64, regret float64) float64 {

	// TODO this equation is still unclear but we'll figure it out
	// sd of inferences? j = worker
	// ...indicates take the standard deviation over all j ∈ {1,...,N(i)}
	// ∈ "is an element of" "belongs to"
	divisor := Sd(inferences) * regret * epsilon

	return regret / divisor
}

func NormalizeRegret(regret float64, sd float64) float64 {
	divisor := sd + epsilon
	normal := regret / divisor
	return Round(normal)
}
