package calc

import (
	"fmt"
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

// Equation (3)
// The losses forecasted by workers during the forecasting tasks
// reflect how accurate worker k expects the inference I(ij)
// to be, given the contextual information D(ijk). The forecasted losses are
// used to obtain the forecast-implied inference of the topic’s
// target variable through a weighted average:
func GetWeightedInference(weights []float64, inference float64) float64 {

	fmt.Println("calculating weighted inference...")
	fmt.Println("base inference: ", inference)

	weightedInferences := make([]float64, len(weights))

	for i, weight := range weights {
		fmt.Println("weight: ", weight)
		weightedInferences[i] = weight * inference
	}

	fmt.Println(weightedInferences)
	fmt.Println(weights)
	fmt.Println("")
	sumWeightedInferences := Sum(weightedInferences)
	sumWeights := Sum(weights)

	result := sumWeightedInferences / sumWeights
	return Round(result)
}

// Equation (6) // not used!
// Phi is a gradient descent function whose differential is used in Equation (7)
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
	return result
}

// Equation (8)
// the standard deviation of the forecasted regrets
// restricts ˆRijk to values in relative proximity to zero.
func NormalizeRegret(regret float64, sd float64) float64 {
	divisor := sd + epsilon
	normal := regret / divisor
	return Round(normal)
}
