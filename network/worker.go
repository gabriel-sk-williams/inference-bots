package network

import (
	"fmt"
	c "inference-bots/calc"
	"math"
	"math/rand/v2"
)

var (
	precision = 1000.0
)

type Worker struct {
	Name               string
	Inferences         []float64
	WeightedInferences []float64
	Losses             []float64
	Forecasts          [][]float64
	Regrets            [][]float64
	NormalizedRegrets  [][]float64
	Weights            [][]float64
	CurrentWeight      float64
}

func CreateWorker(name string, numEpochs int) *Worker {
	return &Worker{
		Name:              name,
		Forecasts:         make([][]float64, numEpochs),
		Regrets:           make([][]float64, numEpochs),
		NormalizedRegrets: make([][]float64, numEpochs),
		Weights:           make([][]float64, numEpochs),
	}
}

func (w *Worker) MakeRandomInference(epoch int) {
	f := rand.Float64() * 10
	inf := (math.Round(f*precision) / precision)
	w.Inferences = append(w.Inferences, inf)
}

// still random lmao
// worker would need a strategy to make a contextual inference
func (w *Worker) MakeContextualInference(epoch int) {
	f := rand.Float64() * 10
	inf := (math.Round(f*precision) / precision)
	w.Inferences = append(w.Inferences, inf)
}

func (w *Worker) ForecastLoss(lastLoss float64, epoch int) float64 {

	var min float64
	var max float64

	lr := 6.0
	ur := 6.0

	if lastLoss-lr < 0 {
		min = 0.0
	} else {
		min = Round(lastLoss - lr)
	}

	if lastLoss+ur > 10.0 {
		max = 10.0
	} else {
		max = Round(lastLoss + ur)
	}

	forecastedLoss := min + rand.Float64()*(max-min) // assumedLoss
	//reduction := weight * 2.5
	//forecast := Round(math.Abs(assumedLoss - reduction))
	log := Round(c.Log(forecastedLoss))

	fmt.Printf("%s: %v—%v -> %v -> %v \n", w.Name, min, max, forecastedLoss, log)

	return forecastedLoss
}

func (w *Worker) MakeWeightedInference(lastNetworkLoss float64, numWorkers int, epoch int) {

	size := numWorkers - 1

	// calculate regret for each forecast using Equation (4)
	// positive regret: expected to outperform network
	// negative regret: indicates network expected to be more accurate
	forecastedRegrets := make([]float64, size)

	for i, fc := range w.Forecasts[epoch] {
		regret := c.Log(lastNetworkLoss) - c.Log(fc)
		forecastedRegrets[i] = regret
	}

	w.Regrets[epoch] = forecastedRegrets

	// get Standard Deviation for all regrets
	sd := c.Sd(forecastedRegrets)

	// normalize regrets in preparation for next step
	normalizedRegrets := make([]float64, size)

	for j, r := range forecastedRegrets {
		normal := c.NormalizeRegret(r, sd)
		normalizedRegrets[j] = normal
	}

	w.NormalizedRegrets[epoch] = normalizedRegrets

	// convert to weights using Equation (7)
	weights := make([]float64, size)

	for k, r := range normalizedRegrets {
		normal := c.PhiPrime(r)
		weights[k] = normal
	}

	w.Weights[epoch] = weights
	w.CurrentWeight = c.Sum(weights)

	weightedInference := c.GetWeightedInference(weights, w.Inferences[epoch])
	w.WeightedInferences = append(w.WeightedInferences, weightedInference)
}

func Round(num float64) float64 {
	return (math.Round(num*100.0) / 100.0)
}
