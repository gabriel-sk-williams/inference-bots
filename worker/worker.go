package worker

import (
	"fmt"
	"math"
	"math/rand/v2"
)

type Worker struct {
	Name              string
	Inferences        []float64
	Losses            []float64
	Regrets           []float64
	NormalizedRegrets []float64
	Forecasts         [][]float64
	Weight            float64
}

func CreateWorker(name string) *Worker {
	return &Worker{
		Name:      name,
		Forecasts: make([][]float64, 2),
	}
}

func (w *Worker) MakeRandomInference(epoch int) {
	f := rand.Float64() * 10
	inf := (math.Round(f*10) / 10)
	w.Inferences = append(w.Inferences, inf)
}

// still random lmao
// worker would need a strategy to make a contextual inference
func (w *Worker) MakeContextualInference(epoch int) {
	f := rand.Float64() * 10
	inf := (math.Round(f*10) / 10)
	w.Inferences = append(w.Inferences, inf)
}

func (w *Worker) ForecastLoss(weight float64, lastLoss float64, epoch int) float64 {

	var min float64
	var max float64

	if lastLoss-1.5 < 0 {
		min = 0.0
	} else {
		min = Round(lastLoss - 1.5)
	}

	if lastLoss+1.0 > 10.0 {
		max = 10.0
	} else {
		max = Round(lastLoss + 1.0)
	}

	forecastedLoss := min + rand.Float64()*(max-min) // assumedLoss
	//reduction := weight * 2.5
	//forecast := Round(math.Abs(assumedLoss - reduction))
	log := Round(math.Log10(forecastedLoss))

	fmt.Printf("%s: %v—%v -> %v -> %v \n", w.Name, min, max, forecastedLoss, log)

	return forecastedLoss
}

func Round(num float64) float64 {
	return (math.Round(num*100.0) / 100.0)
}
