package network

import (
	"fmt"
	w "inference-bots/worker"
)

type Network struct {
	numWorkers int
	numEpochs  int
	Workers    []*w.Worker
	Reputer    *w.Reputer
	Inferences []float64 // Network Inferences
	Losses     []float64 // Network Losses
}

func CreateNetwork(numWorkers int, epochs int, workerNames []string) *Network {
	n := Network{
		numWorkers: numWorkers,
		numEpochs:  epochs,
		Workers:    make([]*w.Worker, numWorkers),
		Inferences: make([]float64, epochs),
		Losses:     make([]float64, epochs),
	}
	n.Reputer = w.NewReputer("Bob", epochs)
	n.Reputer.SayHello()
	n.Reputer.RevealTruths()

	// create workers
	for i := 0; i < numWorkers; i++ {
		name := workerNames[i]
		n.Workers[i] = w.CreateWorker(name)
	}

	return &n
}

func (n *Network) CollectForecasts(worker *w.Worker, epoch int) {
	var forecasts []float64
	currentWeight := worker.Weight
	lastLoss := worker.Losses[epoch-1]

	fmt.Printf("Forecasts for %s: last loss:%v \n", worker.Name, lastLoss)
	for _, forecaster := range n.Workers {
		if forecaster.Name == worker.Name {
			continue
		}

		forecast := forecaster.ForecastLoss(currentWeight, lastLoss, epoch)
		forecasts = append(forecasts, forecast)
	}
	fmt.Println("")
	worker.Forecasts[epoch] = forecasts
}

func (n *Network) CalculateNetworkInference(epoch int) {
	if epoch == 0 {
		n.NaiveNetworkInference(epoch)
	} else {
		n.WeightedNetworkInference(epoch)
	}
}

// Equation (3)
// The losses forecasted by workers during the forecasting tasks
// reflect how accurate worker k expects the inference Iij
// to be, given the contextual information Dijk. The forecasted losses are
// used to obtain the forecast-implied inference of the topic’s
// target variable through a weighted average:
func (n *Network) NaiveNetworkInference(epoch int) {

	inferences := make([]float64, n.numWorkers)

	for pos, w := range n.Workers {
		inferences[pos] = w.Inferences[epoch]
	}

	avg := Avg(inferences)
	n.Inferences[epoch] = Round(avg)
}

func (n *Network) WeightedNetworkInference(epoch int) {

	weightedInferences := make([]float64, n.numWorkers)
	inferences := make([]float64, n.numWorkers)
	weights := make([]float64, n.numWorkers)

	for pos, worker := range n.Workers {
		inferences[pos] = worker.Inferences[epoch]
		weights[pos] = worker.Weight
	}

	for i := 0; i < n.numWorkers; i++ {
		weightedInferences[i] = weights[i] * inferences[i]
	}

	sumOfWeightedInferences := Sum(weightedInferences)
	sumOfWeights := Sum(weights)

	avg := sumOfWeightedInferences / sumOfWeights

	fmt.Println("")
	fmt.Println(inferences)
	fmt.Println(weightedInferences)
	fmt.Println(weights)
	fmt.Println("sumOfWeightedInferences: ", sumOfWeightedInferences)
	fmt.Println("sumOfWeights: ", sumOfWeights)
	fmt.Println("avg: ", avg)
	n.Inferences[epoch] = Round(avg)
}

func (n *Network) CalculateNetworkLoss(epoch int) {
	inf := n.Inferences[epoch]
	truth := n.Reputer.GetTruth(epoch)
	n.Losses[epoch] = Loss(inf, truth)
}

func (n *Network) CalculateWorkerLossesAndRegrets(epoch int) {
	truth := n.Reputer.GetTruth(epoch)

	for _, worker := range n.Workers {
		inf := worker.Inferences[epoch]

		// In the absence of forecasts: Loss is simply the difference between prediction and ground truth
		loss := Loss(inf, truth)
		worker.Losses = append(worker.Losses, loss)

		// with forecasts, Loss is an aggregate of the other forecasts

		// positive regret: expected to outperform network
		// negative regret: indicates network expected to be more accurate
		regret := Regret(n.Losses[epoch], loss)
		worker.Regrets = append(worker.Regrets, regret)
	}
}

func (n *Network) NormalizeWorkerRegrets(epoch int) {

	regrets := make([]float64, n.numEpochs)

	for _, worker := range n.Workers {
		regrets[epoch] = worker.Regrets[epoch]
	}

	sd := Sd(regrets)

	for _, worker := range n.Workers {
		regret := worker.Regrets[epoch]
		normal := NormalizeRegret(regret, sd)
		worker.NormalizedRegrets = append(worker.NormalizedRegrets, normal)
	}
}

func (n *Network) AssignWeights(epoch int) {
	for _, worker := range n.Workers {
		// phiPrime returns a weight
		worker.Weight = PhiPrime(worker.NormalizedRegrets[epoch])
	}
}

/*
func (n *Network) CalculateNetworkRegret(epoch int) {

}
*/

/*
func (n *Network) CalculateWorkerRegrets(epoch int) {

}
*/
