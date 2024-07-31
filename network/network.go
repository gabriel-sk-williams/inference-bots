package network

import (
	"fmt"
	c "inference-bots/calc"
)

type Network struct {
	numWorkers int
	numEpochs  int
	Workers    []*Worker
	Reputer    *Reputer
	Inferences []float64 // Network Inferences
	Losses     []float64 // Network Losses
}

func CreateNetwork(numWorkers int, epochs int, workerNames []string) *Network {
	n := Network{
		numWorkers: numWorkers,
		numEpochs:  epochs,
		Workers:    make([]*Worker, numWorkers),
		Inferences: make([]float64, epochs),
		Losses:     make([]float64, epochs),
	}
	n.Reputer = NewReputer("Bob", epochs)
	n.Reputer.SayHello()
	n.Reputer.RevealTruths()

	// create workers
	for i := 0; i < numWorkers; i++ {
		name := workerNames[i]
		n.Workers[i] = CreateWorker(name, epochs)
	}

	return &n
}

func (n *Network) CollectWorkerForecasts(worker *Worker, epoch int) {
	var forecasts []float64
	lastLoss := worker.Losses[epoch-1]

	fmt.Printf("Forecasts for %s: last loss:%v \n", worker.Name, lastLoss)

	for _, forecaster := range n.Workers {
		if forecaster.Name == worker.Name {
			continue
		}

		forecast := forecaster.ForecastLoss(lastLoss, epoch)
		forecasts = append(forecasts, forecast)
	}
	fmt.Println("")
	worker.Forecasts[epoch] = forecasts
}

func (n *Network) CalculateNetworkInference(epoch int) {
	if epoch == 0 {
		n.NaiveNetworkInference(epoch)
	} else {
		n.NaiveNetworkInference(epoch)
		//n.WeightedNetworkInference(epoch)
	}
}

func (n *Network) NaiveNetworkInference(epoch int) {

	inferences := make([]float64, n.numWorkers)

	for pos, w := range n.Workers {
		inferences[pos] = w.Inferences[epoch]
	}

	avg := c.Avg(inferences)
	n.Inferences[epoch] = c.Round(avg)
}

/*
func (n *Network) WeightedNetworkInference(epoch int) {

}
*/

func (n *Network) CalculateNetworkLoss(epoch int) {
	inf := n.Inferences[epoch]
	truth := n.Reputer.GetTruth(epoch)
	n.Losses[epoch] = c.Loss(inf, truth)
}

// regrets not needed in first round
func (n *Network) CalculateWorkerLosses(epoch int) {
	truth := n.Reputer.GetTruth(epoch)

	for _, worker := range n.Workers {
		inf := worker.Inferences[epoch]

		// In the absence of forecasts: Loss is simply the difference between prediction and ground truth
		loss := c.Loss(inf, truth)
		worker.Losses = append(worker.Losses, loss)

		// with forecasts, Loss is an aggregate of the other forecasts
	}
}
