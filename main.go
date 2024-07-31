package main

import (
	"fmt"
	c "inference-bots/calc"
	nw "inference-bots/network"
	"sort"
)

var (
	numWorkers  = 4
	numEpochs   = 2
	workerNames = []string{"Ralph", "Barto", "Skint", "Morty", "Troym", "Barny"}
)

// first round inferences are made without weight (weight = 1)
// second round inferences are made with weights from the actual losses (from first round)
// third round inferences (and beyond) are made with weights from the forecasted losses

func main() {
	fmt.Println("Inference bots engaged")

	n := nw.CreateNetwork(numWorkers, numEpochs, workerNames)

	for epoch := 0; epoch < numEpochs; epoch++ {

		for _, worker := range n.Workers {
			if epoch == 0 {
				worker.MakeRandomInference(epoch)
			} else {
				lastNetworkLoss := n.Losses[epoch-1]
				n.CollectWorkerForecasts(worker, epoch)
				worker.MakeContextualInference(epoch)
				worker.MakeWeightedInference(lastNetworkLoss, numWorkers, epoch)
			}
		}

		n.CalculateNetworkInference(epoch) // naive or weighted depending on epoch
		n.CalculateNetworkLoss(epoch)
		n.CalculateWorkerLosses(epoch)

		//
		// Sort and print results
		//

		workers := n.Workers
		sort.Slice(workers, func(i, j int) bool {
			return workers[i].Losses[epoch] < workers[j].Losses[epoch]
		})

		fmt.Println("")
		fmt.Printf("Epoch Round %d \n", epoch+1)
		fmt.Printf("Truth: %v \n", n.Reputer.Truths[:epoch+1])
		for _, worker := range workers {
			fmt.Println("")
			if epoch == 0 {
				fmt.Printf("%s - Inference: %v  Loss: %v \n",
					worker.Name,
					worker.Inferences[epoch],
					worker.Losses[epoch],
				)
			} else {
				fmt.Printf("%s - Inference: %v  Loss: %v  PrevLoss: %v vs %v (Network) \n",
					worker.Name,
					worker.Inferences[epoch],
					worker.Losses[epoch],
					worker.Losses[epoch-1],
					n.Losses[epoch-1],
				)
				fmt.Printf("Forecasts: %v \n Regrets: %v \n Normal: %v \n Weights: %v \n",
					worker.Forecasts[epoch],
					worker.Regrets[epoch],
					worker.NormalizedRegrets[epoch],
					worker.Weights[epoch],
				)
				fmt.Printf("WeightedInference: %v \n", worker.WeightedInferences)
			}
		}

		fmt.Println("")
		fmt.Println("Network")
		fmt.Printf("Truth: %v \n", n.Reputer.Truths[:epoch+1])
		fmt.Printf("Inferences: %v \n", n.Inferences) // avg of worker inferences
		fmt.Printf("Losses: %v \n", n.Losses)         // network losses
		fmt.Println("")

	}
}

func normalizeTest() {
	test := []float64{1.0, 3.0, 5.0, 7.0, 8.0, 9.0, 10.2}
	testSD := c.Sd(test)
	fmt.Println("sd: ", testSD)
	allNormalized := c.NormalizeSet(test, testSD)
	fmt.Println(allNormalized)
}
