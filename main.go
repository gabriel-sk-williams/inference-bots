package main

import (
	"fmt"
	nw "inference-bots/network"
	"sort"
)

var (
	numWorkers  = 4
	numEpochs   = 2
	workerNames = []string{"Ralph", "Bart", "Skinner", "Moe", "Troy", "Barney"}
)

// first round inferences are made without weight (weight = 1)
// second round inferences are made with weights from the actual losses (from first round)
// third round inferences (and beyond) are made with weights from the forecasted losses

func main() {
	fmt.Println("Inference bots engaged")

	n := nw.CreateNetwork(numWorkers, numEpochs, workerNames)

	for epoch := 0; epoch < numEpochs; epoch++ {

		truth := n.Reputer.GetTruth(epoch)

		for _, worker := range n.Workers {
			if epoch == 0 {
				worker.MakeRandomInference(epoch)
			} else {
				n.CollectForecasts(worker, epoch)
				worker.MakeContextualInference(epoch)
			}
		}

		n.CalculateNetworkInference(epoch) // naive or weighted depending on epoch
		n.CalculateNetworkLoss(epoch)
		n.CalculateWorkerLossesAndRegrets(epoch)
		n.NormalizeWorkerRegrets(epoch)
		n.AssignWeights(epoch)

		//
		// Sort and print results
		//

		workers := n.Workers
		sort.Slice(workers, func(i, j int) bool {
			return workers[i].Weight > workers[j].Weight
		})

		fmt.Println("")
		fmt.Printf("Epoch Round %d \n", epoch+1)
		fmt.Printf("Truth: %v \n", truth)
		for _, worker := range workers {
			fmt.Printf("%s: Inference: %v Loss: %v Regret: %v Weight: %v \n",
				worker.Name,
				worker.Inferences[epoch],
				worker.Losses[epoch],
				worker.NormalizedRegrets[epoch],
				worker.Weight,
			)
		}

		fmt.Println("")
		fmt.Println("Network")
		fmt.Printf("Truth: %v \n", truth)             // avg of worker inferences
		fmt.Printf("Inferences: %v \n", n.Inferences) // avg of worker inferences
		fmt.Printf("Losses: %v \n", n.Losses)         // network losses
		fmt.Println("")
	}
}

func normalizeTest() {
	test := []float64{1.0, 3.0, 5.0, 7.0, 8.0, 9.0, 10.2}
	testSD := nw.Sd(test)
	fmt.Println("sd: ", testSD)
	allNormalized := nw.NormalizeSet(test, testSD)
	fmt.Println(allNormalized)
}
