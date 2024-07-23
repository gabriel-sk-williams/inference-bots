package workers

import "math/rand/v2"

type Worker struct {
	name string
}

func CreateWorker() Worker {
	return Worker{name: "Ralph"}
}

func Infer() float64 {
	return rand.Float64()
}
