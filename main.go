package main

import (
	"fmt"
	"inference-bots/calc"
)

func main() {
	fmt.Println("inference bots engaged")

	set := calc.CreateRandomFloats(10)
	fmt.Println(set)

	calc.NaturalLogSet(set)

	xx := calc.Log(100)

	fmt.Println(xx)

	calc.Phi(6.2)
}

// Equation (1)
// Inference = Model(Data)

// Equation (2)
// log Loss = Model(Data)

// Equation (3)
// Inference =
// sum of weights * inferences
// sum of weights

// Equation (4)
// Regret = log Loss[i]-1 — log Loss[ijk]

// Equation (6)
// O[p,c](x) = ln [1 + e^p(x-c)]

// Equation (7)
// O'[p,c](x) =
// p
//
