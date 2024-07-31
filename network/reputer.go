package network

import (
	"fmt"
	"math/rand"
	random "math/rand/v2"
)

type Reputer struct {
	Name   string
	Truths []float64
}

func NewReputer(name string, numPoints int) *Reputer {
	reputer := Reputer{Name: name}
	reputer.CreateRandomInts(numPoints)
	return &reputer
}

func (r *Reputer) SayHello() {
	fmt.Println(r.Name, "says Hi!")
}

func (r *Reputer) RevealTruths() {
	fmt.Println("The sacred elements", r.Truths)
}

func (r *Reputer) GetTruth(epoch int) float64 {
	return r.Truths[epoch]
}

func (r *Reputer) CreateRandomInts(length int) {
	set := make([]float64, length)
	for i := 0; i < length; i++ {
		integer := rand.Intn(10)
		set[i] = float64(integer)
	}
	r.Truths = set
}

func (r *Reputer) CreateRandomFloats(length int) {
	set := make([]float64, length)
	for i := 0; i < length; i++ {
		set[i] = random.Float64() * 10
	}
	r.Truths = set
}
