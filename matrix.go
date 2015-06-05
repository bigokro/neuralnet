package neuralnet

import (
	"github.com/gonum/matrix/mat64"
	"math/rand"
)

type Matrix interface {
	mat64.Matrix
	mat64.Vectorer
	mat64.VectorSetter
	mat64.Augmenter
	mat64.Muler
	mat64.Sumer
	mat64.Suber
	mat64.ElemMuler
	mat64.Equaler
	mat64.Applyer
}

func Prepend(sl []float64, val float64) []float64 {
	valSl := []float64{val}
	return append(valSl, sl...)
}

func Zeroes(rows, cols int) Matrix {
	zeroes := make([]float64, rows*cols)
	for i, _ := range zeroes {
		zeroes[i] = 0
	}
	m := mat64.NewDense(rows, cols, zeroes)
	return m
}

func Ones(rows, cols int) Matrix {
	ones := make([]float64, rows*cols)
	for i, _ := range ones {
		ones[i] = 1
	}
	m := mat64.NewDense(rows, cols, ones)
	return m
}

func Rand(rows, cols int) Matrix {
	rands := make([]float64, rows*cols)
	for i, _ := range rands {
		rands[i] = rand.Float64()
	}
	m := mat64.NewDense(rows, cols, rands)
	return m
}
