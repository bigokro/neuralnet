package neuralnet

import (
	"github.com/gonum/matrix/mat64"
	"math"
	"math/rand"
)

type Matrix interface {
	mat64.Matrix
	mat64.Vectorer
	mat64.VectorSetter
	mat64.RowViewer
	mat64.ColViewer
	mat64.Augmenter
	mat64.Muler
	mat64.Sumer
	mat64.Suber
	mat64.Adder
	mat64.ElemMuler
	mat64.ElemDiver
	mat64.Equaler
	mat64.Applyer
}

func Prepend(sl []float64, val float64) []float64 {
	valSl := []float64{val}
	return append(valSl, sl...)
}

func NewZeroes(rows, cols int) Matrix {
	zeroes := make([]float64, rows*cols)
	for i, _ := range zeroes {
		zeroes[i] = 0
	}
	m := mat64.NewDense(rows, cols, zeroes)
	return m
}

func NewOnes(rows, cols int) Matrix {
	ones := make([]float64, rows*cols)
	for i, _ := range ones {
		ones[i] = 1
	}
	m := mat64.NewDense(rows, cols, ones)
	return m
}

func NewRand(rows, cols int) Matrix {
	rands := make([]float64, rows*cols)
	for i, _ := range rands {
		rands[i] = rand.Float64()
	}
	m := mat64.NewDense(rows, cols, rands)
	return m
}

// TODO: turn this into a function that returns a function (via closure)
func NewForValue(rows, cols int, val float64) Matrix {
	vals := make([]float64, rows*cols)
	for i, _ := range vals {
		vals[i] = val
	}
	m := mat64.NewDense(rows, cols, vals)
	return m
}

// The following functions can be passed to a mat64.Applyer as an ApplyFunc

// Calculate the sigmoid of a matrix cell
func Sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

// Calculate the gradient of the sigmoid of a matrix cell
func SigmoidGradient(r, c int, z float64) float64 {
	s := Sigmoid(r, c, z)
	return s * (1 - s)
}

// Calculate the log of a matrix cell
func Log(r, c int, z float64) float64 {
	return math.Log(z)
}

// Calculate the square of a matrix cell
func Square(r, c int, z float64) float64 {
	return math.Pow(z, 2)
}
