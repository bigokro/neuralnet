package neuralnet

import (
	"github.com/gonum/matrix/mat64"
	"testing"
)

func TestNewOnes(t *testing.T) {
	expected := mat64.NewDense(1, 1, []float64{1})
	result := NewOnes(1, 1)
	assertMatrixEquals(t, expected, result)

	expected = mat64.NewDense(2, 2, []float64{1, 1, 1, 1})
	result = NewOnes(2, 2)
	assertMatrixEquals(t, expected, result)

	expected = mat64.NewDense(4, 1, []float64{1, 1, 1, 1})
	result = NewOnes(4, 1)
	assertMatrixEquals(t, expected, result)

	expected = mat64.NewDense(3, 2, []float64{1, 1, 1, 1, 1, 1})
	result = NewOnes(3, 2)
	assertMatrixEquals(t, expected, result)
}

func TestSigmoid(t *testing.T) {
	expected := 0.5
	result := Sigmoid(0, 0, 0)
	assertFloat64Equals(t, expected, result)

	result = Sigmoid(1, 1, 0)
	assertFloat64Equals(t, expected, result)

	result = Sigmoid(5, 43, 0)
	assertFloat64Equals(t, expected, result)

	expected = 0.7310585786300049
	result = Sigmoid(0, 0, 1)
	assertFloat64Equals(t, expected, result)

	expected = 0.2689414213699951
	result = Sigmoid(0, 0, -1)
	assertFloat64Equals(t, expected, result)

	expected = 1.0
	result = Sigmoid(0, 0, 10000)
	assertFloat64Equals(t, expected, result)
}

func TestSigmoidGradient(t *testing.T) {
	expected := 0.2500
	result := SigmoidGradient(0, 0, 0)
	assertFloat64Equals(t, expected, result)

	result = SigmoidGradient(1, 1, 0)
	assertFloat64Equals(t, expected, result)

	result = SigmoidGradient(5, 43, 0)
	assertFloat64Equals(t, expected, result)

	expected = 0.19661193324148185
	result = SigmoidGradient(0, 0, 1)
	assertFloat64Equals(t, expected, result)

	result = SigmoidGradient(0, 0, -1)
	assertFloat64Equals(t, expected, result)

	expected = 0
	result = SigmoidGradient(0, 0, 10000)
	assertFloat64Equals(t, expected, result)
}
