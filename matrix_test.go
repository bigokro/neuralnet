package neuralnet

import (
	"github.com/gonum/matrix/mat64"
	"testing"
)

func TestOnes(t *testing.T) {
	expected := mat64.NewDense(1, 1, []float64{1})
	result := Ones(1, 1)
	assertMatrixEquals(t, expected, result)

	expected = mat64.NewDense(2, 2, []float64{1, 1, 1, 1})
	result = Ones(2, 2)
	assertMatrixEquals(t, expected, result)

	expected = mat64.NewDense(4, 1, []float64{1, 1, 1, 1})
	result = Ones(4, 1)
	assertMatrixEquals(t, expected, result)

	expected = mat64.NewDense(3, 2, []float64{1, 1, 1, 1, 1, 1})
	result = Ones(3, 2)
	assertMatrixEquals(t, expected, result)
}
