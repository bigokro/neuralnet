package neuralnet

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"testing"
)

// Utility methods for this package

func assertMatrixEquals(t *testing.T, expected, actual Matrix) {
	if !actual.Equals(expected) {
		fmt.Printf(">>Expected %+v but got %+v<<\n", expected, actual)
		t.Fail()
	}
}

var EPSILON float64 = 0.00000001

func assertFloat64Equals(t *testing.T, expected, actual float64) {
	if (expected-actual) > EPSILON || (actual-expected) > EPSILON {
		fmt.Printf(">>Expected %+v but got %+v<<\n", expected, actual)
		t.Fail()
	}
}

func assertTrue(t *testing.T, istrue bool, msg string) {
	if !istrue {
		fmt.Println(msg)
		t.Fail()
	}
}

// Tests

func TestNewNeuralNet(t *testing.T) {
	nn := NewNeuralNet([]int{10, 25, 5, 8})
	assertTrue(t, len(nn.Thetas) == 3, "Expected 3 layers of Thetas")

	rows, cols := nn.Thetas[0].Dims()
	assertTrue(t, rows == 25, "Expected 25 rows in layer 0")
	assertTrue(t, cols == 11, "Expected 11 cols in layer 0")

	rows, cols = nn.Thetas[1].Dims()
	assertTrue(t, rows == 5, "Expected 5 rows in layer 1")
	assertTrue(t, cols == 26, "Expected 26 cols in layer 1")

	rows, cols = nn.Thetas[2].Dims()
	assertTrue(t, rows == 8, "Expected 8 rows in layer 2")
	assertTrue(t, cols == 6, "Expected 6 cols in layer 2")
}

func TestNumInputs(t *testing.T) {
	nn := NewNeuralNet([]int{10, 25, 5, 8})
	assertTrue(t, nn.NumInputs() == 10, "Expected 10 input nodes")
}

func TestNumOutputs(t *testing.T) {
	nn := NewNeuralNet([]int{10, 25, 5, 8})
	assertTrue(t, nn.NumOutputs() == 8, "Expected 8 output nodes")
}

func TestCalculate(t *testing.T) {
	nn := NewNeuralNet([]int{2, 2, 2})
	nn.Thetas[0] = mat64.NewDense(2, 3, []float64{1, 1, 1, 1, 1, 1})
	nn.Thetas[1] = mat64.NewDense(2, 3, []float64{1, 1, 1, 1, 1, 1})
	var input, expected Matrix
	input = mat64.NewDense(1, 2, []float64{1, 1})
	expected = mat64.NewDense(1, 2, []float64{0.9481003474891515, 0.9481003474891515})
	result := nn.Calculate(input)
	assertMatrixEquals(t, expected, result)

	nn = NewTestNeuralNet()
	input = NewTestInput()
	expected = NewTestOutput()
	result = nn.Calculate(input)
	assertMatrixEquals(t, expected, result)
}

func TestCategorize(t *testing.T) {
	nn := NewTestNeuralNet()
	m := NewTestInput()
	_, cols := m.Dims()
	input := m.Row(make([]float64, cols), 0)
	expected := 1
	result := nn.Categorize(input)
	assertTrue(
		t,
		expected == result,
		fmt.Sprintf("Expected to choose label=1, but was label=%d", result),
	)
}

func TestFeedForward(t *testing.T) {
	nn := NewTestNeuralNet()
	input := NewTestInput()
	answers := mat64.NewDense(1, 4, []float64{0, 1, 0, 0})
	lambda := 3.0
	expected := 32.58883640171468
	result := nn.feedForward(input, answers, lambda)
	assertFloat64Equals(t, expected, result)

	// TODO: NewTestMultipleInputs (Matrix)
}

// Utility functions

func NewTestNeuralNet() NeuralNet {
	nn := NewNeuralNet([]int{10, 5, 5, 4})
	nn.Thetas[0] = mat64.NewDense(
		5,
		11,
		[]float64{
			0.100, -0.04, 0.260, 0.002, -0.33, -0.04, 0.510, 0.024, -0.12, 0.209, 0.123,
			0.223, 0.255, -0.05, 0.132, 0.083, -0.04, 0.001, 0.133, 0.373, -0.02, -0.53,
			-0.18, 0.429, 0.800, 0.026, 0.321, 0.314, 0.278, -0.59, 0.181, 0.323, -0.45,
			0.107, 0.379, 0.370, -0.08, 0.981, 0.035, -0.25, 0.175, 0.108, 0.462, 0.801,
			-0.12, 0.139, 0.327, -0.97, 0.736, 0.924, 0.033, -0.75, -0.99, 0.919, -0.31,
		})
	nn.Thetas[1] = mat64.NewDense(
		5,
		6,
		[]float64{
			0.435, -0.11, 0.023, -0.32, -0.03, 0.032,
			0.307, -0.55, -0.20, -0.20, 0.399, -0.49,
			0.348, 0.923, -0.21, 0.828, -0.29, 0.218,
			0.929, 0.128, 0.001, -0.92, -0.81, -0.44,
			0.122, 0.483, 0.012, -0.11, 0.111, 0.001,
		})
	nn.Thetas[2] = mat64.NewDense(
		4,
		6,
		[]float64{
			-0.23, -0.11, 0.001, 0.923, 0.190, -0.01,
			0.293, 0.931, 0.837, -0.19, -0.47, 0.984,
			0.440, 0.100, -0.70, -0.08, 0.230, 0.101,
			-0.29, 0.023, 0.834, 0.233, 0.386, 0.011,
		})

	return nn
}

func NewTestInput() Matrix {
	m := mat64.NewDense(
		1,
		10,
		[]float64{
			-0.259, 0.128, 0.131, 0.178, 0.243, 0.256, 0.152, 0.0155, 0.0588, -0.072,
		})
	return m
}

func NewTestOutput() Matrix {
	m := mat64.NewDense(
		1,
		4,
		[]float64{
			0.6192085632020828,
			0.8029390000982674,
			0.5736790525986135,
			0.6110766841737599,
		})
	return m
}
