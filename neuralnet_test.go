package neuralnet

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"testing"
)

// Utility methods for this package

func assertMatrixEquals(t *testing.T, expected, actual Matrix) {
	if !actual.Equals(expected) {
		fmt.Printf(">>Expected: %+v\nBut got: %+v<<\n", expected, actual)
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

func TestBackpropagate(t *testing.T) {
	nn := NewTestNeuralNet()
	input := NewTestInput()
	answers := mat64.NewDense(1, 4, []float64{0, 1, 0, 0})
	lambda := 3.0
	nn.feedForward(input, answers, lambda)
	result := nn.backpropagation(input, answers, nn.CalculatedValues(), lambda)
	assertTrue(
		t,
		len(result) == 3,
		fmt.Sprintf("Expected backpropagation to return three gradients"),
	)
	expected := make([]Matrix, 3)
	expected[0] = mat64.NewDense(5, 11, []float64{
		0, -0.12, 0.78, 0.006, -0.99, -0.12, 1.53, 0.07200000000000001, -0.36, 0.627, 0.369,
		0, 0.765, -0.15000000000000002, 0.396, 0.249, -0.12, 0.003, 0.399, 1.119, -0.06, -1.59,
		0, 1.287, 2.4000000000000004, 0.078, 0.9630000000000001, 0.942, 0.8340000000000001, -1.77, 0.5429999999999999, 0.9690000000000001, -1.35,
		0, 1.137, 1.1099999999999999, -0.24, 2.943, 0.10500000000000001, -0.75, 0.5249999999999999, 0.324, 1.3860000000000001, 2.403,
		0, 0.41700000000000004, 0.9810000000000001, -2.91, 2.208, 2.7720000000000002, 0.099, -2.25, -2.9699999999999998, 2.757, -0.9299999999999999,
	})
	expected[1] = mat64.NewDense(5, 6, []float64{
		-0.04445469259511377, -0.3545661229392681, 0.044160414835666226, -0.9820475409905005, -0.11403056461892552, 0.07299739839626156,
		-0.01385434397512419, -1.6576560537812204, -0.6077412785237413, -0.6068711355057627, 1.1895108534394943, -1.477168780984349,
		0.1316909030292129, 2.841773755141307, -0.5564161506872797, 2.5493128030612393, -0.7988127626077433, 0.7221420385651367,
		0.14399032593002323, 0.4635705434537417, 0.08345629729924436, -2.6885872328014555, -2.3521641489397704, -1.2454937576033351,
		-0.032529947379592174, 1.4310236146088473, 0.017823507234970944, -0.3461334002419627, 0.3154155121333252, -0.013832270702634004,
	})
	expected[2] = mat64.NewDense(4, 6, []float64{
		0.6192085632020828, 0.014854000676620072, 0.2747535398629969, 3.2344254669872865, 0.8608165596394972, 0.3409476659182792,
		-0.19706099990173265, 2.683251397942523, 2.4245153686904675, -0.7181200573679938, -1.5025513719225456, 2.833947178541823,
		0.5736790525986135, 0.6194973844837481, -1.8482281051109348, 0.19120340515929668, 0.9594332383764645, 0.6466724202378563,
		0.6110766841737599, 0.40932513724903175, 2.7701846827072485, 1.1583131749113027, 1.444997353603017, 0.39907612226667477,
	})
	for i, r := range result {
		assertMatrixEquals(t, expected[i], r)
	}

	// TODO: NewTestMultipleInputs (Matrix)
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
