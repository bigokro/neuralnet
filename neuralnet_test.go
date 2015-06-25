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

func assertIntEquals(t *testing.T, expected, actual int) {
	if expected != actual {
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

	nn = NewTestNeuralNet()
	input = NewMultipleTestInputs()
	expected = NewMultipleTestOutputs()
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

	input = NewMultipleTestInputs()
	answers = mat64.NewDense(3, 4, []float64{
		0, 1, 0, 0,
		1, 0, 0, 0,
		0, 0, 0, 1,
	})
	lambda = 1.0
	expected = 6.868093712503938
	result = nn.feedForward(input, answers, lambda)
	assertFloat64Equals(t, expected, result)
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
	expected := NewExpectedGradientsForTestInput()
	for i, r := range result {
		assertMatrixEquals(t, expected[i], r)
	}

	input = NewMultipleTestInputs()
	answers = mat64.NewDense(3, 4, []float64{
		0, 1, 0, 0,
		1, 0, 0, 0,
		0, 0, 0, 1,
	})
	lambda = 1.0
	nn.feedForward(input, answers, lambda)
	result = nn.backpropagation(input, answers, nn.CalculatedValues(), lambda)
	expected = NewExpectedGradientsForMultipleTestInputs()
	for i, r := range result {
		assertMatrixEquals(t, expected[i], r)
	}
}

func TestCalculateCost(t *testing.T) {
	nn := NewTestNeuralNet()
	input := NewTestInput()
	answers := mat64.NewDense(1, 4, []float64{0, 1, 0, 0})
	lambda := 3.0
	expectedCost := 32.58883640171468
	expectedGradients := NewExpectedGradientsForTestInput()
	resultCost, resultGradients := nn.CalculateCost(input, answers, lambda)
	assertFloat64Equals(t, expectedCost, resultCost)
	for i, r := range resultGradients {
		assertMatrixEquals(t, expectedGradients[i], r)
	}

	input = NewMultipleTestInputs()
	answers = mat64.NewDense(3, 4, []float64{
		0, 1, 0, 0,
		1, 0, 0, 0,
		0, 0, 0, 1,
	})
	lambda = 1.0
	expectedCost = 6.868093712503938
	expectedGradients = NewExpectedGradientsForMultipleTestInputs()
	resultCost, resultGradients = nn.CalculateCost(input, answers, lambda)
	assertFloat64Equals(t, expectedCost, resultCost)
	for i, r := range resultGradients {
		assertMatrixEquals(t, expectedGradients[i], r)
	}

	nn = NewNeuralNet([]int{4, 4, 4})
	nn.Thetas[0] = mat64.NewDense(
		4,
		5,
		[]float64{
			0.971951, 0.569477, 0.534057, 0.951718, 0.752634,
			0.313339, 0.961641, 0.661305, 0.020308, 0.663842,
			0.614645, 0.724846, 0.434826, 0.466101, 0.026483,
			0.126076, 0.831136, 0.022930, 0.089201, 0.211449,
		})
	nn.Thetas[1] = mat64.NewDense(
		4,
		5,
		[]float64{
			0.796090, 0.587436, 0.144462, 0.845104, 0.910256,
			0.492499, 0.111620, 0.882888, 0.015276, 0.654968,
			0.411621, 0.440612, 0.454889, 0.754889, 0.105003,
			0.149551, 0.787392, 0.919184, 0.480001, 0.359488,
		})
	input = mat64.NewDense(
		10,
		4,
		[]float64{
			0.140949, 0.160753, 0.151236, 0.292397,
			0.857984, 0.266235, 0.540927, 0.014107,
			0.593759, 0.976316, 0.467458, 0.899625,
			0.012129, 0.240318, 0.417454, 0.104194,
			0.547255, 0.834684, 0.189232, 0.961101,
			0.138613, 0.533119, 0.288622, 0.621332,
			0.337160, 0.461650, 0.837183, 0.209212,
			0.841886, 0.642226, 0.174368, 0.704783,
			0.857430, 0.954894, 0.429795, 0.916772,
			0.489156, 0.054061, 0.039228, 0.541256,
		})
	answers = mat64.NewDense(
		10,
		4,
		[]float64{
			0, 0, 1, 0,
			0, 0, 1, 0,
			0, 1, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 1,
			1, 0, 0, 0,
			0, 0, 1, 0,
			0, 0, 1, 0,
			0, 1, 0, 0,
			0, 0, 0, 1,
		})
	lambda = 1.0
	expectedCost = 7.581017227442351
	expectedGradients = make([]Matrix, 2)
	expectedGradients[0] = mat64.NewDense(
		4,
		5,
		[]float64{
			0.11685355826934, 0.10304531284421498, 0.101659055655753, 0.13561517621523123, 0.12457143567863183,
			0.22663318312471398, 0.18720915839081884, 0.16157529490009193, 0.08494693807087635, 0.1627385020488392,
			0.22373438018555225, 0.17023449654207687, 0.15326321591421635, 0.12315523005302613, 0.1178590375232789,
			0.31164360019090864, 0.23227779688972072, 0.15073478208931565, 0.1201079752523705, 0.17336633025661322,
		})
	expectedGradients[1] = mat64.NewDense(
		4,
		5,
		[]float64{
			0.8369621354738375, 0.8092957147189069, 0.684949378234948, 0.751283163691472, 0.6508013430640769,
			0.5501538490163134, 0.49959773573852767, 0.5227722513549631, 0.4319361958571474, 0.4280413459603802,
			0.4619210093873664, 0.4615907043306723, 0.42331343554474066, 0.44013258196241567, 0.31574127811927005,
			0.6997081567033978, 0.706946399154583, 0.6455851016850678, 0.605440975583609, 0.49643949188024283,
		})
	resultCost, resultGradients = nn.CalculateCost(input, answers, lambda)
	assertFloat64Equals(t, expectedCost, resultCost)
	for i, r := range resultGradients {
		assertMatrixEquals(t, expectedGradients[i], r)
	}

}

func TestPercentCorrect(t *testing.T) {
	expected := []int{1, 2, 3, 2, 3}
	actual := []int{1, 2, 3, 2, 1}
	expectedPercent := 0.80
	expectedPercentByLabel := []float64{0, 1, 1, 0.5}
	actualPercent, actualPercentByLabel := PercentCorrect(4, expected, actual)
	assertFloat64Equals(t, expectedPercent, actualPercent)
	for i := 0; i < len(expectedPercentByLabel); i++ {
		assertFloat64Equals(t, expectedPercentByLabel[i], actualPercentByLabel[i])
	}

	expected = []int{1, 2, 3, 1, 2, 3, 1}
	actual = []int{1, 2, 3, 1, 1, 1, 1}
	expectedPercent = 5.0 / 7.0
	expectedPercentByLabel = []float64{0, 1, 0.5, 0.5}
	actualPercent, actualPercentByLabel = PercentCorrect(4, expected, actual)
	assertFloat64Equals(t, expectedPercent, actualPercent)
	for i := 0; i < len(expectedPercentByLabel); i++ {
		assertFloat64Equals(t, expectedPercentByLabel[i], actualPercentByLabel[i])
	}

	expected = []int{1, 2, 3, 1, 2, 3, 1}
	actual = []int{3, 1, 2, 3, 1, 2, 3}
	expectedPercent = 0
	expectedPercentByLabel = []float64{0, 0, 0, 0}
	actualPercent, actualPercentByLabel = PercentCorrect(4, expected, actual)
	assertFloat64Equals(t, expectedPercent, actualPercent)
	for i := 0; i < len(expectedPercentByLabel); i++ {
		assertFloat64Equals(t, expectedPercentByLabel[i], actualPercentByLabel[i])
	}
}

func TestChooseBest(t *testing.T) {
	values := []float64{0.1, 0.9, 0.3, 0.4, 0.5}
	expected := 1
	actual := ChooseBest(values)
	assertIntEquals(t, expected, actual)

	values = []float64{-0.1, -0.9, -0.3, 0.4, 0.5}
	expected = 4
	actual = ChooseBest(values)
	assertIntEquals(t, expected, actual)

	values = []float64{
		0.21897700730174297, 0.09133133358142083, 0.17227162520454053, 0.06044174984655499, 0.21090492917285905,
		0.012962172064862833, 0.03200441933329224, 0.0030343957738069895, 0.010059393050123757, 0.03693911166947474,
		0.1444435492513144, 0.1525085746435154, 0.008912377810902893, 0.0035007009243766436, 0.11039837873827689, 0.011632145607755892,
	}
	expected = 0
	actual = ChooseBest(values)
	assertIntEquals(t, expected, actual)
}

func TestChooseBestFromEach(t *testing.T) {
	values := mat64.NewDense(
		2,
		5,
		[]float64{
			0.1, 0.9, 0.3, 0.4, 0.5,
			-0.1, -0.9, -0.3, 0.4, 0.5,
		})
	expected := []int{1, 4}
	actual := ChooseBestFromEach(values)
	for i, expectedVal := range expected {
		assertIntEquals(t, expectedVal, actual[i])
	}
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

func NewMultipleTestInputs() Matrix {
	m := mat64.NewDense(
		3,
		10,
		[]float64{
			-0.259, 0.128, 0.131, 0.178, 0.243, 0.256, 0.152, 0.0155, 0.0588, -0.072,
			0.121, 0.202, 0.573, -0.135, 0.861, -0.107, 0.098, 0.208, 0.39, -0.417,
			-0.029, -0.679, 0.317, 0.781, -0.430, -0.568, 0.522, -0.151, -0.586, 0.0563,
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

func NewMultipleTestOutputs() Matrix {
	m := mat64.NewDense(
		3,
		4,
		[]float64{
			0.6192085632020828, 0.8029390000982674, 0.5736790525986135, 0.6110766841737599,
			0.6219883960007477, 0.7996823280371267, 0.5744492092972601, 0.6055849221470206,
			0.5969988254649321, 0.8138660054659164, 0.5649386818032319, 0.6290074867980721,
		})
	return m
}

func NewExpectedGradientsForTestInput() []Matrix {
	expected := make([]Matrix, 3)
	expected[0] = mat64.NewDense(5, 11, []float64{
		0.03381626686391453, -0.12875841311775385, 0.7843284821585811, 0.010429930959172804, -0.9839807044982232, -0.11178264715206876, 1.5386569643171621, 0.07714007256331501, -0.3594758478636093, 0.6289883964915982, 0.36656522878579817,
		-0.00644795673257464, 0.7666700207937368, -0.15082533846176957, 0.39515531766803275, 0.2478522637016017, -0.12156685348601563, 0.0013493230764608922, 0.39801991057664865, 1.118900056670645, -0.06037913985587539, -1.5895357471152547,
		-0.0007140451488353892, 1.2871849376935482, 2.3999086022209495, 0.07790646008550256, 0.9628728999635073, 0.941826487028833, 0.8338172044418982, -1.770108534862623, 0.542988932300193, 0.9689580141452485, -1.349948588749284,
		-0.040389316905360365, 1.1474608330784883, 1.1048301674361138, -0.2452910005146022, 2.935810701590846, 0.09518539599199743, -0.7603396651277723, 0.5188608238303851, 0.32337396558796694, 1.383625108165965, 2.405908030817186,
		-0.007319479491785703, 0.4188957451883725, 0.9800631066250515, -2.910958851813424, 2.2066971326504623, 2.7702213664834963, 0.09712621325010286, -2.2511125608827514, -2.9701134519321224, 2.756569614605883, -0.9294729974765914,
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
	return expected
}

func NewExpectedGradientsForMultipleTestInputs() []Matrix {
	expected := make([]Matrix, 3)
	expected[0] = mat64.NewDense(5, 11, []float64{
		0.012958715790743746, -0.017498062298857097, 0.0799377370710754, 0.0006362076168063592, -0.09937062790580076, -0.02161601432822039, 0.16825088146388129, 0.01397544737730395, -0.04294321743039587, 0.06155687917075615, 0.04405823630307498,
		-0.0037267770843262883, 0.08542793803992413, -0.016895475423881597, 0.042920602632856256, 0.027117595533969606, -0.014679017893541653, 0.0001428630643259894, 0.04367640570758544, 0.12412062185866392, -0.007003951333894589, -0.17605017821860208,
		-0.01137801952768, 0.14141436700063545, 0.26262780589522, 0.0017413933360734789, 0.11028949348248737, 0.09244097970879582, 0.09287774874762246, -0.1969484356418855, 0.057296106704779345, 0.10136037473637673, -0.14439296875843646,
		0.0014509267972148924, 0.1309978103819245, 0.12093972721519232, -0.020954784255910595, 0.3264194511883114, 0.015839172463801197, -0.0903028703357523, 0.059521020853378626, 0.03739266447896083, 0.15494473757345056, 0.2637287919203012,
		-0.006943094567706881, 0.04597299107431753, 0.1051505365852263, -0.32699682082638887, 0.2482388753191244, 0.29967951372651164, 0.009482361906136107, -0.24954763096942137, -0.33204518318147025, 0.3015227623209141, -0.09986822717145927,
	})
	expected[1] = mat64.NewDense(5, 6, []float64{
		0.11597181009006718, 0.01256381720217218, 0.08018306808544562, -0.05664917716265643, 0.05545657818232258, 0.05672617626832944,
		0.056156666775978754, -0.15342690352119964, -0.030226868675135875, -0.0297650752382971, 0.16050312460648955, -0.13000092213714567,
		0.040273530551423464, 0.3260620019852209, -0.048802041582246464, 0.28691281694819926, -0.07191272501103833, 0.08593863040812674,
		0.017913596029747448, 0.05783777871064337, 0.009041976628171163, -0.29203324015118737, -0.2622815156060299, -0.13073594924877924,
		0.12670966552417892, 0.21431545031890917, 0.08263912983798084, 0.01594376125712134, 0.10917827214999604, 0.049215700519625406,
	})
	expected[2] = mat64.NewDense(4, 6, []float64{
		0.2793985948892542, 0.12509411048941327, 0.14299861047407397, 0.49293212968724287, 0.21264287545548785, 0.16192180784247884,
		0.47216244453377015, 0.5778217799362864, 0.5054055765499956, 0.26562343465419336, 0.07677965486573063, 0.6047405644458651,
		0.5710223145663685, 0.35443759129504765, 0.030470947844568524, 0.384646954745718, 0.35277664978337375, 0.37145433450194687,
		0.2818896977062842, 0.1597438614533475, 0.3869625695823985, 0.3071197719291894, 0.2461632371484419, 0.17294901842154642,
	})
	return expected
}
