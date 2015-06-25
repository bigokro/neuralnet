package neuralnet

import (
	"bytes"
	"fmt"
	"github.com/bigokro/go_ml"
	"github.com/gonum/matrix/mat64"
	"io/ioutil"
	"log"
	"strconv"
	"strings"
)

// TODO: make this more generic? Separate NN implementations for categorization vs. calculation?
type NeuralNet struct {
	Thetas  []Matrix
	Zs      []Matrix // Intermediate calculations, pre-sigmoid (Z(n+1) = A(n) * Theta(n)')
	Outputs []Matrix // Calculated values, post-sigmoid (Output(n) = Sigmoid(Z(n))
}

// Creates a new NeuralNet
// The number of layers in the network will be defined by the size of the slice passed to the method
// The number of nodes per layer (starting with the input layer) are defined by the values in the slice
// The values of thetas in each node will be initialized randomly
// (requires the caller to seed the rand package beforehad as needed)
func NewNeuralNet(nodes []int) NeuralNet {
	t := make([]Matrix, len(nodes)-1)
	for i := 1; i < len(nodes); i++ {
		t[i-1] = NewRand(nodes[i], nodes[i-1]+1)
	}
	nn := NeuralNet{Thetas: t}
	return nn
}

// Returns the number of floats expected as inputs to the calculation (number of input nodes)
func (nn *NeuralNet) NumInputs() int {
	_, cols := nn.Thetas[0].Dims()
	return cols - 1
}

// Returns the number of labels in the output layer of the network
func (nn *NeuralNet) NumOutputs() int {
	rows, _ := nn.Thetas[len(nn.Thetas)-1].Dims()
	return rows
}

// Activates the network for the given inputs.
// Returns the calculated confidence scores as a matrix
func (nn *NeuralNet) Calculate(input Matrix) Matrix {
	nn.Zs = make([]Matrix, len(nn.Thetas))
	nn.Outputs = make([]Matrix, len(nn.Thetas))

	a := mat64.DenseCopyOf(input)
	for i, th := range nn.Thetas {
		var x, z mat64.Dense

		//X = [ones(m, 1) X];
		rows, _ := a.Dims()
		ones := NewOnes(rows, 1)
		x.Augment(ones, a)

		//A2 = sigmoid(X * Theta1'); <-- Transpose, multiply, Apply sigmoid
		thRows, thCols := th.Dims()
		thT := mat64.NewDense(thCols, thRows, nil)
		thT.TCopy(th)

		z.Mul(&x, thT)
		nn.Zs[i] = mat64.DenseCopyOf(&z)

		z.Apply(Sigmoid, &z)
		nn.Outputs[i] = mat64.DenseCopyOf(&z)

		a = &z
	}

	return a
}

// Activates the network for the given inputs.
// Based on the calculated results, chooses the best label
// (the output with the highest level of confidence)
// The labels are indexed starting from 0
func (nn *NeuralNet) Categorize(input []float64) int {
	inputMatrix := mat64.NewDense(1, len(input), input)
	calculation := nn.Calculate(inputMatrix)
	best := ChooseBestFromEach(calculation)
	return best[0]
}

func (nn *NeuralNet) CalculatedValues() Matrix {
	return nn.Outputs[len(nn.Outputs)-1]
}

func ChooseBestFromEach(outputs Matrix) []int {
	rows, _ := outputs.Dims()
	best := make([]int, rows)
	for i := 0; i < rows; i++ {
		output := outputs.Row(nil, i)
		best[i] = ChooseBest(output)
	}
	return best
}

func ChooseBest(values []float64) int {
	best := 0
	bestScore := 0.0
	for i, value := range values {
		if value > bestScore {
			best = i
			bestScore = value
		}
	}
	return best
}

// Calculate Cost is the Cost Function for the neural network
// Each call to this method represents a single training step
// in a process of Gradient Descent (or related algorithms)
//
// The method returns the cost ("J"), or error, and a slice of
// matrices corresponding to the gradient for the thetas in
// each layer of the network (from the first hidden layer
// through to the output layer), as in the list of Thetas in
// the network itself
func (nn *NeuralNet) CalculateCost(examples Matrix, answers Matrix, lambda float64) (cost float64, gradients []Matrix) {

	cost = nn.feedForward(examples, answers, lambda)
	gradients = nn.backpropagation(examples, answers, nn.CalculatedValues(), lambda)

	return cost, gradients
}

func (nn *NeuralNet) feedForward(examples Matrix, answers Matrix, lambda float64) (cost float64) {
	rows, cols := answers.Dims()

	calculated := nn.Calculate(examples)

	//JK = sum( ((0-Y).*log(A3)) - ((1-Y).*log(1-A3)) );
	//J = sum(JK);
	var leftHand, all mat64.Dense
	var rightHand, zeroMinusAnswers, oneMinusAnswers Matrix

	zeroMinusAnswers = NewZeroes(rows, cols)
	zeroMinusAnswers.Sub(zeroMinusAnswers, answers)
	leftHand.Apply(Log, calculated)
	leftHand.MulElem(zeroMinusAnswers, &leftHand)

	oneMinusAnswers = NewOnes(rows, cols)
	oneMinusAnswers.Sub(oneMinusAnswers, answers)
	rightHand = NewOnes(calculated.Dims())
	rightHand.Sub(rightHand, calculated)
	rightHand.Apply(Log, rightHand)
	rightHand.MulElem(oneMinusAnswers, rightHand)

	all.Sub(&leftHand, rightHand)

	cost = all.Sum()

	//J = J/m;
	cost = cost / float64(rows)

	// Regularization
	var regCost float64
	var regThetas Matrix
	for _, thetas := range nn.Thetas {
		//RegTheta1 = Theta1;
		//RegTheta1(:,1) = 0;
		r, _ := thetas.Dims()
		zeroes := make([]float64, r)
		regThetas = mat64.DenseCopyOf(thetas)
		regThetas.SetCol(0, zeroes)

		//RegJ += sum(sum( RegTheta1.^2 )) + sum(sum( RegTheta2.^2 ));
		var regSquared mat64.Dense
		regSquared.Apply(Square, regThetas)
		regCost += regSquared.Sum()
	}

	//RegJ = RegJ * (lambda/(2*m));
	regCost = regCost * (lambda / float64(2*rows))

	//J = J + RegJ;
	cost += regCost

	return cost
}

func (nn *NeuralNet) backpropagation(examples Matrix, answers Matrix, calculations Matrix, lambda float64) []Matrix {
	gradients := nn.newGradients()

	// delta3 = a3 - y;
	var errors mat64.Dense
	errors.Sub(calculations, answers)

	inputRows, _ := examples.Dims()

	var delta mat64.Matrix
	// TODO: try this without the dense copy
	delta = mat64.DenseCopyOf(&errors)

	for i := len(nn.Thetas) - 1; i >= 0; i-- {
		var deltaTranspose mat64.Dense
		deltaTranspose.TCopy(delta)

		for j := 0; j < inputRows; j++ {
			var thetaGrad, outputsTranspose, outputsTransposeAugmented mat64.Dense
			if i > 0 {
				outputsTranspose.TCopy(nn.Outputs[i-1].RowView(j))
			} else {
				outputsTranspose.TCopy(examples.RowView(j))
			}

			//a2 = [1 a2];
			ones := NewOnes(1, 1)
			outputsTransposeAugmented.Augment(ones, &outputsTranspose)

			//Theta1_grad = Theta1_grad + (delta2(2:end)' * a1);
			thetaGrad.Mul(deltaTranspose.ColView(j), &outputsTransposeAugmented)
			gradients[i].Add(gradients[i], &thetaGrad)
		}

		if i > 0 {

			var nextDelta, zAugmented, sigmoidGradient mat64.Dense

			//z2 = [1 z2];
			zRows, _ := nn.Zs[i].Dims()
			zAugOnes := NewOnes(zRows, 1)
			zAugmented.Augment(zAugOnes, nn.Zs[i-1])

			// delta2 = (delta3*Theta2) .* sigmoidGradient(z2);
			sigmoidGradient.Apply(SigmoidGradient, &zAugmented)

			nextDelta.Mul(delta, nn.Thetas[i])
			nextDelta.MulElem(&nextDelta, &sigmoidGradient)

			ndRows, ndCols := nextDelta.Dims()
			delta = nextDelta.View(0, 1, ndRows, ndCols-1)
			//delta = &nextDelta
		}
	}

	//Theta2_grad = Theta2_grad / m;
	for _, thetaGrad := range gradients {
		r, c := thetaGrad.Dims()
		m := NewForValue(r, c, float64(inputRows))
		thetaGrad.DivElem(thetaGrad, m)
	}

	// Regularization
	//RegTheta2 = Theta2;
	//RegTheta2(:,1) = 0;
	//Theta2_grad = Theta2_grad + ((lambda/m) * RegTheta2);
	for i, thetaGrad := range gradients {
		r, c := thetaGrad.Dims()
		m := NewForValue(r, c, lambda/float64(inputRows))
		var regTheta mat64.Dense
		regTheta.MulElem(nn.Thetas[i], m)
		zeroVals := make([]float64, r)
		regTheta.SetCol(0, zeroVals)
		thetaGrad.Add(thetaGrad, &regTheta)
	}

	return gradients
}

func (nn *NeuralNet) newGradients() []Matrix {
	gradients := make([]Matrix, len(nn.Thetas))
	for i, layer := range nn.Thetas {
		gradients[i] = NewZeroes(layer.Dims())
	}
	return gradients
}

// Trains the network on the given inputs and expected results
// The inputs and expected slices can be any length, but must be of the same size
// Each pair of input/expected corresponds to a training, testing or validation data set
// Only the first pair will be used for training
//
// The algorithm continues training until the max number of iterations has been reached,
// or until the cost is below the max cost on the training set,
// and the accuracy is above the min percent accuracy (from 0 to 1) for ALL the provided sets
//
// Training will change the Thetas of the neural net.
// The function returns a slice with the percent accuracy (from 0 to 1) of each corresponding training set
func (nn *NeuralNet) Train(inputs []Matrix, expected []Matrix, alpha float64, lambda float64, maxCost float64, minPercentAccuracy float64, maxIterations int) (percentAccuracies []float64, percentAccurateByLabel []float64, scoreByLabel []float64) {
	trainingInputs := inputs[0]
	trainingExpected := expected[0]

	ds := DataSet{nn, trainingInputs, trainingExpected}
	fx, i, err := ml.Fmincg(ds, lambda, maxIterations, true)

	log.Println("Fx: ", fx)
	log.Println("i: ", i)
	log.Println("err: ", err)

	percentAccuracies, percentAccurateByLabel, scoreByLabel = nn.CalculatePercentAccuracies(inputs, expected)

	return
}

func (nn *NeuralNet) CalculatePercentAccuracies(inputs, expected []Matrix) (percentAccuracies []float64, percentAccurateByLabel []float64, scoreByLabel []float64) {
	_, numLabels := expected[0].Dims()
	percentAccuracies = make([]float64, len(inputs))
	percentAccurateByLabel = make([]float64, numLabels)
	scoreByLabel = make([]float64, numLabels)

	for i := 0; i < len(inputs); i++ {
		inputData := inputs[i]
		expectedData := expected[i]

		calculatedData := nn.Calculate(inputData)
		results := ChooseBestFromEach(calculatedData)
		expectedResults := ChooseBestFromEach(expectedData)
		percentAccuracies[i], percentAccurateByLabel = PercentCorrect(numLabels, expectedResults, results)

		tabulation := make([][]int, numLabels)
		for i, _ := range tabulation {
			tabulation[i] = make([]int, numLabels)
		}
		avgScore := make([][]float64, numLabels)
		for i, _ := range avgScore {
			avgScore[i] = make([]float64, numLabels)
		}

		for j := 0; j < len(results); j++ {
			tabulation[expectedResults[j]][results[j]]++
			for k := 0; k < numLabels; k++ {
				avgScore[expectedResults[j]][k] += calculatedData.At(j, k)
			}
		}
		for j := 0; j < len(tabulation); j++ {
			for k := 0; k < len(tabulation[j]); k++ {
				if tabulation[j][k] > 0 {
					avgScore[j][k] = avgScore[j][k] / float64(tabulation[j][k])
				}
			}
		}
		log.Println("Tabulations: ")
		for j := 0; j < len(tabulation); j++ {
			var buffer bytes.Buffer
			for k := 0; k < len(tabulation); k++ {
				buffer.WriteString(fmt.Sprintf("\t%d", tabulation[j][k]))
			}
			log.Println(buffer.String())
		}
		log.Println("Scores: ")
		for j := 0; j < len(avgScore); j++ {
			var buffer bytes.Buffer
			for k := 0; k < len(avgScore); k++ {
				buffer.WriteString(fmt.Sprintf("\t%0.2f", avgScore[j][k]))
			}
			log.Println(buffer.String())
			// TODO: take average of testing and validation results
			scoreByLabel[j] = avgScore[j][j]
		}

	}

	return
}

func PercentCorrect(numLabels int, expected, actual []int) (percentTotal float64, percentCorrectByLabel []float64) {
	correctByLabel := make([]int, numLabels)
	totalByLabel := make([]int, numLabels)
	percentCorrectByLabel = make([]float64, numLabels)
	correct := 0
	for k, result := range actual {
		totalByLabel[expected[k]]++
		if result == expected[k] {
			correct++
			correctByLabel[expected[k]]++
		}
	}
	for i, correctForLabel := range correctByLabel {
		percentCorrectByLabel[i] = float64(correctForLabel) / float64(totalByLabel[i])
	}
	log.Printf("%v correct out of %v\n", correct, len(actual))
	percentTotal = float64(correct) / float64(len(actual))
	return
}

// SaveThetas Store all the current theta values of the instance in the
// "targetDir" directory.
// This method will create a file for each layer of theta called theta_X.txt
// where X is the layer contained on the file.
func (nn *NeuralNet) SaveThetas(targetDir string) (files []string) {
	fileCont := make([]string, len(nn.Thetas))
	for i := 0; i < len(nn.Thetas); i++ {
		thetaValues := MatrixToSlice(nn.Thetas[i])
		for j := 0; j < len(thetaValues); j++ {
			s := []string{}
			for k := 0; k < len(thetaValues[j]); k++ {
				s = append(s, strconv.FormatFloat(thetaValues[j][k], 'e', -1, 64))
			}

			fileCont[i] += strings.Join(s, " ") + "\n"
		}
	}

	files = make([]string, len(nn.Thetas))
	for i := 0; i < len(nn.Thetas); i++ {
		files[i] = fmt.Sprintf("%s/theta_%d.txt", targetDir, i+1)
		ioutil.WriteFile(
			files[i],
			[]byte(fileCont[i]),
			0644)
	}

	return
}

// NewNeuralNetFromFiles Loads the informaton contained in the specified file
// paths and returns a NeuralNet instance.
// Each input file should contain a row by sample, and the values separated by a
// single space.
// To load the thetas specify on thetaSrc the file paths that contains each of
// the layer values. The order of this paths will represent the order of the
// layers.
func NewNeuralNetFromFiles(thetaSrc []string) (result *NeuralNet) {
	result = &NeuralNet{}

	// Parse the Theta params
	for _, thetaNSrc := range thetaSrc {
		strInfo, err := ioutil.ReadFile(thetaNSrc)
		if err != nil {
			panic(err)
		}

		trainingData := strings.Split(string(strInfo), "\n")
		theta := [][]float64{}
		for _, line := range trainingData {
			if line == "" {
				break
			}

			var values []float64
			for _, value := range strings.Split(line, " ") {
				floatVal, err := strconv.ParseFloat(value, 64)
				if err != nil {
					panic(err)
				}
				values = append(values, floatVal)
			}
			theta = append(theta, values)
		}

		// TODO: make this a function or find a way to create a Dense without rolling/unrolling
		rows := len(theta)
		cols := len(theta[0])
		flat := make([]float64, rows*cols)

		idx := 0
		for _, row := range theta {
			for _, val := range row {
				flat[idx] = val
				idx++
			}
		}

		thetaMatrix := mat64.NewDense(rows, cols, flat)
		result.Thetas = append(result.Thetas, thetaMatrix)
	}

	return
}
