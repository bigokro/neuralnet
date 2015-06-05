package neuralnet

import (
	"github.com/gonum/matrix/mat64"
	"math"
)

// TODO: make this more generic? Separate NN implementations for categorization vs. calculation?
type NeuralNet struct {
	Thetas []Matrix
}

// Creates a new NeuralNet
// The number of layers in the network will be defined by the size of the slice passed to the method
// The number of nodes per layer (starting with the input layer) are defined by the values in the slice
// The values of thetas in each node will be initialized randomly
// (requires the caller to seed the rand package beforehad as needed)
func NewNeuralNet(nodes []int) NeuralNet {
	t := make([]Matrix, len(nodes)-1)
	for i := 1; i < len(nodes); i++ {
		t[i-1] = Rand(nodes[i], nodes[i-1]+1)
	}
	nn := NeuralNet{Thetas: t}
	return nn
}

// Returns the number of float64s expected as inputs to the calculation (number of input nodes)
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
	a := mat64.DenseCopyOf(input)
	for _, th := range nn.Thetas {
		var x, z mat64.Dense

		//X = [ones(m, 1) X];
		rows, _ := a.Dims()
		ones := Ones(rows, 1)
		x.Augment(ones, a)

		//A2 = sigmoid(X * Theta1'); <-- Transpose, multiply, Apply sigmoid
		thRows, thCols := th.Dims()
		thT := mat64.NewDense(thCols, thRows, nil)
		thT.TCopy(th)

		z.Mul(&x, thT)
		z.Apply(Sigmoid, &z)

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
	results := calculation.Row(nil, 0)
	best := 0
	bestScore := 0.0
	for i, score := range results {
		if score > bestScore {
			best = i
			bestScore = score
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
	gradients = NewGradients(nn.Thetas)

	cost = nn.feedForward(examples, answers, lambda)

	// TODO

	return cost, gradients
}

func NewGradients(thetas []Matrix) []Matrix {
	gradients := make([]Matrix, len(thetas))
	for i, layer := range thetas {
		r, c := layer.Dims()
		gradients[i] = mat64.NewDense(r, c, nil)
	}
	return gradients
}

func (nn *NeuralNet) feedForward(examples Matrix, answers Matrix, lambda float64) (cost float64) {
	rows, cols := answers.Dims()

	calculated := nn.Calculate(examples)

	//JK = sum( ((0-Y).*log(A3)) - ((1-Y).*log(1-A3)) );
	//J = sum(JK);
	var leftHand, all mat64.Dense
	var rightHand, zeroMinusAnswers, oneMinusAnswers Matrix

	zeroMinusAnswers = Zeroes(rows, cols)
	zeroMinusAnswers.Sub(zeroMinusAnswers, answers)
	leftHand.Apply(Log, calculated)
	leftHand.MulElem(zeroMinusAnswers, &leftHand)

	oneMinusAnswers = Ones(rows, cols)
	oneMinusAnswers.Sub(oneMinusAnswers, answers)
	rightHand = Ones(calculated.Dims())
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

// The following functions can be passed to a mat64.Applyer as an ApplyFunc

// Calculate the sigmoid of a matrix cell
func Sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

// Calculate the log of a matrix cell
func Log(r, c int, z float64) float64 {
	return math.Log(z)
}

// Calculate the square of a matrix cell
func Square(r, c int, z float64) float64 {
	return math.Pow(z, 2)
}
