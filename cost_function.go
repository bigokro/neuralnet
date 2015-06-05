package neuralnet

import (
	"github.com/gonum/matrix/mat64"
)

// A cost function is a function that can be used by the
// gradient descent (or similar) algorithm to measure,
// and thus minimize, the cost (or error) of the network
// by optimizing its parameters.
//
// A cost function takes a slice of mat64.Matrix structs,
// one for each layer of the network (from first hidden layer to the output layer),
// each containing the thetas (weights) for that layer
// a mat64.Matrix containing the training data inputs,
// a mat64.Matrix containing the expected outputs for the training data,
// and a lambda value that is the regularization parameter
// (0.0 would mean no regularization would be performed)
//
// The cost function should return two values:
// The current cost for the network for the given examples,
// and a slice of mat64.Matrix structs with the theta gradients to
// be used for performing gradient descent
type CostFunction interface {
	CalculateCost(thetas []mat64.Matrix, examples mat64.Matrix, answers mat64.Matrix, lambda float64) (cost float64, gradients []mat64.Matrix)
}

type DefaultCostFunction struct{}

/*

func (d *DefaultCostFunction) CalculateCost(thetas []mat64.Matrix, examples mat64.Matrix, answers mat64.Matrix, lambda float64) (cost float64, gradients []mat64.Matrix) {
	gradients = NewGradients(thetas)

	cost = FeedForward(thetas, examples, answers, lambda)
}

func NewGradients(thetas []mat64.Matrix) []mat64.Matrix {
	gradients := make([]mat64.Matrix, len(thetas))
	for i, layer := range thetas {
		r, c := layer.Dims()
		gradients[i] = mat64.NewDense(r, c, nil)
	}
	return gradients
}

func FeedForward() float64 {
}

*/
