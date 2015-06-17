package neuralnet

import ()

// Implementation of a github.com/alonsovidales/go_ml ml.DataSet
type DataSet struct {
	NN       *NeuralNet
	Examples Matrix
	Answers  Matrix
}

// Implementation of the ml.DataSet interface for NeuralNet

// Returns the cost and gradients for the current thetas configuration
func (ds DataSet) CostFunction(lambda float64, calcGrad bool) (j float64, grad [][][]float64, err error) {
	j, gradients := ds.NN.CalculateCost(ds.Examples, ds.Answers, lambda)
	grad = make([][][]float64, len(gradients))
	for i, gradient := range gradients {
		rows, _ := gradient.Dims()
		grad[i] = make([][]float64, rows)
		for j := 0; j < rows; j++ {
			var dst []float64
			grad[i][j] = gradient.Row(dst, j)
		}
	}
	return
}

// Returns the thetas in a 1xn matrix
func (ds DataSet) RollThetasGrad(x [][][]float64) [][]float64 {
	rolled := make([][]float64, 1)

	numElems := 0
	for _, matrix := range x {
		numElems += len(matrix) * len(matrix[0])
	}

	flat := make([]float64, numElems)

	idx := 0
	for _, matrix := range x {
		for _, row := range matrix {
			for _, val := range row {
				flat[idx] = val
				idx++
			}
		}
	}

	rolled[0] = flat
	return rolled
}

// Returns the thetas rolled by the rollThetasGrad method as it original form
func (ds DataSet) UnrollThetasGrad(x [][]float64) [][][]float64 {
	unrolled := make([][][]float64, len(ds.NN.Thetas))

	idx := 0
	for i, theta := range ds.NN.Thetas {
		rows, cols := theta.Dims()
		unrolled[i] = make([][]float64, rows)
		for j := 0; j < rows; j++ {
			unrolled[i][j] = make([]float64, cols)
			for k := 0; k < cols; k++ {
				unrolled[i][j][k] = x[0][idx]
				idx++
			}
		}
	}

	return unrolled
}

// Sets the Theta param after convert it to the corresponding internal data structure
func (ds DataSet) SetTheta(t [][][]float64) {
	for i, matrix := range t {
		for j, row := range matrix {
			ds.NN.Thetas[i].SetRow(j, row)
		}
	}
}

// Returns the theta as a 3 dimensional slice
func (ds DataSet) GetTheta() [][][]float64 {
	th := make([][][]float64, len(ds.NN.Thetas))
	for i, theta := range ds.NN.Thetas {
		rows, _ := theta.Dims()
		th[i] = make([][]float64, rows)
		for j := 0; j < rows; j++ {
			var dst []float64
			th[i][j] = theta.Row(dst, j)
		}
	}

	return th
}
