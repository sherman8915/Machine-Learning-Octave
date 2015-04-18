# Classification - Neural Network, Logistic Regression

Octave is a very powerful and efficent prototyping tool. 
In order to demonstrate its functionality I implemented serveral commonly used 
classification algorithms libraries such as Neural Network and Logistic Regression.
Also there are examples using these libraries showing Neural Network ability to classify handwritten characters. 

Algorithms implemented:

1. Logistic regression - 
  - Cost function with advanced optimization
  - One vs All classification

2. Neural Network - 
  - Cost function
  - BackPropgation and FeedForward algorithms
  - Numerical gradient checking for debugging purposes

## Installation

You need octave to use the libraries or run the examples.

So on a mac(You can probably figure out octave installation on Ubuntu/Windows etc..):
`brew install octave`

Create a project dir:
`mkdir $PROJECT_DIR`
`cd $PROJECT_DIR`

Clone the repo:
`git clone $REPO_URL`

## Usage

Start octave:
`octave`

Within octave:
`cd neural_network`

Run the first Example demonstrating the usage of logistic regression On-Vs-ALl classification:
`ex3`

Run Neural Network classification example:
`ex3_nn`

Run Neural Network training and validation example:
`ex4`

After running the examples I recommend looking through them to understand which 
files store what algorithm or method you are looking for.

## Contributing

If you have more examples you wish to add please feel free to contribute:

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## Credits

The examples and training data set is borrowed from Andrew Ng.

