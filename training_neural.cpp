//Yilun Jiang
//Prof. Sable
//AI project 2 Single hidden layer neural network

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <iomanip>

using namespace std;

//function prototype
void train(vector<vector<double> > &hidden_weights, vector<vector<double> > &output_weights, string train_file, int hidden_count); 
void propogate(vector<vector<double> > &weights, double *inputs, double *outputs, double *weightedInput, int nodeSize, int inputSize); //calculate the output vector given weight vector and input vector 
double sigmoid(double x); //calculate the output of sigmoid function given input x
double sigmoidDeriv(double x); //calculate the derivative of sigmoid function given input x
void update(double *delta, vector<vector<double> > &weight, double *input, double learning_rate, int nodeSize, int inputSize); //weight update function

int main()
{
    string weight_file;
    cout << "Please enter name of the file that contains the initial weights of the neural network: \n";
    cin >> weight_file;

    ifstream inFile;
    inFile.open(weight_file);
    if(!inFile)
    {
		cerr << "Unable to open initial weight file.";
		exit(1);
	}

    int input_count, hidden_count, output_count; //read in number of input node, hidden nodes and output nodes
    inFile >> input_count >> hidden_count >> output_count;

    vector<vector<double> > hidden_weights(hidden_count);
    //read in initial weights of the hidden nodes
    for (int i = 0; i < hidden_count; i++)
    {
        for (int j = 0; j < input_count+1; j++)
        {   
            double temp = 0;
            inFile >> temp;
            hidden_weights[i].push_back(temp);
        }
    }

    vector<vector<double> > output_weights(output_count);
    //read in initial weights of the output nodes
    for (int i = 0; i < output_count; i++)
    {
        for (int j = 0; j < hidden_count+1; j++)
        {
            double temp = 0;
            inFile >> temp;
            output_weights[i].push_back(temp);
        }
            
    }
    inFile.close(); //Done processing the input file

    string train_file, output_file;
    cout << "Please enter name of the file that contains the training data: \n";
    cin >> train_file;
    cout << "Please enter name of the output file(contains the weight of each node after training): \n";
    cin >> output_file;

    train(hidden_weights, output_weights, train_file, hidden_count);

    ofstream outFile;
    outFile.open(output_file);

    outFile << input_count << ' ' << hidden_count << ' ' << output_count << endl;
    outFile.precision(3);

    for(int i = 0; i < hidden_count; i++) //print out weights of hidden nodes
    {
        outFile << fixed << hidden_weights[i][0];
        for(int j = 1; j < input_count+1; j++)
            outFile << ' ' << fixed << hidden_weights[i][j];
        
        outFile << '\n';
    }

    for(int i = 0; i < output_count; i++)
    {
        outFile << fixed << output_weights[i][0];
        for(int j = 1; j < hidden_count+1; j++)
            outFile << ' ' << fixed << output_weights[i][j];
        
        outFile << '\n';    
    }

    outFile.close();
}

void train(vector<vector<double> > &hidden_weights, vector<vector<double> > &output_weights, string train_file, int hidden_count)
{
    double learning_rate = 0;
    int epoch_count = 0;
    cout << "Please enter a learning rate: \n";
    cin >> learning_rate;
    cout << "Please enter number of epoches for training: \n";
    cin >> epoch_count;
    //Initialization is done, ready to train

    ifstream inFile;
    while(epoch_count > 0)
    {
        inFile.open(train_file);
        if(!inFile)
        {
		    cerr << "Unable to open training data file.";
		    exit(1);
	    }
        int example_count = 0, input_count = 0, output_count = 0;
        inFile >> example_count >> input_count >> output_count;
        
        while(example_count > 0)
        {
            double input[input_count]; //read in training inputs
            for(int i = 0; i < input_count; i++)
            {
                input[i] = 0;
                inFile >> input[i];
            }    

            int output[output_count]; //read in training outputs
            for(int i = 0; i < output_count; i++)
            {
                output[i] = 0;
                inFile >> output[i];
            }

            double hiddenNode_output[hidden_count]; //record the outputs from all hidden nodes
            double hiddenNode_weightedInput[hidden_count]; //record the weighted sum of inputs of all hidden nodes
            for(int i = 0; i < hidden_count; i++)
            {
                hiddenNode_output[i] = 0;
                hiddenNode_weightedInput[i] = 0;
            }    
            propogate(hidden_weights, input, hiddenNode_output, hiddenNode_weightedInput, hidden_count, input_count); //propogate in the forward direction

            double outputNode_output[output_count]; //record the outputs in the output nodes
            double outputNode_weightedInput[output_count];
            for(int i = 0; i < output_count; i++) 
            {
                outputNode_output[i] = 0;
                outputNode_weightedInput[i] = 0;
            }    
            propogate(output_weights, hiddenNode_output, outputNode_output, outputNode_weightedInput, output_count, hidden_count); //propogate in the forward direction
            
            //calculate the output node delta
            //delta = sigmoid'(input) * (y - a) 
            //input is the weighted sum of input to the node, y is the actual output from the data, a is the output from the neural network
            double outputNode_delta[output_count];
            for(int i = 0; i < output_count; i++)
                outputNode_delta[i] = 0; 
            //cout << "Output Node delta is ";
            for(int i = 0; i < output_count; i++)
            {
                outputNode_delta[i] = sigmoidDeriv(outputNode_weightedInput[i]) * (output[i] - outputNode_output[i]);
                //cout << outputNode_delta[i] << ' ';
            }

            //Back propogating output node delta for 1 level
            double weighted_delta[hidden_count];
            double hiddenNode_delta[hidden_count];
            for(int i = 0; i < output_count; i++)
            {
                weighted_delta[i] = 0;
                hiddenNode_delta[i] = 0;
            }
            for(int i = 0; i < hidden_count; i++)
            {
                for(int j = 0; j < output_count; j++)
                {
                    weighted_delta[i] += output_weights[j][i+1] * outputNode_delta[j];
                }
            }
            
            for(int i = 0; i < hidden_count; i++)
                hiddenNode_delta[i] = sigmoidDeriv(hiddenNode_weightedInput[i]) * weighted_delta[i];

            //cout << "output node update" << endl;
            update(outputNode_delta, output_weights, hiddenNode_output, learning_rate, output_count, hidden_count); //update the weights of output nodes
            //cout << "hidden node update" << endl;
            update(hiddenNode_delta, hidden_weights, input, learning_rate, hidden_count, input_count); //update the weights of hidden nodes

            example_count--;
        }
        epoch_count--;
        inFile.close();
    } 
}

void propogate(vector<vector<double> > &weights, double *inputs, double *outputs, double *weightedInput, int nodeSize, int inputSize) //calculate the output vector given weight vector and input vector 
{
    for(int i = 0; i < nodeSize; i++) //iterate through each node
    {
        double sum = weights[i][0] * -1; //initialization of activation weight
        for(int j = 0; j < inputSize; j++) //calculated the weighted input to one node
            sum += weights[i][j+1] * inputs[j];

        weightedInput[i] = sum;
        outputs[i] = sigmoid(sum);
    }
}

double sigmoid(double x) //calculate the output of sigmoid function given input x
{
    double y = (double)1 / (1 + exp(-x));
    return y;
}

double sigmoidDeriv(double x) //calculate the derivative of sigmoid function given input x
{
    double y = sigmoid(x) * (1 - sigmoid(x));
    return y;
}

void update(double *delta, vector<vector<double> > &weight, double *input, double learning_rate, int nodeSize, int inputSize) //weight update function
{
    for(int i = 0; i < nodeSize; i++)
    {   
        weight[i][0] += learning_rate * -1 * delta[i];
        //cout << "after is " << weight[i][0] << endl;
        for(int j = 0; j < inputSize; j++)
        {
            //cout << "input[j] is " << input[j] << endl;
            weight[i][j+1] += learning_rate * input[j] * delta[i];
        }
    }
}