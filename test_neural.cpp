//Yilun Jiang
//Prof. Sable
//AI project 2 Single hidden layer neural network 

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <iomanip>

using namespace std;

void test(vector<vector<double> > &hidden_weights, vector<vector<double> > &output_weights, vector<vector<int> > &accuracy, string test_file, int hidden_count);
void propogate(vector<vector<double> > &weights, double *inputs, double *outputs, int nodeSize, int inputSize); //calculate the output vector given weight vector and input vector 
double sigmoid(double x); //calculate the output of sigmoid function given input x

int main()
{
    string weight_file;
    cout << "Please enter name of the file that contains the weights of the trained neural network: \n";
    cin >> weight_file;

    ifstream inFile;
    inFile.open(weight_file);
    if(!inFile)
    {
		cerr << "Unable to open trained weight file.";
		exit(1);
	}

    int input_count, hidden_count, output_count; //read in number of input node, hidden nodes and output nodes
    inFile >> input_count >> hidden_count >> output_count;

    vector<vector<double> > hidden_weights(hidden_count);
    //read in trained weights of the hidden nodes
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
    //read in trained weights of the output nodes
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

    //Initialization of the matrix that records the a,b,c,d value of each output category
    vector<vector<int> > accuracy_count(output_count);
    for(int i = 0; i < output_count; i++)
    {
        for(int j = 0; j < 4; j++)
            accuracy_count[i].push_back(0);
    }
    
    string test_file, output_file;
    cout << "Please enter name of the file that contains the testing data: \n";
    cin >> test_file;
    cout << "Please enter name of the output file(contains different metrics for measuring accuracy): \n";
    cin >> output_file;

    test(hidden_weights, output_weights, accuracy_count, test_file, hidden_count);

    //Initialization of the matrix that records the different metrics value of each output category
    vector<vector<double> > accuracy(output_count);
    for(int i = 0; i < output_count; i++)
    {
        for(int j = 0; j < 4; j++)
            accuracy[i].push_back(0);
    }

    for(int i = 0; i < output_count; i++)
    {
        //first compute the overall accuracy
        //overall accuracy = (A + D) / (A + B + C + D)
        accuracy[i][0] = (double)(accuracy_count[i][0] + accuracy_count[i][3]) / (accuracy_count[i][0] + accuracy_count[i][1] + accuracy_count[i][2] + accuracy_count[i][3]);
        //compute the precision next
        //Precision = A / (A + B)
        accuracy[i][1] = (double)accuracy_count[i][0] / (accuracy_count[i][0] + accuracy_count[i][1]);
        //compute recall next
        //Recall = A / (A + C)
        accuracy[i][2] = (double)accuracy_count[i][0] / (accuracy_count[i][0] + accuracy_count[i][2]);
        //Finally compute F1
        //F1 = (2 * Precision * Recall) / (Precision + Recall)
        accuracy[i][3] = ((double)2 * accuracy[i][1] * accuracy[i][2]) / (accuracy[i][1] + accuracy[i][2]);
    }

    //write to the output file
    ofstream outFile;
    outFile.open(output_file);

    outFile.precision(3);
    //output each category's A,B,C,D and its 4 metrics
    for(int i = 0; i < output_count; i++)
    {
        for(int j = 0; j < 4; j++)
            outFile << fixed << accuracy_count[i][j] << ' ';
        for(int j = 0; j < 4; j++)
            outFile << fixed << accuracy[i][j] << ' ';

        outFile << '\n';
    }

    //Compute micro-averaging of the 4 metrics
    int total_count[4];
    for(int i = 0; i < 4; i++)
    {
        total_count[i] = 0;
        for(int j = 0; j < output_count; j++)
            total_count[i] += accuracy_count[j][i];
    }

    double micro_average[4];
    //compute micro-average overall accuracy
    //overall accuracy = (A + D) / (A + B + C + D)
    micro_average[0] = (double)(total_count[0] + total_count[3]) / (total_count[0] + total_count[1] + total_count[2] + total_count[3]);
    //compute micro-average precision
    //Precision = A / (A + B)
    micro_average[1] = (double)total_count[0] / (total_count[0] + total_count[1]);
    //compute micro-average recall
    //Recall = A / (A + C)
    micro_average[2] = (double)total_count[0] / (total_count[0] + total_count[2]);
    //compute micro-average F1
    //F1 = (2 * Precision * Recall) / (Precision + Recall)
    micro_average[3] = (double)(2 * micro_average[1] * micro_average[2]) /  (micro_average[1] + micro_average[2]);

    //Compute macro-average of the 4 metrics
    double macro_average[4];
    for(int i = 0; i < 3; i++)
    {
        macro_average[i] = 0;
        for(int j = 0; j < output_count; j++)
            macro_average[i] += accuracy[j][i];
        
        macro_average[i] /= output_count;
    }

    //compute macro-average F1
    macro_average[3] = (2 * macro_average[1] * macro_average[2]) /  (macro_average[1] + macro_average[2]);

    //write the computed result of micro-average and macro-average to the output file
    outFile << fixed << micro_average[0] << ' ' << fixed << micro_average[1] << ' ' << fixed << micro_average[2] << ' ' << fixed << micro_average[3] << endl;
    outFile << fixed << macro_average[0] << ' ' << fixed << macro_average[1] << ' ' << fixed << macro_average[2] << ' ' << fixed << macro_average[3] << endl;

    outFile.close();
}

void test(vector<vector<double> > &hidden_weights, vector<vector<double> > &output_weights, vector<vector<int> > &accuracy_count, string test_file, int hidden_count)
{
    ifstream inFile;
    inFile.open(test_file);
    if(!inFile)
    {
		cerr << "Unable to open test data file.";
		exit(1);
	}

    int example_count = 0, input_count = 0, output_count = 0;
    inFile >> example_count >> input_count >> output_count;

    while(example_count > 0)
    {
        double input[input_count]; //read in testing inputs
        for(int i = 0; i < input_count; i++)
        {
            inFile >> input[i];
        }    

        int output[output_count]; //read in testing outputs
        for(int i = 0; i < output_count; i++)
        {
            inFile >> output[i];
        }

        double hiddenNode_output[hidden_count]; //record the outputs from all hidden nodes

        for(int i = 0; i < hidden_count; i++)
        {
            hiddenNode_output[i] = 0;
        }    
        propogate(hidden_weights, input, hiddenNode_output, hidden_count, input_count); //propogate in the forward direction

        double outputNode_output[output_count]; //record the outputs in the output nodes
        for(int i = 0; i < output_count; i++) 
        {
            outputNode_output[i] = 0;
        }    
        propogate(output_weights, hiddenNode_output, outputNode_output, output_count, hidden_count); //propogate in the forward direction

        //if the output nodes give answer above 0.5, we classify it as 1 else we classify it as 0
        for(int i = 0; i < output_count; i++)
        {
            if(outputNode_output[i] >= 0.5)
                outputNode_output[i] = 1;
            else
                outputNode_output[i] = 0;
        }

        //comparing the predicted result with the actual result from data
        for(int i = 0; i < output_count; i++)
        {
            if(outputNode_output[i] == 1 && output[i] == 1) //situation A
                accuracy_count[i][0] += 1;
            else if(outputNode_output[i] == 1 && output[i] == 0) //situation B
                accuracy_count[i][1] += 1;
            else if(outputNode_output[i] == 0 && output[i] == 1) //situation C
                accuracy_count[i][2] += 1;
            else //situation D
                accuracy_count[i][3] += 1; 
        }

        example_count--;
    }
}

void propogate(vector<vector<double> > &weights, double *inputs, double *outputs, int nodeSize, int inputSize) //calculate the output vector given weight vector and input vector 
{
    for(int i = 0; i < nodeSize; i++) //iterate through each node
    {
        double sum = weights[i][0] * -1; //initialization of activation weight
        for(int j = 0; j < inputSize; j++) //calculated the weighted input to one node
            sum += weights[i][j+1] * inputs[j];

        outputs[i] = sigmoid(sum);
    }
}

double sigmoid(double x) //calculate the output of sigmoid function given input x
{
    double y = 1 / (1 + exp(-x));
    return y;
}
