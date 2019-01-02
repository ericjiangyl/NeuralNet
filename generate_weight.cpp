#include <iostream>
#include <iomanip>

using namespace std;

int main()
{
    //generate the random weights for the hidden nodes
    hidden_count = 10; //number of hidden nodes count
    cout.precision(3);
    for(int i = 0; i < hidden_count; i++)
    {
        for(int j = 0; j < 12; j++) //12 is the number of input nodes
        {
            double r = ((double) rand() / (RAND_MAX));
            cout << fixed << r << ' ';
        }
        cout << '\n' ;
    }

    //generate the random weights for the output nodes
    for(int i = 0; i < hidden_count; i++)
    {
        double temp = ((double) rand() / (RAND_MAX));
        cout << fixed << temp << ' ';
    }
        
}