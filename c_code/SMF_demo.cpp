#include "SMF_lib.h"
#include <iostream>

int main(int argc, char** argv)
{
	int* user;
	int* item;
	double* rating;
	int N = 3;
	int K = 5;
	int MAX_ITER = 100;
	double lr = 0.01;
	double *user_factor = new double[3];
	double *item_factor = new double[3];
	load_sparse_matrix("data_file.csv", user, item, rating);	
	std::cout << "Data loaded\n";
	SMF_sgd(user, item, rating, 3, 3, N, K, MAX_ITER, 1.0, lr, 500.0, user_factor, item_factor);
}