#include "SMF_lib.h"
#include <iostream>
#include <Eigen/Dense>
//#include <Eigen/Sparse>

#include <fstream>
#include <vector>
//#include <strtk.hpp>
#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>

using namespace std;
using namespace Eigen;
using namespace boost;

void load_sparse_matrix(string csv_filename, int* user, int* item, double* rating) {
	ifstream in(csv_filename.c_str());
	string line;
	if (in.is_open()) {
		vector<int> u;
		vector<int> v;
		vector<double> r;
		char_separator<char> sep(";");

		while(1) 
		{
			getline(in, line);
			if (!in.good()) {
				break;
			}
			// Parse line
<<<<<<< Updated upstream
			tokenizer< char_separator<char> > tokens(line, sep);	
			tokenizer< char_separator<char> >::iterator it = tokens.begin();
			cout << (string) *(it++) << '\t' << (string) *(it++) << '\t' << (string) *(it++) << endl;
=======
			

>>>>>>> Stashed changes
		}

		in.close();
	}
	else cout << "Can't open file " + csv_filename;
	return;
}

void SMF_sgd(int* user, int* item, double* rating, int N_users, int N_items, int N_ratings, int K, int MAX_ITER, double lambda, double lr, double t0, double* user_factor, double* item_factor)
{
	double THRESH = 1e-3;
	double BACKOFF_RATE = 0.75;

	// Initialization
	int iter = 0;
	MatrixXd u_f = MatrixXd::Random(N_users,K);
	MatrixXd v_f = MatrixXd::Random(N_ratings,K);
	MatrixXd u_old;
	MatrixXd v_old;
	double err_old = 1e10;
	double cum_err = 0;
	double err = err_old;
	double anneal_rate = 1;	

	// Main loop	
	while(1)
	{
		iter++;
		cout << "Iteration " << iter << '\t';
		cum_err = 0;
		// Store old previous values of the factors
		u_old = MatrixXd(u_f);
		v_old = MatrixXd(v_f);

		// Once pass of stochastic gradient descent over the whole dataset
		for (int n = 0; n < N_ratings; n++)
		{
			double r = rating[n];
			int u = user[n];
			int v = item[n];
			err = r - u_f.row(u).dot(v_f.row(v));
			cum_err += err;
			u_f.row(u) += -lambda*lr*u_f.row(u)+lr*v_f.row(v)*err;
			v_f.row(v) += -lambda*lr*v_f.row(v)+lr*u_f.row(u)*err;
		}
		cum_err/=N_ratings;
		cout << "Average error: " << cum_err << '\n';

		// Check conditions for ending the loop

		if (cum_err > err_old) {
			cout << "Backing off\t" << cum_err << '\t' << err_old << '\n';
			lr = lr/anneal_rate;
			lr = lr*anneal_rate*BACKOFF_RATE;
			continue;
		}

		if (iter >= MAX_ITER) {
			cout << "Maximum number of iterations reached\n";
			return;
		}
		if (fabs(cum_err-err_old)<THRESH) {
			cout << "Convergence\n";
			cout << fabs(cum_err-err_old);
			return;
		}

		// Bookkeeping
		err_old = cum_err;
		anneal_rate = sqrt(t0)/(double)sqrt(t0+iter);
		lr = lr*anneal_rate;
	}
}
