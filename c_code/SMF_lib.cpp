#include "SMF_lib.h"
#include <iostream>
//#include <Eigen/Dense>
//#include <Eigen/Sparse>

#include <fstream>
//#include <vector>
//#include <string>
//#include <strtk.hpp>
#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>

// Dario Garcia, 2013

// TODO
// - Implement non-negative matrix factorization, since ratings are always going to be non-negative
// - It makes sense to use the unconstrained version (SMF_sgd) if we remove the mean of the ratings

using namespace std;
using namespace Eigen;
using namespace boost;

namespace SMF {

void RatingMatrix::loadFromFile(string csv_filename) {
	ifstream in(csv_filename.c_str());
	string line;
	if (in.is_open()) {
		int u;
		int v;
		double r;
		char_separator<char> sep(";");
		tokenizer< char_separator<char> >::iterator it;
		this->clear();
		while(1) 
		{
			getline(in, line);
			if (!in.good()) {
				break;
			}
			// Parse line
			tokenizer< char_separator<char> > tokens(line, sep);	
			it = tokens.begin();
			u = atoi( (*(it++)).c_str());
			v = atoi( (*(it++)).c_str());
			r = atoi( (*(it)).c_str());
			this->insert(u,v,r);
		}

		in.close();
	}
	else cout << "Can't open file " + csv_filename;
	return;
};

  void SMF_sgd(const RatingMatrix &M, int K, int MAX_ITER, double lambda, double lr, double t0, Eigen::MatrixXd &user_factor, Eigen::MatrixXd &item_factor, bool removeMean = false)
{
	double THRESH = 1e-3;
	double BACKOFF_RATE = 0.75;

	// Initialization
	int iter = 0;
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
		u_old = MatrixXd(user_factor);
		v_old = MatrixXd(item_factor);

		// One pass of stochastic gradient descent over the whole dataset
		vector<triplet>::const_iterator it = M.data.begin();

		for (; it < M.data.end(); it++)
		{
			SMF::triplet tr = *it;
			double r = tr.value;
			long u = tr.i;
			long v = tr.j;
//			cout << u << '\t' << v << '\t' << r << endl;
			if (removeMean)
			  r-=M.mean();
			err = r - user_factor.row(u-1).dot(item_factor.row(v-1));
			cum_err += abs(err);
			user_factor.row(u-1) += -lambda*lr*user_factor.row(u-1)+lr*item_factor.row(v-1)*err;
			item_factor.row(v-1) += -lambda*lr*item_factor.row(v-1)+lr*user_factor.row(u-1)*err;
		}

		cum_err/=M.numRatings();
		cout << "Average error: " << cum_err << '\t' << "Learning rate: " << lr << endl;


		// Check conditions for ending the loop or backing off 

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
			//cout << fabs(cum_err-err_old);
			return;
		}

		// Bookkeeping
		err_old = cum_err;
		anneal_rate = sqrt(t0)/(double)sqrt(t0+iter);
		lr = lr*anneal_rate;
	}
}

} //end namespace SMF
