#include <string>
#include <vector>
#include <set>
#include <Eigen/Dense>

using namespace std;

namespace SMF {

struct triplet {
	long i;
	long j;
	double value;
	
	triplet(long _i, long _j, long _value) {
		i = _i;
		j = _j;
		value = _value;
	}
};

class RatingMatrix {
// MEMBERS
public:
	vector<triplet> data;

private:
	long _nRows;
	long _nCols;
	long _nnz;
	set<long> users;
	set<long> items;

// METHODS
public:
	// Constructors
	RatingMatrix() {
		_nRows = 0;
		_nCols = 0;
		_nnz = 0;
	}

	// Getters
	long numUsers() const { return _nRows; }
	long numItems() const { return _nCols; }
	long numRatings() const { return _nnz; }

	//
	void insert(long i, long j, long k) {
		data.push_back(triplet(i,j,k));
		_nnz++;
		if (! users.count(i) ) {
			users.insert(i);
			_nRows++;
		}
		if (! items.count(j) ) {
			items.insert(j);
			_nCols++;
		}
	}

	void clear() {
		data.clear();
		_nnz = 0;
		_nRows = 0;
		_nCols = 0;
	}
	void loadFromFile(string csv_filename);
};

void SMF_sgd(const RatingMatrix &M, int K, int MAX_ITER, double lambda, double lr, double t0, Eigen::MatrixXd &user_factor, Eigen::MatrixXd &item_factor);
//void load_sparse_matrix(std::string csv_filename, std::vector<int> &user, std::vector<int> &item, std::vector<double> &rating);
}