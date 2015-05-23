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

/** 
 * The RatingMatrix class is basically a very simple sparse matrix
 */
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
	double _sum;

// METHODS
public:
	// Constructors
	RatingMatrix() {
		_nRows = 0;
		_nCols = 0;
		_nnz = 0;
	}

	// Overloaded operators
	
	// Getters
	long numUsers() const { return _nRows; }
	long numItems() const { return _nCols; }
	long numRatings() const { return _nnz; }

	// Insert a new element at position (i,j) with value k
    // NOTE: Doesn't check if there already exists another
    // element with those indices
	void insert(long i, long j, long k) {
		data.push_back(triplet(i,j,k));
		_nnz++;
		if (! users.count(i) ) {
			users.insert(i);
			_nRows = max(_nRows, i);
		}
		if (! items.count(j) ) {
			items.insert(j);
			_nCols = max(_nCols, j);
		}
		_sum+=k;
	}

	// Clear the matrix data
	void clear() {
		data.clear();
		_nnz = 0;
		_nRows = 0;
		_nCols = 0;
		_sum = 0;
	}
	
	// Load matrix data from a CSV file
	void loadFromFile(string csv_filename);

	// Get average value
    // TODO: Cache the value, invalidate / recalculate
    // when the matrix is updated
	double mean() const { return _sum/_nnz; };

};

void SMF_sgd(const RatingMatrix &M, int K, int MAX_ITER, double lambda, 
        double lr, double t0, Eigen::MatrixXd &user_factor, 
        Eigen::MatrixXd &item_factor, bool removeMean);
//void load_sparse_matrix(std::string csv_filename, std::vector<int> &user, std::vector<int> &item, std::vector<double> &rating);
}
