#include "SMF_lib.h"
#include <Eigen/src/Core/IO.h>
#include <iostream>
#include <fstream>

int main(int argc, char* argv[])
{
	std::vector<int> user;
	std::vector<int> item;
	std::vector<double> rating;
	char OUT_USER[] = "user_factor";
	char OUT_ITEM[] = "item_factor";
	int K = 50;
	int MAX_ITER = 100;
	double lr = 0.02;
	double lambda = 0.1;
	double t0 = 500;
	bool removeMean = false;

	SMF::RatingMatrix M;

	M.loadFromFile("data_file.csv");	
	std::cout << "Data loaded\n";
	std::cout << "Number of users: " << M.numUsers() << std::endl;
	std::cout << "Number of items: " << M.numItems() << std::endl;
	std::cout << "Number of ratings: " << M.numRatings() << std::endl;
	std::cout << "Average rating: " << M.mean() << std::endl;
	
	Eigen::MatrixXd user_factor = Eigen::MatrixXd::Random(M.numUsers(), K);
	Eigen::MatrixXd item_factor = Eigen::MatrixXd::Random(M.numItems(), K);
	
//	std::cout << user[0] << '\t' << item[0] << '\t' << rating[0] << std::endl;
	SMF_sgd(M, K, MAX_ITER, lambda, lr, t0, user_factor, item_factor, removeMean);

// Save results
	Eigen::IOFormat test(Eigen::StreamPrecision, 0, ";", "\n", "", "","",""); 
	ofstream out(OUT_USER);
	out << user_factor.format(test);
	out.close();

	ofstream out_item(OUT_ITEM);
	out_item << item_factor.format(test);
	out_item.close();

}
