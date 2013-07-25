#include <string>
void SMF_sgd(int* user, int* item, double* rating, int N_users, int N_items, int N_ratings, int K, int MAX_ITER, double lambda, double lr, double t0, double* user_factor, double* item_factor);
void load_sparse_matrix(std::string csv_filename, int* user, int* item, double* rating);
