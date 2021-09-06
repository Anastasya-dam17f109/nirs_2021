#pragma once
#pragma once
#include <iostream>
#include "omp.h"
#include <fstream>
#include <cmath>
#include <ctime>
#include <chrono>
#include <thread>
#include <boost/math/distributions/normal.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <boost/random.hpp>
#include <boost/math/distributions/rayleigh.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/students_t.hpp>

using namespace boost::math;
using  namespace std;
using boost::math::normal;

struct mix_elem {
	double value;
	int class_flag = 0;
};

struct target {
	int x;
	int y;
	int size;
	int brightness;
	int mix_type;

};

struct mesh_elem {
	vector<double> lefts;
	vector<double> lefts_SKO;
	vector<double> rights;
	vector<double> rights_SKO;
	int knot_elem;
	unsigned amount = 0;
	friend std::ostream& operator << (std::ostream &out, const mesh_elem &elem) {
		unsigned i;
		out << " knot number: " << elem.knot_elem << "\n";
		out << "lefts " << "\n";
		for (int i = 0; i < elem.lefts.size(); ++i)
			out << elem.lefts[i] << " ";
		out << "\n";
		out << "rights " << "\n";
		for (i = 0; i < elem.rights.size(); ++i)
			out << elem.rights[i] << " ";
		return out;
	}
};

class SEM_games
{
	double** mixture_image;
	double* mixture_image_one_mass;
	mesh_elem* dist_mesh;
	int **     class_flag;
	double *   re_mix_shift;
	double *   re_mix_scale;

	double * mix_shift;
	double * mix_scale;
	double * mix_weight;

	double **g_i_j;
	double *target_pixels;
	double **g_i_j_0;
	double accuracy = 0.1;

	double re_targ_shift = 200;
	unsigned image_len = 15;

	string mixture_type = "";
	unsigned class_amount = 1;
	unsigned hyp_cl_amount = 1;
	unsigned re_cl_amount = 1;

	double* x_coords;
	double* y_coords;
	target* targs;
	unsigned amount_trg = 1;
	unsigned min_targ_size = 17;
	unsigned backg_size = 35;

	int mix_params_amount = 1;
	double * nu_i;
	double * interval_bounds;
	double * nu_i_bounds;
	int intervals_amount = 30;
	double rfar;
	float all_mistakes = 0;
	float* mistake_mix;
	int* buf_numbs;
	string filename_gen_image = "D:\\generated_image.txt";
	string filename_split_image = "D:\\splitted_image.txt";
	std::ofstream out;
	std::ifstream in_;
	const double pi = boost::math::constants::pi<double>();
public:
	SEM_games(int img_size, string mix_t, int amount_targets, int classes, unsigned h_classes, double acc, bool file_flag, bool draw_flag);
	~SEM_games();

	void copy_in_one_mass(double* img, int x_c, int y_c, int x_l, int y_l);
	void create_splitted_img();
	void draw_graphics();
	void img_generator();
	void image_reader();
	void memory_allocation();
	void mixture_inicalization();
	void split_image();
	
	int chi_square_stats(double* data, int data_size);
	int kolmogorov_stats(double* data, int data_size, double* mix_shift, double* mix_scale, int  hyp_cl_amount);
	double find_k_stat(double * data, int wind_size, int k_stat);
	double find_med(double* window, int wind_size);
	std::pair<int, int> partition(double* mass, int left, int right, int  ind_pivot);
	int  mean(double**);
	
	void BIC();
	void detect_results();
	void th_detect_results(int beg, int end);
	void FAR_computation();
	void dist_computation();
	void statistics_creation(string filename);

	void SEMalgorithm();
	void SEMalgorithm_OMP();
	void SEMalgorithm_combinated();
	void SEMalgorithm_median();
	void SEMalgorithm_median_mean();
	void SEMalgorithm_median_mean_OMP();
	void SEMalgorithm_median_mean2_OMP();
	void SEMalgorithm_median_mean_raygh_OMP();
	void SEMalgorithm_raygh();
	void SEMalgorithm_raygh12();
};



