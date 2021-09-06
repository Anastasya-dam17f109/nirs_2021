#include "pch.h"
#include "SEM_games.h"
#include <typeinfo>
boost::random::mt19937 generator_S{ static_cast<std::uint32_t>(time(0)) };

SEM_games::SEM_games(int img_size, string mix_t, int amount_targets, int classes, unsigned h_classes, double acc, bool file_flag, bool draw_flag)
{
	//std::cout << "Hello World!\n";
	image_len = img_size * amount_targets;
	mixture_type = mix_t;
	amount_trg = amount_targets;
	class_amount = classes;
	hyp_cl_amount = h_classes;
	accuracy = acc;
	re_targ_shift = 200;
	memory_allocation();
	mixture_inicalization();
	
	if (file_flag)
		image_reader();
	else
		img_generator();
	cout << "mixture_type: " << mixture_type << endl;

	copy_in_one_mass(mixture_image_one_mass, 0, 0, image_len, image_len);
	if (mixture_type == "raygleigh_with_shift") {
		mix_params_amount = 2;
		SEMalgorithm_raygh12();
		SEMalgorithm_raygh12();
		SEMalgorithm_raygh12();
		SEMalgorithm_raygh12();
		SEMalgorithm_raygh12();
	}
	else {
		if (mixture_type == "rayleigh") {
			mix_params_amount = 1;
			//SEMalgorithm_raygh();
			SEMalgorithm_median_mean_raygh_OMP();
			dist_computation();
		}
		else
		{
			mix_params_amount = 2;
			for (int i = 0; i < 100; ++i) {
				SEMalgorithm_OMP();
				dist_computation();
				statistics_creation("D:\\SEM_statistics.txt");
				//BIC();
				//SEMalgorithm_median();
				//BIC();
				/*SEMalgorithm_median_mean_OMP();
				dist_computation();
				statistics_creation("D:\\SEM_mean_statistics.txt");*/

				SEMalgorithm_median_mean2_OMP();
				dist_computation();
				statistics_creation("D:\\SEM_mean2_statistics.txt");
			}
			//BIC();
		}
	}
	statistics_to_csv("D:\\SEM_statistics.txt","SEM");
	statistics_to_csv("D:\\SEM_mean2_statistics.txt", "SEM_mean2");
	//cout << "Chi_stat for estimated paramas: class_number " << chi_square_stats(mixture_image_one_mass, image_len*image_len) << endl;
	split_image();
	create_splitted_img();
	FAR_computation();
	//BIC();
	if(draw_flag)
		draw_graphics();
}

void SEM_games::memory_allocation() {
	unsigned i, j;
	nu_i = new double[intervals_amount];
	interval_bounds = new double[intervals_amount];
	nu_i_bounds = new double[intervals_amount];
	re_mix_shift = new double[class_amount + 1];
	re_mix_scale = new double[class_amount + 1];
	mistake_mix = new float[class_amount + 1];

	mix_shift = new double[hyp_cl_amount];
	mix_scale = new double[hyp_cl_amount];
	mix_weight = new double[hyp_cl_amount];

	buf_numbs = new int[hyp_cl_amount];
	////
	g_i_j = new double *[image_len*image_len];
	g_i_j_0 = new double *[image_len*image_len];
	for (i = 0; i < image_len*image_len; i++) {
		g_i_j[i] = new double[hyp_cl_amount];
		g_i_j_0[i] = new double[hyp_cl_amount];
		for (j = 0; j < hyp_cl_amount; j++)
			g_i_j_0[i][j] = 0.0;
	}
	targs = new target[amount_trg*amount_trg];
	x_coords = new double[amount_trg*amount_trg];
	y_coords = new double[amount_trg*amount_trg];
	mixture_image_one_mass = new double[image_len*image_len];
	mixture_image = new double *[image_len];    // массив указателей (2)
	class_flag = new int    *[image_len];
	for (int i = 0; i < image_len; i++) {
		mixture_image[i] = new double[image_len];     // инициализация указателей
		class_flag[i] = new int[image_len];
		for (int j = 0; j < image_len; j++)
			class_flag[i][j] = 0;
	}
}

//чтение изображения из файла

void SEM_games::image_reader() {
	in_.open(filename_gen_image);
	for (int i = 0; i < image_len; i++) {
		for (int j = 0; j < image_len; j++)
			in_ >> mixture_image[i][j];
	}
	in_.close();
}

//инициализация параметров смеси

void SEM_games::mixture_inicalization() {
	double last_shift = 0;
	double last_scale = 10;
	unsigned i, j;

	for (i = 0; i < hyp_cl_amount; i++) {
		if (i > 0) {
			last_shift = mix_shift[i - 1];
			last_scale = 1;
		}
		mix_shift[i] = pow(0.5, i + 1)*250.0 + last_shift;
		if (hyp_cl_amount == 2)
			mix_shift[1] = 200;
		mix_scale[i] = (2.0*(i + 1))*last_scale;

		//mix_weight[i] = 0.0
		//////
		mix_weight[i] = 1.0 / double(hyp_cl_amount);
	}

	for (i = 0; i < class_amount + 1; ++i) {
		re_mix_shift[i] = 0.0;
		re_mix_scale[i] = 0.0;
		mistake_mix[i] = 0.0;
	}

	// для двукомпонентного случая
	if (mixture_type == "normal") {
		re_mix_shift[0] = 128.0;
		re_mix_scale[0] = 37.0;
		re_mix_shift[1] = re_targ_shift;
		re_mix_scale[1] = 30.0;
	}
	else {
		if (mixture_type == "rayleigh") {
			re_mix_shift[0] = 0;
			re_mix_scale[0] = 20;
			re_mix_shift[1] = 0;
			re_targ_shift = 40.0;
			re_mix_scale[1] = 40.0;
		}
	}
	targs[0].brightness = re_targ_shift;
	targs[0].size = 25;
	targs[0].x = backg_size / 2 - 1 - targs[0].size / 2;
	targs[0].y = backg_size / 2 - 1 - targs[0].size / 2;
	targs[0].mix_type = 2;
	re_cl_amount = 2;
	
}

//генерация изображения в зависимости от типа изображения

void SEM_games::img_generator() {
	boost::random::normal_distribution <> dist_norm_bcg{ 128, 37.0 };
	boost::random::normal_distribution <> dist_norm_trg{ 200, 1.5 };
	boost::random::uniform_01 <> dist_rel;
	int x = 0;
	int y = 0;
	int t_coord_x = 0;
	int t_coord_y = 0;
	int mix_number = 1;

	int    * targ_size = new int[amount_trg];
	double * targ_bright = new double[amount_trg];
	auto dist_gen_bcg = [&]() {
		if (mixture_type == "normal")
			return dist_norm_bcg(generator_S);
		else {
			if (mixture_type == "raygleigh_with_shift")
				return 30 + sqrt(-2 * pow(30, 2.0) *log(1 - dist_rel(generator_S)));
			else {
				if (mixture_type == "rayleigh")
					return sqrt(-2 * pow(20, 2.0) *log(1 - dist_rel(generator_S)));
			}
		}

	};

	auto dist_gen_trg = [&](int i, int j) {
		/*if (mixture_type == "normal") {

			if (class_amount > 1)
				dist_norm_trg.param(boost::random::normal_distribution <>::param_type(150, 1.5*mix_number));
			return dist_norm_trg(generator);
		}
		else
		{*/

		if (mixture_type == "normal") {
			dist_norm_trg.param(boost::random::normal_distribution <>::param_type(re_targ_shift, 30));
			return dist_norm_trg(generator_S);
		}
		else {
			if (mixture_type == "raygleigh_with_shift")
				return 60 + sqrt(-2 * pow(40, 2.0) *log(1 - dist_rel(generator_S)));
			else {
				if (mixture_type == "rayleigh")
					return sqrt(-2 * pow(re_targ_shift, 2.0) *log(1 - dist_rel(generator_S)));
			}
		}
	};


	for (int i = 0; i < image_len; i++) {
		for (int j = 0; j < image_len; j++)
			mixture_image[i][j] = dist_gen_bcg();
	}

	double sred = mean(mixture_image);
	int bright_step = (255 - sred - 40) / class_amount;
	int amount_brigh_trg = amount_trg / class_amount;
	if (amount_brigh_trg == 0)
		amount_brigh_trg = 1;
	//
	min_targ_size = 25;
	//
	target_pixels = new double[min_targ_size*min_targ_size];
	for (int i = 0; i < amount_trg; i++) {
		targ_size[i] = min_targ_size + i * 2;
		targ_bright[i] = sred + 40 + (int(i / amount_brigh_trg) + 1) * bright_step;
	}
	targ_bright[0] = 200;
	re_mix_shift[0] = 128.0;
	re_mix_scale[0] = 37.0;
	re_mix_shift[1] = targ_bright[0];
	re_mix_scale[1] = 30.0;
	int itr = 0;
	for (int i = 0; i < amount_trg; i++) {
		if (i > 0 && targ_bright[i] != targ_bright[i - 1]) {
			mix_number++;
			cout << mix_number << endl;
			re_mix_shift[mix_number] = targ_bright[i];
			re_mix_scale[mix_number] = 1.5*mix_number;
		}
		for (int j = 0; j < amount_trg; j++) {
			x = i * backg_size;
			y = j * backg_size;
			t_coord_x = x + backg_size / 2 - 1 - targ_size[j] / 2;
			t_coord_y = y + backg_size / 2 - 1 - targ_size[j] / 2;
			/*x_coords[i*amount_trg + j] = t_coord_x;
			y_coords[i*amount_trg + j] = t_coord_y;*/

			targs[i*amount_trg + j].brightness = targ_bright[i];
			targs[i*amount_trg + j].size = targ_size[j];
			targs[i*amount_trg + j].x = t_coord_x;
			targs[i*amount_trg + j].y = t_coord_y;
			//dist_unif.param(boost::random::uniform_int_distribution<>::param_type(targ_bright[i] - targ_size[j] / 3, targ_bright[i] + targ_size[j] / 3));
			for (int k = 0; k < targ_size[j]; k++) {
				for (int l = 0; l < targ_size[j]; l++) {
					mixture_image[t_coord_x + k][t_coord_y + l] = dist_gen_trg(i, j);
					cout << "targ_bright[i]" << mixture_image[t_coord_x + k][t_coord_y + l] << endl;
					target_pixels[itr] = mixture_image[t_coord_x + k][t_coord_y + l];
					itr++;
				}
			}
		}
	}

	for (int i = 0; i < amount_trg*amount_trg; i++) {
		for (int j = 1; j < class_amount + 1; j++) {
			if (targs[i].brightness == re_mix_shift[j])
				targs[i].mix_type = j + 1;
		}

	}
	out.open(filename_gen_image);
	for (int i = 0; i < image_len; i++) {
		for (int j = 0; j < image_len; j++)
			out << mixture_image[i][j] << " ";
		out << std::endl;
	}
	out.close();

	cout << "re_mix_shift values:" << endl;
	for (int i = 0; i < class_amount + 1; i++)
		cout << re_mix_shift[i] << "  ";
	cout << endl;
	cout << "generated mix_shift values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_shift[i] << "  ";
	cout << endl;
	cout << "re_mix_scale values:" << endl;
	for (int i = 0; i < class_amount + 1; i++)
		cout << re_mix_scale[i] << "  ";
	cout << endl;
	cout << "generated mix_scale values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_scale[i] << "  ";
	cout << endl;
	delete[] targ_size;
	delete[] targ_bright;

}

//обычный алгоритм SEM

void SEM_games::SEMalgorithm() {
	//вначале итеративная реализация
	cout <<  endl;
	cout << endl;
	cout << "classical SEM  " << endl;
	boost::random::uniform_01 <> dist_poly;

	int idx_i = 0;
	int idx_j = 0;
	int thr_nmb = 5;
	double summ = 0;
	double cur_max = 0;
	int itr = 0;
	bool stop_flag = true;
	double n = double(image_len * image_len);
	//int k = class_amount + 1;
	//int k = hyp_cl_amount;
	auto begin = std::chrono::steady_clock::now();
	double** buff = new double*[thr_nmb];
	double** buff1 = new double*[thr_nmb];
	double * max = new double[thr_nmb];
	int** y_i_j = new int *[image_len*image_len];
	double** val_i_j = new double *[image_len];

	for (int i = 0; i < image_len; ++i)
		val_i_j[i] = new double[image_len];
	for (int i = 0; i < image_len*image_len; i++) {
		y_i_j[i] = new int[hyp_cl_amount];
		for (int j = 0; j < hyp_cl_amount; j++) {
			y_i_j[i][j] = 0.0;
			g_i_j[i][j] = 1.0 / double(hyp_cl_amount);
		}
	}

	for (int i = 0; i < thr_nmb; i++) {
		buff[i] = new double[hyp_cl_amount];
		buff1[i] = new double[hyp_cl_amount];
		max[i] = 0.0;
		for (int l = 0; l < hyp_cl_amount; l++) {
			buff[i][l] = 0;
			buff1[i][l] = 0;
		}
	}

	auto mix_weight_computation = [&](int beg, int end, int nmb) {
		int idx_i = 0;
		int idx_j = 0;
		unsigned i, t, j;
		double summ1 = 0;
		double val, bound_d, bound_u;

		for (i = beg; i < end; i++) {
			summ1 = 0;
			idx_i = unsigned(i / image_len);
			idx_j = i - idx_i * image_len;
			summ1 = 0;
			idx_i = unsigned(i / image_len);
			idx_j = i - idx_i * image_len;
			bound_d = 0;
			bound_u = g_i_j[i][0];
			for (t = 0; t < hyp_cl_amount; ++t) {
				if (val_i_j[idx_i][idx_j] < bound_u && val_i_j[idx_i][idx_j] >= bound_d) {
					y_i_j[i][t] = 1;
					buff[nmb][t] += 1;
					if (mixture_type == "normal")
						buff1[nmb][t] += mixture_image[idx_i][idx_j];
					else {
						if (mixture_type == "log_normal")
							buff1[nmb][t] += log(mixture_image[idx_i][idx_j]);
					}

				}
				else
					y_i_j[i][t] = 0;
				bound_d += g_i_j[i][t];
				if (t == hyp_cl_amount - 2) {
					//cout << bound_u << endl;
					bound_u = 1;
				}
				else
					if (t == hyp_cl_amount - 1)
						bound_u += 0;
					else
						bound_u += g_i_j[i][t + 1];
			}
		}

	};

	auto g_i_j_recomputation = [&](int beg, int end) {
		int idx_i = 0;
		int idx_j = 0;
		unsigned i, t, j;
		double summ1 = 0;
		for (i = beg; i < end; ++i) {
			summ1 = 0;
			idx_i = unsigned(i / image_len);
			idx_j = i - idx_i * image_len;

			for (t = 0; t < hyp_cl_amount; t++)
				if (mix_scale[t] != 0 && mix_weight[t] != 0)
					if (mixture_type == "normal")
						summ1 += mix_weight[t] * (1 / (mix_scale[t] * sqrt(2 * pi)))*exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[t], 2)) /
						(2.0 * mix_scale[t] * mix_scale[t]));
					else {
						if (mixture_type == "log_normal")
							summ1 += mix_weight[t] * (1 / (mix_scale[t] * mixture_image[idx_i][idx_j] * sqrt(2 * pi)))*exp(-(pow(log(mixture_image[idx_i][idx_j]) - mix_shift[t], 2)) /
							(2.0 * mix_scale[t] * mix_scale[t]));
					}

			for (j = 0; j < hyp_cl_amount; j++) {
				if (mix_scale[j] != 0 && mix_weight[j] != 0)
					if (mixture_type == "normal")
						g_i_j[i][j] = mix_weight[j] * (1 / (mix_scale[j] * sqrt(2 * pi)*summ1))*exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[j], 2))
							/ (2.0 * mix_scale[j] * mix_scale[j]));
					else {
						if (mixture_type == "log_normal")
							g_i_j[i][j] = mix_weight[j] * (1 / (mix_scale[j] * mixture_image[idx_i][idx_j] * sqrt(2 * pi)*summ1))*exp(-(pow(log(mixture_image[idx_i][idx_j]) - mix_shift[j], 2))
								/ (2.0 * mix_scale[j] * mix_scale[j]));
					}
			}
		}
	};

	auto mix_scale_computation = [&](int beg, int end, int nmb) {
		int idx_i = 0;
		int idx_j = 0;
		unsigned i, j, l;

		for (l = 0; l < hyp_cl_amount; l++) {
			for (i = beg; i < end; i++) {
				idx_i = int(i / image_len);
				idx_j = i - idx_i * image_len;
				if (mixture_type == "normal")
					buff[nmb][l] += y_i_j[i][l] * pow(mixture_image[idx_i][idx_j] - mix_shift[l], 2);
				else {
					if (mixture_type == "log_normal")
						buff[nmb][l] += y_i_j[i][l] * pow(log(mixture_image[idx_i][idx_j]) - mix_shift[l], 2);
				}
				if (max[nmb] < abs(g_i_j[i][l] - g_i_j_0[i][l]))
					max[nmb] = abs(g_i_j[i][l] - g_i_j_0[i][l]);
				g_i_j_0[i][l] = g_i_j[i][l];

			}
		}
	};

	auto mix_split = [&](int beg, int end) {
		for (int i = beg; i < end; i++) {
			int idx_i = int(i / image_len);
			int idx_j = i - idx_i * image_len;
			double buf_max = g_i_j[i][0];
			int idx_max = 0;
			for (int l = 0; l < hyp_cl_amount; l++) {
				if (buf_max < g_i_j[i][l]) {
					buf_max = g_i_j[i][l];
					idx_max = l;
				}
			}
			class_flag[idx_i][idx_j] = idx_max + 1;
		}
	};

	while (stop_flag && itr < 400) {
		itr++;

		for (int i = 0; i < image_len; ++i)
			for (int j = 0; j < image_len; ++j)
				val_i_j[i][j] = dist_poly(generator_S);

		std::thread threadObj1(mix_weight_computation, 0, n / thr_nmb, 0);
		std::thread threadObj2(mix_weight_computation, n / thr_nmb, 2 * n / thr_nmb, 1);
		std::thread threadObj3(mix_weight_computation, 2 * n / thr_nmb, 3 * n / thr_nmb, 2);
		std::thread threadObj4(mix_weight_computation, 3 * n / thr_nmb, 4 * n / thr_nmb, 3);
		std::thread threadObj5(mix_weight_computation, 4 * n / thr_nmb, n, 4);
		threadObj1.join();
		threadObj2.join();
		threadObj3.join();
		threadObj4.join();

		threadObj5.join();
		//cout << "itr:" << itr << endl;
		for (int l = 0; l < hyp_cl_amount; l++) {
			mix_weight[l] = 0.0;
			mix_shift[l] = 0.0;
			for (int m = 0; m < thr_nmb; m++) {
				mix_weight[l] += buff[m][l];

				buff[m][l] = 0.0;
				mix_shift[l] += buff1[m][l];
				buff1[m][l] = 0.0;
			}

			mix_shift[l] = mix_shift[l] / mix_weight[l];
			//cout << mix_shift[l] << endl;
		}

		std::thread threadObj11(mix_scale_computation, 0, n / thr_nmb, 0);
		std::thread threadObj12(mix_scale_computation, n / thr_nmb, 2 * n / thr_nmb, 1);
		std::thread threadObj13(mix_scale_computation, 2 * n / thr_nmb, 3 * n / thr_nmb, 2);
		std::thread threadObj14(mix_scale_computation, 3 * n / thr_nmb, 4 * n / thr_nmb, 3);
		std::thread threadObj15(mix_scale_computation, 4 * n / thr_nmb, n, 4);
		threadObj11.join();
		threadObj12.join();
		threadObj13.join();
		threadObj14.join();
		threadObj15.join();

		for (int l = 0; l < hyp_cl_amount; l++) {
			mix_scale[l] = 0.0;
			for (int m = 0; m < thr_nmb; m++) {
				mix_scale[l] += buff[m][l];
				buff[m][l] = 0.0;
				if (cur_max < max[m])
					cur_max = max[m];
				max[m] = 0;
			}
			mix_scale[l] = sqrt(mix_scale[l] / mix_weight[l]);
			//cout << mix_weight[l] << " ";
			mix_weight[l] = mix_weight[l] / n;
			//cout << mix_scale[l] << " " << mix_weight[l] << endl;

		}
		std::thread threadObj16(g_i_j_recomputation, 0, n / thr_nmb);
		std::thread threadObj17(g_i_j_recomputation, n / thr_nmb, 2 * n / thr_nmb);
		std::thread threadObj18(g_i_j_recomputation, 2 * n / thr_nmb, 3 * n / thr_nmb);
		std::thread threadObj19(g_i_j_recomputation, 3 * n / thr_nmb, 4 * n / thr_nmb);
		std::thread threadObj199(g_i_j_recomputation, 4 * n / thr_nmb, n);
		threadObj16.join();
		threadObj17.join();
		threadObj18.join();
		threadObj19.join();
		threadObj199.join();

		//cout << "max: " << cur_max << endl;

		if (cur_max < accuracy)
			stop_flag = false;
		else
			cur_max = 0;
	}
	
	auto end = std::chrono::steady_clock::now();
	auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	cout << "elapsed_ms  " << elapsed_ms.count() << endl;
	
	cout << "re_mix_shift values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << re_mix_shift[i] << "  ";
	cout << endl;
	cout << "EM mix_shift values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_shift[i] << "  ";
	cout << endl;
	cout << "re_mix_scale values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << re_mix_scale[i] << "  ";
	cout << endl;
	cout << "EM mix_scale values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_scale[i] << "  ";
	cout << endl;

	cout << "EM mix_weight values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_weight[i] << "  ";
	cout << endl;
	/*cout << "EM max deviation: " << cur_max << " , iterations: " << itr << endl;

	cout << endl;*/

	/*std::thread threadObj1(mix_split, 0, n / thr_nmb);
	std::thread threadObj2(mix_split, n / thr_nmb, 2 * n / thr_nmb);
	std::thread threadObj3(mix_split, 2 * n / thr_nmb, 3 * n / thr_nmb);
	std::thread threadObj4(mix_split, 3 * n / thr_nmb, 4 * n / thr_nmb);
	std::thread threadObj5(mix_split, 4 * n / thr_nmb, n);
	threadObj1.join();
	threadObj2.join();
	threadObj3.join();
	threadObj4.join();
	threadObj5.join();*/

	for (int i = 0; i < thr_nmb; i++) {
		delete[] buff[i];
		delete[] buff1[i];

	}
	for (int i = 0; i < image_len*image_len; i++)
		delete[]  y_i_j[i];
	for (int i = 0; i < image_len; i++)
		delete[]  val_i_j[i];
	delete[] y_i_j;
	delete[] val_i_j;
	delete[] buff;
	delete[] buff1;
	delete[] max;

}

//обычный алгоритм SEM - omp version

void SEM_games::SEMalgorithm_OMP() {
	//вначале итеративная реализация
	cout << endl;
	cout << endl;
	cout << "classical SEM with OMP " << endl;
	boost::random::uniform_01 <> dist_poly;

	int idx_i = 0;
	int idx_j = 0;
	int thr_nmb = 5;
	double summ = 0;
	double cur_max = 0;
	int itr = 0;
	bool stop_flag = true;
	double n = double(image_len * image_len);
	//int k = class_amount + 1;
	//int k = hyp_cl_amount;
	auto begin = std::chrono::steady_clock::now();
	double** buff = new double*[thr_nmb];
	double** buff1 = new double*[thr_nmb];
	double * max = new double[thr_nmb];
	int** y_i_j = new int *[image_len*image_len];
	double** val_i_j = new double *[image_len];

	for (int i = 0; i < image_len; ++i)
		val_i_j[i] = new double[image_len];
	for (int i = 0; i < image_len*image_len; i++) {
		y_i_j[i] = new int[hyp_cl_amount];
		for (int j = 0; j < hyp_cl_amount; j++) {
			y_i_j[i][j] = 0.0;
			g_i_j[i][j] = 1.0 / double(hyp_cl_amount);
		}
	}

	for (int i = 0; i < thr_nmb; i++) {
		buff[i] = new double[hyp_cl_amount];
		buff1[i] = new double[hyp_cl_amount];
		max[i] = 0.0;
		for (int l = 0; l < hyp_cl_amount; l++) {
			buff[i][l] = 0;
			buff1[i][l] = 0;
		}
	}

	

	while (stop_flag && itr < 400) {
		itr++;
		cur_max = 0;
		for (int i = 0; i < image_len; ++i)
			for (int j = 0; j < image_len; ++j)
				val_i_j[i][j] = dist_poly(generator_S);
		for (int l = 0; l < hyp_cl_amount; l++) {
			mix_weight[l] = 0.0;
			mix_shift[l] = 0.0;
		}

		#pragma omp parallel 
		{
			int idx_i = 0;
			int idx_j = 0;
			double summ1 = 0;
			double val, bound_d, bound_u;
			#pragma omp for
			for (int i = 0; i < image_len*image_len; i++) {
				idx_i = unsigned(i / image_len);
				idx_j = i - idx_i * image_len;
				summ1 = 0;
				bound_d = 0;
				bound_u = g_i_j[i][0];
				for (int t = 0; t < hyp_cl_amount; ++t) {
					if (val_i_j[idx_i][idx_j] < bound_u && val_i_j[idx_i][idx_j] >= bound_d) {
						y_i_j[i][t] = 1;
						#pragma omp critical 
						{
							mix_weight[t] += 1;
							if (mixture_type == "normal")
								mix_shift[t] += mixture_image[idx_i][idx_j];
							else {
								if (mixture_type == "log_normal")
									mix_shift[t] += log(mixture_image[idx_i][idx_j]);
							}
						}
					}
					else
						y_i_j[i][t] = 0;
					#pragma omp critical
					{
						bound_d += g_i_j[i][t];
						if (t == hyp_cl_amount - 2)
							bound_u = 1;
						else
							if (t == hyp_cl_amount - 1)
								bound_u += 0;
							else
								bound_u += g_i_j[i][t + 1];
					}
				}
			}
		}

		for (int l = 0; l < hyp_cl_amount; l++) {
			mix_shift[l] = mix_shift[l] / mix_weight[l];
			mix_scale[l] = 0;
		}

		#pragma omp parallel
		{
			int idx_i1, idx_j1, idx_med;
			#pragma omp for
			for (int l = 0; l < hyp_cl_amount; l++) {
				idx_med = 0;
				for (int m = 0; m < image_len*image_len; m++) {
					idx_i1 = m / image_len;
					idx_j1 = m % image_len;

					#pragma omp critical
					{
						if (mixture_type == "normal"){
							mix_scale[l] += y_i_j[m][l] * pow(mixture_image[idx_i1][idx_j1] - mix_shift[l], 2);
						
						}
						
					}
				}
			}
		}

		for (int l = 0; l < hyp_cl_amount; l++) {
			mix_scale[l] = sqrt(mix_scale[l] / mix_weight[l]);
			mix_weight[l] = mix_weight[l] / n;
		}
		#pragma omp parallel
		{
			int idx_i = 0;
			int idx_j = 0;
			//int i, t, j;
			double summ1 = 0;

			#pragma omp for
			for (int i = 0; i < image_len*image_len; i++) {
				summ1 = 0;
				idx_i = unsigned(i / image_len);
				idx_j = i - idx_i * image_len;

				for (int t = 0; t < hyp_cl_amount; t++) {
					#pragma omp critical
					{
						if (mix_scale[t] != 0 && mix_weight[t] != 0)
							if (mixture_type == "normal")
								summ1 += mix_weight[t] * (1 / (mix_scale[t] * sqrt(2 * pi)))*exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[t], 2)) /
								(2.0 * mix_scale[t] * mix_scale[t]));
							else {
								if (mixture_type == "log_normal")
									summ1 += mix_weight[t] * (1 / (mix_scale[t] * mixture_image[idx_i][idx_j] * sqrt(2 * pi)))*exp(-(pow(log(mixture_image[idx_i][idx_j]) - mix_shift[t], 2)) /
									(2.0 * mix_scale[t] * mix_scale[t]));
							}
					}
				}


				for (int j = 0; j < hyp_cl_amount; j++) {
					if (mix_scale[j] != 0 && mix_weight[j] != 0)
						if (mixture_type == "normal")
							g_i_j[i][j] = mix_weight[j] * (1 / (mix_scale[j] * sqrt(2 * pi)*summ1))*exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[j], 2))
								/ (2.0 * mix_scale[j] * mix_scale[j]));
						else {
							if (mixture_type == "log_normal")
								g_i_j[i][j] = mix_weight[j] * (1 / (mix_scale[j] * mixture_image[idx_i][idx_j] * sqrt(2 * pi)*summ1))*exp(-(pow(log(mixture_image[idx_i][idx_j]) - mix_shift[j], 2))
									/ (2.0 * mix_scale[j] * mix_scale[j]));
						}
					/*#pragma omp critical (cout)
					cout << "  idx_i " << g_i_j[i][j] << "  " << idx_j << endl;*/
				}
			}
		}

		#pragma omp parallel
		{
			int idx_i1, idx_j1, idx_med;
			#pragma omp for
			for (int l = 0; l < hyp_cl_amount; l++) {

				for (int m = 0; m < image_len*image_len; m++) {

					#pragma omp critical
					{

						if (cur_max < abs(g_i_j[m][l] - g_i_j_0[m][l]))
							cur_max = abs(g_i_j[m][l] - g_i_j_0[m][l]);
						g_i_j_0[m][l] = g_i_j[m][l];
					}
				}
			}
		}
		//cout << "max: " << cur_max << endl;

		if (cur_max < accuracy)
			stop_flag = false;
		else
			cur_max = 0;
	}

	auto end = std::chrono::steady_clock::now();
	auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	cout << "elapsed_ms  " << elapsed_ms.count() << endl;

	cout << "re_mix_shift values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << re_mix_shift[i] << "  ";
	cout << endl;
	cout << "EM mix_shift values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_shift[i] << "  ";
	cout << endl;
	cout << "re_mix_scale values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << re_mix_scale[i] << "  ";
	cout << endl;
	cout << "EM mix_scale values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_scale[i] << "  ";
	cout << endl;

	cout << "EM mix_weight values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_weight[i] << "  ";
	cout << endl;
	
	for (int i = 0; i < thr_nmb; i++) {
		delete[] buff[i];
		delete[] buff1[i];

	}
	for (int i = 0; i < image_len*image_len; i++)
		delete[]  y_i_j[i];
	for (int i = 0; i < image_len; i++)
		delete[]  val_i_j[i];
	delete[] y_i_j;
	delete[] val_i_j;
	delete[] buff;
	delete[] buff1;
	delete[] max;

}

//реализация SEM для нормальнного распределения с оценкой среднего
// через медиану а СКО  - через выборочную дсперсию

void SEM_games::SEMalgorithm_combinated() {
	cout << " SEM  with combine estimation" << endl;
	boost::random::uniform_01 <> dist_poly;
	int idx_med = 0;
	int idx_i = 0;
	int idx_j = 0;
	int thr_nmb = 5;
	double summ = 0;
	double cur_max = 0;
	int itr = 0;
	bool stop_flag = true;
	double n = double(image_len * image_len);
	
	double** buf_median = new double*[hyp_cl_amount];
	for (int i = 0; i < hyp_cl_amount; ++i)
		buf_median[i] = new double[image_len*image_len];
	auto begin = std::chrono::steady_clock::now();
	double** buff = new double*[thr_nmb];
	double** buff1 = new double*[thr_nmb];
	double * max = new double[thr_nmb];
	int** y_i_j = new int *[image_len*image_len];
	double** val_i_j = new double *[image_len];

	for (int i = 0; i < image_len; ++i)
		val_i_j[i] = new double[image_len];
	for (int i = 0; i < image_len*image_len; i++) {
		y_i_j[i] = new int[hyp_cl_amount];
		for (int j = 0; j < hyp_cl_amount; j++) {
			y_i_j[i][j] = 0.0;
			g_i_j[i][j] = 1.0 / double(hyp_cl_amount);
		}
	}

	for (int i = 0; i < thr_nmb; i++) {
		buff[i] = new double[hyp_cl_amount];
		buff1[i] = new double[hyp_cl_amount];
		max[i] = 0.0;
		for (int l = 0; l < hyp_cl_amount; l++) {
			buff[i][l] = 0;
			buff1[i][l] = 0;
		}
	}

	auto mix_weight_computation = [&](int beg, int end, int nmb) {
		int idx_i = 0;
		int idx_j = 0;
		unsigned i, t, j;
		double summ1 = 0;
		double val, bound_d, bound_u;

		for (i = beg; i < end; i++) {
			summ1 = 0;
			idx_i = unsigned(i / image_len);
			idx_j = i - idx_i * image_len;
			summ1 = 0;
			idx_i = unsigned(i / image_len);
			idx_j = i - idx_i * image_len;
			bound_d = 0;
			bound_u = g_i_j[i][0];
			for (t = 0; t < hyp_cl_amount; ++t) {
				if (val_i_j[idx_i][idx_j] < bound_u && val_i_j[idx_i][idx_j] >= bound_d) {
					y_i_j[i][t] = 1;
					buff[nmb][t] += 1;
					if (mixture_type == "normal")
						buff1[nmb][t] += mixture_image[idx_i][idx_j];
					else {
						if (mixture_type == "log_normal")
							buff1[nmb][t] += log(mixture_image[idx_i][idx_j]);
					}

				}
				else
					y_i_j[i][t] = 0;
				bound_d += g_i_j[i][t];
				if (t == hyp_cl_amount - 2) {
					//cout << bound_u << endl;
					bound_u = 1;
				}
				else
					if (t == hyp_cl_amount - 1)
						bound_u += 0;
					else
						bound_u += g_i_j[i][t + 1];
			}
		}

	};

	auto g_i_j_recomputation = [&](int beg, int end) {
		int idx_i = 0;
		int idx_j = 0;
		unsigned i, t, j;
		double summ1 = 0;
		for (i = beg; i < end; ++i) {
			summ1 = 0;
			idx_i = unsigned(i / image_len);
			idx_j = i - idx_i * image_len;

			for (t = 0; t < hyp_cl_amount; t++)
				if (mix_scale[t] != 0 && mix_weight[t] != 0)
					if (mixture_type == "normal")
						summ1 += mix_weight[t] * (1 / (mix_scale[t] * sqrt(2 * pi)))*exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[t], 2)) /
						(2.0 * mix_scale[t] * mix_scale[t]));
					else {
						if (mixture_type == "log_normal")
							summ1 += mix_weight[t] * (1 / (mix_scale[t] * mixture_image[idx_i][idx_j] * sqrt(2 * pi)))*exp(-(pow(log(mixture_image[idx_i][idx_j]) - mix_shift[t], 2)) /
							(2.0 * mix_scale[t] * mix_scale[t]));
					}

			for (j = 0; j < hyp_cl_amount; j++) {
				if (mix_scale[j] != 0 && mix_weight[j] != 0)
					if (mixture_type == "normal")
						g_i_j[i][j] = mix_weight[j] * (1 / (mix_scale[j] * sqrt(2 * pi)*summ1))*exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[j], 2))
							/ (2.0 * mix_scale[j] * mix_scale[j]));
					else {
						if (mixture_type == "log_normal")
							g_i_j[i][j] = mix_weight[j] * (1 / (mix_scale[j] * mixture_image[idx_i][idx_j] * sqrt(2 * pi)*summ1))*exp(-(pow(log(mixture_image[idx_i][idx_j]) - mix_shift[j], 2))
								/ (2.0 * mix_scale[j] * mix_scale[j]));
					}
			}
		}
	};

	auto mix_scale_computation = [&](int beg, int end, int nmb) {
		int idx_i = 0;
		int idx_j = 0;
		unsigned i, j, l;

		for (l = 0; l < hyp_cl_amount; l++) {
			for (i = beg; i < end; i++) {
				idx_i = int(i / image_len);
				idx_j = i - idx_i * image_len;
				if (mixture_type == "normal")
					buff[nmb][l] += y_i_j[i][l] * pow(mixture_image[idx_i][idx_j] - mix_shift[l], 2);
				else {
					if (mixture_type == "log_normal")
						buff[nmb][l] += y_i_j[i][l] * pow(log(mixture_image[idx_i][idx_j]) - mix_shift[l], 2);
				}
				if (max[nmb] < abs(g_i_j[i][l] - g_i_j_0[i][l]))
					max[nmb] = abs(g_i_j[i][l] - g_i_j_0[i][l]);
				g_i_j_0[i][l] = g_i_j[i][l];

			}
		}
	};

	auto mix_split = [&](int beg, int end) {
		for (int i = beg; i < end; i++) {
			int idx_i = int(i / image_len);
			int idx_j = i - idx_i * image_len;
			double buf_max = g_i_j[i][0];
			int idx_max = 0;
			for (int l = 0; l < hyp_cl_amount; l++) {
				if (buf_max < g_i_j[i][l]) {
					buf_max = g_i_j[i][l];
					idx_max = l;
				}
			}
			class_flag[idx_i][idx_j] = idx_max + 1;
		}
	};

	while (stop_flag && itr < 400) {
		itr++;

		for (int i = 0; i < image_len; ++i)
			for (int j = 0; j < image_len; ++j)
				val_i_j[i][j] = dist_poly(generator_S);

		std::thread threadObj1(mix_weight_computation, 0, n / thr_nmb, 0);
		std::thread threadObj2(mix_weight_computation, n / thr_nmb, 2 * n / thr_nmb, 1);
		std::thread threadObj3(mix_weight_computation, 2 * n / thr_nmb, 3 * n / thr_nmb, 2);
		std::thread threadObj4(mix_weight_computation, 3 * n / thr_nmb, 4 * n / thr_nmb, 3);
		std::thread threadObj5(mix_weight_computation, 4 * n / thr_nmb, n, 4);
		threadObj1.join();
		threadObj2.join();
		threadObj3.join();
		threadObj4.join();

		threadObj5.join();
		//cout << "itr:" << itr << endl;
		for (int l = 0; l < hyp_cl_amount; l++) {
			mix_weight[l] = 0.0;
			mix_shift[l] = 0.0;
			for (int m = 0; m < thr_nmb; m++) {
				mix_weight[l] += buff[m][l];

				buff[m][l] = 0.0;
				//mix_shift[l] += buff1[m][l];
				buff1[m][l] = 0.0;
			}

			//mix_shift[l] = mix_shift[l] / mix_weight[l];
			//cout << mix_shift[l] << endl;
			idx_med = 0;
			for (int m = 0; m < image_len*image_len; m++) {
				if (y_i_j[m][l] == 1) {
					buf_median[l][idx_med] = mixture_image[m / image_len][m % image_len];
					idx_med++;
				}
			}
			mix_shift[l] = find_med(buf_median[l], mix_weight[l]);
		}
		
		std::thread threadObj11(mix_scale_computation, 0, n / thr_nmb, 0);
		std::thread threadObj12(mix_scale_computation, n / thr_nmb, 2 * n / thr_nmb, 1);
		std::thread threadObj13(mix_scale_computation, 2 * n / thr_nmb, 3 * n / thr_nmb, 2);
		std::thread threadObj14(mix_scale_computation, 3 * n / thr_nmb, 4 * n / thr_nmb, 3);
		std::thread threadObj15(mix_scale_computation, 4 * n / thr_nmb, n, 4);
		threadObj11.join();
		threadObj12.join();
		threadObj13.join();
		threadObj14.join();
		threadObj15.join();

		for (int l = 0; l < hyp_cl_amount; l++) {
			mix_scale[l] = 0.0;
			for (int m = 0; m < thr_nmb; m++) {
				mix_scale[l] += buff[m][l];
				buff[m][l] = 0.0;
				if (cur_max < max[m])
					cur_max = max[m];
				max[m] = 0;
			}
			mix_scale[l] = sqrt(mix_scale[l] / mix_weight[l]);
			//cout << mix_weight[l] << " ";
			mix_weight[l] = mix_weight[l] / n;
			//cout << mix_scale[l] << " " << mix_weight[l] << endl;

		}
		std::thread threadObj16(g_i_j_recomputation, 0, n / thr_nmb);
		std::thread threadObj17(g_i_j_recomputation, n / thr_nmb, 2 * n / thr_nmb);
		std::thread threadObj18(g_i_j_recomputation, 2 * n / thr_nmb, 3 * n / thr_nmb);
		std::thread threadObj19(g_i_j_recomputation, 3 * n / thr_nmb, 4 * n / thr_nmb);
		std::thread threadObj199(g_i_j_recomputation, 4 * n / thr_nmb, n);
		threadObj16.join();
		threadObj17.join();
		threadObj18.join();
		threadObj19.join();
		threadObj199.join();

		if (cur_max < accuracy)
			stop_flag = false;
		else
			cur_max = 0;
	}
	
	auto end = std::chrono::steady_clock::now();
	auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	cout << "elapsed_ms  " << elapsed_ms.count() << endl;
	cout << "re_mix_shift values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << re_mix_shift[i] << "  ";
	cout << endl;
	cout << "EM mix_shift values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_shift[i] << "  ";
	cout << endl;
	cout << "re_mix_scale values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << re_mix_scale[i] << "  ";
	cout << endl;
	cout << "EM mix_scale values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_scale[i] << "  ";
	cout << endl;

	cout << "SEM mix_weight values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_weight[i] << "  ";
	cout << endl;
	/*cout << "EM max deviation: " << cur_max << " , iterations: " << itr << endl;

	cout << endl;*/

	std::thread threadObj1(mix_split, 0, n / thr_nmb);
	std::thread threadObj2(mix_split, n / thr_nmb, 2 * n / thr_nmb);
	std::thread threadObj3(mix_split, 2 * n / thr_nmb, 3 * n / thr_nmb);
	std::thread threadObj4(mix_split, 3 * n / thr_nmb, 4 * n / thr_nmb);
	std::thread threadObj5(mix_split, 4 * n / thr_nmb, n);
	threadObj1.join();
	threadObj2.join();
	threadObj3.join();
	threadObj4.join();
	threadObj5.join();

	for (int i = 0; i < thr_nmb; i++) {
		delete[] buff[i];
		delete[] buff1[i];

	}
	for (int i = 0; i < hyp_cl_amount; i++) 
		delete[] buf_median[i];
	delete[] buf_median;
	for (int i = 0; i < image_len*image_len; i++)
		delete[]  y_i_j[i];
	for (int i = 0; i < image_len; i++)
		delete[]  val_i_j[i];
	delete[] y_i_j;
	delete[] val_i_j;
	delete[] buff;
	delete[] buff1;
	delete[] max;
}

//реализация SEM для нормальнного распределения с оценкой среднего и СКО
// через медиану 

void SEM_games::SEMalgorithm_median() {
	//вначале итеративная реализация
	cout << "SEM with median estimation " << endl;
	boost::random::uniform_01 <> dist_poly;

	int idx_i = 0;
	int idx_j = 0;
	int thr_nmb = 5;
	double summ = 0;
	double cur_max = 0;
	int itr = 0;
	bool stop_flag = true;
	double n = double(image_len * image_len);
	//int k = class_amount + 1;
	//int k = hyp_cl_amount;
	auto begin = std::chrono::steady_clock::now();
	double** buff = new double*[thr_nmb];
	double** buff1 = new double*[thr_nmb];
	double * max = new double[thr_nmb];
	double** buf_median = new double*[hyp_cl_amount];
	for (int i = 0; i < hyp_cl_amount; ++i)
		buf_median[i] = new double[image_len*image_len];
	int** y_i_j = new int *[image_len*image_len];
	double** val_i_j = new double *[image_len];
	for (int i = 0; i < image_len; ++i)
		val_i_j[i] = new double[image_len];
	for (int i = 0; i < image_len*image_len; i++) {
		y_i_j[i] = new int[hyp_cl_amount];
		for (int j = 0; j < hyp_cl_amount; j++) {
			y_i_j[i][j] = 0.0;
			g_i_j[i][j] = 1.0 / double(hyp_cl_amount);
		}
	}

	for (int i = 0; i < thr_nmb; i++) {
		buff[i] = new double[hyp_cl_amount];
		buff1[i] = new double[hyp_cl_amount];
		max[i] = 0.0;
		for (int l = 0; l < hyp_cl_amount; l++) {
			buff[i][l] = 0;
			buff1[i][l] = 0;
		}
	}

	auto mix_weight_computation = [&](int beg, int end, int nmb) {
		int idx_i = 0;
		int idx_j = 0;
		unsigned i, t, j;
		double summ1 = 0;
		double val, bound_d, bound_u;

		for (i = beg; i < end; i++) {
			summ1 = 0;
			idx_i = unsigned(i / image_len);
			idx_j = i - idx_i * image_len;
			summ1 = 0;
			idx_i = unsigned(i / image_len);
			idx_j = i - idx_i * image_len;
			bound_d = 0;
			bound_u = g_i_j[i][0];
			for (t = 0; t < hyp_cl_amount; ++t) {
				if (val_i_j[idx_i][idx_j] < bound_u && val_i_j[idx_i][idx_j] >= bound_d) {
					y_i_j[i][t] = 1;
					buff[nmb][t] += 1;
					if (mixture_type == "normal")
						buff1[nmb][t] += mixture_image[idx_i][idx_j];
					else {
						if (mixture_type == "log_normal")
							buff1[nmb][t] += log(mixture_image[idx_i][idx_j]);
					}
				}
				else
					y_i_j[i][t] = 0;
				bound_d += g_i_j[i][t];
				if (t == hyp_cl_amount - 2) {
					bound_u = 1;
				}
				else
					if (t == hyp_cl_amount - 1)
						bound_u += 0;
					else
						bound_u += g_i_j[i][t + 1];
			}
		}

	};

	auto g_i_j_recomputation = [&](int beg, int end) {
		int idx_i = 0;
		int idx_j = 0;
		unsigned i, t, j;
		double summ1 = 0;
		for (i = beg; i < end; ++i) {
			summ1 = 0;
			idx_i = unsigned(i / image_len);
			idx_j = i - idx_i * image_len;

			for (t = 0; t < hyp_cl_amount; t++)
				if (mix_scale[t] != 0 && mix_weight[t] != 0)
					if (mixture_type == "normal")
						summ1 += mix_weight[t] * (1 / (mix_scale[t] * sqrt(2 * pi)))*exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[t], 2)) /
						(2.0 * mix_scale[t] * mix_scale[t]));
					else {
						if (mixture_type == "log_normal")
							summ1 += mix_weight[t] * (1 / (mix_scale[t] * mixture_image[idx_i][idx_j] * sqrt(2 * pi)))*exp(-(pow(log(mixture_image[idx_i][idx_j]) - mix_shift[t], 2)) /
							(2.0 * mix_scale[t] * mix_scale[t]));
					}


			for (j = 0; j < hyp_cl_amount; j++) {
				if (mix_scale[j] != 0 && mix_weight[j] != 0)
					if (mixture_type == "normal")
						g_i_j[i][j] = mix_weight[j] * (1 / (mix_scale[j] * sqrt(2 * pi)*summ1))*exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[j], 2))
							/ (2.0 * mix_scale[j] * mix_scale[j]));
					else {
						if (mixture_type == "log_normal")
							g_i_j[i][j] = mix_weight[j] * (1 / (mix_scale[j] * mixture_image[idx_i][idx_j] * sqrt(2 * pi)*summ1))*exp(-(pow(log(mixture_image[idx_i][idx_j]) - mix_shift[j], 2))
								/ (2.0 * mix_scale[j] * mix_scale[j]));
					}
			}
		}
	};

	auto mix_scale_computation = [&](int beg, int end, int nmb) {
		int idx_i = 0;
		int idx_j = 0;
		unsigned i, j, l;

		for (l = 0; l < hyp_cl_amount; l++) {
			for (i = beg; i < end; i++) {
				idx_i = int(i / image_len);
				idx_j = i - idx_i * image_len;
				/*if (mixture_type == "normal")
					buff[nmb][l] += y_i_j[i][l] * pow(mixture_image[idx_i][idx_j] - mix_shift[l], 2);
				else {
					if (mixture_type == "log_normal")
						buff[nmb][l] += y_i_j[i][l] * pow(log(mixture_image[idx_i][idx_j]) - mix_shift[l], 2);
				}*/
				if (max[nmb] < abs(g_i_j[i][l] - g_i_j_0[i][l]))
					max[nmb] = abs(g_i_j[i][l] - g_i_j_0[i][l]);
				g_i_j_0[i][l] = g_i_j[i][l];

			}
		}
	};

	auto mix_split = [&](int beg, int end) {
		for (int i = beg; i < end; i++) {
			int idx_i = int(i / image_len);
			int idx_j = i - idx_i * image_len;
			double buf_max = g_i_j[i][0];
			int idx_max = 0;
			for (int l = 0; l < hyp_cl_amount; l++) {
				if (buf_max < g_i_j[i][l]) {
					buf_max = g_i_j[i][l];
					idx_max = l;
				}
			}
			class_flag[idx_i][idx_j] = idx_max + 1;
		}
	};
	int idx_med = 0;
	while (stop_flag && itr < 400) {
		itr++;

		for (int i = 0; i < image_len; ++i)
			for (int j = 0; j < image_len; ++j)
				val_i_j[i][j] = dist_poly(generator_S);

		std::thread threadObj1(mix_weight_computation, 0, n / thr_nmb, 0);
		std::thread threadObj2(mix_weight_computation, n / thr_nmb, 2 * n / thr_nmb, 1);
		std::thread threadObj3(mix_weight_computation, 2 * n / thr_nmb, 3 * n / thr_nmb, 2);
		std::thread threadObj4(mix_weight_computation, 3 * n / thr_nmb, 4 * n / thr_nmb, 3);
		std::thread threadObj5(mix_weight_computation, 4 * n / thr_nmb, n, 4);
		threadObj1.join();
		threadObj2.join();
		threadObj3.join();
		threadObj4.join();
		threadObj5.join();
		//cout << "itr:" << itr << endl;
		for (int l = 0; l < hyp_cl_amount; l++) {
			mix_weight[l] = 0.0;
			mix_shift[l] = 0.0;
			for (int m = 0; m < thr_nmb; m++) {
				mix_weight[l] += buff[m][l];

				buff[m][l] = 0.0;
				//mix_shift[l] += buff1[m][l];
				buff1[m][l] = 0.0;
			}
			idx_med = 0;
			for (int m = 0; m < image_len*image_len; m++) {
				if (y_i_j[m][l] == 1) {
					buf_median[l][idx_med] = mixture_image[m/ image_len][m % image_len];
					idx_med++;
				}
			}
			mix_shift[l] = find_med(buf_median[l], mix_weight[l]);
			//for (int m = 0; m < idx_med; m++) {
			//	cout << "mix_shift[l]  "<< m<<"  "  << buf_median[l][m] << endl;
			//}
			//cout << "median  " << mix_shift[l] << endl;
			////mix_shift[l] = mix_shift[l] / mix_weight[l];
			idx_med = 0;
			for (int m = 0; m < image_len*image_len; m++) {
				if (y_i_j[m][l] == 1) {
					buf_median[l][idx_med] = abs(mixture_image[m / image_len][m%image_len]- mix_shift[l]);
					idx_med++;
				}
			}
			mix_scale[l] = find_med(buf_median[l], mix_weight[l])*1.4826;
			//mix_shift[l] = mix_shift[l] / mix_weight[l];
			//cout << mix_shift[l] << endl;
		}

		std::thread threadObj11(mix_scale_computation, 0, n / thr_nmb, 0);
		std::thread threadObj12(mix_scale_computation, n / thr_nmb, 2 * n / thr_nmb, 1);
		std::thread threadObj13(mix_scale_computation, 2 * n / thr_nmb, 3 * n / thr_nmb, 2);
		std::thread threadObj14(mix_scale_computation, 3 * n / thr_nmb, 4 * n / thr_nmb, 3);
		std::thread threadObj15(mix_scale_computation, 4 * n / thr_nmb, n, 4);
		threadObj11.join();
		threadObj12.join();
		threadObj13.join();
		threadObj14.join();
		threadObj15.join();

		for (int l = 0; l < hyp_cl_amount; l++) {
			//mix_scale[l] = 0.0;
			for (int m = 0; m < thr_nmb; m++) {
				//mix_scale[l] += buff[m][l];
				buff[m][l] = 0.0;
				if (cur_max < max[m])
					cur_max = max[m];
				max[m] = 0;
			}
			//mix_scale[l] = sqrt(mix_scale[l] / mix_weight[l]);
			//cout << mix_weight[l] << " ";
			mix_weight[l] = mix_weight[l] / n;
			//cout << mix_scale[l] << " " << mix_weight[l] << endl;

		}
		std::thread threadObj16(g_i_j_recomputation, 0, n / thr_nmb);
		std::thread threadObj17(g_i_j_recomputation, n / thr_nmb, 2 * n / thr_nmb);
		std::thread threadObj18(g_i_j_recomputation, 2 * n / thr_nmb, 3 * n / thr_nmb);
		std::thread threadObj19(g_i_j_recomputation, 3 * n / thr_nmb, 4 * n / thr_nmb);
		std::thread threadObj199(g_i_j_recomputation, 4 * n / thr_nmb, n);
		threadObj16.join();
		threadObj17.join();
		threadObj18.join();
		threadObj19.join();
		threadObj199.join();

		//cout << "max: " << cur_max << endl;

		if (cur_max < accuracy)
			stop_flag = false;
		else
			cur_max = 0;
	}
	
	auto end = std::chrono::steady_clock::now();
	auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	cout << "elapsed_ms  " << elapsed_ms.count() << endl;
	cout << endl;
	cout << endl;
	cout << "re_mix_shift values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << re_mix_shift[i] << "  ";
	cout << endl;
	cout << "EM mix_shift values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_shift[i] << "  ";
	cout << endl;
	cout << "re_mix_scale values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << re_mix_scale[i] << "  ";
	cout << endl;
	cout << "EM mix_scale values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_scale[i] << "  ";
	cout << endl;
	cout << "EM mix_weight values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_weight[i] << "  ";
	cout << endl;
	/*cout << "EM max deviation: " << cur_max << " , iterations: " << itr << endl;

	cout << endl;*/

	/*std::thread threadObj1(mix_split, 0, n / thr_nmb);
	std::thread threadObj2(mix_split, n / thr_nmb, 2 * n / thr_nmb);
	std::thread threadObj3(mix_split, 2 * n / thr_nmb, 3 * n / thr_nmb);
	std::thread threadObj4(mix_split, 3 * n / thr_nmb, 4 * n / thr_nmb);
	std::thread threadObj5(mix_split, 4 * n / thr_nmb, n);
	threadObj1.join();
	threadObj2.join();
	threadObj3.join();
	threadObj4.join();
	threadObj5.join();*/

	for (int i = 0; i < thr_nmb; i++) {
		delete[] buff[i];
		delete[] buff1[i];
	}
	for (int i = 0; i < hyp_cl_amount; i++) {
		delete[] buf_median[i];
		
	}

	for (int i = 0; i < image_len*image_len; i++)
		delete[]  y_i_j[i];
	for (int i = 0; i < image_len; i++)
		delete[]  val_i_j[i];
	delete[] y_i_j;
	delete[] val_i_j;
	delete[] buf_median;
	delete[] buff;
	delete[] buff1;
	delete[] max;
}

//реализация SEM для нормальнного распределения с оценкой среднего и СКО
//медианой и обычным образом и выбор из них наиболее правдоподобной оценки

void SEM_games::SEMalgorithm_median_mean() {
	//вначале итеративная реализация
	cout << "SEM with median_mean estimation " << endl;
	boost::random::uniform_01 <> dist_poly;
	auto begin = std::chrono::steady_clock::now();

	int idx_i      = 0;
	int idx_j      = 0;
	int thr_nmb    = 5;
	int est_amount = 2;
	double summ    = 0;
	double cur_max = 0;
	int itr        = 0;
	bool stop_flag = true;
	double n       = double(image_len * image_len);
	
	double** buff       = new double*[thr_nmb];
	double** buff1      = new double*[thr_nmb];
	double * max        = new double[thr_nmb];
	double** buf_median = new double*[hyp_cl_amount];
	int** y_i_j         = new int *[image_len*image_len];
	double** val_i_j    = new double *[image_len];
	double* max_shift   = new double[est_amount*hyp_cl_amount];
	double* max_scale   = new double[est_amount*hyp_cl_amount];
	double* max_L_mass  = new double[est_amount];

	double*mix_shift_1 = new double[hyp_cl_amount];
	double*mix_scale_1 = new double[hyp_cl_amount];

	for (int i = 0; i < image_len; ++i)
		val_i_j[i] = new double[image_len];
	for (int i = 0; i < image_len*image_len; i++) {
		y_i_j[i] = new int[hyp_cl_amount];
		for (int j = 0; j < hyp_cl_amount; j++) {
			y_i_j[i][j] = 0.0;
			g_i_j[i][j] = 1.0 / double(hyp_cl_amount);
		}
	}

	for (int i = 0; i < hyp_cl_amount; ++i)
		buf_median[i] = new double[image_len*image_len];
	for (int i = 0; i < thr_nmb; i++) {
		buff[i] = new double[hyp_cl_amount];
		buff1[i] = new double[hyp_cl_amount];
		max[i] = 0.0;
		for (int l = 0; l < hyp_cl_amount; l++) {
			buff[i][l] = 0;
			buff1[i][l] = 0;
		}
	}

	auto mix_weight_computation = [&](int beg, int end, int nmb) {
		int idx_i = 0;
		int idx_j = 0;
		unsigned i, t, j;
		double summ1 = 0;
		double val, bound_d, bound_u;

		for (i = beg; i < end; i++) {
			//summ1 = 0;
			idx_i = unsigned(i / image_len);
			idx_j = i - idx_i * image_len;
			summ1 = 0;
			idx_i = unsigned(i / image_len);
			idx_j = i - idx_i * image_len;
			bound_d = 0;
			bound_u = g_i_j[i][0];
			for (t = 0; t < hyp_cl_amount; ++t) {
				if (val_i_j[idx_i][idx_j] < bound_u && val_i_j[idx_i][idx_j] >= bound_d) {
					y_i_j[i][t] = 1;
					buff[nmb][t] += 1;
					if (mixture_type == "normal")
						buff1[nmb][t] += mixture_image[idx_i][idx_j];
					else {
						if (mixture_type == "log_normal")
							buff1[nmb][t] += log(mixture_image[idx_i][idx_j]);
					}
				}
				else
					y_i_j[i][t] = 0;
				bound_d += g_i_j[i][t];
				if (t == hyp_cl_amount - 2) {
					bound_u = 1;
				}
				else
					if (t == hyp_cl_amount - 1)
						bound_u += 0;
					else
						bound_u += g_i_j[i][t + 1];
			}
		}
	};

	auto L_max_calculation = [&](int iter, double* mix_shift, double*mix_scale) {
		double buf_max_l = 0;
		bool pre_flag    = true;
		bool flag        = false;
		double max_L     = 0;
		int idx_i, idx_j;

		for (int m = 0; m < image_len*image_len; m++) {
			idx_i = unsigned(m / image_len);
			idx_j = m - idx_i * image_len;
			double B = 0;
			for (int t = 0; t < hyp_cl_amount; ++t) {
				if (mix_scale[t] != 0 && mix_weight[t] != 0) 
					B += mix_weight[t] * (1.0 / (mix_scale[t] ))*
						exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[t], 2)) /
							(2.0 * mix_scale[t] * mix_scale[t]));
				else {
					flag = true;
					break;
				}
			}
			if (B > 0)
				buf_max_l = buf_max_l + log(B / (image_len*image_len));
		}
		for (int m = 0; m < hyp_cl_amount; ++m) {
			max_shift[hyp_cl_amount*iter + m] = mix_shift[m];
			max_scale[hyp_cl_amount * iter + m] = mix_scale[m];
		}
		
		max_L_mass[iter] = buf_max_l;
	};

	auto g_i_j_recomputation = [&](int beg, int end) {
		int idx_i = 0;
		int idx_j = 0;
		unsigned i, t, j;
		double summ1 = 0;
		for (i = beg; i < end; ++i) {
			summ1 = 0;
			idx_i = unsigned(i / image_len);
			idx_j = i - idx_i * image_len;

			for (t = 0; t < hyp_cl_amount; t++)
				if (mix_scale[t] != 0 && mix_weight[t] != 0)
					if (mixture_type == "normal")
						summ1 += mix_weight[t] * (1 / (mix_scale[t] * sqrt(2 * pi)))*exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[t], 2)) /
						(2.0 * mix_scale[t] * mix_scale[t]));
					else {
						if (mixture_type == "log_normal")
							summ1 += mix_weight[t] * (1 / (mix_scale[t] * mixture_image[idx_i][idx_j] * sqrt(2 * pi)))*exp(-(pow(log(mixture_image[idx_i][idx_j]) - mix_shift[t], 2)) /
							(2.0 * mix_scale[t] * mix_scale[t]));
					}


			for (j = 0; j < hyp_cl_amount; j++) {
				if (mix_scale[j] != 0 && mix_weight[j] != 0)
					if (mixture_type == "normal")
						g_i_j[i][j] = mix_weight[j] * (1 / (mix_scale[j] * sqrt(2 * pi)*summ1))*exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[j], 2))
							/ (2.0 * mix_scale[j] * mix_scale[j]));
					else {
						if (mixture_type == "log_normal")
							g_i_j[i][j] = mix_weight[j] * (1 / (mix_scale[j] * mixture_image[idx_i][idx_j] * sqrt(2 * pi)*summ1))*exp(-(pow(log(mixture_image[idx_i][idx_j]) - mix_shift[j], 2))
								/ (2.0 * mix_scale[j] * mix_scale[j]));
					}
			}
		}
	};

	auto mix_scale_computation = [&](int beg, int end, int nmb) {
		int idx_i = 0;
		int idx_j = 0;
		unsigned i, j, l;

		for (l = 0; l < hyp_cl_amount; l++) {
			for (i = beg; i < end; i++) {
				
				if (max[nmb] < abs(g_i_j[i][l] - g_i_j_0[i][l]))
					max[nmb] = abs(g_i_j[i][l] - g_i_j_0[i][l]);
				g_i_j_0[i][l] = g_i_j[i][l];

			}
		}
	};

	auto mix_split = [&](int beg, int end) {
		for (int i = beg; i < end; i++) {
			int idx_i = int(i / image_len);
			int idx_j = i - idx_i * image_len;
			double buf_max = g_i_j[i][0];
			int idx_max = 0;
			for (int l = 0; l < hyp_cl_amount; l++) {
				if (buf_max < g_i_j[i][l]) {
					buf_max = g_i_j[i][l];
					idx_max = l;
				}
			}
			class_flag[idx_i][idx_j] = idx_max + 1;
		}
	};
	int idx_med = 0;
	while (stop_flag && itr < 400) {
		itr++;

		for (int i = 0; i < image_len; ++i)
			for (int j = 0; j < image_len; ++j)
				val_i_j[i][j] = dist_poly(generator_S);

		std::thread threadObj1(mix_weight_computation, 0, n / thr_nmb, 0);
		std::thread threadObj2(mix_weight_computation, n / thr_nmb, 2 * n / thr_nmb, 1);
		std::thread threadObj3(mix_weight_computation, 2 * n / thr_nmb, 3 * n / thr_nmb, 2);
		std::thread threadObj4(mix_weight_computation, 3 * n / thr_nmb, 4 * n / thr_nmb, 3);
		std::thread threadObj5(mix_weight_computation, 4 * n / thr_nmb, n, 4);
		threadObj1.join();
		threadObj2.join();
		threadObj3.join();
		threadObj4.join();
		threadObj5.join();
		//cout << "itr:" << itr << endl;
		for (int l = 0; l < hyp_cl_amount; l++) {
			mix_weight[l] = 0.0;
			mix_shift[l]  = 0.0;
			for (int m = 0; m < thr_nmb; m++) {
				mix_weight[l] += buff[m][l];
				buff[m][l] = 0.0;
				mix_shift[l] += buff1[m][l];
				buff1[m][l] = 0.0;
			}
			mix_shift[l] = mix_shift[l] / mix_weight[l];
			
			mix_scale[l] = 0;
			for (int m = 0; m < image_len*image_len; m++) {
				idx_i = int(m / image_len);
				idx_j = m - idx_i * image_len;
				if (mixture_type == "normal")
					mix_scale[l] += y_i_j[m][l] * pow(mixture_image[idx_i][idx_j] - mix_shift[l], 2);
			}
			mix_scale[l] = sqrt(mix_scale[l] / mix_weight[l]);

			// оценивание с помощью медианы
			idx_med = 0;
			for (int m = 0; m < image_len*image_len; m++) {
				if (y_i_j[m][l] == 1) {
					buf_median[l][idx_med] = mixture_image[m / image_len][m % image_len];
					idx_med++;
				}
			}
			mix_shift_1[l] = find_med(buf_median[l], mix_weight[l]);
			
			idx_med = 0;
			for (int m = 0; m < image_len*image_len; m++) {
				if (y_i_j[m][l] == 1) {
					buf_median[l][idx_med] = abs(mixture_image[m / image_len][m % image_len] - mix_shift_1[l]);
					idx_med++;
				}
			}
			mix_scale_1[l] = find_med(buf_median[l], mix_weight[l])*1.4826;
		}
		L_max_calculation(0, mix_shift, mix_scale);
		L_max_calculation(1, mix_shift_1, mix_scale_1);
		int beg_idx = 0;
		int max_idx = 0;
		
		for (int m = beg_idx ; m < est_amount; ++m) {
			if (max_L_mass[max_idx] < max_L_mass[m]) 
				max_idx = m;
		}
		//max_L_mass[0] = max_L_mass[max_idx];
		
		for (int m = 0; m < hyp_cl_amount; ++m) {
			mix_shift[m] = max_shift[hyp_cl_amount*max_idx + m] ;
			mix_scale[m] = max_scale[hyp_cl_amount *max_idx + m];
			//cout << "mix_shift_new[l]   " << mix_shift[m] << endl;
			/*max_shift[0 + m] = max_shift[hyp_cl_amount*max_idx + m];
			max_scale[0 + m] = max_scale[hyp_cl_amount *max_idx + m];*/
		}
		std::thread threadObj11(mix_scale_computation, 0, n / thr_nmb, 0);
		std::thread threadObj12(mix_scale_computation, n / thr_nmb, 2 * n / thr_nmb, 1);
		std::thread threadObj13(mix_scale_computation, 2 * n / thr_nmb, 3 * n / thr_nmb, 2);
		std::thread threadObj14(mix_scale_computation, 3 * n / thr_nmb, 4 * n / thr_nmb, 3);
		std::thread threadObj15(mix_scale_computation, 4 * n / thr_nmb, n, 4);
		threadObj11.join();
		threadObj12.join();
		threadObj13.join();
		threadObj14.join();
		threadObj15.join();

		for (int l = 0; l < hyp_cl_amount; l++) {
			for (int m = 0; m < thr_nmb; m++) {
				buff[m][l] = 0.0;
				if (cur_max < max[m])
					cur_max = max[m];
				max[m] = 0;
			}
			mix_weight[l] = mix_weight[l] / n;
		}
		std::thread threadObj16(g_i_j_recomputation, 0, n / thr_nmb);
		std::thread threadObj17(g_i_j_recomputation, n / thr_nmb, 2 * n / thr_nmb);
		std::thread threadObj18(g_i_j_recomputation, 2 * n / thr_nmb, 3 * n / thr_nmb);
		std::thread threadObj19(g_i_j_recomputation, 3 * n / thr_nmb, 4 * n / thr_nmb);
		std::thread threadObj199(g_i_j_recomputation, 4 * n / thr_nmb, n);
		threadObj16.join();
		threadObj17.join();
		threadObj18.join();
		threadObj19.join();
		threadObj199.join();

		if (cur_max < accuracy)
			stop_flag = false;
		else
			cur_max = 0;
	}

	auto end = std::chrono::steady_clock::now();
	auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	cout << "elapsed_ms  " << elapsed_ms.count() << endl;
	
	cout << "re_mix_shift values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << re_mix_shift[i] << "  ";
	cout << endl;
	cout << "EM mix_shift values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_shift[i] << "  ";
	cout << endl;
	cout << "re_mix_scale values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << re_mix_scale[i] << "  ";
	cout << endl;
	cout << "EM mix_scale values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_scale[i] << "  ";
	cout << endl;
	cout << "EM mix_weight values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_weight[i] << "  ";
	cout << endl;
	/*cout << "EM max deviation: " << cur_max << " , iterations: " << itr << endl;

	cout << endl;*/

	/*std::thread threadObj1(mix_split, 0, n / thr_nmb);
	std::thread threadObj2(mix_split, n / thr_nmb, 2 * n / thr_nmb);
	std::thread threadObj3(mix_split, 2 * n / thr_nmb, 3 * n / thr_nmb);
	std::thread threadObj4(mix_split, 3 * n / thr_nmb, 4 * n / thr_nmb);
	std::thread threadObj5(mix_split, 4 * n / thr_nmb, n);
	threadObj1.join();
	threadObj2.join();
	threadObj3.join();
	threadObj4.join();
	threadObj5.join();*/

	for (int i = 0; i < thr_nmb; i++) {
		delete[] buff[i];
		delete[] buff1[i];
	}
	for (int i = 0; i < hyp_cl_amount; i++) 
		delete[] buf_median[i];
	for (int i = 0; i < image_len*image_len; i++)
		delete[]  y_i_j[i];
	for (int i = 0; i < image_len; i++)
		delete[]  val_i_j[i];

	delete[] y_i_j;
	delete[] val_i_j;
	delete[] max_shift;
	delete[] max_scale;
	delete[] max_L_mass;
	delete[] buf_median;
	delete[] buff;
	delete[] buff1;
	delete[] max;
	delete[] mix_shift_1;
	delete[] mix_scale_1;
}

void SEM_games::SEMalgorithm_median_mean_OMP() {
	//вначале итеративная реализация
	cout << endl;
	cout << endl;
	cout << "SEM with median_mean estimation - omp version" << endl;
	boost::random::uniform_01 <> dist_poly;
	auto begin = std::chrono::steady_clock::now();
	int thr_nmb = 5;
	int est_amount = 2;
	double summ = 0;
	double cur_max = 0;
	int itr = 0;
	bool stop_flag = true;
	double n = double(image_len * image_len);

	double** buff = new double*[thr_nmb];
	double** buff1 = new double*[thr_nmb];
	double * max = new double[thr_nmb];
	double** buf_median = new double*[hyp_cl_amount];
	int** y_i_j = new int *[image_len*image_len];
	double** val_i_j = new double *[image_len];
	double* max_shift = new double[est_amount*hyp_cl_amount];
	double* max_scale = new double[est_amount*hyp_cl_amount];
	double* max_L_mass = new double[est_amount];

	double*mix_shift_1 = new double[hyp_cl_amount];
	double*mix_scale_1 = new double[hyp_cl_amount];

	for (int i = 0; i < image_len; ++i)
		val_i_j[i] = new double[image_len];
	for (int i = 0; i < image_len*image_len; i++) {
		y_i_j[i] = new int[hyp_cl_amount];
		for (int j = 0; j < hyp_cl_amount; j++) {
			y_i_j[i][j] = 0.0;
			g_i_j[i][j] = 1.0 / double(hyp_cl_amount);
		}
	}

	for (int i = 0; i < hyp_cl_amount; ++i)
		buf_median[i] = new double[image_len*image_len];
	for (int i = 0; i < thr_nmb; i++) {
		buff[i] = new double[hyp_cl_amount];
		buff1[i] = new double[hyp_cl_amount];
		max[i] = 0.0;
		for (int l = 0; l < hyp_cl_amount; l++) {
			buff[i][l] = 0;
			buff1[i][l] = 0;
		}
	}

	auto L_max_calculation = [&](int iter, double* mix_shift, double*mix_scale) {
		double buf_max_l = 0;
		bool pre_flag = true;
		bool flag = false;
		double max_L = 0;
		int idx_i, idx_j;

		for (int m = 0; m < image_len*image_len; m++) {
			idx_i = unsigned(m / image_len);
			idx_j = m - idx_i * image_len;
			double B = 0;
			for (int t = 0; t < hyp_cl_amount; ++t) {
				if (mix_scale[t] != 0 && mix_weight[t] != 0)
					B += mix_weight[t] * (1.0 / (mix_scale[t]))*
					exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[t], 2)) /
					(2.0 * mix_scale[t] * mix_scale[t]));
				else {
					flag = true;
					break;
				}
			}
			if (B > 0)
				buf_max_l = buf_max_l + log(B / (image_len*image_len));
		}
		for (int m = 0; m < hyp_cl_amount; ++m) {
			max_shift[hyp_cl_amount*iter + m] = mix_shift[m];
			max_scale[hyp_cl_amount * iter + m] = mix_scale[m];
		}

		max_L_mass[iter] = buf_max_l;
	};

	int idx_med = 0;
	while (stop_flag && itr < 400) {
		itr++;
		cur_max = 0;
		for (int i = 0; i < image_len; ++i)
			for (int j = 0; j < image_len; ++j)
				val_i_j[i][j] = dist_poly(generator_S);
		for (int l = 0; l < hyp_cl_amount; l++) {
			mix_weight[l] = 0.0;
			mix_shift[l] = 0.0;
		}
		
		#pragma omp parallel 
		{
			int idx_i = 0;
			int idx_j = 0;
			double summ1 = 0;
			double val, bound_d, bound_u;
			#pragma omp for
			for (int i = 0; i < image_len*image_len; i++) {
				idx_i = unsigned(i / image_len);
				idx_j = i - idx_i * image_len;
				summ1 = 0;
				bound_d = 0;
				bound_u = g_i_j[i][0];
				for (int t = 0; t < hyp_cl_amount; ++t) {
					if (val_i_j[idx_i][idx_j] < bound_u && val_i_j[idx_i][idx_j] >= bound_d) {
						y_i_j[i][t] = 1;
						#pragma omp critical 
						{
							mix_weight[t] += 1;
							if (mixture_type == "normal")
								mix_shift[t] += mixture_image[idx_i][idx_j];
							else {
								if (mixture_type == "log_normal")
									mix_shift[t] += log(mixture_image[idx_i][idx_j]);
							}
						}
					}
					else
						y_i_j[i][t] = 0;
					#pragma omp critical
					{
						bound_d += g_i_j[i][t];
						if (t == hyp_cl_amount - 2) 
							bound_u = 1;
						else
							if (t == hyp_cl_amount - 1)
								bound_u += 0;
							else
								bound_u += g_i_j[i][t + 1];
					}
				}
			}
		}

		for (int l = 0; l < hyp_cl_amount; l++) {
			mix_shift[l] = mix_shift[l] / mix_weight[l];
			mix_scale[l] = 0;
		}

		#pragma omp parallel
		{
			int idx_i1, idx_j1, idx_med;
			#pragma omp for
				for (int l = 0; l < hyp_cl_amount; l++) {
						idx_med = 0;
				for (int m = 0; m < image_len*image_len; m++) {
					idx_i1 = m / image_len;
					idx_j1 = m % image_len;
					
					#pragma omp critical
					{
						if (mixture_type == "normal")
							mix_scale[l] += y_i_j[m][l] * pow(mixture_image[idx_i1][idx_j1] - mix_shift[l], 2);
						if (y_i_j[m][l] == 1) {
							buf_median[l][idx_med] = mixture_image[idx_i1][idx_j1];
							idx_med++;
						}
						/*if (cur_max < abs(g_i_j[m][l] - g_i_j_0[m][l]))
							cur_max = abs(g_i_j[m][l] - g_i_j_0[m][l]);
						g_i_j_0[m][l] = g_i_j[m][l];*/
					}
				}
			}
		}

		for (int l = 0; l < hyp_cl_amount; l++) {
			// оценивание с помощью медианы
			mix_scale[l] = sqrt(mix_scale[l] / mix_weight[l]);
			mix_shift_1[l] = find_med(buf_median[l], mix_weight[l]);
			//mix_shift_1[l] = mix_shift[l];
			idx_med = 0;
			for (int m = 0; m < image_len*image_len; m++) {
				if (y_i_j[m][l] == 1) {
					buf_median[l][idx_med] = abs(mixture_image[m / image_len][m % image_len] - mix_shift_1[l]);
					idx_med++;
				}
			}
			mix_scale_1[l] = find_med(buf_median[l], mix_weight[l])*1.4826;
		}
		//for (int l = 0; l < hyp_cl_amount; l++) {
		//		// оценивание с помощью медианы
		//		mix_scale[l] = sqrt(mix_scale[l] / mix_weight[l]);
		//		mix_shift_1[l] = find_med(buf_median[l], mix_weight[l]);
		//		mix_scale_1[l] = 0;
		//		idx_med = 0;
		//		for (int m = 0; m < image_len*image_len; m++) {
		//			mix_scale_1[l] += y_i_j[m][l]* abs(mixture_image[m / image_len][m % image_len] - mix_shift_1[l]);
		//			
		//		}
		//		mix_scale_1[l] = mix_scale_1[l] *sqrt(3.1415926/2.0)/ mix_weight[l];
		//}
		L_max_calculation(0, mix_shift, mix_scale);
		L_max_calculation(1, mix_shift_1, mix_scale_1);
		int beg_idx = 0;
		int max_idx = 0;

		for (int m = beg_idx; m < est_amount; ++m) {
			if (max_L_mass[max_idx] < max_L_mass[m])
				max_idx = m;
		}
		
		for (int m = 0; m < hyp_cl_amount; ++m) {
			mix_shift[m] = max_shift[hyp_cl_amount*max_idx + m];
			mix_scale[m] = max_scale[hyp_cl_amount *max_idx + m];
		}

		for (int l = 0; l < hyp_cl_amount; l++) {
			
			mix_weight[l] = mix_weight[l] / n;
		}
		
		#pragma omp parallel
		{
			int idx_i = 0;
			int idx_j = 0;
			//int i, t, j;
			double summ1 = 0;

			#pragma omp for
			for (int i = 0; i < image_len*image_len;i++) {
				summ1 = 0;
				idx_i = unsigned(i / image_len);
				idx_j = i - idx_i * image_len;
				
				for (int t = 0; t < hyp_cl_amount; t++)
				#pragma omp critical
				{
					if (mix_scale[t] != 0 && mix_weight[t] != 0) {
						if (mixture_type == "normal")
							summ1 += mix_weight[t] * (1 / (mix_scale[t] * sqrt(2 * pi)))*exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[t], 2)) /
							(2.0 * mix_scale[t] * mix_scale[t]));
						else {
							if (mixture_type == "log_normal")
								summ1 += mix_weight[t] * (1 / (mix_scale[t] * mixture_image[idx_i][idx_j] * sqrt(2 * pi)))*exp(-(pow(log(mixture_image[idx_i][idx_j]) - mix_shift[t], 2)) /
								(2.0 * mix_scale[t] * mix_scale[t]));
						}
					}
				}


				for (int j = 0; j < hyp_cl_amount; j++) {
					if (mix_scale[j] != 0 && mix_weight[j] != 0)
						if (mixture_type == "normal")
							g_i_j[i][j] = mix_weight[j] * (1 / (mix_scale[j] * sqrt(2 * pi)*summ1))*exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[j], 2))
								/ (2.0 * mix_scale[j] * mix_scale[j]));
						else {
							if (mixture_type == "log_normal")
								g_i_j[i][j] = mix_weight[j] * (1 / (mix_scale[j] * mixture_image[idx_i][idx_j] * sqrt(2 * pi)*summ1))*exp(-(pow(log(mixture_image[idx_i][idx_j]) - mix_shift[j], 2))
									/ (2.0 * mix_scale[j] * mix_scale[j]));
						}
					/*#pragma omp critical (cout)
					cout << "  idx_i " << g_i_j[i][j] << "  " << idx_j << endl;*/
				}
			}
		}

		#pragma omp parallel
		{
			int idx_i1, idx_j1, idx_med;
			#pragma omp for
			for (int l = 0; l < hyp_cl_amount; l++) {
				
				for (int m = 0; m < image_len*image_len; m++) {
					
					#pragma omp critical
					{
						
						if (cur_max < abs(g_i_j[m][l] - g_i_j_0[m][l]))
							cur_max = abs(g_i_j[m][l] - g_i_j_0[m][l]);
						g_i_j_0[m][l] = g_i_j[m][l];
					}
				}
			}
		}
		if (cur_max < accuracy)
			stop_flag = false;
		else
			cur_max = 0;
	}

	auto end = std::chrono::steady_clock::now();
	auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	cout << "elapsed_ms  " << elapsed_ms.count() << endl;

	cout << "re_mix_shift values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << re_mix_shift[i] << "  ";
	cout << endl;
	cout << "EM mix_shift values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_shift[i] << "  ";
	cout << endl;
	cout << "re_mix_scale values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << re_mix_scale[i] << "  ";
	cout << endl;
	cout << "EM mix_scale values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_scale[i] << "  ";
	cout << endl;
	cout << "EM mix_weight values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_weight[i] << "  ";
	cout << endl;
	/*cout << "EM max deviation: " << cur_max << " , iterations: " << itr << endl;

	cout << endl;*/

	/*std::thread threadObj1(mix_split, 0, n / thr_nmb);
	std::thread threadObj2(mix_split, n / thr_nmb, 2 * n / thr_nmb);
	std::thread threadObj3(mix_split, 2 * n / thr_nmb, 3 * n / thr_nmb);
	std::thread threadObj4(mix_split, 3 * n / thr_nmb, 4 * n / thr_nmb);
	std::thread threadObj5(mix_split, 4 * n / thr_nmb, n);
	threadObj1.join();
	threadObj2.join();
	threadObj3.join();
	threadObj4.join();
	threadObj5.join();
*/
	for (int i = 0; i < thr_nmb; i++) {
		delete[] buff[i];
		delete[] buff1[i];
	}
	for (int i = 0; i < hyp_cl_amount; i++)
		delete[] buf_median[i];
	for (int i = 0; i < image_len*image_len; i++)
		delete[]  y_i_j[i];
	for (int i = 0; i < image_len; i++)
		delete[]  val_i_j[i];

	delete[] y_i_j;
	delete[] val_i_j;
	delete[] max_shift;
	delete[] max_scale;
	delete[] max_L_mass;
	delete[] buf_median;
	delete[] buff;
	delete[] buff1;
	delete[] max;
	delete[] mix_shift_1;
	delete[] mix_scale_1;
}

//2 medians

void SEM_games::SEMalgorithm_median_mean2_OMP() {
	//вначале итеративная реализация
	cout << endl;
	cout << endl;
	cout << "SEM with median_mean estimation - omp version" << endl;
	boost::random::uniform_01 <> dist_poly;
	auto begin = std::chrono::steady_clock::now();
	int thr_nmb = 5;
	int est_amount = 3;
	double summ = 0;
	double cur_max = 0;
	int itr = 0;
	bool stop_flag = true;
	double n = double(image_len * image_len);

	double** buff = new double*[thr_nmb];
	double** buff1 = new double*[thr_nmb];
	double * max = new double[thr_nmb];
	double** buf_median = new double*[hyp_cl_amount];
	int** y_i_j = new int *[image_len*image_len];
	double** val_i_j = new double *[image_len];
	double* max_shift = new double[est_amount*hyp_cl_amount];
	double* max_scale = new double[est_amount*hyp_cl_amount];
	double* max_L_mass = new double[est_amount];

	double*mix_shift_1 = new double[hyp_cl_amount];
	double*mix_scale_1 = new double[hyp_cl_amount];

	double*mix_shift_2 = new double[hyp_cl_amount];
	double*mix_scale_2 = new double[hyp_cl_amount];

	for (int i = 0; i < image_len; ++i)
		val_i_j[i] = new double[image_len];
	for (int i = 0; i < image_len*image_len; i++) {
		y_i_j[i] = new int[hyp_cl_amount];
		for (int j = 0; j < hyp_cl_amount; j++) {
			y_i_j[i][j] = 0.0;
			g_i_j[i][j] = 1.0 / double(hyp_cl_amount);
		}
	}

	for (int i = 0; i < hyp_cl_amount; ++i)
		buf_median[i] = new double[image_len*image_len];
	for (int i = 0; i < thr_nmb; i++) {
		buff[i] = new double[hyp_cl_amount];
		buff1[i] = new double[hyp_cl_amount];
		max[i] = 0.0;
		for (int l = 0; l < hyp_cl_amount; l++) {
			buff[i][l] = 0;
			buff1[i][l] = 0;
		}
	}

	auto L_max_calculation = [&](int iter, double* mix_shift, double*mix_scale) {
		double buf_max_l = 0;
		bool pre_flag = true;
		bool flag = false;
		double max_L = 0;
		int idx_i, idx_j;

		for (int m = 0; m < image_len*image_len; m++) {
			idx_i = unsigned(m / image_len);
			idx_j = m - idx_i * image_len;
			double B = 0;
			for (int t = 0; t < hyp_cl_amount; ++t) {
				if (mix_scale[t] != 0 && mix_weight[t] != 0)
					B += mix_weight[t] * (1.0 / (mix_scale[t]))*
					exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[t], 2)) /
					(2.0 * mix_scale[t] * mix_scale[t]));
				else {
					flag = true;
					break;
				}
			}
			if (B > 0)
				buf_max_l = buf_max_l + log(B / (image_len*image_len));
		}
		for (int m = 0; m < hyp_cl_amount; ++m) {
			max_shift[hyp_cl_amount*iter + m] = mix_shift[m];
			max_scale[hyp_cl_amount * iter + m] = mix_scale[m];
		}

		max_L_mass[iter] = buf_max_l;
	};

	int idx_med = 0;
	while (stop_flag && itr < 400) {
		itr++;
		cur_max = 0;
		for (int i = 0; i < image_len; ++i)
			for (int j = 0; j < image_len; ++j)
				val_i_j[i][j] = dist_poly(generator_S);
		for (int l = 0; l < hyp_cl_amount; l++) {
			mix_weight[l] = 0.0;
			mix_shift[l] = 0.0;
		}

		#pragma omp parallel 
		{
			int idx_i = 0;
			int idx_j = 0;
			double summ1 = 0;
			double val, bound_d, bound_u;
			#pragma omp for
			for (int i = 0; i < image_len*image_len; i++) {
				idx_i = unsigned(i / image_len);
				idx_j = i - idx_i * image_len;
				summ1 = 0;
				bound_d = 0;
				bound_u = g_i_j[i][0];
				for (int t = 0; t < hyp_cl_amount; ++t) {
					if (val_i_j[idx_i][idx_j] < bound_u && val_i_j[idx_i][idx_j] >= bound_d) {
						y_i_j[i][t] = 1;
						#pragma omp critical 
						{
							mix_weight[t] += 1;
							if (mixture_type == "normal")
								mix_shift[t] += mixture_image[idx_i][idx_j];
							else {
								if (mixture_type == "log_normal")
									mix_shift[t] += log(mixture_image[idx_i][idx_j]);
							}
						}
					}
					else
						y_i_j[i][t] = 0;
					#pragma omp critical
					{
						bound_d += g_i_j[i][t];
						if (t == hyp_cl_amount - 2)
							bound_u = 1;
						else
							if (t == hyp_cl_amount - 1)
								bound_u += 0;
							else
								bound_u += g_i_j[i][t + 1];
					}
				}
			}
		}

		for (int l = 0; l < hyp_cl_amount; l++) {
			mix_shift[l] = mix_shift[l] / mix_weight[l];
			mix_scale[l] = 0;
		}

		#pragma omp parallel
		{
			int idx_i1, idx_j1, idx_med;
			#pragma omp for
			for (int l = 0; l < hyp_cl_amount; l++) {
				idx_med = 0;
				for (int m = 0; m < image_len*image_len; m++) {
					idx_i1 = m / image_len;
					idx_j1 = m % image_len;

					#pragma omp critical
					{
						if (mixture_type == "normal")
							mix_scale[l] += y_i_j[m][l] * pow(mixture_image[idx_i1][idx_j1] - mix_shift[l], 2);
						if (y_i_j[m][l] == 1) {
							buf_median[l][idx_med] = mixture_image[idx_i1][idx_j1];
							idx_med++;
						}
						/*if (cur_max < abs(g_i_j[m][l] - g_i_j_0[m][l]))
							cur_max = abs(g_i_j[m][l] - g_i_j_0[m][l]);
						g_i_j_0[m][l] = g_i_j[m][l];*/
					}
				}
			}
		}

		for (int l = 0; l < hyp_cl_amount; l++) {
			// оценивание с помощью медианы
			mix_scale[l] = sqrt(mix_scale[l] / mix_weight[l]);
			mix_shift_1[l] = find_med(buf_median[l], mix_weight[l]);
			mix_shift_2[l] = mix_shift_1[l];
			mix_scale_2[l] = 0;
			//mix_shift_1[l] = mix_shift[l];
			idx_med = 0;
			for (int m = 0; m < image_len*image_len; m++) {
				if (y_i_j[m][l] == 1) {
					buf_median[l][idx_med] = abs(mixture_image[m / image_len][m % image_len] - mix_shift_1[l]);
					idx_med++;
					mix_scale_2[l] += y_i_j[m][l] * abs(mixture_image[m / image_len][m % image_len] - mix_shift_2[l]);
				}
			}
			mix_scale_1[l] = find_med(buf_median[l], mix_weight[l])*1.4826;
			mix_scale_2[l] = mix_scale_2[l] * (sqrt( pi/ 2.0)) / mix_weight[l];
		}
		
		L_max_calculation(0, mix_shift, mix_scale);
		L_max_calculation(1, mix_shift_1, mix_scale_1);
		L_max_calculation(2, mix_shift_2, mix_scale_2);
		int beg_idx = 0;
		int max_idx = 0;
		/*cout << "mean = " << mix_shift[0] << " " << mix_scale[0] << endl;
		cout << "median1 = " << mix_shift_1[0] << " " << mix_scale_1[0] << endl;
		cout << "median2 = " << mix_shift_2[0] << " " << mix_scale_2[0] << endl;*/
		for (int m = beg_idx; m < est_amount; ++m) {
			if (max_L_mass[max_idx] < max_L_mass[m])
				max_idx = m;
		}

		for (int m = 0; m < hyp_cl_amount; ++m) {
			mix_shift[m] = max_shift[hyp_cl_amount*max_idx + m];
			mix_scale[m] = max_scale[hyp_cl_amount *max_idx + m];
		}

		for (int l = 0; l < hyp_cl_amount; l++) {

			mix_weight[l] = mix_weight[l] / n;
		}

#pragma omp parallel
		{
			int idx_i = 0;
			int idx_j = 0;
			//int i, t, j;
			double summ1 = 0;

#pragma omp for
			for (int i = 0; i < image_len*image_len; i++) {
				summ1 = 0;
				idx_i = unsigned(i / image_len);
				idx_j = i - idx_i * image_len;

				for (int t = 0; t < hyp_cl_amount; t++) {
					#pragma omp critical
					{
						if (mix_scale[t] != 0 && mix_weight[t] != 0)
							if (mixture_type == "normal")
								summ1 += mix_weight[t] * (1 / (mix_scale[t] * sqrt(2 * pi)))*exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[t], 2)) /
								(2.0 * mix_scale[t] * mix_scale[t]));
							else {
								if (mixture_type == "log_normal")
									summ1 += mix_weight[t] * (1 / (mix_scale[t] * mixture_image[idx_i][idx_j] * sqrt(2 * pi)))*exp(-(pow(log(mixture_image[idx_i][idx_j]) - mix_shift[t], 2)) /
									(2.0 * mix_scale[t] * mix_scale[t]));
							}
					}
				}


				for (int j = 0; j < hyp_cl_amount; j++) {
					if (mix_scale[j] != 0 && mix_weight[j] != 0)
						if (mixture_type == "normal")
							g_i_j[i][j] = mix_weight[j] * (1 / (mix_scale[j] * sqrt(2 * pi)*summ1))*exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[j], 2))
								/ (2.0 * mix_scale[j] * mix_scale[j]));
						else {
							if (mixture_type == "log_normal")
								g_i_j[i][j] = mix_weight[j] * (1 / (mix_scale[j] * mixture_image[idx_i][idx_j] * sqrt(2 * pi)*summ1))*exp(-(pow(log(mixture_image[idx_i][idx_j]) - mix_shift[j], 2))
									/ (2.0 * mix_scale[j] * mix_scale[j]));
						}
					/*#pragma omp critical (cout)
					cout << "  idx_i " << g_i_j[i][j] << "  " << idx_j << endl;*/
				}
			}
		}

#pragma omp parallel
		{
			int idx_i1, idx_j1, idx_med;
#pragma omp for
			for (int l = 0; l < hyp_cl_amount; l++) {

				for (int m = 0; m < image_len*image_len; m++) {

#pragma omp critical
					{

						if (cur_max < abs(g_i_j[m][l] - g_i_j_0[m][l]))
							cur_max = abs(g_i_j[m][l] - g_i_j_0[m][l]);
						g_i_j_0[m][l] = g_i_j[m][l];
					}
				}
			}
		}
		if (cur_max < accuracy)
			stop_flag = false;
		else
			cur_max = 0;
	}

	auto end = std::chrono::steady_clock::now();
	auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	cout << "elapsed_ms  " << elapsed_ms.count() << endl;

	cout << "re_mix_shift values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << re_mix_shift[i] << "  ";
	cout << endl;
	cout << "EM mix_shift values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_shift[i] << "  ";
	cout << endl;
	cout << "re_mix_scale values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << re_mix_scale[i] << "  ";
	cout << endl;
	cout << "EM mix_scale values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_scale[i] << "  ";
	cout << endl;
	cout << "EM mix_weight values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_weight[i] << "  ";
	cout << endl;
	/*cout << "EM max deviation: " << cur_max << " , iterations: " << itr << endl;

	cout << endl;*/

	/*std::thread threadObj1(mix_split, 0, n / thr_nmb);
	std::thread threadObj2(mix_split, n / thr_nmb, 2 * n / thr_nmb);
	std::thread threadObj3(mix_split, 2 * n / thr_nmb, 3 * n / thr_nmb);
	std::thread threadObj4(mix_split, 3 * n / thr_nmb, 4 * n / thr_nmb);
	std::thread threadObj5(mix_split, 4 * n / thr_nmb, n);
	threadObj1.join();
	threadObj2.join();
	threadObj3.join();
	threadObj4.join();
	threadObj5.join();
*/
	for (int i = 0; i < thr_nmb; i++) {
		delete[] buff[i];
		delete[] buff1[i];
	}
	for (int i = 0; i < hyp_cl_amount; i++)
		delete[] buf_median[i];
	for (int i = 0; i < image_len*image_len; i++)
		delete[]  y_i_j[i];
	for (int i = 0; i < image_len; i++)
		delete[]  val_i_j[i];

	delete[] y_i_j;
	delete[] val_i_j;
	delete[] max_shift;
	delete[] max_scale;
	delete[] max_L_mass;
	delete[] buf_median;
	delete[] buff;
	delete[] buff1;
	delete[] max;
	delete[] mix_shift_1;
	delete[] mix_scale_1;
	delete[] mix_shift_2;
	delete[] mix_scale_2;
}

//2 medians - для распределения релея

void SEM_games::SEMalgorithm_median_mean_raygh_OMP() {
	//вначале итеративная реализация
	cout << endl;
	cout << endl;
	cout << "SEM with median_mean estimation - o_mp version. handling a rayleigh distribution case" << endl;
	boost::random::uniform_01 <> dist_poly;
	auto begin = std::chrono::steady_clock::now();
	int est_amount = 3;
	double summ = 0;
	double cur_max = 0;
	int itr = 0;
	bool stop_flag = true;
	double n = double(image_len * image_len);

	/*double** buff = new double*[thr_nmb];
	double** buff1 = new double*[thr_nmb];*/
	//double * max = new double[thr_nmb];
	double** buf_median = new double*[hyp_cl_amount];
	int** y_i_j = new int *[image_len*image_len];
	double** val_i_j = new double *[image_len];

	double* max_shift = new double[est_amount*hyp_cl_amount];
	double* max_scale = new double[est_amount*hyp_cl_amount];
	double* max_L_mass = new double[est_amount];

	double*mix_shift_1 = new double[hyp_cl_amount];
	double*mix_scale_1 = new double[hyp_cl_amount];

	double*mix_shift_2 = new double[hyp_cl_amount];
	double*mix_scale_2 = new double[hyp_cl_amount];

	for (int i = 0; i < image_len; ++i)
		val_i_j[i] = new double[image_len];
	for (int i = 0; i < image_len*image_len; i++) {
		y_i_j[i] = new int[hyp_cl_amount];
		for (int j = 0; j < hyp_cl_amount; j++) {
			y_i_j[i][j] = 0.0;
			g_i_j[i][j] = 1.0 / double(hyp_cl_amount);
			g_i_j_0[i][j] = 0;
		}
	}

	for (int i = 0; i < hyp_cl_amount; ++i)
		buf_median[i] = new double[image_len*image_len];

	auto L_max_calculation = [&](int iter, double* mix_shift, double*mix_scale) {
		double buf_max_l = 0;
		bool flag = false;
		double max_L = 0;
		int idx_i, idx_j;

		for (int m = 0; m < image_len*image_len; m++) {
			idx_i = unsigned(m / image_len);
			idx_j = m - idx_i * image_len;
			double B = 0;
			for (int t = 0; t < hyp_cl_amount; ++t) {
				if (mix_scale[t] != 0 && mix_weight[t] != 0) {
					if (mixture_type == "normal")
					B += mix_weight[t] * (1.0 / (mix_scale[t]))*
						exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[t], 2)) /
						(2.0 * mix_scale[t] * mix_scale[t]));
					if (mixture_type == "rayleigh")
						B += mix_weight[t] * (mixture_image[idx_i][idx_j] / pow(mix_scale[t],2))*
							exp(-(pow(mixture_image[idx_i][idx_j], 2)) /
							(2.0 * mix_scale[t] * mix_scale[t]));
				}
				else {
					flag = true;
					break;
				}
			}
			if (B > 0)
				buf_max_l = buf_max_l + log(B / (image_len*image_len));
		}
		for (int m = 0; m < hyp_cl_amount; ++m) {
			max_shift[hyp_cl_amount*iter + m] = mix_shift[m];
			max_scale[hyp_cl_amount * iter + m] = mix_scale[m];
		}

		max_L_mass[iter] = buf_max_l;
	};

	int idx_med = 0;
	while (stop_flag && itr < 400) {
		itr++;
		cout << "itr " << itr << endl;
		cur_max = 0;
		for (int i = 0; i < image_len; ++i)
			for (int j = 0; j < image_len; ++j)
				val_i_j[i][j] = dist_poly(generator_S);
		for (int l = 0; l < hyp_cl_amount; l++) {
			mix_weight[l]  = 0.0;
			mix_shift[l]   = 0.0;
			mix_shift_1[l] = 0;
			mix_shift_2[l] = 0;
			mix_scale[l]   = 0;
			mix_scale_1[l] = 0;
			mix_scale_2[l] = 0;
		}

		#pragma omp parallel 
		{
			int idx_i = 0;
			int idx_j = 0;
			double summ1 = 0;
			double val, bound_d, bound_u;
			#pragma omp for
			for (int i = 0; i < image_len*image_len; i++) {
				idx_i = unsigned(i / image_len);
				idx_j = i - idx_i * image_len;
				summ1 = 0;
				bound_d = 0;
				bound_u = g_i_j[i][0];
				for (int t = 0; t < hyp_cl_amount; ++t) {
					if (val_i_j[idx_i][idx_j] < bound_u && val_i_j[idx_i][idx_j] >= bound_d) {
						y_i_j[i][t] = 1;
						#pragma omp critical 
						{
							mix_weight[t] += 1;
							if (mixture_type == "normal")
								mix_shift[t] += mixture_image[idx_i][idx_j];
							else {
								if (mixture_type == "log_normal")
									mix_shift[t] += log(mixture_image[idx_i][idx_j]);
								else {
									if (mixture_type == "rayleigh") {
										mix_scale[t] += mixture_image[idx_i][idx_j];
										mix_scale_2[t] += mixture_image[idx_i][idx_j]* mixture_image[idx_i][idx_j];
									}
								}

							}
						}
					}
					else
						y_i_j[i][t] = 0;
					#pragma omp critical
					{
						bound_d += g_i_j[i][t];
						if (t == hyp_cl_amount - 2)
							bound_u = 1;
						else
							if (t == hyp_cl_amount - 1)
								bound_u += 0;
							else
								bound_u += g_i_j[i][t + 1];
					}
				}
			}
		}

		for (int l = 0; l < hyp_cl_amount; l++) {
			if (mixture_type == "normal") {
				mix_shift[l] = mix_shift[l] / mix_weight[l];
				mix_scale[l] = 0;
			}
			else {
				if (mixture_type == "rayleigh") {
					mix_scale[l] =sqrt(2.0/pi)* mix_scale[l]/ mix_weight[l];
					mix_scale_1[l] = 0;
					mix_scale_2[l] = sqrt(mix_scale_2[l] / double(2 * mix_weight[l]));;
				}
			}

		}

		#pragma omp parallel
		{
			int idx_i1, idx_j1, idx_med;
			#pragma omp for
			for (int l = 0; l < hyp_cl_amount; l++) {
				idx_med = 0;
				for (int m = 0; m < image_len*image_len; m++) {
					idx_i1 = m / image_len;
					idx_j1 = m % image_len;

					#pragma omp critical
					{
						if (mixture_type == "normal")
							mix_scale[l] += y_i_j[m][l] * pow(mixture_image[idx_i1][idx_j1] - mix_shift[l], 2);
						/*else {
							if (mixture_type == "rayleigh") {
								mix_scale_1[l] += y_i_j[m][l] * pow(mixture_image[idx_i1][idx_j1] - mix_scale[l] *sqrt(pi/2.0 ), 2);
							}
						}*/
						if (y_i_j[m][l] == 1) {
							buf_median[l][idx_med] = mixture_image[idx_i1][idx_j1];
							idx_med++;
						}
						
					}
				}
			}
		}

		for (int l = 0; l < hyp_cl_amount; l++) {
			// оценивание с помощью медианы
			if (mixture_type == "normal") {
				mix_scale[l] = sqrt(mix_scale[l] / mix_weight[l]);
				mix_shift_1[l] = find_med(buf_median[l], mix_weight[l]);
				mix_shift_2[l] = mix_shift_1[l];
				mix_scale_2[l] = 0;
				idx_med = 0;
				for (int m = 0; m < image_len*image_len; m++) {
					if (y_i_j[m][l] == 1) {
						buf_median[l][idx_med] = abs(mixture_image[m / image_len][m % image_len] - mix_shift_1[l]);
						idx_med++;
						mix_scale_2[l] += y_i_j[m][l] * abs(mixture_image[m / image_len][m % image_len] - mix_shift_2[l]);
					}
				}
				mix_scale_1[l] = find_med(buf_median[l], mix_weight[l])*1.4826;
				mix_scale_2[l] = mix_scale_2[l] * (sqrt(2.0 / pi)) / mix_weight[l];
			}
			else {
				if (mixture_type == "rayleigh") {
					
					
					mix_scale_1[l] = find_med(buf_median[l], mix_weight[l])/sqrt(log(4.0));
					/*idx_med = 0;
					for (int m = 0; m < image_len*image_len; m++) {
						if (y_i_j[m][l] == 1) {
							buf_median[l][idx_med] = abs(mixture_image[m / image_len][m % image_len] - mix_scale[l] * sqrt(pi / 2.0));
							idx_med++;
							
						}
					}
					mix_scale_1[l] = find_med(buf_median[l], mix_weight[l])*1.4826;
					mix_scale_2[l] = mix_scale_2[l] * (sqrt(2.0 / pi)) / mix_weight[l];*/
				}
			}
		
			
		}

		L_max_calculation(0, mix_shift, mix_scale);
		L_max_calculation(1, mix_shift_1, mix_scale_1);
		L_max_calculation(2, mix_shift_2, mix_scale_2);
		
		cout << "estimations " << mix_scale[0] << " " << mix_scale_1[0] << " " << mix_scale_2[0]  << endl;
		int beg_idx = 0;
		int max_idx = 0;

		for (int m = beg_idx; m < est_amount; ++m) {
			if (max_L_mass[max_idx] < max_L_mass[m])
				max_idx = m;
		}

		for (int m = 0; m < hyp_cl_amount; ++m) {
			mix_shift[m] = max_shift[hyp_cl_amount*max_idx + m];
			mix_scale[m] = max_scale[hyp_cl_amount *max_idx + m];
		}

		for (int l = 0; l < hyp_cl_amount; l++) {

			mix_weight[l] = mix_weight[l] / n;
		}

		#pragma omp parallel
		{
			int idx_i = 0;
			int idx_j = 0;
			//int i, t, j;
			double summ1 = 0;

			#pragma omp for
			for (int i = 0; i < image_len*image_len; i++) {
				summ1 = 0;
				idx_i = unsigned(i / image_len);
				idx_j = i - idx_i * image_len;

				for (int t = 0; t < hyp_cl_amount; t++) {
					#pragma omp critical
					{
						if (mix_scale[t] != 0 && mix_weight[t] != 0)
							if (mixture_type == "normal")
								summ1 += mix_weight[t] * (1 / (mix_scale[t] * sqrt(2 * pi)))*exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[t], 2)) /
								(2.0 * mix_scale[t] * mix_scale[t]));
							else {
								if (mixture_type == "log_normal")
									summ1 += mix_weight[t] * (1 / (mix_scale[t] * mixture_image[idx_i][idx_j] * sqrt(2 * pi)))*exp(-(pow(log(mixture_image[idx_i][idx_j]) - mix_shift[t], 2)) /
									(2.0 * mix_scale[t] * mix_scale[t]));
								else {
									if (mixture_type == "rayleigh")
										summ1 += mix_weight[t] * (mixture_image[idx_i][idx_j] / (mix_scale[t] * mix_scale[t]))*exp(-(pow(mixture_image[idx_i][idx_j], 2)) /
										(2.0 * mix_scale[t] * mix_scale[t]));
								}
							}
					}
				}


				for (int j = 0; j < hyp_cl_amount; j++) {
					if (mix_scale[j] != 0 && mix_weight[j] != 0)
						if (mixture_type == "normal")
							g_i_j[i][j] = mix_weight[j] * (1 / (mix_scale[j] * sqrt(2 * pi)*summ1))*exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[j], 2))
								/ (2.0 * mix_scale[j] * mix_scale[j]));
						else {
							if (mixture_type == "log_normal")
								g_i_j[i][j] = mix_weight[j] * (1 / (mix_scale[j] * mixture_image[idx_i][idx_j] * sqrt(2 * pi)*summ1))*exp(-(pow(log(mixture_image[idx_i][idx_j]) - mix_shift[j], 2))
									/ (2.0 * mix_scale[j] * mix_scale[j]));
							else {
								if (mixture_type == "rayleigh")
									g_i_j[i][j] = mix_weight[j] * (mixture_image[idx_i][idx_j] / (mix_scale[j] * mix_scale[j] * summ1))*exp(-(pow(mixture_image[idx_i][idx_j], 2)) /
									(2.0 * mix_scale[j] * mix_scale[j]));
							}
						}
					/*#pragma omp critical (cout)
					cout << "  idx_i " << g_i_j[i][j] << "  " << idx_j << endl;*/
				}
			}
		}

		#pragma omp parallel
		{
			int idx_i1, idx_j1, idx_med;
			#pragma omp for
			for (int l = 0; l < hyp_cl_amount; l++) {
				for (int m = 0; m < image_len*image_len; m++) {
					#pragma omp critical
					{
						if (cur_max < abs(g_i_j[m][l] - g_i_j_0[m][l]))
							cur_max = abs(g_i_j[m][l] - g_i_j_0[m][l]);
						g_i_j_0[m][l] = g_i_j[m][l];
					}
				}
			}
		}
		if (cur_max < accuracy)
			stop_flag = false;
		else
			cur_max = 0;
	}

	auto end = std::chrono::steady_clock::now();
	auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	cout << "elapsed_ms  " << elapsed_ms.count() << endl;

	cout << "re_mix_shift values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << re_mix_shift[i] << "  ";
	cout << endl;
	cout << "EM mix_shift values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_shift[i] << "  ";
	cout << endl;
	cout << "re_mix_scale values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << re_mix_scale[i] << "  ";
	cout << endl;
	cout << "EM mix_scale values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_scale[i] << "  ";
	cout << endl;
	cout << "EM mix_weight values:" << endl;
	for (int i = 0; i < hyp_cl_amount; i++)
		cout << mix_weight[i] << "  ";
	cout << endl;
	
	for (int i = 0; i < hyp_cl_amount; i++)
		delete[] buf_median[i];
	for (int i = 0; i < image_len*image_len; i++)
		delete[]  y_i_j[i];
	for (int i = 0; i < image_len; i++)
		delete[]  val_i_j[i];

	delete[] y_i_j;
	delete[] val_i_j;
	delete[] max_shift;
	delete[] max_scale;
	delete[] max_L_mass;
	delete[] buf_median;
	
	delete[] mix_shift_1;
	delete[] mix_scale_1;
	delete[] mix_shift_2;
	delete[] mix_scale_2;
}

// раскраска изображения с использованием или критерия хи-квадрат, или колмогорова

void SEM_games::split_image() {

	int length_ = 5;
	int iters_i = image_len / length_;
	int x_l, y_l, idx_class;
	double* buf_img = new double [4* length_*length_];
	double* buf_mix_shift = new double[hyp_cl_amount + 1];
	double* buf_mix_scale = new double[hyp_cl_amount + 1];

	for (int i = 0; i < hyp_cl_amount; ++i) {
		buf_mix_shift[i] = mix_shift[i];
		buf_mix_scale[i] = mix_scale[i];
	}
	buf_mix_shift[hyp_cl_amount] = re_mix_shift[0];
	buf_mix_scale[hyp_cl_amount] = re_mix_scale[0];

	auto mix_split_classic = [&](int x_c, int y_c, int x_l, int y_l) {
		for (int i = x_c; i < x_c+ x_l; i++) {
			for (int j = y_c; j < y_c+ y_l; j++) {
				int idx_i = i * image_len+j;
				double buf_max = g_i_j[idx_i][0];
				int idx_max = 0;
				for (int l = 0; l < hyp_cl_amount; l++) {
					if (buf_max < g_i_j[idx_i][l]) {
						buf_max = g_i_j[idx_i][l];
						idx_max = l;
					}
				}
				class_flag[i][j] = idx_max + 1;
			}
		}
	};
	
	
	for (int i = 0; i < iters_i; ++i) {
		for (int j = 0; j < iters_i; ++j) {
			if (i == iters_i - 1)
				x_l = length_ + image_len % length_;
			else
				x_l = length_;
			if (j == iters_i - 1)
				y_l = length_ + image_len % length_;
			else
				y_l = length_;
			 copy_in_one_mass(buf_img, i*length_, j*length_, x_l, y_l);
			//idx_class = chi_square_stats(buf_img,  x_l*y_l);
			idx_class = kolmogorov_stats(buf_img, x_l*y_l, buf_mix_shift, buf_mix_scale, hyp_cl_amount + 1);
			
			//cout << " idx_class  " << idx_class << endl;
			if (idx_class == -1)
				mix_split_classic(i*length_, j*length_, x_l, y_l);
			else {
				for (int l = i * length_; l < i*length_ + x_l; l++) 
					for (int m = j * length_; m < j*length_ + y_l; m++) 
						class_flag[l][m] = idx_class + 1;
			}
			
		}
	}
	delete[] buf_mix_shift;
	delete[] buf_mix_scale;
	delete[] buf_img;

}

// перепечатывание в файл разделенной картинки

void SEM_games::create_splitted_img() {
	out.open(filename_split_image);
	for (int i = 0; i < image_len; i++) {
		for (int j = 0; j < image_len; j++)
			out << class_flag[i][j] << " ";
		out << std::endl;
	}
	out.close();
}

// перевод статистики в формат csv

void SEM_games::statistics_to_csv(string filename, string alg_nme) {

	out.open("D:\\"+mixture_type+"_"+ alg_nme + "_statistics.csv");
	ifstream f;                     // создаем поток
	string buffer;
	std::string tmp;
	out << "size;";
	for (int i = 0; i < hyp_cl_amount; ++i) 
		out << "shift_"<<i+1<<";";
	for (int i = 0; i < hyp_cl_amount; ++i)
		out << "scale_" << i + 1 << ";";
	out << endl;
	out << 0<<";";
	f.open(filename);
	
	getline(f,buffer);
	getline(f, buffer);
	std::istringstream ist0(buffer);
	string print_buf0 = "";
	while (ist0 >> tmp)
		print_buf0 += tmp + ";";
	print_buf0 = print_buf0.substr(0, print_buf0.length() - 1);
	out << print_buf0 << endl;
	while (getline(f, buffer)) {
		std::istringstream ist_i(buffer);
		string print_buf = "";
		while (ist_i >> tmp)
			print_buf+= tmp + ";";
		print_buf = print_buf.substr(0, print_buf.length() - 1);
		
		out << print_buf<< endl;
	}

	f.close();
	out.close();
}

// накопление статистики

void SEM_games::statistics_creation(string filename) {
	ifstream f;                     // создаем поток
	f.open(filename);
	string m_type;
	double* buf_mix_shift = new double[re_cl_amount];
	double* buf_mix_scale = new double[re_cl_amount];
	bool re_wr_flag = false;
	int diff_amount = 0;
	//проверяем, есть ли такой файл, если нет - создаем 
	if (f) {
		// файл есть, проверяем, соответствует ли распределение смеси в нем наблюдаемому в текущем запуске программы...
		f >> m_type;
		
		if (m_type == mixture_type) {
			for (int i = 0; i < re_cl_amount; ++i)
				f >> buf_mix_shift[i] ;
			for (int i = 0; i < re_cl_amount; ++i)
				f >> buf_mix_scale[i] ;
			for (int i = 0; i < re_cl_amount; ++i) {
				if (buf_mix_scale[i] != re_mix_scale[i])
					diff_amount++;
				if (buf_mix_shift[i] != re_mix_shift[i])
					diff_amount++;
			}
			//различий нет, делаем дозапись
			if (diff_amount == 0) {
				ofstream f1;
				f1.open(filename, std::ios::app);
				f1 << targs[0].size << " ";
				for (int i = 0; i < hyp_cl_amount; ++i)
					f1 << mix_shift[buf_numbs[i] - 1] << " ";
				for (int i = 0; i < hyp_cl_amount; ++i)
					f1 << mix_scale[buf_numbs[i] - 1] << " ";
				f1 << endl;
				f1.close();
			}
			else {
				re_wr_flag = true;
			}
		}
		else {
			re_wr_flag = true;
		}
	}
	else
		re_wr_flag = true;
	//различия есть, переписываем все нафиг
	if(re_wr_flag){
		ofstream f1;
		f1.open(filename);
		f1 << mixture_type<<endl;
		for(int i =0; i< re_cl_amount; ++i)
			f1 << re_mix_shift[i]<< " ";
		for (int i = 0; i < re_cl_amount; ++i)
			f1 << re_mix_scale[i] << " ";
		f1 << endl;
		f1 << targs[0].size << " ";
		for (int i = 0; i < hyp_cl_amount; ++i)
			f1 << mix_shift[buf_numbs[i]-1] << " ";
		for (int i = 0; i < hyp_cl_amount; ++i)
			f1 << mix_scale[buf_numbs[i] - 1] << " ";
		f1 << endl;
		f1.close();
	}
	f.close();
}

//рабочая версия для релея без сдвига

void SEM_games::SEMalgorithm_raygh() {
	//вначале итеративная реализация
	boost::random::uniform_01 <> dist_poly;
	int idx_i = 0;
	int idx_j = 0;
	int thr_nmb = 5;
	double summ = 0;
	double cur_max = 0;
	int itr = 0;
	bool stop_flag = true;
	double n = double(image_len * image_len);
	int k = class_amount + 1;
	auto begin = std::chrono::steady_clock::now();
	double** buff = new double*[thr_nmb];
	double** buff1 = new double*[thr_nmb];
	double * max = new double[thr_nmb];
	int** y_i_j = new int *[image_len*image_len];
	double** val_i_j = new double *[image_len];

	for (int i = 0; i < image_len; ++i)
		val_i_j[i] = new double[image_len];
	for (int i = 0; i < image_len*image_len; i++) {
		y_i_j[i] = new int[hyp_cl_amount];
		for (int j = 0; j < hyp_cl_amount; j++) {
			y_i_j[i][j] = 0.0;
			g_i_j[i][j] = 1.0 / double(hyp_cl_amount);
			g_i_j_0[i][j] = 0;
		}
	}

	for (int i = 0; i < thr_nmb; i++) {
		buff[i] = new double[k];
		buff1[i] = new double[k];
		max[i] = 0.0;
		for (int l = 0; l < k; l++) {
			buff[i][l] = 0;
			buff1[i][l] = 0;
		}
	}

	auto mix_weight_computation = [&](int beg, int end, int nmb) {
		int idx_i = 0;
		int idx_j = 0;
		unsigned i, t, j;
		double summ1 = 0;
		double val, bound_d, bound_u;

		for (i = beg; i < end; i++) {
			summ1 = 0;
			idx_i = unsigned(i / image_len);
			idx_j = i - idx_i * image_len;
			summ1 = 0;
			idx_i = unsigned(i / image_len);
			idx_j = i - idx_i * image_len;
			bound_d = 0;
			bound_u = g_i_j[i][0];
			for (t = 0; t < k; ++t) {
				if (val_i_j[idx_i][idx_j] < bound_u && val_i_j[idx_i][idx_j] >= bound_d) {
					y_i_j[i][t] = 1;
					buff[nmb][t] += 1;
					/*if (buff1[nmb][t] != -1) {
						if (buff1[nmb][t] > mixture_image[idx_i][idx_j])
							buff1[nmb][t] = mixture_image[idx_i][idx_j];
					}
					else*/
					buff1[nmb][t] += mixture_image[idx_i][idx_j] * mixture_image[idx_i][idx_j];
				}
				else
					y_i_j[i][t] = 0;
				bound_d += g_i_j[i][t];
				if (t == k - 2) {
					bound_u = 1;
				}
				else
					if (t == k - 1)
						bound_u += 0;
					else
						bound_u += g_i_j[i][t + 1];
				if (max[nmb] < abs(g_i_j[i][t] - g_i_j_0[i][t]))
					max[nmb] = abs(g_i_j[i][t] - g_i_j_0[i][t]);
				g_i_j_0[i][t] = g_i_j[i][t];
			}
		}

	};

	auto g_i_j_recomputation = [&](int beg, int end) {
		int idx_i = 0;
		int idx_j = 0;
		unsigned i, t, j;
		double summ1 = 0;
		for (i = beg; i < end; ++i) {
			summ1 = 0;
			idx_i = unsigned(i / image_len);
			idx_j = i - idx_i * image_len;

			for (t = 0; t < k; t++)
				if (mix_scale[t] != 0
					&& mix_weight[t] != 0)
					summ1 += mix_weight[t] * ((mixture_image[idx_i][idx_j]) / (mix_scale[t] * mix_scale[t]))*exp(-(pow(mixture_image[idx_i][idx_j], 2)) /
					(2.0 * mix_scale[t] * mix_scale[t]));

			for (j = 0; j < k; j++) {
				if (mix_scale[j] != 0
					&& mix_weight[j] != 0)
					g_i_j[i][j] = mix_weight[j] * (((mixture_image[idx_i][idx_j]) / (mix_scale[j] * mix_scale[j] * summ1))*exp(-(pow(mixture_image[idx_i][idx_j], 2))
						/ (2.0 * mix_scale[j] * mix_scale[j])));
			}
		}
	};

	auto mix_scale_computation = [&](int beg, int end, int nmb) {
		int idx_i = 0;
		int idx_j = 0;
		unsigned i, j, l;

		for (l = 0; l < k; l++) {
			for (i = beg; i < end; i++) {
				idx_i = int(i / image_len);
				idx_j = i - idx_i * image_len;
				buff[nmb][l] += y_i_j[i][l] * pow(mixture_image[idx_i][idx_j], 2);
				if (max[nmb] < abs(g_i_j[i][l] - g_i_j_0[i][l]))
					max[nmb] = abs(g_i_j[i][l] - g_i_j_0[i][l]);
				g_i_j_0[i][l] = g_i_j[i][l];
			}
		}
	};


	auto mix_split = [&](int beg, int end) {
		for (int i = beg; i < end; i++) {
			int idx_i = int(i / image_len);
			int idx_j = i - idx_i * image_len;
			double buf_max = g_i_j[i][0];
			int idx_max = 0;
			for (int l = 0; l < k; l++) {
				if (buf_max < g_i_j[i][l]) {
					buf_max = g_i_j[i][l];
					idx_max = l;
				}
			}
			class_flag[idx_i][idx_j] = idx_max + 1;
		}
	};

	while (stop_flag && itr < 800) {
		itr++;
		if (mixture_type != "normal") {
			for (int i = 0; i < image_len; ++i)
				for (int j = 0; j < image_len; ++j)
					val_i_j[i][j] = dist_poly(generator_S);

			std::thread threadObj1(mix_weight_computation, 0, n / thr_nmb, 0);
			std::thread threadObj2(mix_weight_computation, n / thr_nmb, 2 * n / thr_nmb, 1);
			std::thread threadObj3(mix_weight_computation, 2 * n / thr_nmb, 3 * n / thr_nmb, 2);
			std::thread threadObj4(mix_weight_computation, 3 * n / thr_nmb, 4 * n / thr_nmb, 3);
			std::thread threadObj5(mix_weight_computation, 4 * n / thr_nmb, n, 4);
			threadObj1.join();
			threadObj2.join();
			threadObj3.join();
			threadObj4.join();

			threadObj5.join();
			cout << "itr:" << itr << endl;
			for (int l = 0; l < k; l++) {
				mix_weight[l] = 0.0;
				mix_scale[l] = 0;
				for (int m = 0; m < thr_nmb; m++) {
					mix_weight[l] += buff[m][l];
					buff[m][l] = 0.0;
					mix_scale[l] += buff1[m][l];
					buff1[m][l] = 0;
					if (cur_max < max[m])
						cur_max = max[m];
					max[m] = 0;
				}
				mix_scale[l] = sqrt(mix_scale[l] / double(2 * mix_weight[l]));
				cout << mix_scale[l] << endl;
			}

			std::thread threadObj21(g_i_j_recomputation, 0, n / thr_nmb);
			std::thread threadObj22(g_i_j_recomputation, n / thr_nmb, 2 * n / thr_nmb);
			std::thread threadObj23(g_i_j_recomputation, 2 * n / thr_nmb, 3 * n / thr_nmb);
			std::thread threadObj24(g_i_j_recomputation, 3 * n / thr_nmb, 4 * n / thr_nmb);
			std::thread threadObj25(g_i_j_recomputation, 4 * n / thr_nmb, n);
			threadObj21.join();
			threadObj22.join();
			threadObj23.join();
			threadObj24.join();
			threadObj25.join();

			cout << "max: " << cur_max << endl;

			if (cur_max < accuracy)
				stop_flag = false;
			else
				cur_max = 0;

		}
	}

	auto end = std::chrono::steady_clock::now();
	auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	cout << "elapsed_ms  " << elapsed_ms.count() << endl;
	cout << endl;
	cout << endl;
	cout << "re_mix_shift values:" << endl;
	for (int i = 0; i < k; i++)
		cout << re_mix_shift[i] << "  ";
	cout << endl;
	cout << "EM mix_shift values:" << endl;
	for (int i = 0; i < k; i++)
		cout << mix_shift[i] << "  ";
	cout << endl;
	cout << "re_mix_scale values:" << endl;
	for (int i = 0; i < k; i++)
		cout << re_mix_scale[i] << "  ";
	cout << endl;
	cout << "EM mix_scale values:" << endl;
	for (int i = 0; i < k; i++)
		cout << mix_scale[i] << "  ";
	cout << endl;
	cout << "EM max deviation: " << cur_max << " , iterations: " << itr << endl;

	std::thread threadObj1(mix_split, 0, n / thr_nmb);
	std::thread threadObj2(mix_split, n / thr_nmb, 2 * n / thr_nmb);
	std::thread threadObj3(mix_split, 2 * n / thr_nmb, 3 * n / thr_nmb);
	std::thread threadObj4(mix_split, 3 * n / thr_nmb, 4 * n / thr_nmb);
	std::thread threadObj5(mix_split, 4 * n / thr_nmb, n);
	threadObj1.join();
	threadObj2.join();
	threadObj3.join();
	threadObj4.join();
	threadObj5.join();
	/*double min1 = 1000;
	double min2 = 1000;
	out.open(filename_split_image);
	for (int i = 0; i < image_len; i++) {
		for (int j = 0; j < image_len; j++) {
			out << class_flag[i][j] << " ";
			if (class_flag[i][j] == 1) {
				if (mixture_image[i][j] < min1)
					min1 = mixture_image[i][j];
			}
			if (class_flag[i][j] == 2) {
				if (mixture_image[i][j] < min2)
					min2 = mixture_image[i][j];
			}
		}
		out << std::endl;
	}
	out.close();
	cout << "min1  " << min1 << "   " << "min2  " << min2 << endl;*/
	for (int i = 0; i < thr_nmb; i++) {
		delete[] buff[i];
		delete[] buff1[i];
	}
	delete[] buff;
	delete[] buff1;
	delete[] max;

}

//  рабочая версия программы для распределения Релея со сдвигом

void SEM_games::SEMalgorithm_raygh12() {
	boost::random::uniform_01 <> dist_poly;
	int idx_i = 0;
	int idx_j = 0;
	int thr_nmb = 5;
	double summ = 0;
	double cur_max = 0;
	double shift_max = 100;

	int itr = 0;
	bool stop_flag = true;
	double n = double(image_len * image_len);
	int k = class_amount + 1;
	auto begin = std::chrono::steady_clock::now();
	double** buff = new double*[thr_nmb];
	double** buff1 = new double*[thr_nmb];
	double * max = new double[thr_nmb];
	int** y_i_j = new int *[image_len*image_len];
	int** y_i_j_var = new int *[image_len*image_len];
	int** y_i_j_const = new int *[image_len*image_len];
	double** g_i_j_00 = new double *[image_len*image_len];
	double** val_i_j = new double *[image_len];
	double*dj = new double[k];
	double*buf_weight = new double[k];
	double*d_prob = new double[k];
	double*m2 = new double[k];
	double*m3 = new double[k];
	double *buf_max_l_mass = new double[20];
	double*mix_mid = new double[k];
	double*mix_mid2 = new double[k];
	double*max_sh = new double[21 * k];
	double* last_max_sh = new double[k];
	double* pre_last_max_sh = new double[k];
	double* pre_last_max_sсale = new double[k];
	double last_max_L = 0;
	double *med_vars = new double[k];
	double ** med_class = new double*[k];

	double *max_shift = new double[5 * k];
	double *max_scale = new double[5 * k];
	double *max_weight = new double[5 * k];
	double * iter_l = new double[2];

	int itr_g = 0;
	unsigned iter_med_mass = 0;
	bool step_flag;
	double min_element;
	for (int i = 0; i < image_len; ++i)
		val_i_j[i] = new double[image_len];

	for (int i = 0; i < image_len*image_len; i++) {
		y_i_j[i] = new int[hyp_cl_amount];
		y_i_j_var[i] = new int[hyp_cl_amount];
		y_i_j_const[i] = new int[hyp_cl_amount];
		g_i_j_00[i] = new double[hyp_cl_amount];
		for (int j = 0; j < hyp_cl_amount; j++) {
			y_i_j[i][j] = 0.0;
			y_i_j_var[i][j] = 0.0;
			y_i_j_const[i][j] = 0.0;

			g_i_j[i][j] = 1.0 / double(hyp_cl_amount);
			g_i_j_0[i][j] = 0;
			g_i_j_00[i][j] = 0;
		}
	}

	for (int j = 0; j < hyp_cl_amount; j++) {
		med_class[j] = new double[image_len*image_len];

	}

	for (int i = 0; i < thr_nmb; i++) {
		buff[i] = new double[k];
		buff1[i] = new double[k];
		max[i] = 0.0;
		for (int l = 0; l < k; l++) {
			buff[i][l] = 0;
			buff1[i][l] = 0;
		}
	}

	auto mix_weight_computation = [&](int beg, int end, int nmb) {
		int idx_i = 0;
		int idx_j = 0;
		unsigned i, t, j;
		double summ1 = 0;
		double val, bound_d, bound_u;

		for (i = beg; i < end; i++) {
			summ1 = 0;
			idx_i = unsigned(i / image_len);
			idx_j = i - idx_i * image_len;
			summ1 = 0;
			idx_i = unsigned(i / image_len);
			idx_j = i - idx_i * image_len;
			bound_d = 0;
			bound_u = g_i_j[i][0];
			for (t = 0; t < k; ++t) {
				if (val_i_j[idx_i][idx_j] < bound_u && val_i_j[idx_i][idx_j] >= bound_d) {
					y_i_j[i][t] = 1;
					buff[nmb][t] += 1;
					buff1[nmb][t] += mixture_image[idx_i][idx_j];
				}
				else
					y_i_j[i][t] = 0;
				bound_d += g_i_j[i][t];
				if (t == k - 2)
					bound_u = 1;
				else
					if (t == k - 1)
						bound_u += 0;
					else
						bound_u += g_i_j[i][t + 1];
			}
		}
	};

	auto g_i_j_recomputation = [&](int beg, int end, bool end_flag) {
		int idx_i = 0;
		int idx_j = 0;
		unsigned i, t, j;
		double summ1 = 0;
		for (i = beg; i < end; ++i) {
			summ1 = 0;
			idx_i = unsigned(i / image_len);
			idx_j = i - idx_i * image_len;
			for (t = 0; t < k; t++)
				if (mix_scale[t] != 0
					&& mix_weight[t] != 0) {
					if (mixture_image[idx_i][idx_j] - mix_shift[t] > 0)
						summ1 += mix_weight[t] * ((mixture_image[idx_i][idx_j] - mix_shift[t]) / (mix_scale[t] * mix_scale[t]))*
						exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[t], 2)) /
						(2.0 * mix_scale[t] * mix_scale[t]));
					else {
						if (end_flag)
							summ1 += 0;
					}
				}
			for (j = 0; j < k; j++) {
				if (mix_scale[j] != 0
					&& mix_weight[j] != 0) {
					if (mixture_image[idx_i][idx_j] - mix_shift[j] > 0 && summ1 > 0)
						g_i_j[i][j] = mix_weight[j] * (((mixture_image[idx_i][idx_j] - mix_shift[j]) / (mix_scale[j] * mix_scale[j] * summ1))*
							exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[j], 2)) / (2.0 * mix_scale[j] * mix_scale[j])));
					else {
						if (end_flag)
							g_i_j[i][j] = 0;
						else
							g_i_j[i][j] = 0;
					}
				}
				else
					g_i_j[i][j] = 0;
			}
		}
	};

	auto mix_scale_computation = [&](int beg, int end, int nmb) {
		int idx_i = 0;
		int idx_j = 0;
		unsigned i, j, l;
		for (l = 0; l < k; l++) {
			for (i = beg; i < end; i++) {
				idx_i = int(i / image_len);
				idx_j = i - idx_i * image_len;
				buff[nmb][l] += y_i_j[i][l] * pow(mixture_image[idx_i][idx_j] - mix_shift[l], 2);
				if (max[nmb] < abs(g_i_j[i][l] - g_i_j_0[i][l]))
					max[nmb] = abs(g_i_j[i][l] - g_i_j_0[i][l]);
				g_i_j_0[i][l] = g_i_j[i][l];

			}
		}
	};

	auto mix_split = [&](int beg, int end) {
		for (int i = beg; i < end; i++) {
			int idx_i = int(i / image_len);
			int idx_j = i - idx_i * image_len;
			double buf_max = g_i_j[i][0];
			int idx_max = 0;
			for (int l = 0; l < k; l++) {
				if (buf_max < g_i_j[i][l]) {
					buf_max = g_i_j[i][l];
					idx_max = l;
				}
			}
			class_flag[idx_i][idx_j] = idx_max + 1;
		}
	};

	auto calc_radical1 = [&](int l) {
		double a1 = mix_shift[l];
		double sigma = mix_scale[l];
		double a0 = 0;
		int itr = 0;
		int counter = 0;
		double a2 = 0;
		double a3 = 0;
		//cout << "sigma " << mix_shift[l] << endl;
		/*for (int m = 0; m < image_len*image_len; m++) {
			a3 += y_i_j[m][l] * (mixture_image[idx_i][idx_j] - a1)* (mixture_image[idx_i][idx_j] - a1);
			counter += y_i_j[m][l];
		}
		a3 = a3 / (2.0*counter);*/
		while ((itr < 100) && ((a1 > 0) && (abs(a1 - a0) > 0.001))) {
			/*if (a1 < min_element)
				a1 = min_element - 0.001;*/
			a0 = a1;
			a1 = 0;
			//cout << "a0 " << a0 << endl;
			a2 = 0;
			a3 = 0;
			counter = 0;
			for (int m = 0; m < image_len*image_len; m++) {
				idx_i = unsigned(m / image_len);
				idx_j = m - idx_i * image_len;

				if ((mixture_image[idx_i][idx_j] - a0 > 0) && (sigma != 0)) {
					a1 += y_i_j[m][l] * mixture_image[idx_i][idx_j];
					a2 += double(y_i_j[m][l]) / (mixture_image[idx_i][idx_j] - a0);
					counter += y_i_j[m][l];
					a3 += y_i_j[m][l] * (mixture_image[idx_i][idx_j] - a0)* (mixture_image[idx_i][idx_j] - a0);
				}

				if ((mixture_image[idx_i][idx_j] - a0 == 0) && (sigma != 0)) {
					a1 += y_i_j[m][l] * mixture_image[idx_i][idx_j];
					a2 += double(y_i_j[m][l]) / (0.001);
					counter += y_i_j[m][l];
					a3 += y_i_j[m][l] * (0.001)* (0.001);
				}
			}
			//a3 = pre_last_max_sсale[l];
			//cout << "a2 " << a2 << " " << a3 << " " << counter << " " << a0 << " " << endl;
			//a1 = a2 * (a1 / a2 - a3 / (2 * counter)) / counter;
			a1 = (a1 - a2 * a3 / (2.0*counter)) / double(counter);
			//a1 = a2 * (a1 / a2 - sigma*sigma) / double(counter);

				//a1 = a1 / counter;
			//a0 = a1;
			//a1 = 
			itr++;
		}
		if (a1 < 0)
			a1 = a0;
		//cout << "a1 " << a1 <<" "<< itr<< endl;
		if (a1 < min_element)
			a1 = min_element - 0.001;
		//cout << "a1 " << a1 <<" "<< itr<< endl;
		return a1;
		//a1
	};

	auto f_computation = [&](double sigma, double y, int l) {
		double result = 0;
		double a1 = 0;
		double a2 = 0;
		double a3 = 0;
		int count = 0;
		for (int m = 0; m < image_len*image_len; m++) {
			idx_i = unsigned(m / image_len);
			idx_j = m - idx_i * image_len;

			if ((mixture_image[idx_i][idx_j] - y > 0) && (sigma != 0)) {
				a1 += y_i_j[m][l] * mixture_image[idx_i][idx_j];
				a2 += y_i_j[m][l] / (mixture_image[idx_i][idx_j] - y);
				count += y_i_j[m][l];
				a3 += y_i_j[m][l] * (mixture_image[idx_i][idx_j] - y)* (mixture_image[idx_i][idx_j] - y);

			}
			/*result += y_i_j[m][l] * ((mixture_image[idx_i][idx_j] - y) *(mixture_image[idx_i][idx_j] - y) - sigma * sigma)
			/ ((mixture_image[idx_i][idx_j] - y)*sigma*sigma);*/
			/*result +=  ((mixture_image[idx_i][idx_j] - y) *(mixture_image[idx_i][idx_j] - y) - sigma * sigma)
			/ ((mixture_image[idx_i][idx_j] - y)*sigma*sigma);*/
			if ((mixture_image[idx_i][idx_j] - y == 0) && (sigma != 0))
				/*result += y_i_j[m][l] * (0.001 *0.001 - sigma * sigma)
				/ (0.001*sigma*sigma);*/
				/*result += (0.001 *0.001 - sigma * sigma)
					/ (0.001*sigma*sigma);*/
			{
				a1 += y_i_j[m][l] * mixture_image[idx_i][idx_j];
				a2 += y_i_j[m][l] / (0.001);
				count += y_i_j[m][l];
				a3 += y_i_j[m][l] * (0.001)* (0.001);

			}
		}
		result = count * (a1 / count - y) / a2 - a3 / (2 * count);

		return result;
	};


	auto calc_radical = [&](int l) {
		double x_min = mix_shift[l] - 20;
		double x_max = mix_shift[l] + 20;
		double y_last = 0;
		if (x_max != x_min) {
			while (f_computation(mix_scale[l], x_min, l)*f_computation(mix_scale[l], x_max, l) > 0 && x_max < m3[l])
				x_max += 1;

			if (x_max < m3[l] && f_computation(mix_scale[l], x_min, l)*f_computation(mix_scale[l], x_max, l) < 0) {
				double x_mid = (x_min + x_max) / 2;
				y_last = x_mid;
				if (f_computation(mix_scale[l], x_min, l)*f_computation(mix_scale[l], x_mid, l) < 0)
					x_max = x_mid;
				else
					if (f_computation(mix_scale[l], x_max, l)*f_computation(mix_scale[l], x_mid, l) < 0)
						x_min = x_mid;

				x_mid = (x_min + x_max) / 2;
				//cout << "x_mid " << x_mid << endl;
				while (abs(x_mid - y_last) < 0.0001) {
					y_last = x_mid;
					if (f_computation(mix_scale[l], x_min, l)*f_computation(mix_scale[l], x_mid, l) < 0)
						x_max = x_mid;
					else {
						if (f_computation(mix_scale[l], x_max, l)*f_computation(mix_scale[l], x_mid, l) < 0)
							x_min = x_mid;
					}
					x_mid = (x_min + x_max) / 2;
				}
				med_vars[l] = x_mid;

				//cout << "!" << endl;
				//cout << "mix_shift[x_mid] " << mix_shift[l]<<" "<< med_vars[l]<< endl;
			}
			else {
				med_vars[l] = mix_shift[l];
				//x_max = mix_shift[l]+20;
				//x_min = mix_shift[l] - 20;
				//while (f_computation(mix_scale[l], x_min, l)*f_computation(mix_scale[l], x_max, l) > 0 && x_min > min_element)
				//	x_min -= 1;
				///*med_vars[l] = mix_shift[l];*/
				//cout << "x_min " << x_min << endl;

				//double x_mid = (x_min + x_max) / 2;
				//y_last = x_mid;
				//if (f_computation(mix_scale[l], x_min, l)*f_computation(mix_scale[l], x_mid, l) < 0)
				//	x_max = x_mid;
				//else
				//	if (f_computation(mix_scale[l], x_max, l)*f_computation(mix_scale[l], x_mid, l) < 0)
				//		x_min = x_mid;

				//x_mid = (x_min + x_max) / 2;
				//while (abs(x_mid - y_last) < 0.0001) {
				//	y_last = x_mid;
				//	if (f_computation(mix_scale[l], x_min, l)*f_computation(mix_scale[l], x_mid, l) < 0)
				//		x_max = x_mid;
				//	else {
				//		if (f_computation(mix_scale[l], x_max, l)*f_computation(mix_scale[l], x_mid, l) < 0)
				//			x_min = x_mid;
				//	}
				//	x_mid = (x_min + x_max) / 2;
				//}
				//med_vars[l] = x_mid;

				//cout << "!!" << endl;
				//cout << "mix_shift[x_mid] " << mix_shift[l] << " " << med_vars[l] << endl;
			}
		}
		else
			mix_shift[l] = x_min;
	};

	auto weights_refresh = [&](int ** y_i_j, double* mix_weight) {
		int itr_fr = 0;
		bool stop_flag1 = true;
		mix_weight[0] = 0.5;
		mix_weight[1] = 0.5;
		g_i_j_recomputation(0, (image_len * image_len), false);
		for (int i = 0; i < image_len*image_len; i++)
			for (int j = 0; j < hyp_cl_amount; j++)
				g_i_j_00[i][j] = 0;
		while (stop_flag1 && itr_fr < 400) {
			itr_fr++;
			cur_max = 0;
			if (mixture_type != "normal") {
				for (int i = 0; i < image_len; ++i)
					for (int j = 0; j < image_len; ++j)
						val_i_j[i][j] = dist_poly(generator_S);
				shift_max = 100;
				std::thread threadObj1(mix_weight_computation, 0, n / thr_nmb, 0);
				std::thread threadObj2(mix_weight_computation, n / thr_nmb, 2 * n / thr_nmb, 1);
				std::thread threadObj3(mix_weight_computation, 2 * n / thr_nmb, 3 * n / thr_nmb, 2);
				std::thread threadObj4(mix_weight_computation, 3 * n / thr_nmb, 4 * n / thr_nmb, 3);
				std::thread threadObj5(mix_weight_computation, 4 * n / thr_nmb, n, 4);
				threadObj1.join();
				threadObj2.join();
				threadObj3.join();
				threadObj4.join();
				threadObj5.join();
				bool mix_w_zero = false;
				for (int l = 0; l < k; l++) {
					mix_weight[l] = 0.0;
					for (int m = 0; m < thr_nmb; m++) {
						mix_weight[l] += buff[m][l];
						buff[m][l] = 0.0;
					}
					mix_weight[l] = mix_weight[l] / (image_len*image_len);
					if (mix_weight[l] == 0)
						mix_w_zero = true;
				}
				if (!mix_w_zero) {
					for (int l = 0; l < k; l++) {
						for (int i = 0; i < image_len*image_len; i++) {
							if (cur_max < abs(g_i_j[i][l] - g_i_j_00[i][l]))
								cur_max = abs(g_i_j[i][l] - g_i_j_00[i][l]);
							g_i_j_00[i][l] = g_i_j[i][l];
						}
					}
					std::thread threadObj21(g_i_j_recomputation, 0, n / thr_nmb, false);
					std::thread threadObj22(g_i_j_recomputation, n / thr_nmb, 2 * n / thr_nmb, false);
					std::thread threadObj23(g_i_j_recomputation, 2 * n / thr_nmb, 3 * n / thr_nmb, false);
					std::thread threadObj24(g_i_j_recomputation, 3 * n / thr_nmb, 4 * n / thr_nmb, false);
					std::thread threadObj25(g_i_j_recomputation, 4 * n / thr_nmb, n, false);
					threadObj21.join();
					threadObj22.join();
					threadObj23.join();
					threadObj24.join();
					threadObj25.join();
					if (cur_max < accuracy)
						stop_flag1 = false;
					else
						cur_max = 0;
				}
			}
		}
	};

	auto throw_out_useless_pix = [&](int ** y_i_j, double *mix_shift, double * mix_weight) {
		int amount = 0;
		int id = 0;
		int _id = 1;

		if (mix_shift[0] > mix_shift[1]) {
			id = 1;
			_id = 0;

		}

		double bound_u, bound_d;
		bound_d = 256;
		bound_u = 0;
		int m = 0;

		for (int l = 0; l < k; l++) {
			iter_med_mass = 0;
			for (int m = 0; m < image_len*image_len; ++m) {
				idx_i = unsigned(m / image_len);
				idx_j = m - idx_i * image_len;
				if (y_i_j[m][id] != 0) {
					if (mixture_image[idx_i][idx_j] > bound_u)
						bound_u = mixture_image[idx_i][idx_j];
				}
				if ((y_i_j[m][l] == 1) && (mixture_image[idx_i][idx_j] < mix_shift[l])) {
					y_i_j[m][l] = 0;
					amount++;
					mix_weight[l] --;
					int u = 0;
					unsigned min_id = 0;
					bool flag = true;
					while ((mixture_image[idx_i][idx_j] < mix_shift[u]) && (u < k)) {
						if (mix_shift[u] < mix_shift[min_id])
							min_id = u;
						u++;
					}
					if (u == k) {
						y_i_j[m][min_id] = 1;
						mix_weight[min_id] ++;
						mix_shift[min_id] = mixture_image[idx_i][idx_j] - 0.001;
					}
					else {
						y_i_j[m][u] = 1;
						mix_weight[u] ++;
					}
				}
			}
		}
		for (int l = 0; l < k; l++) {
			mix_scale[l] = 0.0;
			for (int m = 0; m < image_len*image_len; m++) {
				idx_i = unsigned(m / image_len);
				idx_j = m - idx_i * image_len;
				mix_scale[l] += y_i_j[m][l] * pow(mixture_image[idx_i][idx_j] - mix_shift[l], 2);
			}
		}

		bool step_flag = true;
		bool indicator = true;
		//while (step_flag) {
		indicator = true;
		for (int l = 0; l < k; l++) {
			for (int m = 0; m < image_len*image_len; m++) {
				idx_i = unsigned(m / image_len);
				idx_j = m - idx_i * image_len;
				if (y_i_j[m][l] == 1 && mixture_image[idx_i][idx_j] > mix_shift[_id]) {
					//cout << "mix_shift[_id] " << mix_shift[_id] <<" mixture_image[idx_i][idx_j] "<< mixture_image[idx_i][idx_j]<< endl;
					double sigma_1 = sqrt(mix_scale[id] / double(2 * mix_weight[id]));
					double sigma_2 = sqrt(mix_scale[_id] / double(2 * mix_weight[_id]));
					double l_otn = 2 * log(sigma_2 / sigma_1) + log((mixture_image[idx_i][idx_j] - mix_shift[id]) / (mixture_image[idx_i][idx_j] - mix_shift[_id])) +
						0.5*((mixture_image[idx_i][idx_j] - mix_shift[_id])*(mixture_image[idx_i][idx_j] - mix_shift[_id]) / (sigma_2*sigma_2) -
						(mixture_image[idx_i][idx_j] - mix_shift[id])*(mixture_image[idx_i][idx_j] - mix_shift[id]) / (sigma_1*sigma_1));
					//log(mix_weight[id]/ double(mix_weight[_id]))+
					//cout << "l_otn " << l_otn<< endl;
					if (l_otn < 0) {
						if (_id != l) {
							y_i_j[m][id] = 0;
							y_i_j[m][_id] = 1;

							mix_weight[id] --;
							mix_weight[_id] ++;
							mix_scale[_id] += pow(mixture_image[idx_i][idx_j] - mix_shift[_id], 2);
							mix_scale[id] -= pow(mixture_image[idx_i][idx_j] - mix_shift[id], 2);
							indicator = false;
						}
					}
					else {
						if (id != l) {
							y_i_j[m][_id] = 0;
							y_i_j[m][id] = 1;
							indicator = false;
							mix_weight[_id] --;
							mix_weight[id] ++;
							mix_scale[id] += pow(mixture_image[idx_i][idx_j] - mix_shift[id], 2);
							mix_scale[_id] -= pow(mixture_image[idx_i][idx_j] - mix_shift[_id], 2);
						}
					}
				}
			}
		}
		if (indicator != false)
			step_flag = false;
		//}

		bound_u = 256;
		for (int m = 0; m < image_len*image_len; m++) {
			idx_i = unsigned(m / image_len);
			idx_j = m - idx_i * image_len;
			if ((y_i_j[m][_id] == 1) && (mixture_image[idx_i][idx_j] < bound_u))
				bound_u = mixture_image[idx_i][idx_j];
		}
		double sigma_2 = sqrt(mix_scale[_id] / double(2 * mix_weight[_id]));
		double sigma_1 = sqrt(mix_scale[id] / double(2 * mix_weight[id]));
		double log_bound = 0.99;
		double id_bound = 1 - exp(-(bound_u - mix_shift[id])*(bound_u - mix_shift[id]) / (2 * sigma_1*sigma_1));
		//log_bound = id_bound + 0.03;
		if (log_bound > 1)
			log_bound = 0.99;
		cout << "bound_u  " << bound_u << " " << mix_shift[_id] << endl;
		while (mix_shift[_id] + sqrt(-2 * log(log_bound)*sigma_2*sigma_2) < bound_u) {
			mix_shift[_id] += bound_u - (mix_shift[_id] + sqrt(-2 * log(log_bound)*sigma_2*sigma_2));
			mix_scale[_id] = 0.0;
			for (int m = 0; m < image_len*image_len; m++) {
				idx_i = unsigned(m / image_len);
				idx_j = m - idx_i * image_len;
				mix_scale[_id] += y_i_j[m][_id] * pow(mixture_image[idx_i][idx_j] - mix_shift[_id], 2);
			}
			sigma_2 = sqrt(mix_scale[_id] / double(2 * mix_weight[_id]));
			//}
			mix_shift[_id] += bound_u - (mix_shift[_id] + sqrt(-2 * log(log_bound)*sigma_2*sigma_2));
			//mix_shift[_id] -= 1;
			cout << "new sigma_2  " << bound_u << " " << mix_shift[_id] << " " << mix_shift[_id] + sqrt(-2 * log(log_bound)*sigma_2*sigma_2) << endl;
			//cout << "up_bound  " << 1 - exp(-(bound_u - mix_shift[id])*(bound_u - mix_shift[id]) / (2 * sigma_1*sigma_1)) <<
				//" " << 1 - exp(-(mix_shift[_id] - mix_shift[id])*(bound_u - mix_shift[id]) / (2 * sigma_1*sigma_1)) << endl;
			for (int l = 0; l < k; l++) {
				mix_scale[l] = 0.0;
				for (int m = 0; m < image_len*image_len; m++) {
					idx_i = unsigned(m / image_len);
					idx_j = m - idx_i * image_len;
					mix_scale[l] += y_i_j[m][l] * pow(mixture_image[idx_i][idx_j] - mix_shift[l], 2);
				}
			}
			bool step_flag = true;
			bool indicator = true;
			//while (step_flag) {
			indicator = true;
			for (int l = 0; l < k; l++) {
				for (int m = 0; m < image_len*image_len; m++) {
					idx_i = unsigned(m / image_len);
					idx_j = m - idx_i * image_len;
					if (y_i_j[m][l] == 1 && mixture_image[idx_i][idx_j] > mix_shift[_id]) {
						//cout << "mix_shift[_id] " << mix_shift[_id] <<" mixture_image[idx_i][idx_j] "<< mixture_image[idx_i][idx_j]<< endl;
						double sigma_1 = sqrt(mix_scale[id] / double(2 * mix_weight[id]));
						double sigma_2 = sqrt(mix_scale[_id] / double(2 * mix_weight[_id]));
						double l_otn = log(mix_weight[id] / double(mix_weight[_id])) + 2 * log(sigma_2 / sigma_1) + log((mixture_image[idx_i][idx_j] - mix_shift[id]) / (mixture_image[idx_i][idx_j] - mix_shift[_id])) +
							0.5*((mixture_image[idx_i][idx_j] - mix_shift[_id])*(mixture_image[idx_i][idx_j] - mix_shift[_id]) / (sigma_2*sigma_2) -
							(mixture_image[idx_i][idx_j] - mix_shift[id])*(mixture_image[idx_i][idx_j] - mix_shift[id]) / (sigma_1*sigma_1));
						//log(mix_weight[id]/ double(mix_weight[_id]))+
						//cout << "l_otn " << l_otn<< endl;
						if (l_otn < 0) {
							if (_id != l) {
								y_i_j[m][id] = 0;
								y_i_j[m][_id] = 1;

								mix_weight[id] --;
								mix_weight[_id] ++;
								mix_scale[_id] += pow(mixture_image[idx_i][idx_j] - mix_shift[_id], 2);
								mix_scale[id] -= pow(mixture_image[idx_i][idx_j] - mix_shift[id], 2);
								indicator = false;
							}
						}
						else {
							if (id != l) {
								y_i_j[m][_id] = 0;
								y_i_j[m][id] = 1;
								indicator = false;
								mix_weight[_id] --;
								mix_weight[id] ++;
								mix_scale[id] += pow(mixture_image[idx_i][idx_j] - mix_shift[id], 2);
								mix_scale[_id] -= pow(mixture_image[idx_i][idx_j] - mix_shift[_id], 2);
							}
						}
					}
				}
			}
			if (indicator != false)
				step_flag = false;
			//}

			bound_u = 256;
			for (int m = 0; m < image_len*image_len; m++) {
				idx_i = unsigned(m / image_len);
				idx_j = m - idx_i * image_len;
				if ((y_i_j[m][_id] == 1) && (mixture_image[idx_i][idx_j] < bound_u))
					bound_u = mixture_image[idx_i][idx_j];
				//if ((y_i_j[m][_id] == 1 )&& (mixture_image[idx_i][idx_j] < mix_shift[_id])) {
				//	y_i_j[m][_id] = 0;
				//	y_i_j[m][id] = 1;
				//	//indicator = false;
				//	mix_weight[_id] --;
				//	mix_weight[id] ++;
				//	bound_u --;
				//}
			}
			sigma_2 = sqrt(mix_scale[_id] / double(2 * mix_weight[_id]));
			sigma_1 = sqrt(mix_scale[id] / double(2 * mix_weight[id]));
			//log_bound = 0.99;
			id_bound = 1 - exp(-(bound_u - mix_shift[id])*(bound_u - mix_shift[id]) / (2 * sigma_1*sigma_1));
			//log_bound = id_bound + 0.03;
			if (log_bound > 1)
				log_bound = 0.99;
			//cout << "bound_u  " << bound_u << " " << mix_shift[_id]<<" " << sigma_2<<" "<< mix_shift[_id] +sqrt(-2*log(0.99)*sigma_2*sigma_2)<< endl;
			while (mix_shift[_id] + sqrt(-2 * log(log_bound)*sigma_2*sigma_2) < bound_u) {
				mix_shift[_id] += bound_u - (mix_shift[_id] + sqrt(-2 * log(log_bound)*sigma_2*sigma_2));
				mix_scale[_id] = 0.0;
				for (int m = 0; m < image_len*image_len; m++) {
					idx_i = unsigned(m / image_len);
					idx_j = m - idx_i * image_len;
					mix_scale[_id] += y_i_j[m][_id] * pow(mixture_image[idx_i][idx_j] - mix_shift[_id], 2);
				}
				sigma_2 = sqrt(mix_scale[_id] / double(2 * mix_weight[_id]));
			}
			cout << "new sigma_2  " << bound_u << " " << mix_shift[_id] << " " << mix_shift[_id] + sqrt(-2 * log(log_bound)*sigma_2*sigma_2) << endl;

		}
	};

	auto a_max_calculation = [&](int iter, bool scale_flag) {
		double buf_max_l = 0;
		int l0_max, l1_max;
		bool pre_flag = true;
		bool flag = false;
		double max_L = 0;
		mix_shift[0] = med_vars[0];
		mix_shift[1] = med_vars[1];
		mix_scale[0] = max_scale[2 * iter];
		mix_scale[1] = max_scale[2 * iter + 1];
		int idx_i, idx_j;
		for (int t = 0; t < k; ++t) {
			buf_weight[t] = mix_weight[t];
			for (int m = 0; m < image_len*image_len; m++)
				y_i_j_var[m][t] = y_i_j[m][t];

		}

		if (scale_flag) {
			throw_out_useless_pix(y_i_j_var, mix_shift, buf_weight);

			for (int t = 0; t < k; ++t) {
				mix_scale[t] = 0.0;
				for (int m = 0; m < image_len*image_len; m++) {
					idx_i = unsigned(m / image_len);
					idx_j = m - idx_i * image_len;
					mix_scale[t] += y_i_j_var[m][t] * pow(mixture_image[idx_i][idx_j] - mix_shift[t], 2);
				}
				mix_scale[t] = sqrt(mix_scale[t] / (buf_weight[t] * 2));

			}
			/*weights_refresh(y_i_j_var, buf_weight);
			buf_weight[0] = buf_weight[0] * (image_len*image_len);
			buf_weight[1] = buf_weight[1] * (image_len*image_len);*/
		}
		/*if(mix_shift[0]< mix_shift[1])
			mix_shift[1] += 0.01;
		else
			mix_shift[0] += 0.01;
*/
		for (int m = 0; m < image_len*image_len; m++) {
			idx_i = unsigned(m / image_len);
			idx_j = m - idx_i * image_len;
			double B = 0;
			for (int t = 0; t < k; ++t) {
				if (mix_scale[t] != 0 && buf_weight[t] != 0) {
					if (mixture_image[idx_i][idx_j] - mix_shift[t] > 0) {
						B += buf_weight[t] * ((mixture_image[idx_i][idx_j] - mix_shift[t]) / (mix_scale[t] * mix_scale[t]))*
							exp(-(pow(mixture_image[idx_i][idx_j] - mix_shift[t], 2)) /
							(2.0 * mix_scale[t] * mix_scale[t]));
					}
					else
						B -= 0.000;
				}
				else {
					flag = true;
					break;
				}
			}
			if (B > 0)
				buf_max_l = buf_max_l + log(B / (image_len*image_len));
		}

		max_sh[k*iter + 0] = mix_shift[0];
		max_sh[k*iter + 1] = mix_shift[1];
		max_scale[2 * iter] = mix_scale[0];
		max_scale[2 * iter + 1] = mix_scale[1];
		max_weight[2 * iter] = buf_weight[0];
		max_weight[2 * iter + 1] = buf_weight[1];
		return  buf_max_l;
	};

	auto low_est_destroy = [&](bool step_flag) {
		bool indicator = true;
		if ((med_vars[0] < med_vars[1]) != step_flag)
			indicator = !indicator;
		//cout<<
		/*for (int l = 0; l < k; l++) {
			if (med_vars[l] < min_element)
				med_vars[l] = min_element - 0.001;

		}*/
		/*if ((med_vars[0] == med_vars[1]) || (abs(med_vars[0] - med_vars[1]) < 10))
			if (indicator)
				med_vars[1] += 20;
			else
				med_vars[0] += 20;*/
	};

	auto max_l_sigma_computation = [&]() {
		for (int t = 0; t < k; ++t) {
			mix_scale[t] = 0.0;
			for (int m = 0; m < image_len*image_len; m++) {
				idx_i = unsigned(m / image_len);
				idx_j = m - idx_i * image_len;
				mix_scale[t] += y_i_j[m][t] * pow(mixture_image[idx_i][idx_j] - mix_shift[t], 2);
			}
			mix_scale[t] = sqrt(mix_scale[t] / (mix_weight[t] * 2));
		}

	};



	//определяем минимальныйй элемент во всей совокупности. один из классов обязан иметь это значение сдвига 
	min_element = mixture_image[0][0];
	for (int l = 0; l < image_len; l++) {
		for (int m = 0; m < image_len; m++)
			if (min_element > mixture_image[l][m])
				min_element = mixture_image[l][m];
	}

	int pre_itr = 0;
	bool mix_w_zero = true;

	while (mix_w_zero && pre_itr < 5) {
		stop_flag = true;
		itr = 0;
		mix_w_zero = false;
		pre_itr++;
		for (int i = 0; i < image_len*image_len; i++) {
			for (int j = 0; j < hyp_cl_amount; j++) {
				pre_last_max_sh[j] = 0;
				y_i_j[i][j] = 0.0;
				y_i_j_var[i][j] = 0.0;
				y_i_j_const[i][j] = 0.0;

				g_i_j[i][j] = 1.0 / double(hyp_cl_amount);
				g_i_j_0[i][j] = 0;
				g_i_j_00[i][j] = 0;
			}
		}
		while (stop_flag && itr < 300) {
			//while (itr < 80) {
			itr++;
			for (int i = 0; i < image_len; ++i)
				for (int j = 0; j < image_len; ++j)
					val_i_j[i][j] = dist_poly(generator_S);
			shift_max = 100;
			std::thread threadObj1(mix_weight_computation, 0, n / thr_nmb, 0);
			std::thread threadObj2(mix_weight_computation, n / thr_nmb, 2 * n / thr_nmb, 1);
			std::thread threadObj3(mix_weight_computation, 2 * n / thr_nmb, 3 * n / thr_nmb, 2);
			std::thread threadObj4(mix_weight_computation, 3 * n / thr_nmb, 4 * n / thr_nmb, 3);
			std::thread threadObj5(mix_weight_computation, 4 * n / thr_nmb, n, 4);
			threadObj1.join();
			threadObj2.join();
			threadObj3.join();
			threadObj4.join();
			threadObj5.join();
			//cout << "itr:" << itr << endl;

			for (int l = 0; l < k; l++) {
				mix_weight[l] = 0.0;
				for (int m = 0; m < thr_nmb; m++) {
					mix_weight[l] += buff[m][l];
					buff[m][l] = 0.0;
					buff1[m][l] = 0;
				}
				if (mix_weight[l] == 0)
					mix_w_zero = true;
			}

			itr_g = 0;
			if (!mix_w_zero) {
				//начальное приближение на сигма

				for (int l = 0; l < k; l++) {
					mix_shift[l] = 0.0;
					iter_med_mass = 0;
					m2[l] = 0;
					for (int m = 0; m < image_len*image_len; m++) {
						idx_i = unsigned(m / image_len);
						idx_j = m - idx_i * image_len;
						mix_shift[l] += y_i_j[m][l] * mixture_image[idx_i][idx_j];
						if (y_i_j[m][l] != 0) {
							med_class[l][iter_med_mass] = mixture_image[idx_i][idx_j];
							if (iter_med_mass == 0)
								m2[l] = med_class[l][iter_med_mass];

							else
								if (m2[l] > med_class[l][iter_med_mass])
									m2[l] = med_class[l][iter_med_mass];

							iter_med_mass++;
						}
					}
					mix_shift[l] = mix_shift[l] / double(mix_weight[l]);
				}
				for (int l = 0; l < k; ++l) {
					iter_med_mass = 0;
					mix_scale[l] = 0;
					double med = find_med(med_class[l], mix_weight[l]);
					for (int m = 0; m < image_len*image_len; m++) {
						idx_i = unsigned(m / image_len);
						idx_j = m - idx_i * image_len;
						mix_scale[l] += y_i_j[m][l] * pow(mixture_image[idx_i][idx_j] - mix_shift[l], 2);
						if (y_i_j[m][l] != 0) {
							med_class[l][iter_med_mass] = abs(mixture_image[idx_i][idx_j] - mix_shift[l]);
							//med_class[l][iter_med_mass] = abs(mixture_image[idx_i][idx_j] - med);
							iter_med_mass++;
						}
					}
					double med1 = find_med(med_class[l], mix_weight[l]);
					mix_scale[l] = sqrt(mix_scale[l] / (mix_weight[l] * (2 - 3.1415926 / 2.0)));
					mix_mid[l] = mix_scale[l];
					mix_mid2[l] = abs(med1) / (0.460434);
					//mix_mid2[l] = abs(med1) / (0.448453);
					max_scale[l] = mix_scale[l];
					max_scale[2 + l] = mix_scale[l];
					max_scale[4 + l] = mix_mid2[l];
					max_scale[6 + l] = mix_mid2[l];
				}


				shift_max = 0;
				itr_g++;
				// оценим а

				for (int l = 0; l < k; l++) {
					mix_shift[l] = 0.0;
					iter_med_mass = 0;
					m2[l] = 0;
					for (int m = 0; m < image_len*image_len; m++) {
						idx_i = unsigned(m / image_len);
						idx_j = m - idx_i * image_len;
						mix_shift[l] += y_i_j[m][l] * mixture_image[idx_i][idx_j];
						if (y_i_j[m][l] != 0) {
							med_class[l][iter_med_mass] = mixture_image[idx_i][idx_j];
							if (iter_med_mass == 0)
								m3[l] = med_class[l][iter_med_mass];
							else
								if (m3[l] < med_class[l][iter_med_mass])
									m3[l] = med_class[l][iter_med_mass];
							iter_med_mass++;
						}
					}
					mix_shift[l] = mix_shift[l] / double(mix_weight[l]);
					m2[l] = abs(mix_mid2[l] * sqrt(3.1415926 / 2) - mix_shift[l]);
					mix_shift[l] = abs(mix_mid[l] * sqrt(3.1415926 / 2) - mix_shift[l]);
					mix_scale[l] = max_scale[2 + l];
					//calc_radical(l);
					med_vars[l] = calc_radical1(l);
				}
				bool radical_flag = (med_vars[1] != med_vars[0]);
				bool _flag = med_vars[0] < med_vars[1];
				step_flag = _flag;
				low_est_destroy(step_flag);
				buf_max_l_mass[0] = a_max_calculation(0, true);


				for (int l = 0; l < k; l++) {
					mix_scale[l] = mix_mid[l];
					double med = find_med(med_class[l], mix_weight[l]);
					mix_shift[l] = abs(mix_mid[l] * sqrt(log(4)) - med);
					mix_scale[l] = max_scale[4 + l];
					//calc_radical(l);
					med_vars[l] = calc_radical1(l);
				}
				if ((!radical_flag) && (med_vars[1] == med_vars[0]))
					radical_flag = false;
				else
					radical_flag = true;
				low_est_destroy(step_flag);
				buf_max_l_mass[1] = a_max_calculation(1, true);

				for (int l = 0; l < k; l++) {
					mix_scale[l] = mix_mid2[l];
					mix_shift[l] = m2[l];
					med_vars[l] = calc_radical1(l);
					/*mix_scale[l] = max_scale[4 + l];
					calc_radical(l);*/
				}
				low_est_destroy(step_flag);
				buf_max_l_mass[2] = a_max_calculation(2, true);

				for (int l = 0; l < k; l++) {
					mix_scale[l] = mix_mid2[l];
					double med = find_med(med_class[l], mix_weight[l]);
					mix_shift[l] = abs(mix_mid2[l] * sqrt(log(4)) - med);
					mix_scale[l] = max_scale[6 + l];
					//calc_radical(l);
					med_vars[l] = calc_radical1(l);
				}
				low_est_destroy(step_flag);
				buf_max_l_mass[3] = a_max_calculation(3, true);

				if (itr > 1) {
					med_vars[0] = last_max_sh[0];
					med_vars[1] = last_max_sh[1];
					last_max_L = a_max_calculation(4, true);
				}

				int max_ind = 0;
				for (int l = 0; l < 4; ++l)
					if (buf_max_l_mass[max_ind] < buf_max_l_mass[l]) {
						max_ind = l;
					}
				//max_ind = 0;
				mix_shift[0] = max_sh[k* max_ind + 0];
				mix_shift[1] = max_sh[2 * max_ind + 1];
				mix_scale[0] = max_scale[k* max_ind + 0];
				mix_scale[1] = max_scale[k* max_ind + 1];
				mix_weight[0] = max_weight[k* max_ind + 0];
				mix_weight[1] = max_weight[k* max_ind + 1];
				pre_last_max_sсale[0] = max_scale[k* max_ind + 0];
				pre_last_max_sсale[1] = max_scale[k* max_ind + 1];
				double buf_max_l = buf_max_l_mass[max_ind];
				//last_max_L = buf_max_l - 1;
			   /* cout << "pereotsenca " << mix_shift[0] << " " << mix_shift[1] << " " << endl;
				cout << " buf_max_l " << buf_max_l_mass[0] << " " << buf_max_l_mass[1] << " " << buf_max_l_mass[2] << " " << buf_max_l_mass[3] << " " << endl;
				cout << "last_max_L " << last_max_L << endl;*/
				//cout << "y_last" << min_element << endl;
				if (itr > 1) {
					if (buf_max_l > last_max_L) {
						//cout << "last_max_L " << last_max_L << endl;
						//if (step_flag == (mix_shift[0] < mix_shift[1])) {
						last_max_L = buf_max_l;
						cout << "max_ind " << max_ind << endl;
						last_max_sh[0] = mix_shift[0];
						last_max_sh[1] = mix_shift[1];
						/*}
						else {
							if (last_max_sh[0] != last_max_sh[1]) {
								mix_shift[0] = last_max_sh[0];
								mix_shift[1] = last_max_sh[1];
							}
							else {
								last_max_L = buf_max_l;
								last_max_sh[0] = mix_shift[0];
								last_max_sh[1] = mix_shift[1];
								step_flag = (mix_shift[0] < mix_shift[1]);
							}
						}*/
					}
					else {
						//cout << "wwwwwwwww" << endl;
						//cout << "last_max_L " << last_max_L << endl;
						if ((last_max_sh[0] != last_max_sh[1]) && (radical_flag)) {
							//if ((last_max_sh[0] != last_max_sh[1])) {
							cout << "max_ind " << 4 << endl;
							mix_shift[0] = last_max_sh[0];
							mix_shift[1] = last_max_sh[1];
							mix_scale[0] = max_scale[k * 4 + 0];
							mix_scale[1] = max_scale[k * 4 + 1];
							pre_last_max_sсale[0] = max_scale[k * 4 + 0];
							pre_last_max_sсale[1] = max_scale[k * 4 + 1];
							mix_weight[0] = max_weight[k * 4 + 0];
							mix_weight[1] = max_weight[k * 4 + 1];
						}
						else {
							/*if (step_flag == (mix_shift[0] < mix_shift[1])) {
								last_max_L = buf_max_l;
								last_max_sh[0] = mix_shift[0];
								last_max_sh[1] = mix_shift[1];
							}
							else {*/
							if (last_max_sh[0] != last_max_sh[1]) {
								mix_shift[0] = last_max_sh[0];
								mix_shift[1] = last_max_sh[1];
								mix_scale[0] = max_scale[k * 4 + 0];
								mix_scale[1] = max_scale[k * 4 + 1];
								cout << "max_ind " << 4 << endl;
								pre_last_max_sсale[0] = max_scale[k * 4 + 0];
								pre_last_max_sсale[1] = max_scale[k * 4 + 1];
							}
							else {
								last_max_L = buf_max_l;
								last_max_sh[0] = mix_shift[0];
								last_max_sh[1] = mix_shift[1];
								step_flag = (mix_shift[0] < mix_shift[1]);
							}
							//}
						}
					}
				}
				else {
					last_max_L = buf_max_l;
					last_max_sh[0] = mix_shift[0];
					last_max_sh[1] = mix_shift[1];
					step_flag = (mix_shift[0] < mix_shift[1]);
					cout << "max_ind " << max_ind << endl;
				}

				//throw_out_useless_pix(y_i_j, mix_shift, mix_weight);
				if (mix_weight[0] == 0 || mix_weight[1] == 0) {
					mix_w_zero = true;
					break;

				}
				else {
					//max_l_sigma_computation();

					pre_last_max_sh[0] = mix_shift[0];
					pre_last_max_sh[1] = mix_shift[1];
					if (mix_scale[0] < 2 || mix_scale[1] < 2) {
						mix_w_zero = true;
						break;
					}
				}
			}
			else {
				if (mix_shift[0] > mix_shift[1]) {
					mix_shift[0] -= 10;
					mix_scale[0] += 10;
					mix_scale[1] -= 10;
				}
				else {
					mix_shift[1] -= 10;
					mix_scale[1] += 10;
					mix_scale[0] -= 10;
				}

			}
			if (mix_w_zero)
				break;
			else {
				//cout << "itr_g   " << itr_g << endl;
				std::thread threadObj16(mix_scale_computation, 0, n / thr_nmb, 0);
				std::thread threadObj17(mix_scale_computation, n / thr_nmb, 2 * n / thr_nmb, 1);
				std::thread threadObj18(mix_scale_computation, 2 * n / thr_nmb, 3 * n / thr_nmb, 2);
				std::thread threadObj19(mix_scale_computation, 3 * n / thr_nmb, 4 * n / thr_nmb, 3);
				std::thread threadObj20(mix_scale_computation, 4 * n / thr_nmb, n, 4);
				threadObj16.join();
				threadObj17.join();
				threadObj18.join();
				threadObj19.join();
				threadObj20.join();
				//weights_refresh(y_i_j, mix_weight);
				cur_max = 0;
				for (int l = 0; l < k; l++) {
					//cout << " mix_shift[l] " << mix_shift[l] << endl;
					for (int m = 0; m < thr_nmb; m++) {
						buff[m][l] = 0.0;
						if (cur_max < max[m])
							cur_max = max[m];
						max[m] = 0;
					}
					//cout << "mix_scale[l]" << mix_scale[l] << " " << endl;
					//mix_weight[l] = mix_weight[l] / (image_len*image_len);

				}


				std::thread threadObj21(g_i_j_recomputation, 0, n / thr_nmb, false);
				std::thread threadObj22(g_i_j_recomputation, n / thr_nmb, 2 * n / thr_nmb, false);
				std::thread threadObj23(g_i_j_recomputation, 2 * n / thr_nmb, 3 * n / thr_nmb, false);
				std::thread threadObj24(g_i_j_recomputation, 3 * n / thr_nmb, 4 * n / thr_nmb, false);
				std::thread threadObj25(g_i_j_recomputation, 4 * n / thr_nmb, n, false);
				threadObj21.join();
				threadObj22.join();
				threadObj23.join();
				threadObj24.join();
				threadObj25.join();
				//if ((cur_max < accuracy)&&(cur_max !=0))
				if ((cur_max < accuracy))
					stop_flag = false;
				else
					cur_max = 0;
			}

		}
		if (abs(mix_shift[0] - mix_shift[1]) < 5)
			mix_w_zero = true;
	}
	auto end = std::chrono::steady_clock::now();
	auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	cout << endl;
	cout << endl;
	cout << "elapsed_ms  " << elapsed_ms.count() << endl;
	cout << "last_max_L  " << last_max_L << endl;
	cout << "EM mix_shift values:" << endl;
	for (int i = 0; i < k; i++) {
		cout << mix_shift[i] << "  ";
		cout << "quantille: " << mix_shift[i] + sqrt(-2 * log(0.99)*mix_scale[i] * mix_scale[i]) << "  ";
	}
	cout << endl;
	cout << "EM mix_scale values:" << endl;
	for (int i = 0; i < k; i++)
		cout << mix_scale[i] << "  ";
	cout << endl;
	cout << "EM max deviation: " << cur_max << " , iterations: " << itr << endl;
	cout << endl;
	//переоценка весов
	med_vars[0] = mix_shift[0];
	med_vars[1] = mix_shift[1];
	weights_refresh(y_i_j, mix_weight);

	std::thread threadObj1(mix_split, 0, n / thr_nmb);
	std::thread threadObj2(mix_split, n / thr_nmb, 2 * n / thr_nmb);
	std::thread threadObj3(mix_split, 2 * n / thr_nmb, 3 * n / thr_nmb);
	std::thread threadObj4(mix_split, 3 * n / thr_nmb, 4 * n / thr_nmb);
	std::thread threadObj5(mix_split, 4 * n / thr_nmb, n);
	threadObj1.join();
	threadObj2.join();
	threadObj3.join();
	threadObj4.join();
	threadObj5.join();
	//out.open(filename_split_image);
	cout << "last_max_L1" << last_max_L << endl;
	int *c_i = new int[2];
	double *a_i = new double[2];
	c_i[0] = 0;
	c_i[1] = 0;
	a_i[0] = 0;
	a_i[1] = 0;
	iter_med_mass = 0;
	int iter_med_mass1 = 0;
	for (int i = 0; i < image_len; i++) {
		for (int j = 0; j < image_len; j++) {
			if (class_flag[i][j] == 1) {
				a_i[0] += mixture_image[i][j];
				c_i[0]++;
				med_class[0][iter_med_mass] = (mixture_image[i][j]);
				iter_med_mass++;
			}
			else {
				a_i[1] += mixture_image[i][j];
				c_i[1]++;
				med_class[1][iter_med_mass1] = (mixture_image[i][j]);
				iter_med_mass1++;
			}
		}
	}
	cout << "mids: " << mix_shift[0] + mix_scale[0] * sqrt(3.1415926 / 2.0) << "   " << mix_shift[1] + mix_scale[1] * sqrt(3.1415926 / 2.0) << endl;
	cout << "meds: " << mix_shift[0] + mix_scale[0] * sqrt(log(4)) << "   " << mix_shift[1] + mix_scale[1] * sqrt(log(4)) << endl;

	double min1 = 1000;
	double min2 = 1000;
	/*out.open(filename_split_image);
	for (int i = 0; i < image_len; i++) {
		for (int j = 0; j < image_len; j++) {
			out << class_flag[i][j] << " ";
			if (class_flag[i][j] == 1) {
				if (mixture_image[i][j] < min1)
					min1 = mixture_image[i][j];
			}
			if (class_flag[i][j] == 2) {
				if (mixture_image[i][j] < min2)
					min2 = mixture_image[i][j];
			}
		}
		out << std::endl;
	}
	out.close();*/
	cout << "min1  " << min1 << "   " << "min2  " << min2 << endl;

	for (int i = 0; i < thr_nmb; i++) {
		delete[] buff[i];
		delete[] buff1[i];
	}
	delete[] buff;
	delete[] buff1;
	delete[] max;

}

//вычисление медианы

double SEM_games::find_med(double* window, int wind_size) {
	int med_index = (wind_size ) / 2-1;
	bool flag = true;
	int left = 0;
	int right = wind_size - 1;

	while (flag) {
		//cout << "ddd";
		std::pair<int, int> result = partition(window, left, right, med_index);
		if (result.first< med_index && result.second > med_index) {
			flag = false;

		}
		else {
			if (result.first > med_index)
				right = result.first;
			else {
				if (result.second < med_index)
					left = result.second;
			}
		}
	}
	return 	window[med_index];
}

//копирование участка изображения в одномерный массив

void SEM_games::copy_in_one_mass(double* image_one_mass , int x_c, int y_c, int x_l, int y_l) {
	int idx = 0;
	int i, j ;
	for (i = x_c; i < x_c+ x_l; ++i) {
		for (j = y_c; j < y_c + y_l; ++j) {
			image_one_mass[idx] = mixture_image[i][j];
			idx++;
		}
	}
}

//вычисление k-той порядковой статистики в массиве

double SEM_games::find_k_stat(double * data, int wind_size, int k_stat) {
	bool flag = true;
	int  left = 0;
	int  right = wind_size - 1;

	while (flag) {
		std::pair<int, int> result = partition(data, left, right, k_stat);
		if (result.first< k_stat && result.second > k_stat)
			flag = false;
		else {
			if (result.first > k_stat)
				right = result.first;
			else {
				if (result.second < k_stat)
					left = result.second;
			}
		}
	}
	
	return 	data[k_stat];
}
//double SEM_games::quick_sort(double* window, int wind_size) {
//	int med_index = (wind_size - 1) / 2;
//	bool flag = true;
//	int left = 0;
//	int right = wind_size - 1;
//
//	while (flag) {
//		//cout << "ddd";
//		std::pair<int, int> result = partition(window, left, right, med_index);
//		if (result.first< med_index && result.second > med_index) {
//			flag = false;
//
//		}
//		else {
//			if (result.first > med_index)
//				right = result.first;
//			else {
//				if (result.second < med_index)
//					left = result.second;
//			}
//		}
//	}
//	return 	window[med_index];
//}

// раскраска через хи-квадрат

int SEM_games::chi_square_stats(double* data, int data_size) {
	unsigned i, j;
	int k;
	bool flag = true;
	int buf_intervals_amount;
	//intervals_amount =20;
	/*double max_value = find_k_stat(mixture_image_one_mass, image_len*image_len,image_len*image_len - 1);
	double min_value = find_k_stat(mixture_image_one_mass, image_len*image_len, 0);*/
	double max_value = find_k_stat(data, data_size, data_size - 1) + 1;
	double min_value = find_k_stat(data, data_size, 0) - 1;
	double len_interval = (max_value - min_value) / intervals_amount;

	for (i = 0; i < intervals_amount; ++i)
		nu_i[i] = 0;
	for (i = 0; i < data_size; ++i) {
		k = 0;
		flag = true;
		while (flag) {
			if (((len_interval * k + min_value) <= data[i]) && ((len_interval * (k + 1) + min_value) > data[i]))
				flag = false;
			else
				k++;
		}
		nu_i[k] = nu_i[k] + 1;
	}
	flag = true;
	int r = data_size;
	int buf_nu = 0;
	buf_intervals_amount = 0;
	for (int j = 0; j < intervals_amount; ++j) {
		cout << "nu_i[k] " << nu_i[j] << endl;
		if ((nu_i[j] > 5) && flag) {
			if (j < intervals_amount - 1) {
				if ((r - nu_i[j] > 5)) {
					interval_bounds[buf_intervals_amount] = len_interval * (j + 1) + min_value;
					nu_i_bounds[buf_intervals_amount] = nu_i[j];
					r -= nu_i[j];
					buf_intervals_amount++;
				}
				else {
					flag = false;
					buf_nu = nu_i[j];
					r -= nu_i[j];
				}
			}
			else {
				interval_bounds[buf_intervals_amount] = len_interval * (j + 1) + min_value;
				nu_i_bounds[buf_intervals_amount] = nu_i[j];
				r -= nu_i[j];
				buf_intervals_amount++;
			}
		}
		else {
			if (flag) {
				buf_nu = nu_i[j];
				r -= nu_i[j];
				flag = false;
			}
			else {
				buf_nu += nu_i[j];
				r -= nu_i[j];
			}
			if (buf_nu > 5) {
				if (j < intervals_amount - 1) {
					if (r > 5)
					{
						cout << "j " << j << " buf_nu " << buf_nu << endl;
						interval_bounds[buf_intervals_amount] = len_interval * (j + 1) + min_value;
						nu_i_bounds[buf_intervals_amount] = buf_nu;
						buf_intervals_amount++;
						buf_nu = 0;

						flag = true;
					}
				}
				else {
					cout << "j " << j << " buf_nu " << buf_nu << endl;
					interval_bounds[buf_intervals_amount] = len_interval * (j + 1) + min_value;
					nu_i_bounds[buf_intervals_amount] = buf_nu;
					buf_intervals_amount++;
					buf_nu = 0;

					flag = true;
				}
			}
		}

	}

	

	cout << "buf_intervals_amount " << buf_intervals_amount << endl;
	for (int j = 0; j < buf_intervals_amount; ++j) {
		cout << " j " << interval_bounds[j] <<" "<< nu_i_bounds[j]<< endl;
	}

	auto chi_stat_computation = [&](double* shift, double* scale, int size_) {
		unsigned i, j;
		double chi_stat;
		double teor_nu;
	
		double quant_chi = quantile(chi_squared(intervals_amount - 1 - 1), 0.95);
		int* flag_mass = new int[size_];
		int flag_summ = 0;
		int flag_idx = 0;

		//cout << " quant_chi  - " << quant_chi << endl;

		for (i = 0; i < size_; ++i) {
			chi_stat = 0;
		
			for (j = 0; j < buf_intervals_amount; ++j) {
				if (j == 0) {
					if (mixture_type == "normal")
						teor_nu = cdf(normal(shift[i], scale[i]), interval_bounds[j]);
					if (mixture_type == "rayleigh")
						teor_nu = cdf(rayleigh(scale[i]), interval_bounds[j]);
				}
				else {
					if (j != buf_intervals_amount - 1) {
						if (mixture_type == "normal")
							teor_nu = cdf(normal(shift[i], scale[i]), interval_bounds[j])
							- cdf(normal(shift[i], scale[i]), interval_bounds[j-1]);
						if (mixture_type == "rayleigh")
							teor_nu = cdf(rayleigh(scale[i]), interval_bounds[j])
							- cdf(rayleigh(scale[i]), interval_bounds[j-1]);
					}
					else {
						if (mixture_type == "normal")
							teor_nu = 1 - cdf(normal(shift[i], scale[i]), interval_bounds[j - 1]);
						if (mixture_type == "rayleigh")
							teor_nu = 1 - cdf(rayleigh(scale[i]), interval_bounds[j - 1]);
					}
				}
				cout << "teor_nu " << teor_nu << endl;
				teor_nu = teor_nu * (data_size);
			
			
					chi_stat += (nu_i_bounds[j] - teor_nu)* (nu_i_bounds[j] - teor_nu) / teor_nu;
			
			}
			quant_chi = quantile(chi_squared(buf_intervals_amount - 1 - mix_params_amount), 0.95);
			cout << "classs " << i << ": chi_stat - " << chi_stat << " quant_chi = " << quant_chi << endl;
			if (chi_stat < quant_chi)
				flag_mass[i] = 1;
			else
				flag_mass[i] = 0;
		}
		for (i = 0; i < size_; ++i) {
			flag_summ += flag_mass[i];
			if (flag_mass[i] == 1)
				flag_idx = i;
		}
		delete[] flag_mass;
		if (flag_summ == 1)
			return flag_idx;
		else return -1;
};
	//auto chi_stat_computation = [&](double* shift, double* scale, int size_) {
	//	unsigned i, j;
	//	double chi_stat;
	//	double teor_nu;
	//	double buf_nu = 0;
	//	double buf_teor_nu = 0;
	//	bool nu_flag = true;
	//	double quant_chi = quantile(chi_squared(intervals_amount - 1 - 1), 0.95);
	//	int* flag_mass = new int[size_];
	//	int flag_summ = 0;
	//	int flag_idx = 0;
	//	int buf_intervals_amount;
	//	//cout << " quant_chi  - " << quant_chi << endl;

	//	for (i = 0; i < size_; ++i) {
	//		chi_stat = 0;
	//		buf_intervals_amount = 1;
	//		buf_nu = 0;
	//		buf_teor_nu = 0;
	//		nu_flag = true;
	//		for (j = 0; j < intervals_amount; ++j) {
	//			if (j == 0) {
	//				if (mixture_type == "normal")
	//					teor_nu = cdf(normal(shift[i], scale[i]), len_interval*(j + 1) + min_value);
	//				if (mixture_type == "rayleigh")
	//					teor_nu = cdf(rayleigh(scale[i]), len_interval*(j + 1) + min_value);
	//			}
	//			else {
	//				if (j != intervals_amount - 1) {
	//					if (mixture_type == "normal")
	//					teor_nu = cdf(normal(shift[i], scale[i]), len_interval *(j + 1) + min_value)
	//						- cdf(normal(shift[i], scale[i]), len_interval * (j)+min_value);
	//					if (mixture_type == "rayleigh")
	//						teor_nu = cdf(rayleigh(scale[i]), len_interval *(j + 1) + min_value)
	//						- cdf(rayleigh(scale[i]), len_interval * (j)+min_value);
	//				}
	//				else {
	//					if (mixture_type == "normal")
	//					teor_nu = 1 - cdf(normal(shift[i], scale[i]), len_interval * (j)+min_value);
	//					if (mixture_type == "rayleigh")
	//						teor_nu = 1 - cdf(rayleigh(scale[i]), len_interval * (j)+min_value);
	//				}
	//			}
	//			//cout << "teor_nu " << teor_nu << endl;
	//			teor_nu = teor_nu * (data_size);
	//			if ((j == intervals_amount - 2) && (nu_i[j + 1] <= 5)) {
	//				nu_flag = false;
	//			}
	//			if ((nu_i[j] > 5) && nu_flag) {
	//				chi_stat += (nu_i[j] - teor_nu)* (nu_i[j] - teor_nu) / teor_nu;
	//				buf_intervals_amount++;
	//			}
	//			else {
	//				if (nu_flag) {
	//					buf_nu = nu_i[j];
	//					buf_teor_nu = teor_nu;
	//					nu_flag = false;
	//				}
	//				else {
	//					buf_nu += nu_i[j];
	//					//buf_intervals_amount--;
	//					buf_teor_nu += teor_nu;
	//				}
	//				if (buf_nu > 5) {
	//					if ((j == intervals_amount - 2) && (nu_i[j + 1] <= 5)) {
	//						//cout << "nice" << endl;
	//					}
	//					else {
	//						chi_stat += (buf_nu - buf_teor_nu)* (buf_nu - buf_teor_nu) / buf_teor_nu;
	//						cout << "j " << j << " buf_nu " << buf_nu << endl;
	//						buf_intervals_amount++;
	//						buf_nu = 0;
	//						buf_teor_nu = 0;
	//						nu_flag = true;
	//					}
	//				}
	//			}
	//		}
	//		quant_chi = quantile(chi_squared(buf_intervals_amount - 1 - 2), 0.95);
	//		cout << "classs " << i << ": chi_stat - " << chi_stat << " quant_chi = "<< quant_chi<<endl;
	//		if (chi_stat < quant_chi)
	//			flag_mass[i] = 1;
	//		else
	//			flag_mass[i] = 0;
	//	}
	//	for (i = 0; i < size_; ++i) {
	//		flag_summ += flag_mass[i];
	//		if(flag_mass[i] == 1)
	//			flag_idx = i;
	//	}
	//	delete[] flag_mass;
	//	if (flag_summ == 1)
	//		return flag_idx;
	//	else return -1;
	//};
	return chi_stat_computation(mix_shift, mix_scale, hyp_cl_amount);
}

//вычисление критерия колмогорова

int SEM_games::kolmogorov_stats(double* data, int data_size, double* mix_shift, double* mix_scale, int  hyp_cl_amount) {
	unsigned i, j;
	int k, buf_intervals_amount;
	bool flag = true;
	double max_d_n, buf_d_n, F_n_curr;
	int* flag_mass = new int[hyp_cl_amount];
	double* stats_mass = new double[hyp_cl_amount];
	int flag_summ = 0;
	int flag_idx = 0;
	double max_value = find_k_stat(data, data_size, data_size - 1) + 1;
	double min_value = find_k_stat(data, data_size, 0);
	double len_interval = (max_value - min_value) / intervals_amount;

	double *max_L_mass = new double[hyp_cl_amount];
	for (i = 0; i < intervals_amount; ++i)
		nu_i[i] = 0;
	for (i = 0; i < data_size; ++i) {
		k = 0;
		flag = true;
		while (flag) {
			if (((len_interval * k + min_value) <= data[i]) && ((len_interval * (k + 1) + min_value) > data[i]))
				flag = false;
			else
				k++;
		}
		nu_i[k] = nu_i[k] + 1;
	}
	
	auto L_max_calculation = [&](int iter, double mix_shift, double mix_scale) {
		double buf_max_l = 0;
		bool flag = false;
		double B;
		for (int m = 0; m < data_size; m++) {
			B = 0;
			if (mix_scale != 0) {
				if (mixture_type == "normal")
					B = (1.0 / (mix_scale))*
					exp(-(pow(data[m] - mix_shift, 2)) /
					(2.0 * mix_scale * mix_scale));
				if (mixture_type == "rayleigh")
					B = (data[m] / pow(mix_scale, 2))*
					exp(-(pow(data[m], 2)) /
					(2.0 * mix_scale * mix_scale));
			}
			else {
				flag = true;
				break;
			}
			
			if (B > 0)
				buf_max_l = buf_max_l + log(B / (data_size));
		}
		
		max_L_mass[iter] = buf_max_l;
	};

	
	
	double dn_bound =0.264 ;
	//double dn_bound = 0.238;

	
	for (i = 0; i < hyp_cl_amount; ++i) {
		F_n_curr = 0;
		max_d_n = 0;
		if (mixture_type == "normal")
			max_d_n = cdf(normal(mix_shift[i], mix_scale[i]), min_value);
		if (mixture_type == "rayleigh")
			max_d_n = cdf(rayleigh(mix_scale[i]), min_value);

		for (j = 0; j < intervals_amount; ++j) {
			F_n_curr += nu_i[j]/ data_size;
			if (j != intervals_amount - 1) {
				for ( k = 0; k < 2; ++k) {
					if (mixture_type == "normal")
						buf_d_n = abs(cdf(normal(mix_shift[i], mix_scale[i]), (len_interval *(j + k) + min_value))- F_n_curr);
					if (mixture_type == "rayleigh")
						buf_d_n = abs(cdf(rayleigh(mix_scale[i]), (len_interval *(j + k) + min_value))- F_n_curr);
					if (buf_d_n > max_d_n)
						max_d_n = buf_d_n;
				}
			}
			else {
				if (mixture_type == "normal")
					buf_d_n = abs(cdf(normal(mix_shift[i], mix_scale[i]), (len_interval *(j +1) + min_value)) - 1);
				if (mixture_type == "rayleigh")
					buf_d_n = abs(cdf(rayleigh(mix_scale[i]), (len_interval *(j + 1) + min_value)) - 1);
				if (buf_d_n > max_d_n)
					max_d_n = buf_d_n;
			}
			

		}
		
		//cout << "classs " << i << ": max_d_n - " << max_d_n << "  dn_bound = " << dn_bound << endl;
		stats_mass[i] = max_d_n;
		if (max_d_n < dn_bound)
			flag_mass[i] = 1;
		else
			flag_mass[i] = 0;
	}
	for (i = 0; i < hyp_cl_amount; ++i) {
		flag_summ += flag_mass[i];
		if (flag_mass[i] == 1)
			flag_idx = i;
	}
	if (flag_summ > 1) {
		cout << "conflict! bound value - " << dn_bound<< endl;
		for (int i = 0; i < hyp_cl_amount; ++i) {
			cout << "class " << i << " - " << stats_mass[i] << " , l_stats = ";
			L_max_calculation(i, mix_shift[i], mix_scale[i]);
			cout << max_L_mass[i] << endl;
		}
		int beg_idx = 0;
		int max_idx = 0;

		for (int m = beg_idx; m < hyp_cl_amount; ++m) {
			if (max_L_mass[max_idx] < max_L_mass[m])
				max_idx = m;
		}
		flag_summ = 1;
		flag_idx = max_idx;
	}
	delete[] flag_mass;
	delete[] stats_mass;
	delete[] max_L_mass;

	if (flag_summ == 1)
		return flag_idx;
	else return -1;
}

// процедура partition для вычисления прядковых статистик

std::pair<int, int> SEM_games::partition(double* mass, int left, int right, int  ind_pivot) {
	double pivot = mass[ind_pivot];
	double buf = mass[left];
	mass[left] = pivot;
	mass[ind_pivot] = buf;
	int j = left;
	int	k = right + 1;
	int	iter_l = left + 1;
	int	iter_r = right;

	while (iter_l <= iter_r) {
		while (iter_l <= iter_r && mass[iter_l] < mass[left]) {
			if (j == iter_l - 1) {
				j++;
				iter_l++;
			}
			else {
				j++;
				buf = mass[iter_l];
				mass[iter_l] = mass[j];
				mass[j] = buf;
				iter_l++;
			}
		}
		while (iter_l <= iter_r && mass[iter_r] > mass[left]) {
			if (k == iter_r + 1) {
				k -= 1;
				iter_r -= 1;
			}
			else {
				k -= 1;
				buf = mass[iter_r];
				mass[iter_r] = mass[k];
				mass[k] = buf;
				iter_r -= 1;
			}
		}
		if (iter_l <= iter_r) {
			if (mass[iter_l] != mass[left] && mass[iter_r] != mass[left]) {
				buf = mass[iter_r];
				mass[iter_r] = mass[iter_l];
				mass[iter_l] = buf;
				j++;
				buf = mass[iter_l];
				mass[iter_l] = mass[j];
				mass[j] = buf;
				iter_l++;
				k -= 1;
				buf = mass[iter_r];
				mass[iter_r] = mass[k];
				mass[k] = buf;
				iter_r -= 1;
			}
			else {
				if (mass[iter_l] == mass[left] && mass[iter_r] != mass[left]) {
					buf = mass[iter_r];
					mass[iter_r] = mass[iter_l];
					mass[iter_l] = buf;
					j++;
					buf = mass[j];
					mass[j] = mass[iter_l];
					mass[iter_l] = buf;
					iter_l++;
				}
				else {
					if (mass[iter_l] != mass[left] && mass[iter_r] == mass[left]) {
						buf = mass[iter_r];
						mass[iter_r] = mass[iter_l];
						mass[iter_l] = buf;
						k -= 1;
						buf = mass[iter_r];
						mass[iter_r] = mass[k];
						mass[k] = buf;
						iter_r -= 1;
					}
					else {
						if (mass[iter_l] == mass[left] && mass[iter_r] == mass[left]) {
							iter_r -= 1;
							while (iter_l <= iter_r && mass[iter_r] == mass[left])
								iter_r -= 1;
							if (iter_l < iter_r) {
								if (mass[iter_r] < mass[left]) {
									buf = mass[iter_r];
									mass[iter_r] = mass[j + 1];
									mass[j + 1] = buf;
									j++;
									iter_l++;
								}
								else {
									buf = mass[iter_r];
									mass[iter_r] = mass[k - 1];
									mass[k - 1] = buf;
									k -= 1;
									iter_r -= 1;
								}
							}
						}
					}
				}
			}
		}
	}

	buf = mass[left];
	mass[left] = mass[j];
	mass[j] = buf;

	return std::pair<int, int>(j - 1, k);
}

// вычисление bic

void SEM_games::BIC() {
	double summ = 0;
	double big_summ = 0;
	long double f_i_j, pix_buf;
	unsigned i, j, idx_i, idx_j;
	
	cout << "mix_weight[j] " << mix_weight[0] << endl;
	for (i = 0; i < image_len*image_len; ++i) {
		idx_i = i / image_len;
		idx_j = i % image_len;
		summ = 0;
		pix_buf = mixture_image[idx_i][idx_j];
		for (j = 0; j < hyp_cl_amount; ++j)
			summ += mix_weight[j] * (1 / (mix_scale[j] * sqrt(2 * pi)))*exp(-((pix_buf
				- mix_shift[j])*(pix_buf - mix_shift[j])) / (2.0 * mix_scale[j] * mix_scale[j]));

		big_summ += log(summ);
	}
	unsigned count_n_z = 0;
	/*for (j = 0; j < hyp_cl_amount; ++j)
		if (mix_weight[j] != 0)
			++count_n_z;*/
	count_n_z = hyp_cl_amount;
	double bic = +2 * big_summ - log(image_len*image_len)*(3 * count_n_z - 1);

	cout << "BIC:  " << bic << "     " << big_summ << "\n" << "\n" << endl;
}

//оценка точности - ошибка 2го рода. распараллеливание через std:: thread

void SEM_games::detect_results() {
	int thr_nmb = 5;
	std::thread threadObj1(&SEM_games::th_detect_results, this, 0, image_len / thr_nmb);
	std::thread threadObj2(&SEM_games::th_detect_results, this, image_len / thr_nmb, 2 * image_len / thr_nmb);
	std::thread threadObj3(&SEM_games::th_detect_results, this, 2 * image_len / thr_nmb, 3 * image_len / thr_nmb);
	std::thread threadObj4(&SEM_games::th_detect_results, this, 3 * image_len / thr_nmb, 4 * image_len / thr_nmb);
	std::thread threadObj5(&SEM_games::th_detect_results, this, 4 * image_len / thr_nmb, image_len);
	threadObj1.join();
	threadObj2.join();
	threadObj3.join();
	threadObj4.join();
	threadObj5.join();

	int trg_pix = 0;
	for (int i = 0; i < amount_trg*amount_trg; i++)
		trg_pix += targs[i].size*targs[i].size;

	cout << "real far: " << rfar / (image_len*image_len - trg_pix) << endl;
	cout << "all_mistakes: " << all_mistakes << endl;
	cout << "mistakes for each mixture: ";
	for (int i = 0; i < class_amount + 1; i++) {
		cout << mistake_mix[i] / all_mistakes << "  ";
	}
	cout << endl;
}

//обработка результатов -  версия для потока std:: thread

void SEM_games::th_detect_results(int beg, int end) {
	int l_numb = 0;
	float n_hit = 0.0;
	float n_miss = 0.0;
	float n_detect = 0.0;
	int   n_block = 0;

	for (int i = 0; i < amount_trg; i++) {
		n_hit = 0.0;
		n_miss = 0.0;
		n_detect = 0.0;

		for (int j = beg; j < end; j++) {
			n_block = 0;
			while (j < n_block + 1 && j >= n_block)
				n_block++;
			for (int k = i * 110; k < (i + 1) * 110; k++) {
				double x_min = targs[n_block*amount_trg + i].x;
				double x_max = x_min + targs[n_block*amount_trg + i].size;
				double y_min = targs[n_block*amount_trg + i].y;
				double y_max = y_min + targs[n_block*amount_trg + i].size;

				if (j < x_max &&j >= x_min) {
					if (k < y_max &&k >= y_min) {
						if (class_flag[j][k] > 1) {
							rfar++;
							all_mistakes++;
							mistake_mix[class_flag[j][k] - 1] ++;
						}
					}
					else {
						if (class_flag[j][k] != targs[n_block*amount_trg + i].mix_type) {
							all_mistakes++;
							mistake_mix[class_flag[j][k] - 1] ++;
						}
					}
				}
			}
		}
	}
}

//вычисление отклонений полученных оценок от реальных распределений - для того, чтобы можно было провести
// между ними сопоставление и вычислить ошибку второго рода

void SEM_games::dist_computation() {
	unsigned comp_numb = 0;
	double min_dist, buf_dist;
	for (int i = 0; i < hyp_cl_amount; i++) {
		min_dist = sqrt((mix_shift[0] - re_mix_shift[i])*(mix_shift[0] - re_mix_shift[i])
			+ (mix_scale[0] - re_mix_scale[i])*(mix_scale[0] - re_mix_scale[i]));

		for (int j = 0; j < hyp_cl_amount; ++j) {
			buf_dist = sqrt((mix_shift[j] - re_mix_shift[i])*(mix_shift[j] - re_mix_shift[i])
				+ (mix_scale[j] - re_mix_scale[i])*(mix_scale[j] - re_mix_scale[i]));
			if (min_dist >= buf_dist) {
				min_dist = buf_dist;
				comp_numb = j + 1;
			}
		}
		buf_numbs[i] = comp_numb;
	}

	cout << " relative error: " << endl;
	cout << " shifts: " <<abs(re_mix_shift[0]- mix_shift[buf_numbs[0]-1])/ re_mix_shift[0] <<" , "<< abs(re_mix_shift[1] - mix_shift[buf_numbs[1]-1]) / re_mix_shift[1] << endl;
	cout << " scales: " << abs(re_mix_scale[0] - mix_scale[buf_numbs[0]-1]) / re_mix_scale[0] << " , " << abs(re_mix_scale[1] - mix_scale[buf_numbs[1]-1]) / re_mix_scale[1] << endl;
}

// вычисление ошибки 2го рода - распараллеливание чере open mp

void SEM_games::FAR_computation() {
	mistake_mix = new float[hyp_cl_amount];
	for (int i = 0; i < hyp_cl_amount; i++)
		mistake_mix[i] = 0;
	rfar = 0;
	int l_numb = 0;
	float n_hit = 0.0;
	float n_miss = 0.0;
	float n_detect = 0.0;
	double x_min, x_max, y_min, y_max;
	x_min = targs[0].x;
	x_max = x_min + targs[0].size;
	y_min = targs[0].y;
	y_max = y_min + targs[0].size;
	int   n_block = 0;
	
	/*n_hit = 0.0;
	n_miss = 0.0;
	n_detect = 0.0;*/
	
    #pragma omp parallel
	{
		#pragma omp for
		for (int j = 0; j < image_len; j++) {
			for (int k = 0; k < image_len; k++) {
				#pragma omp critical
				{
					if (j < x_max &&j >= x_min) {
						if (k < y_max &&k >= y_min) {
							if (class_flag[j][k] != buf_numbs[targs[0].mix_type - 1]) {
								all_mistakes++;
								mistake_mix[class_flag[j][k] - 1] ++;
							}
						}
						else {
							if (class_flag[j][k] != buf_numbs[0]) {
								rfar += 1;
								all_mistakes++;
								mistake_mix[class_flag[j][k] - 1] ++;
							}
						}
					}
					else {
						if (class_flag[j][k] != buf_numbs[0]) {
							rfar += 1;
							all_mistakes++;
							mistake_mix[class_flag[j][k] - 1] ++;
						}
					}
				}
			}
		}
	}
	 
	
	cout << "real far: " << rfar/ (image_len*image_len - targs[0].size * targs[0].size) << endl;
	cout << "all_mistakes: " << all_mistakes << endl;
	cout << "mistakes for each mixture: ";
	for (int i = 0; i < hyp_cl_amount; i++) {
		cout << mistake_mix[i] / all_mistakes << "  ";
	}
	cout << endl;
	delete[] mistake_mix;
	
}

// отрисовка на python

void SEM_games::draw_graphics() {
	cout << "end 2" << endl;
	string cmd = "echo python  C:\\Users\\anastasya\\PycharmProjects\\untitled5\\mixture_vizualization.py " + filename_gen_image + " " + filename_split_image +
		" | %windir%\\system32\\cmd.exe \"/K\" C:\\Users\\anastasya\\Anaconda3\\Scripts\\activate.bat  ";
	system(cmd.c_str());
}

//вычисение среднего арифметического

int SEM_games::mean(double** data) {
	double result = 0;
	for (int k = 0; k < image_len; k++) {
		for (int l = 0; l < image_len; l++)
			result += data[k][l];
	}
	return int(result / (image_len*image_len));
}

SEM_games::~SEM_games()
{
	cout << "dell" << endl;
	for (int i = 0; i < image_len*image_len; i++) {
		if (i < image_len)
			delete[] mixture_image[i];
		delete[] g_i_j_0[i];
		delete[] g_i_j[i];
	}

	delete[] mixture_image;
	delete[] g_i_j_0;
	delete[] g_i_j;
	delete[] x_coords;
	delete[] y_coords;
	for (int i = 0; i < intervals_amount; ++i)
		cout << nu_i[i] << "  ";
	delete[] nu_i;
	delete[] buf_numbs;
	/*delete[] re_mix_shift;
	delete[] re_mix_scale;*/
	delete[] mix_shift;
	delete[] mix_scale;
	delete[] mix_weight;

}
