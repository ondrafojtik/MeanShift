#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <float.h>
#include <chrono>
#include <vector>
#include <thread>
#include <functional>
#include <omp.h>
#include <random>
#include <iomanip>

#define M_PI       3.14159265358979323846   // pi


// expected-2
//const int dims = 4;
//const int n = 2048;
//const std::string filepath = "g2-4-50.txt";
//const double same_cluster_threshold = 0.65;
//const double cluster_aoe = 2.0;

// expected-15
const int dims = 2;
const int n = 5000;
const std::string filepath = "s1.txt";
const double same_cluster_threshold = 0.21;
const double cluster_aoe = 0.1;

// expected-16
//const int dims = 32;
//const int n = 1024;
//const std::string filepath = "dim032.txt";
//const double same_cluster_threshold = 1.38;
//const double cluster_aoe = 0.1;

// expected-100
//const int dims = 2;
//const int n = 100000;
//const std::string filepath = "birch2.txt";
//const double same_cluster_threshold = 0.06;
//const double cluster_aoe = 0.01;

// expected-100
//const int dims = 2;
//const int n = 100000;
//const std::string filepath = "birch3.txt";
//const double same_cluster_threshold = 0.085;
//const double cluster_aoe = 0.01;


struct Point
{
	double coords[dims];

	Point(double number)
	{
		for (int i = 0; i < dims; i++)
			coords[i] = number;
	}
};

double compute_distance(Point p1, Point p2)
{
	double sum = 0;

	for (int dim = 0; dim < dims; dim++)
	{
		double dif = p1.coords[dim] - p2.coords[dim];
		sum += pow(dif, 2);
	}

	return sqrt(sum);
}

struct DataStructure
{
	std::vector<Point> points;
	int* same_cluster_to_calc = (int*)malloc(sizeof(int) * n);

	~DataStructure() { free(same_cluster_to_calc); }

	void add2(Point p)
	{
		int same_cluster = 0;
		for (int i = 0; i < points.size(); i++)
		{
			double d = compute_distance(p, points.at(i));
			if (d <= same_cluster_threshold) same_cluster += 1;
		}
		if (same_cluster == 0) points.push_back(p);
	}

	// faster with more clusteres (to test -> lower "same_cluster_threshold")
	void add(Point p)
	{
		int _n = this->points.size();
		int same_cluster = 0;

		#pragma omp parallel for
		for (int i = 0; i < _n; i++)
		{
			double d = compute_distance(p, points.at(i));
			if (d <= same_cluster_threshold) same_cluster_to_calc[i] = 1;
			else same_cluster_to_calc[i] = 0;
		}

		#pragma omp simd reduction(+, same_cluster)
		for (int i = 0; i < _n; i++)
		{
			same_cluster += same_cluster_to_calc[i];
		}

		if (same_cluster == 0) points.push_back(p);

	}

};

std::chrono::high_resolution_clock::time_point t1;
std::chrono::high_resolution_clock::time_point t2;


void start_timer() {
	//std::cout << "Timer started" << std::endl;
	t1 = std::chrono::high_resolution_clock::now();
}

void end_timer() {
	t2 = std::chrono::high_resolution_clock::now();
	//std::cout << "Timer ended" << std::endl;
}
void print_elapsed_time() {
	std::cout << "\nTime elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " milliseconds\n" << std::endl;
}



double* points_x = (double*)malloc(sizeof(double) * n);
double* points_y = (double*)malloc(sizeof(double) * n);
Point* points = (Point*)malloc(sizeof(Point) * n);
double x_min = DBL_MAX;
double x_max = DBL_MIN;
double y_min = DBL_MAX;
double y_max = DBL_MIN;

double* kernel_upper_x = (double*)malloc(sizeof(double) * n);
double* kernel_upper_y = (double*)malloc(sizeof(double) * n);
Point* kernel_upper = (Point*)malloc(sizeof(Point) * n);
double* kernel_bottom = (double*)malloc(sizeof(double) * n);

double cluster_x;
double cluster_y;

Point cluster(0);
//Point* clusters = (Point*)malloc(sizeof(Point) * n);

DataStructure data_structure;

bool flag = true;


static double lerp(double value, double min, double max)
{
	double val = (value - min) / (max - min);
	return val;
}

// alpha = compute_distance(x1 - x2)
double kernel(double alpha)
{
	double bot = pow(sqrt(2 * M_PI), dims);
	double exp_exp = pow(alpha, 2) / 2;
	exp_exp = -1 * exp_exp;

	double result = (1 / bot) * exp(exp_exp);
	return result;
}

void preproces_kernels()
{
	#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		double d = compute_distance(points[i], cluster);


		if (d > cluster_aoe)
		{
			for (int dim = 0; dim < dims; dim++)
				kernel_upper[i].coords[dim] = 0;

			kernel_bottom[i] = 0;
		}
		else
		{
			double k = kernel(d);

			for (int dim = 0; dim < dims; dim++)
				kernel_upper[i].coords[dim] = k * points[i].coords[dim];

			kernel_bottom[i] = k;
		}

	}

}

Point compute_shift()
{
	Point p(0);
	double sum_bottom = 0;
	#pragma omp simd reduction(+, sum_bottom)
	for (int i = 0; i < n; i++)
	{
		sum_bottom += kernel_bottom[i];
	}

	for (int dim = 0; dim < dims; dim++)
	{
		double sum_kernel_upper_per_dim = 0;

	#pragma omp simd reduction(+, sum_kernel_upper_per_dim)
		for (int i = 0; i < n; i++)
		{
			sum_kernel_upper_per_dim += kernel_upper[i].coords[dim];
		}

		p.coords[dim] = sum_kernel_upper_per_dim / sum_bottom;

	}

	return p;
}

void mean_shift()
{

	preproces_kernels();

	Point p = compute_shift();

	double fault_rate = 0;
	if (abs(cluster.coords[0] - p.coords[0]) <= fault_rate && abs(cluster.coords[1] - p.coords[1]) <= fault_rate)
		flag = false;
	else
		flag = true;

	cluster = p;

}

void read_file(std::string fname)
{
	std::ifstream file(fname);

	int index = 0;

	if (file.is_open())
	{

		std::string line;


		while (std::getline(file, line)) {

			std::stringstream sin(line);

			for (int i = 0; i < dims; i++)
			{
				double value;
				sin >> value;
				points[index].coords[i] = value;
			}

			index += 1;
		}

		file.close();

	}
	else
	{
		std::cout << fname << " file not open" << std::endl;
		return;
	}

	// normalize
	for (int dim = 0; dim < dims; dim++)
	{
		double dim_min = DBL_MAX;
		double dim_max = DBL_MIN;

		#pragma omp simd reduction(min:dim_min)
		for (int i = 0; i < n; i++)
		{
			if (points[i].coords[dim] < dim_min) dim_min = points[i].coords[dim];
		}

		#pragma omp simd reduction(max:dim_max)
		for (int i = 0; i < n; i++)
		{
			if (points[i].coords[dim] > dim_max) dim_max = points[i].coords[dim];
		}

		#pragma omp parallel for
		for (int i = 0; i < n; i++)
		{
			points[i].coords[dim] = lerp(points[i].coords[dim], dim_min, dim_max);
		}

	}

	// algo
	for (int cluster_id = 0; cluster_id < n; cluster_id++)
	{

		cluster = points[cluster_id];

		while (flag)
		{
			flag = false;
			mean_shift();
		}

		data_structure.add(cluster);

	}

	end_timer();

	std::cout << "Number of clusteres: " << data_structure.points.size() << std::endl;

	for (int i = 0; i < data_structure.points.size(); i++)
	{
		for (int dim = 0; dim < dims; dim++)
			std::cout << std::setprecision(6) << std::fixed << data_structure.points[i].coords[dim] << ", ";
		std::cout << std::endl;
	}

}

int main()
{
	start_timer();
	read_file(filepath);
	print_elapsed_time();
	return 0;
}
