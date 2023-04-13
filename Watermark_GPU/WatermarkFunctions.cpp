#pragma warning(disable:4996)
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include "UtilityFunctions.h"
#include "WatermarkFunctions.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <arrayfire.h>
#include <af/opencl.h>
#include <cmath>
#include <vector>

//min και max είναι defined ως macro στα windows, οπότε οποιαδήποτε συνάρτηση ονομάζεται min/max δε θα δουλέψει
//ακόμα και αν εφαρμόζουμε μπροστά το namespace, για αυτό τα κάνουμε undefine
#undef min
#undef max

using std::cout;

//συνάρτηση που διαβάζει τον W πίνακα
af::array load_W(std::string w_file, const int rows, const int cols) {
	float* w_ptr = NULL;
	int i = 0;
	std::ifstream w_stream;

	w_stream.open(w_file.c_str(), std::ios::binary);
	if (!w_stream.is_open()) {
		cout << "\nError opening \"" << w_file << "\" file.";
		system("pause");
		exit(-1);
	}
	w_ptr = new float[rows * cols];
	while (!w_stream.eof()) {
		w_stream.read(reinterpret_cast<char*>(&w_ptr[i]), sizeof(float));
		i++;
	}
	af::array w = af::array(cols, rows, w_ptr).as(af::dtype::f16);
	w = af::transpose(w);
	delete[] w_ptr;
	return w;
}

void compute_NVF_mask(af::array& image, af::array& padded, af::array& m, const int p, const int pad, const dim_t rows, const dim_t cols, cl::CommandQueue& queue, cl::Context& context, cl::Program& program_nvf)
{
	size_t padded_rows = rows + 2 * pad;
	const size_t padded_cols = cols + 2 * pad;
	const size_t elems = rows * cols;
	//εκτέλεση kernel
	try {
		cl_int err = 0;
		cl::Buffer padded_buff(context, CL_MEM_READ_ONLY, sizeof(cl_half) * padded_rows * padded_cols, NULL, &err);
		padded_buff = *padded.device<cl_mem>();

		//αντιγραφή της εικόνας μέσω του Buffer ο οποίος αντιγράφτηκε από το padded array (gpu->gpu), αντι να αντιγράψουμε
		//σε pointer στη CPU και ξανα πίσω στη GPU
		cl::Image2D padded_image2d(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_LUMINANCE, CL_HALF_FLOAT), padded_cols, padded_rows, 0, NULL, &err);
		size_t orig[] = { 0,0,0 };
		size_t des[] = { padded_cols, padded_rows, 1 };
		err = clEnqueueCopyBufferToImage(queue(), padded_buff(), padded_image2d(), 0, orig, des, NULL, NULL, NULL);
		cl::Buffer nvf_buff(context, CL_MEM_WRITE_ONLY, sizeof(cl_half) * elems, NULL, &err);

		//δημιουργία, αρχικοποίηση και τρέξιμο του kernel
		cl::Kernel kernel = cl::Kernel(program_nvf, "nvf", &err);
		err = kernel.setArg(0, padded_image2d);
		err = kernel.setArg(1, nvf_buff);
		err = kernel.setArg(2, p);

		//τρέξιμο του kernel
		//fix για NVIDIA (OpenCL 1.2): Πρέπει GlobalGroupSize % LocalGroupSize == 0. Απλώς κάνουμε padding στο GlobalGroupSize (πχ στις γραμμες)
		//ωστε να ισχυει η συνθηκη (πχ image width=3620, height=2028 -> globalSize= 7341360, localSize (πχ) = 256. 7341360%256= 48, οποτε ΔΕΝ θα τρεξει ο kernel!
		//θα κανουμε padding σε ενα απο τα στοιχεια της εικονας (πχ height) την τιμή -> localSize-(height%localSize) -> 256-(3620%256) = 256-36 = 220
		//οποτε πλεον height = 3620 + 220 = 3840, οποτε (3840*2028)%256 = 0
		//BOUND CHECKING γινεται μεσα στον kernel, οποτε ειμαστε ΟΚ που το global εχει out of bound τιμες
		if (padded_cols * padded_rows % UtilityFunctions::max_workgroup_size != 0)
			padded_rows += UtilityFunctions::max_workgroup_size - (padded_rows % UtilityFunctions::max_workgroup_size);

		err = queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(padded_cols, padded_rows), cl::NDRange(1, UtilityFunctions::max_workgroup_size));
		queue.finish();
		//γρήγορο διάβασμα της εικόνας που επιστρέφει το kernel
		m = afcl::array(rows, cols, nvf_buff(), af::dtype::f16, true);
	}

	catch (const af::exception & e) {
		cout << "ERROR in compute_nvf_mask(): " << e.what() << "\n";
	}
	catch (const std::exception & ex) {
		cout << "ERROR in compute_nvf_mask(): " << ex.what() << "\n";
	}
}
//συνάρτηση που ενθέτει το watermark με χρήση της NVF mask και το επιστρέφει.
af::array make_and_add_watermark_NVF(af::array& image, const af::array& w, const int p, const float psnr, float* a, cl::CommandQueue& queue, cl::Context& context, cl::Program& program_nvf)
{
	//για να μην κάνουμε query συνέχεια τις στήλες/γραμμές, τις αποθηκεύουμε
	const auto rows = image.dims(0);
	const auto cols = image.dims(1);
	const auto elems = rows * cols;

	//padding
	const int pad = p / 2;
	const auto padded_rows = rows + 2 * pad;
	const auto padded_cols = cols + 2 * pad;
	af::array padded = af::constant(0, padded_cols, padded_rows, af::dtype::f16);
	padded(af::seq(static_cast<double>(pad), static_cast<double>(padded_cols - pad - 1)), af::seq(static_cast<double>(pad), static_cast<double>(padded_rows - pad - 1))) = image.T();
	af::array m_nvf;
	compute_NVF_mask(image, padded, m_nvf, p, pad, rows, cols, queue, context, program_nvf);

	//η τελική μάσκα u = m*w (point wise πολλαπλασιασμός)
	af::array u = m_nvf * w;

	//υπολογισμός του α παραμέτρου
	float divv = std::sqrt(af::sum<float>(af::pow(u, 2)) / (elems));
	*a = (255.0f / std::sqrt(std::pow(10.0f, psnr / 10.0f))) / divv;

	//ένθεση υδατογραφήματος
	af::array y = image + (*a * u);

	//επιστροφή υδατογραφήματος
	return y;

	//αν πεταχτεί exception τότε θα φτάσουμε εδώ, επιστρέφουμε null array
	return NULL;
}

//συνάρτηση που ενθέτει το υδατογράφημα στην original_image με χρήση της ME mask  και το επιστρέφει.
af::array make_and_add_watermark_ME(af::array& image, const af::array& w, const int p, const float psnr, float* a, cl::CommandQueue& queue, cl::Context& context, cl::Program& program)
{
	//για να μην κάνουμε query συνέχεια τις στήλες/γραμμές, τις αποθηκεύουμε
	const auto rows = image.dims(0);
	const auto cols = image.dims(1);
	const auto elems = rows * cols;

	//padding
	const int pad = p / 2;
	const auto padded_rows = rows + 2 * pad;
	const auto padded_cols = cols + 2 * pad;
	af::array padded = af::constant(0, padded_cols, padded_rows, af::dtype::f16);
	padded(af::seq(static_cast<double>(pad), static_cast<double>(padded_cols - pad - 1)), af::seq(static_cast<double>(pad), static_cast<double>(padded_rows - pad - 1))) = image.T();

	//εδώ γίνεται ο υπολογισμός της μάσκας (μάσκα χρειάζεται αλλά όχι το φιλτρο)
	af::array m_e, e_x, a_x;
	compute_ME_mask(image, padded, m_e, e_x, a_x, p, pad, rows, cols, queue, context, program, true);

	//η τελική μάσκα u = m*w (point wise πολλαπλασιασμός)
	af::array u = m_e * w;

	//υπολογισμός του strength
	float divv = std::sqrt(af::sum<float>(af::pow(u, 2)) / elems);
	*a = (255.0f / std::sqrt(std::pow(10.0f, psnr / 10.0f))) / divv;

	//ένθεση υδατογραφήματος
	af::array y = image + (*a * u);

	//επιστροφή υδατογραφήματος
	return y;
}


//overloaded ώστε να επιστρέφει το φίλτρο και τη γειτονιά
af::array make_and_add_watermark_ME(af::array& image, const af::array& w, af::array& a_x, const int p, const float psnr, float* a, cl::CommandQueue& queue, cl::Context& context, cl::Program& program)
{
	const auto rows = image.dims(0);
	const auto cols = image.dims(1);
	const auto elems = rows * cols;
	const int pad = p / 2;
	const auto padded_rows = rows + 2 * pad;
	const auto padded_cols = cols + 2 * pad;
	af::array padded = af::constant(0, padded_cols, padded_rows, af::dtype::f16);
	padded(af::seq(static_cast<double>(pad), static_cast<double>(padded_cols - pad - 1)), af::seq(static_cast<double>(pad), static_cast<double>(padded_rows - pad - 1))) = image.T();
	af::array m_e, e_x;
	compute_ME_mask(image, padded, m_e, e_x, a_x, p, pad, rows, cols, queue, context, program, true);
	af::array u = m_e * w;
	float divv = std::sqrt(af::sum<float>(af::pow(u, 2)) / elems);
	*a = (255.0f / std::sqrt(std::pow(10.0f, psnr / 10.0f))) / divv;
	af::array y = image + (*a * u);
	return y;
}


//συνάρτηση που υπολογίζει τη ME μάσκα της image. Χρησιμοποιείται τόσο στη δημιουργία watermark
//όσο και από τον detector. Πέρα από τη μάσκα υπολογίζει και την error sequence και μπορεί να επιστραφεί και το φίλτρο prediction error
void compute_ME_mask(af::array& image, af::array& padded, af::array& m_e, af::array& e_x, af::array& a_x, const int p, const int pad, const dim_t rows, const dim_t cols, cl::CommandQueue& queue, cl::Context& context, cl::Program& program, const bool mask_needed)
{
	const auto elems = rows * cols;
	//padded είναι transposed, οπότε 0 dimension = στήλες και 1st = γραμμές
	auto padded_rows = padded.dims(1);
	auto padded_cols = padded.dims(0);
	const int p_squared = static_cast<int>(std::pow(p, 2));
	const int p_squared_1 = p_squared - 1;

	cl_int err = 0;
	try {
		cl::Buffer padded_buff(context, CL_MEM_READ_ONLY, sizeof(cl_float) * padded_rows * padded_cols);
		padded_buff = *padded.device<cl_mem>();

		//αντιγραφή της εικόνας μέσω του Buffer ο οποίος αντιγράφτηκε από το padded array (gpu->gpu), αντι να αντιγράψουμε
		//σε pointer στη CPU και ξαναπίσω στη GPU, πολύ γρήγορη διαδικασία
		cl::Image2D padded_image2d(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_LUMINANCE, CL_HALF_FLOAT), padded_cols, padded_rows, 0, NULL, &err);
		size_t orig[] = { 0,0,0 };
		size_t des[] = { static_cast<size_t>(padded_cols), static_cast<size_t>(padded_rows), 1 };

		err = clEnqueueCopyBufferToImage(queue(), padded_buff(), padded_image2d(), 0, orig, des, NULL, NULL, NULL);
		//δημιουργία buffers για τα Rx,rx και γειτονιά (κενά, θα γραφτούν δεδομένα από το kernel)
		cl::Buffer neighb_buff(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * elems * (p_squared_1), NULL, &err);
		cl::Buffer Rx_buff(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * elems * (p_squared_1) * (p_squared_1), NULL, &err);
		cl::Buffer rx_buff(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * elems * (p_squared_1), NULL, &err);

		//δημιουργία και αρχικοποίηση του kernel
		cl::Kernel kernel = cl::Kernel(program, "me", &err);
		err = kernel.setArg(0, padded_image2d);
		err = kernel.setArg(1, Rx_buff);
		err = kernel.setArg(2, rx_buff);
		err = kernel.setArg(3, neighb_buff);
		//err = kernel.setArg(4, p);

		//τρέξιμο του kernel
		//fix για NVIDIA (OpenCL 1.2): Πρέπει GlobalGroupSize % LocalGroupSize == 0. Απλώς κάνουμε padding στο GlobalGroupSize (πχ στις γραμμες)
		//ωστε να ισχυει η συνθηκη (πχ image width=3620, height=2028 -> globalSize= 7341360, localSize (πχ) = 256. 7341360%256= 48, οποτε ΔΕΝ θα τρεξει ο kernel!
		//θα κανουμε padding σε ενα απο τα στοιχεια της εικονας (πχ height) την τιμή -> localSize-(height%localSize) -> 256-(3620%256) = 256-36 = 220
		//οποτε πλεον height = 3620 + 220 = 3840, οποτε (3840*2028)%256 = 0
		//BOUND CHECKING γινεται μεσα στον kernel, οποτε ειμαστε ΟΚ που το global εχει out of bound τιμες
		if (padded_cols * padded_rows % UtilityFunctions::max_workgroup_size != 0)
			padded_rows += UtilityFunctions::max_workgroup_size - (padded_rows % UtilityFunctions::max_workgroup_size);
		err = queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(padded_cols, padded_rows), cl::NDRange(1, UtilityFunctions::max_workgroup_size));
		queue.finish();

		//θέτουμε στα arrayfire arrays τα δεδομένα απευθείας από τη GPU, δεν υπάρχει μεταφορά από GPU σε RAM και από RAM
		//πάλι σε GPU, διαβάζονται από τη GPU και "μετατρέπονται" σε arrayfire arrays.
		af::array Rx_all = afcl::array(rows * p_squared_1 * p_squared_1, cols, Rx_buff(), af::dtype::f16, true);
		af::array rx_all = afcl::array(rows * p_squared_1, cols, rx_buff(), af::dtype::f16, true);
		af::array x_ = af::moddims(afcl::array(rows * p_squared_1, cols, neighb_buff(), af::dtype::f16, true), p_squared_1, elems);

		//τώρα θα χρειαστεί να κάνουμε reduction αθροίσματα των blocks
		//όλα τα [p^2-1,1] blocks θα αθροιστούν στην rx
		//όλα τα [p^2-1, p^2-1] blocks θα αθροιστούν στην Rx
		af::array rx = af::sum(af::moddims(rx_all, p_squared_1, elems), 1);
		af::array Rx = af::moddims(af::sum(af::moddims(Rx_all, p_squared_1 * p_squared_1, elems), 1), p_squared_1, p_squared_1);

		//επίλυση του συστήματος
		a_x = af::solve(Rx, rx);
		//υπολογισμός παράλληλα του dot product μεταξύ της γειτονιάς και της a_x ώστε να προκύψει
		//το prediction error sequence, είναι πολλαπλασιασμός πινάκων
		e_x = af::moddims(af::flat(image).T() - af::matmul(a_x, x_, AF_MAT_TRANS), rows, cols);

		//τελικός υπολογισμός μάσκας (αν χρειάζεται, στο detection μια φορά χρειάζεται)
		if (mask_needed) {
			af::array e_x_abs = af::abs(e_x);
			m_e = e_x_abs / af::max<float>(e_x_abs);
		}
	}
	//οποιοδήποτε σφάλμα στη δημιουργία μάσκας είναι μη-αναστρέψιμο, οπότε τερματίζουμε το πρόγραμμα
	catch (const af::exception & e) {
		cout << e.what();
		exit(-1);
	}
	catch (const cl::Error & cle) {
		cout << cle.what();
		exit(-1);
	}
	catch (const std::exception & ex) {
		cout << ex.what();
		exit(-1);
	}
}

//overloaded, γρήγορος υπολογισμός μάσκας μιας εικόνας εφαρμόζοντας έτοιμο prediction filter
void compute_ME_mask(af::array& image, af::array& a_x, af::array& m_e, af::array& e_x, const dim_t rows, const dim_t cols, const int p, const int pad)
{
	const int p_squared = static_cast<int>(std::pow(p, 2));
	af::array padded_image_all = af::moddims(af::unwrap(image, p, p, 1, 1, pad, pad, true), p_squared, rows * cols);
	af::array x_ = af::join(0, padded_image_all.rows(0, (p_squared / 2) - 1), padded_image_all.rows((p_squared / 2) + 1, af::end));
	//το prediction error sequence
	e_x = af::moddims(af::flat(image).T() - af::matmul(a_x, x_, AF_MAT_TRANS), rows, cols);
	//τελικός υπολογισμός μάσκας
	af::array e_x_abs = af::abs(e_x);
	m_e = e_x_abs / af::max<float>(e_x_abs);
}
//overloaded, γρήγορος υπολογισμός prediction error sequence μιας εικόνας εφαρμόζοντας έτοιμο prediction filter
void compute_error_sequence(af::array& image, af::array& a_x, af::array& e_x, const dim_t rows, const dim_t cols, const int p, const int pad)
{
	const int p_squared = static_cast<int>(std::pow(p, 2));
	af::array padded_image_all = af::moddims(af::unwrap(image, p, p, 1, 1, pad, pad, true), p_squared, rows * cols);
	af::array x_ = af::join(0, padded_image_all.rows(0, (p_squared / 2) - 1), padded_image_all.rows((p_squared / 2) + 1, af::end));
	//το prediction error sequence
	e_x = af::moddims(af::flat(image).T() - af::matmul(a_x, x_, AF_MAT_TRANS), rows, cols);
}

//συνάρτηση που υλοποιεί τον watermark detector
float mask_detector(af::array& image, const af::array& w, const int p, const float psnr, cl::CommandQueue& queue, cl::Context& context, cl::Program& program_nvf, cl::Program& program_me, bool is_nvf)
{
	//για να μην κάνουμε query συνέχεια τις στήλες/γραμμές, τις αποθηκεύουμε
	const auto rows = image.dims(0);
	const auto cols = image.dims(1);
	const auto elems = rows * cols;

	//padding
	const int p_squared = static_cast<int>(pow(p, 2));
	const int pad = p / 2;
	const auto padded_rows = rows + 2 * pad;
	const auto padded_cols = cols + 2 * pad;

	af::array m, e_z, a_z;
	if (is_nvf) {
		af::array padded = af::constant(0.0f, padded_cols, padded_rows);
		padded(af::seq(pad, static_cast<double>(padded_cols - pad - 1)), af::seq(pad, static_cast<double>(padded_rows - pad - 1))) = image.T();
		//υπολογισμός του e_z και a_z μόνο. στη συνέχεια υπολογισμός NVF mask
		compute_ME_mask(image, padded, m, e_z, a_z, p, pad, rows, cols, queue, context, program_me, false);
		padded = af::constant(0.0f, padded_cols, padded_rows);
		padded(af::seq(pad, static_cast<double>(padded_cols - pad - 1)), af::seq(pad, static_cast<double>(padded_rows - pad - 1))) = image.T();
		compute_NVF_mask(image, padded, m, p, pad, rows, cols, queue, context, program_nvf);
	}
	else {
		af::array padded = af::constant(0.0f, padded_cols, padded_rows);
		padded(af::seq(pad, static_cast<double>(padded_cols - pad - 1)), af::seq(pad, static_cast<double>(padded_rows - pad - 1))) = image.T();
		//υπολογισμός e_z, a_z καθώς και της Me mask
		compute_ME_mask(image, padded, m, e_z, a_z, p, pad, rows, cols, queue, context, program_me, true);
	}
	//η τελική μάσκα u = m*w (point wise πολλαπλασιασμός)
	af::array u = m * w;

	//μένει να υπολογίσουμε το eu
	af::array padded = af::constant(0.0f, padded_cols, padded_rows);
	padded(af::seq(pad, static_cast<double>(padded_cols - pad - 1)), af::seq(pad, static_cast<double>(padded_rows - pad - 1))) = u.T();

	//η μάσκα m_eu δε χρειάζεται για τον υπολογισμό του correlation αλλά το error_u
	af::array e_u;
	compute_error_sequence(u, a_z, e_u, rows, cols, p, pad);

	//υπολογισμός των d_ez, d_eu και dot_ez_eu
	float dot_ez_eu, d_ez, d_eu, correlation;
	dot_ez_eu = af::dot<float>(af::flat(e_u), af::flat(e_z)); //dot() θέλει vectors, και για αυτό γίνεται flattening
	d_ez = std::sqrt(af::sum<float>(af::pow(e_z, 2)));
	d_eu = std::sqrt(af::sum<float>(af::pow(e_u, 2)));

	//υπολογισμός correlation
	correlation = dot_ez_eu / (d_ez * d_eu);
	return correlation;
}

//overloaded για να ανιχνεύει με βάση προηγούμενο video frame (a_x, x_)
float mask_detector(af::array& image, const af::array& w, af::array& a_x, const int p, const float psnr, cl::CommandQueue& queue, cl::Context& context, cl::Program& program_me)
{
	//για να μην κάνουμε query συνέχεια τις στήλες/γραμμές, τις αποθηκεύουμε
	const auto rows = image.dims(0);
	const auto cols = image.dims(1);
	const auto elems = rows * cols;

	//padding
	const int p_squared = static_cast<int>(pow(p, 2));
	const int pad = p / 2;
	const auto padded_rows = rows + 2 * pad;
	const auto padded_cols = cols + 2 * pad;
	af::array padded = af::constant(0.0f, padded_cols, padded_rows);
	padded(af::seq(pad, static_cast<double>(padded_cols - pad - 1)), af::seq(pad, static_cast<double>(padded_rows - pad - 1))) = image.T();

	af::array m_e, e_z;
	//γρήγορος υπολογισμός μάσκας και error με βάση τα a_x/x_ του προηγούμενου frame
	compute_ME_mask(image, a_x, m_e, e_z, rows, cols, p, pad);

	//η τελική μάσκα u = m*w (point wise πολλαπλασιασμός)
	af::array u = m_e * w;

	//μένει να υπολογίσουμε το eu
	padded = af::constant(0.0f, padded_cols, padded_rows);
	padded(af::seq(pad, static_cast<double>(padded_cols - pad - 1)), af::seq(pad, static_cast<double>(padded_rows - pad - 1))) = u.T();

	//μόνο το e_u χρειάζεται
	af::array m_eu, e_u, a_u;
	compute_ME_mask(u, padded, m_eu, e_u, a_u, p, pad, rows, cols, queue, context, program_me, false);

	//υπολογισμός των d_ez, d_eu και dot_ez_eu
	float dot_ez_eu, d_ez, d_eu, correlation;
	dot_ez_eu = af::dot<float>(af::flat(e_u), af::flat(e_z)); //dot() θέλει vectors, και για αυτό γίνεται flattening
	d_ez = std::sqrt(af::sum<float>(af::pow(e_z, 2)));
	d_eu = std::sqrt(af::sum<float>(af::pow(e_u, 2)));

	//υπολογισμός correlation
	correlation = dot_ez_eu / (d_ez * d_eu);
	return correlation;
}


//βοηθητική συνάρτηση για να θέσουμε τις περιοχές της εικόνας
//χρειάστηκε να πειραχτεί το cl2.hpp αρχείο ώστε να γίνει define
//ένα κατάλληλο macro
cl::size_t<3> range3(size_t a, size_t b, size_t c) {
	cl::size_t<3> range;
	range[0] = a; range[1] = b; range[2] = c;
	return range;
}