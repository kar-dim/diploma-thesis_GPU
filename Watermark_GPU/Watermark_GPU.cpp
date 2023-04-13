#pragma warning(disable:4996)
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include "INIReader.h"
#include "UtilityFunctions.h"
#include "WatermarkFunctions.h"
#define cimg_use_opencv
#define cimg_use_cpp11 1
#define cimg_use_png
#include "CImg.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <arrayfire.h>
#include <af/opencl.h>
#include <af/util.h>
#include <thread>
#include <omp.h>

using std::cout;
using std::string;
using namespace cimg_library;

int main(void)
{
	cl_int platform_id, device_id, err = 0;
	int p, rows, cols;
	unsigned int frames;
	float psnr, fps, frame_period;
	string image_path, video_path, video_path_w, w_file;

	//διάβασμα των παραμέτρων του αρχείου settings.ini ώστε να πάρουμε
	//το platform και το device id, χρησιμοποίησα την inih βιβλιοθήκη με c++ bindings
	//τη τροποποίησα ώστε να μπορεί να διαβάσει και uint64 τιμές (για το watermark key)
	INIReader inir("settings.ini");
	if (inir.ParseError() < 0) {
		cout << "Could not load opencl configuration file\n";
		return -1;
	}

	//διάβασμα των platform και device ID από το αρχέιο
	platform_id = inir.GetInteger("opencl", "platform_id", 0);
	device_id = inir.GetInteger("opencl", "device_id", 0);

	//διαβάζουμε πρώτα το OpenCL platform και ελέγχουμε για τυχόν exceptions, ειδικά αν έχει δοθεί 
	//τιμή από το αρχείο μεγαλύτερη από τον αριθμό των πιθανών platforms
	std::vector<cl::Platform> platforms;
	err = cl::Platform::get(&platforms);
	if (platforms.size() == 0 || err != CL_SUCCESS) {
		cout << "No OpenCL platforms found, exiting\n";
		return -1;
	}
	cout << "\nNumber of OpenCL platforms: " << platforms.size();
	cout << "\n\n-------------------------\n";

	cl::Platform plat;
	try {
		plat = platforms.at(platform_id);
	}
	//πιθανό να συμβεί αυτό το exception λόγω λανθασμένης τιμής στο .ini file
	catch (std::out_of_range& e) {
		(void)e;
		cout << "OpenCL exception on platform selection was thrown. Reason:\nNo platform exists with id = " << platform_id;
		return -1;
	}

	//παίρνουμε την gpu device αντίστοιχα
	std::vector<cl::Device> devices;
	err = plat.getDevices(CL_DEVICE_TYPE_GPU, &devices);
	if (devices.size() == 0 || err != CL_SUCCESS) {
		cout << "No GPUs found, exiting\n";
		return -1;
	}

	cl::Device device;
	try {
		device = devices.at(device_id);
	}
	catch (std::out_of_range& e) {
		(void)e;
		cout << "OpenCL exception on device selection was thrown. Reason:\nNo device exists with id = " << device_id;
		return -1;
	}
	//δημιουργία opencl context και queue
	cl::Context context(device);
	cl::CommandQueue queue(context, device);

	//εκτύπωση μηνυμάτων για τo platform και gpu device
	UtilityFunctions::print_opencl_info(plat, device);

	//πέρασμα των opencl context,device και queue στην ArrayFire
	afcl::addDevice(device(), context(), queue());
	afcl::setDevice(device(), context());

	//διάβασμα παραμέτρων p και psnr
	p = inir.GetInteger("parameters", "p", -1);
	psnr = static_cast<float>(inir.GetReal("parameters", "psnr", -1.0f));

	//διαβασμα του MAX_WORKGROUP_SIZE
	UtilityFunctions::max_workgroup_size = inir.GetInteger("opencl", "max_workgroup_size", -1);
	//ΔΕΝ πρεπει να υπερβαίνει το MAX_WORKGROUP_SIZE της συσκευής!
	//Αν δεν δοθεί τιμή, τότε το θέτουμε με βάση το max της συσκευής
	size_t device_maxWorkGroupSize;
	device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &device_maxWorkGroupSize);
	if (UtilityFunctions::max_workgroup_size == -1)
		UtilityFunctions::max_workgroup_size = static_cast<int>(device_maxWorkGroupSize);
	else {
		if (UtilityFunctions::max_workgroup_size < 2 || (UtilityFunctions::max_workgroup_size % 2 != 0) || UtilityFunctions::max_workgroup_size > device_maxWorkGroupSize) {
			cout << " ERROR: MAX_WORKGROUP_SIZE parameter must NOT exceed selected device's MAX_WORKGROUP_SIZE limitation and must be a positive number and power of 2\n";
			return -1;
		}
	}

	//για p>3 υπάρχουν θέματα με την αποθήκευση των Rx buffers στην Me masking
	if (p != 3) {
		cout << "For now, only p=3 is allowed\n";
		return -1;
	}
	/*if (p <= 0 || p % 2 != 1 || p > 9) {
		cout << "p parameter must be a positive odd number less than 9\n";
		return -1;
	}*/

	//compile των δικών μου opencl kernels
	std::string program_data;
	cl::Program program_nvf, program_me;
	try {
		program_data = UtilityFunctions::loadProgram("kernels/nvf.cl");
		program_nvf = cl::Program(context, program_data);
		program_nvf.build({ device }, "-cl-fast-relaxed-math");
		program_data = UtilityFunctions::loadProgram("kernels/me_p3.cl");
		program_me = cl::Program(context, program_data);
		program_me.build({ device }, "-cl-fast-relaxed-math");
	}
	catch (cl::Error& e) {
		cout << "Could not build a kernel, Reason:\n\n";
		cout << e.what();
		if (program_nvf.get() != NULL)
		    cout << program_nvf.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
		if (program_me.get() != NULL)
		    cout << program_me.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
		return -1;
	}
	//ο πίνακας που θα εμπεριέχει το υδατογράφημα
	af::array w;

	//διάβασμα του path της εικόνας
	image_path = inir.Get("paths", "image", "NO_IMAGE");
	const char* image_path_c = image_path.c_str();

	//διάβασμα των παραμέτρων video
	rows = inir.GetInteger("parameters_video", "rows", -1);
	cols = inir.GetInteger("parameters_video", "cols", -1);
	frames = (unsigned int)inir.GetInteger("parameters_video", "frames", -1);
	fps = (float)inir.GetReal("parameters_video", "fps", -1);

	//διάβασμα του path των video
	video_path = inir.Get("paths", "video", "NO_VIDEO");
	const char* video_path_c = video_path.c_str();
	video_path_w = inir.Get("paths", "video_compressed", "NO_VIDEO");
	const char* video_path_w_c = video_path_w.c_str();

	//διάβασμα του ονόματος του αρχείου που έχει το W πίνακα
	w_file = inir.Get("paths", "w_path", "w.txt");

	//στο σημείο αυτό θα γίνειη προσθήκη της μάσκας σε εικόνες
	//οι εικόνες βρίσκονται στον φάκελο images και έχουν διάφορες αναλύσεις από μικρές έως 4K
	//ενώ μπορούν να πειραχτούν παρακάτω και διάφοροι παραμέτροι όπως το PSNR και το p (μέγεθος μάσκας στη μια διάσταση)
	try {
		//δεν έχει νόημα η υδατογράφηση όταν τα p και psnr έχουν μη έγκυρες τιμές
		if (psnr <= 0) {
			cout << "PSNR must be a positive number\n";
			return -1;
		}

		bool image_mode = true;

		//εφαρμογή σε μια εικόνα
		if (image_mode == true) {
			//φορτώνουμε την εικόνα από τον δίσκο στην ram (CImg object), φορτώνουμε την αρχική RGB (3 κανάλια) εικόνα.
			timer::start();
			CImg<unsigned char> rgb_image_cimg(image_path_c);
			unsigned char* rgb_img_vals = rgb_image_cimg.data();
			timer::end();
			cout << "Time to load and tranfer RGB (uint8 *3 channels) image from disk to RAM: " << timer::secs_passed() << "\n";

			//αν η εικόνα είναι πολύ μικρή ή μεγάλη δεν εφαρμόζουμε υδατογράφηση
			if (rgb_image_cimg.width() <= 16 || rgb_image_cimg.height() <= 16) {
				cout << "Image dimensions too low\n";
				return -1;
			}
			if (rgb_image_cimg.width() > device.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>() || rgb_image_cimg.width() > 3840 || rgb_image_cimg.height() > device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>() || rgb_image_cimg.height() > 2160) {
				cout << "Image dimensions too high for this GPU\n";
				return -1;
			}
			const int rows = rgb_image_cimg.height();
			const int cols = rgb_image_cimg.width();
			//διάβασμα του W πίνακα
			w = load_W(w_file, rows, cols);
			//αντιγράφουμε την εικόνα από τη RAM (CImg object) στη GPU. Για εικόνα μπορούσαμε απευθείας από δίσκο<->gpu
			//απλώς ήθελα να δείξω το κόστος της μεταφοράς στη gpu από τον δίσκο σε σχέση με τη ram.
			timer::start();
			af::array rgb_image = af::array(cols, rows, 3, rgb_img_vals).as(af::dtype::f16);
			af::sync();
			timer::end();
			cout << "Time to transfer RGB (uint8 *3 channels) image from RAM to VRAM: " << timer::secs_passed() << "\n\n";
			rgb_image = af::transpose(rgb_image);

			//μετατροπή σε grayscale (αντίστοιχος matlab/cpu code για τα βάρη)
			af::array image = af::round(0.299 * rgb_image(af::span, af::span, 0)) + af::round(0.587 * rgb_image(af::span, af::span, 1)) + af::round(0.114 * rgb_image(af::span, af::span, 2));
			af::print("image", image);

			float a;
			//warmup μια φορά (ΝVF) ώστε να δημιουργηθούν τα kernels
			make_and_add_watermark_NVF(image, w, p, psnr, &a, queue, context, program_nvf);
			//δημιουργία watermark (NVF)
			timer::start();
			af::array watermark_NVF = make_and_add_watermark_NVF(image, w, p, psnr, &a, queue, context, program_nvf);
			timer::end();
			cout << "a: " << std::fixed << std::setprecision(8) << a << "\n";
			cout << "Time to calculate NVF mask of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << timer::secs_passed() << " seconds.\n\n";

			//warmup μια φορά (ME)
			make_and_add_watermark_ME(image, w, p, psnr, &a, queue, context, program_me);
			//δημιουργία watermark (ME)
			timer::start();
			af::array watermark_ΜΕ = make_and_add_watermark_ME(image, w, p, psnr, &a, queue, context, program_me);
			timer::end();
			cout << "a: " << std::fixed << std::setprecision(8) << a << "\n";
			cout << "Time to calculate ME mask of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << timer::secs_passed() << " seconds.\n\n";

			//correlation μεταξύ των τελικών υδατογραφημένων εικόνων, για NVF και ΜΕ
			//warmup πρώτα
			mask_detector(watermark_NVF, w, p, psnr, queue, context, program_nvf, program_me, true);
			mask_detector(watermark_ΜΕ, w, p, psnr, queue, context, program_nvf, program_me, false);

			//χρονομέτρηση detection
			timer::start();
			float correlation_nvf = mask_detector(watermark_NVF, w, p, psnr, queue, context, program_nvf, program_me, true);
			timer::end();
			cout << "Time to calculate correlation (NVF) of an image of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << timer::secs_passed() << " seconds.\n\n";

			timer::start();
			float correlation_me = mask_detector(watermark_ΜΕ, w, p, psnr, queue, context, program_nvf, program_me, false);
			timer::end();
			cout << "Time to calculate correlation (ME) of an image of " << rows << " rows and " << cols << " columns with parameters:\np= " << p << "\tPSNR(dB)= " << psnr << "\n" << timer::secs_passed() << " seconds.\n\n";

			cout << "Correlation [NVF]: " << std::fixed << std::setprecision(16) << correlation_nvf << "\n";
			cout << "Correlation [ME]: " << std::fixed << std::setprecision(16) << correlation_me << "\n";

		}

		//εφαρμογή σε video
		else {
			//έλεγχος παραμέτρων video

			if (rows <= 16 || cols <= 16) {
				cout << "Video dimensions too low\n";
				return -1;
			}
			if (rows > device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>() || rows > 2160 || cols > device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>() || cols > 3840) {
				cout << "Video dimensions too high for this GPU\n";
				return -1;
			}
			if (fps <= 1 || fps > 60) {
				cout << "Video FPS is too low or too high\n";
				return -1;
			}
			if (frames <= 1) {
				cout << "Frame count too low\n";
				return -1;
			}
			if (video_path_w == "NO_VIDEO") {
				cout << "Invalid video file names\n";
			}
			//πόση ώρα ανάμεσα σε κάθε frame
			frame_period = 1.0f / fps;
			float time_diff;
			w = load_W(w_file, rows, cols);
			/*

							REALTIME ΥΔΑΤΟΓΡΑΦΗΣΗ (RAW VIDEO)

			*/
			//επιλογή είτε να υδατογραφήσουμε ΟΛΟ ΤΟ ΒΙΝΤΕΟ ειτε να υδατογραφήσουμε μόνο το πρώτο frame
			bool first_frame_w = false, two_frames_watermark = false;
			float a;
			//φόρτωση του video
			//CImgDisplay disp;
			CImgList<unsigned char> video_cimg;
			video_cimg = video_cimg.load_yuv(video_path_c, cols, rows, 420, 0, frames - 1, 1, false);
			//οι δυο vectors θα έχουν όλα τα watermarked frames.
			//std::vector<af::array> frames_nvf;
			//frames_nvf.reserve(frames);
			std::vector<af::array> frames_me;
			frames_me.reserve(frames);
			std::vector<af::array> a_x, x_;
			for (unsigned int i = 0; i < (frames / 2) + 1; i++) {
				a_x.push_back(af::constant(0, 1, 1));
			}
			int counter = 0;
			if (!first_frame_w) {
				//af::Window window1(1280, 720, "Watermarked Video");
				for (unsigned int i = 0; i < frames; i++) {
					//διάβασμα και μεταφορά από τη RAM -> VRAM της Υ συνιστώσας
					//timer::start();
					CImg<unsigned char> temp(video_cimg.at(i).get_channel(0));
					unsigned char* y_vals = temp.data();
					af::array gpu_frame(cols, rows, y_vals);
					//af::print("ar", gpu_frame);
					gpu_frame = af::transpose(gpu_frame).as(af::dtype::f32);
					//υπολογισμός ME watermarked frame.
					if (i % 2 && two_frames_watermark == false)
						frames_me.push_back(make_and_add_watermark_ME(gpu_frame, w, p, psnr, &a, queue, context, program_me));
					else {
						frames_me.push_back(make_and_add_watermark_ME(gpu_frame, w, a_x[counter], p, psnr, &a, queue, context, program_me));
						counter++;
					}
					//timer::end();
					//cout << timer::secs_passed() << "\n";

					//θα δείξω σε realtime το τελικό watermarked video.
					//για να κάνουμε "simulate" το realtime, θα πρέπει να αφαιρεθεί ο χρόνος που σπαταλήθηκε στον υπολογισμό
					//αν είναι μεγαλύτερος τότε δε πετυχαίνεται realtime απεικόνιση, οπότε δείχνουμε απευθείας χωρίς sleep
					//if ((time_diff = frame_period - timer::secs_passed()) > 0) {
						//std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(1000 * time_diff)));
					//}
					//window1.image(normalize_to_f32(frames_me[i]));
				}
			}
			else {
				//υδατογράφηση ΜΟΝΟ του πρώτου frame
				CImg<unsigned char> temp(video_cimg.at(0).get_channel(0));
				unsigned char* y_vals = temp.data();
				af::array gpu_frame(cols, rows, y_vals);
				gpu_frame = af::transpose(gpu_frame).as(af::dtype::f32);

				//frames_nvf.push_back(make_and_add_watermark_NVF(gpu_frame, w, p, psnr, queue, context, program_nvf, false));
				frames_me.push_back(make_and_add_watermark_ME(gpu_frame, w, p, psnr, &a, queue, context, program_me));
				//τα υπόλοιπα frames θα μπουν όπως έχουν
				for (unsigned int i = 1; i < frames; i++) {
					CImg<unsigned char> temp(video_cimg.at(i).get_channel(0));
					unsigned char* y_vals = temp.data();
					af::array gpu_frame(cols, rows, y_vals);
					gpu_frame = af::transpose(gpu_frame);
					frames_me.push_back(gpu_frame.as(af::dtype::f32));
				}
			}

			/*

						ΑΠΟΘΗΚΕΥΣΗ RAW->WATERMARKED ΒΙΝΤΕΟ

			*/

			//			//εδώ αποθηκεύω σε αρχείο το watermark ME video
			//			{
			//				CImgList<unsigned char> video_cimg_to_w(frames, cols, rows, 1, 3);
			//				for (int i = 0; i < frames; i++) {
			//					unsigned char* ptr = af::clamp(frames_me[i].T(),0,255).as(af::dtype::u8).host<unsigned char>();
			//					unsigned char* ptr2 = new unsigned char[rows * cols];
			//#pragma omp parallel for
			//					for (int i = 0; i < rows * cols; i++)
			//						ptr2[i] = ptr[i];
			//
			//					af::freeHost(ptr);
			//					ptr = NULL;
			//					CImg<unsigned char> temp0(cols, rows);
			//					delete[] temp0._data;
			//					//temp0._data δε χρειάζεται delete[], γίνεται από τον CImg destructor
			//					temp0._data = new unsigned char[rows * cols];
			//#pragma omp parallel for
			//					for (int i = 0; i < rows * cols; i++)
			//						temp0._data[i] = ptr2[i];
			//					//temp0._data = ptr2;
			//					video_cimg_to_w.at(i).draw_image(0, 0, 0, 0, temp0);
			//					delete[] ptr2;
			//					ptr2 = NULL;
			//
			//					CImg<unsigned char> temp1(video_cimg.at(i).get_channel(1));
			//					CImg<unsigned char> temp2(video_cimg.at(i).get_channel(2));
			//					video_cimg_to_w.at(i).draw_image(0, 0, 0, 1, temp1);
			//					video_cimg_to_w.at(i).draw_image(0, 0, 0, 2, temp2);
			//				}
			//
			//				video_cimg_to_w.save_yuv("videos\\watermarked.yuv", 420, false);
			//				/*CImgDisplay disp;
			//				for (int i = 0; i < frames; i++) {
			//					timer::start();
			//					CImg<unsigned char> temp(video_cimg_to_w.at(i).get_channel(0));
			//					timer::end();
			//					if ((time_diff = frame_period - timer::secs_passed()) > 0) {
			//						std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(1000 * time_diff)));
			//					}
			//					disp.display(temp);
			//				}*/
			//				
			//			}


			/*

						REALTIME(?) RAW->WATERMARKED VIDEO DETECTION

			*/

			//Φτιάχνουμε άλλους δυο vectors που θα κρατάνε το correlation
			//std::vector<float> frames_nvf_cor(frames);
			std::vector<float> frames_me_cor(frames);
			//παρακάτω είναι realtime detection (θα δείξω στην οθόνη τη ΜΕ watermark sequence)
			//μπορεί να τροποποιηθεί ώστε να βρίσκει correlation κάθε 2ο frame (τότε δε δείχνω στην οθόνη προφανώς τίποτα)
			CImgDisplay cimgd0;
			for (unsigned int i = 0; i < frames; i++) {
				//frames_nvf_cor[i] = mask_detector(frames_nvf[i], w, p, psnr, queue, context, program_me);
				timer::start();
				frames_me_cor[i] = mask_detector(frames_me[i], w, p, psnr, queue, context, program_nvf, program_me, false);
				timer::end();
				//εφαρμόζω clamping στο 0-255 μόνο για εμφάνιση της εικόνας! (αλλιώς αλλάζουν τα raw data που είναι λάθος)
				af::array clamped = af::clamp(frames_me[i], 0, 255);
				unsigned char* ptr = af::clamp(clamped.T(), 0, 255).as(af::dtype::u8).host<unsigned char>();
				unsigned char* ptr2 = new unsigned char[rows * cols];
#pragma omp parallel for
				for (int i = 0; i < rows * cols; i++)
					ptr2[i] = ptr[i];

				af::freeHost(ptr);
				ptr = NULL;
				CImg<unsigned char> temp0(cols, rows);
				delete[] temp0._data;
				//temp0._data δε χρειάζεται delete[], γίνεται από τον CImg destructor
				temp0._data = new unsigned char[rows * cols];
#pragma omp parallel for
				for (int i = 0; i < rows * cols; i++)
					temp0._data[i] = ptr2[i];

				if ((time_diff = frame_period - timer::secs_passed()) > 0) {
					std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(1000 * time_diff)));
				}
				cimgd0.display(temp0);

				cout << timer::secs_passed() << "\n";
				//cout << frames_me_cor[i] << "\n";
			}

			/*

						REALTIME (?) DETECTION RAW->WATERMARKED ΑΝΑ ΔΥΟ FRAMES


			*/


			//std::vector<float> frames_me_cor2(frames);
			//counter = 0;
			//for (unsigned int i = 0; i < frames; i++) {
			//	timer::start();
			//	if (i % 2 != 0){
			//		frames_me_cor2[i] = mask_detector(frames_me[i], w, a_x[counter], p, psnr, queue, context, program_me);
			//		counter++;
			//	}
			//	else {
			//		frames_me_cor2[i] = mask_detector(frames_me[i], w, p, psnr, queue, context, program_me);
			//	}
			//	timer::end();
			//	cout << timer::secs_passed() << "\n";
			//}



/*

			REALTIME (?) DETECTION ΣΥΜΠΙΕΣΜΕΝΟΥ VIDEO FILE


*/

//αρχικά διαβάζουμε το συμπιεσμένο βίντεο.
			CImgList<unsigned char> video_cimg_w;
			video_cimg_w = video_cimg_w.load_video(video_path_w_c, 0, frames - 1);
			//std::vector<float> frames_nvf_cor_w(frames);
			std::vector<float> frames_me_cor_w(frames);
			//std::vector<af::array> frames_nvf_w;
			std::vector<af::array> frames_me_w;
			//frames_nvf_w.reserve(frames);
			frames_me_w.reserve(frames);
			for (unsigned int i = 0; i < frames; i++) {
				CImg<unsigned char> temp(video_cimg_w.at(i).get_channel(0));
				unsigned char* y_vals = temp.data();
				af::array gpu_frame(cols, rows, y_vals);
				//διάβασμα (συμπιεσμένων frames)
				//frames_nvf_w.push_back(gpu_frame.as(af::dtype::f32));
				frames_me_w.push_back(gpu_frame.T().as(af::dtype::f32));
			}

			//detection
			af::array clamped(rows, cols);
			//af::Window window3(cols, rows, "Watermarked Video");
			CImgDisplay cimgd;
			for (unsigned int i = 0; i < frames; i++) {
				timer::start();
				//frames_nvf_cor_w[i] = mask_detector(frames_nvf[i], w, p, psnr, queue, context, program_me);
				frames_me_cor_w[i] = mask_detector(frames_me_w[i], w, p, psnr, queue, context, program_nvf, program_me, false);
				timer::end();
				//εφαρμόζω clamping στο 0-255 μόνο για εμφάνιση της εικόνας! (αλλιώς αλλάζουν τα raw data που είναι λάθος)
				clamped = af::clamp(frames_me_w[i], 0, 255);
				unsigned char* ptr = af::clamp(clamped.T(), 0, 255).as(af::dtype::u8).host<unsigned char>();
				unsigned char* ptr2 = new unsigned char[rows * cols];
#pragma omp parallel for
				for (int i = 0; i < rows * cols; i++)
					ptr2[i] = ptr[i];

				af::freeHost(ptr);
				ptr = NULL;
				CImg<unsigned char> temp0(cols, rows);
				delete[] temp0._data;
				//temp0._data δε χρειάζεται delete[], γίνεται από τον CImg destructor
				temp0._data = new unsigned char[rows * cols];
#pragma omp parallel for
				for (int i = 0; i < rows * cols; i++)
					temp0._data[i] = ptr2[i];

				if ((time_diff = frame_period - timer::secs_passed()) > 0) {
					std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(1000 * time_diff)));
				}
				cimgd.display(temp0);
				//window3.image(normalize_to_f32(clamped));
				//cout << timer::secs_passed() << "\n";
				//cout << frames_me_cor_w[i] << "\n";
			}

		}

		system("pause");
		return 0;
	}
	catch (const af::exception& e) {
		cout << e.what();
		return -1;
	}
}