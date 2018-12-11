//#include "opencv2/video/tracking.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/cudaoptflow.hpp"
//#include <opencv2\opencv.hpp>
//#include <opencv2\imgproc\types_c.h>
//#include <boost\filesystem.hpp>
//#include<boost\algorithm\string.hpp>
//using namespace boost;
//using namespace boost::filesystem;
//
//#include <stdio.h>
//#include <iostream>
//#include <fstream>
//using namespace cv;
//using namespace std;
//using namespace cv::cuda;
//
//int bound = 20;
//int type = 1;
//int device_id = 0;
//int step = 1;
//std::string outcatalog = "G:\\paper\\UCF101\\UCF101_Img\\";
//
//static void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y, double lowerBound, double higherBound) {
//#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
//	for (int i = 0; i < flow_x.rows; ++i) {
//		for (int j = 0; j < flow_y.cols; ++j) {
//			float x = flow_x.at<float>(i, j);
//			float y = flow_y.at<float>(i, j);
//			img_x.at<uchar>(i, j) = CAST(x, lowerBound, higherBound);
//			img_y.at<uchar>(i, j) = CAST(y, lowerBound, higherBound);
//		}
//	}
//#undef CAST
//}
//
//static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, double, const Scalar& color)
//{
//	for (int y = 0; y < cflowmap.rows; y += step)
//		for (int x = 0; x < cflowmap.cols; x += step)
//		{
//			const Point2f& fxy = flow.at<Point2f>(y, x);
//			line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),
//				color);
//			circle(cflowmap, Point(x, y), 2, color, -1);
//		}
//}
//
//bool HandleVideo(std::string videopath, std::string videoname) {
//	std::vector<std::string> splitPath;
//	boost::split(splitPath, videopath, is_any_of("_"), token_compress_on);
//	std::string BigClass = splitPath.at(1);
//	
//	std::string outsubcatalog_flowx = outcatalog + BigClass + "\\" + videoname + "\\flowx";
//	std::string outsubcatalog_flowy = outcatalog + BigClass + "\\" + videoname + "\\flowy";
//	std::string outsubcatalog_image = outcatalog + BigClass + "\\" + videoname + "\\image";
//	
//	if (!exists(path(outsubcatalog_flowx))) {
//		create_directories(path(outsubcatalog_flowx));
//	}
//	if (!exists(path(outsubcatalog_flowy))) {
//		create_directories(path(outsubcatalog_flowy));
//	}
//	if (!exists(path(outsubcatalog_image))) {
//		create_directories(path(outsubcatalog_image));
//	}
//	std::string writePathflow_x = outsubcatalog_flowx + "\\" + videoname + "_flow_x";
//	std::string writePathflow_y = outsubcatalog_flowy + "\\" + videoname + "_flow_y";
//	std::string writePathiamge = outsubcatalog_image + "\\" + videoname + "_iamge";
//
//	cout << videopath << endl;
//	//cout << writePathflow_x << endl << writePathflow_y << endl << writePathiamge << endl;
//	//cout << "============================================================" << endl;
//
//	VideoCapture capture(videopath);
//	if (!capture.isOpened()) {
//		printf("Could not initialize capturing..\n");
//		return false;
//	}
//
//	int frame_num = 0;
//	Mat image, prev_image, prev_grey, grey, frame, flow_x, flow_y;
//
//	GpuMat frame_0, frame_1, flow_u, flow_v;
//
//	setDevice(device_id);
//	//cv::cuda::FarnebackOpticalFlow alg_farn;
//	//Ptr<OpticalFlowDual_TVL1> alg_tvl1 = createOptFlow_DualTVL1();
//	Ptr<OpticalFlowDual_TVL1> alg_tvl1 = cv::cuda::OpticalFlowDual_TVL1::create();
//	//cv::cuda::OpticalFlowDual_TVL1 alg_tvl1;
//	//cv::cuda::BroxOpticalFlow alg_brox(0.197f, 50.0f, 0.8f, 10, 77, 10);
//
//	while (true) {
//		capture >> frame;
//		if (frame.empty())
//			break;
//		if (frame_num == 0) {
//			image.create(frame.size(), CV_8UC3);
//			grey.create(frame.size(), CV_8UC1);
//			prev_image.create(frame.size(), CV_8UC3);
//			prev_grey.create(frame.size(), CV_8UC1);
//
//			frame.copyTo(prev_image);
//			cv::cvtColor(prev_image, prev_grey, CV_BGR2GRAY);
//
//			frame_num++;
//
//			int step_t = step;
//			while (step_t > 1) {
//				capture >> frame;
//				step_t--;
//			}
//			continue;
//		}
//
//		frame.copyTo(image);
//		cv::cvtColor(image, grey, CV_BGR2GRAY);
//
//		//  Mat prev_grey_, grey_;
//		//  resize(prev_grey, prev_grey_, Size(453, 342));
//		//  resize(grey, grey_, Size(453, 342));
//		frame_0.upload(prev_grey);
//		frame_1.upload(grey);
//
//
//		// GPU optical flow
//		switch (type) {
//		case 0:
//			//alg_farn(frame_0, frame_1, flow_u, flow_v);
//			break;
//		case 1:
//			//alg_tvl1->calc(frame_0, frame_1, flow_u, flow_v);
//			alg_tvl1->calc(frame_0, frame_1, flow_u);
//			break;
//		case 2:
//			GpuMat d_frame0f, d_frame1f;
//			frame_0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
//			frame_1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);
//			//alg_brox(d_frame0f, d_frame1f, flow_u, flow_v);
//			break;
//		}
//
//		flow_u.download(flow_x);
//		Mat flows[2];
//		split(flow_x, flows);
//
//		//flow_v.download(flow_y);
//
//		//// Output optical flow
//		Mat imgX(flows[0].size(), CV_8UC1);
//		Mat imgY(flows[0].size(), CV_8UC1);
//		convertFlowToImage(flows[0], flows[1], imgX, imgY, -bound, bound);
//
//		//imshow("out", imgX);
//		//cv::waitKey(1);
//		char tmp[20];
//		sprintf(tmp, "_%05d.jpg", int(frame_num));
//
//		imwrite(writePathflow_x + tmp, imgX);
//		imwrite(writePathflow_y + tmp, imgY);
//		imwrite(writePathiamge + tmp, image);
//
//		std::swap(prev_grey, grey);
//		std::swap(prev_image, image);
//		frame_num = frame_num + 1;
//
//		int step_t = step;
//		while (step_t > 1) {
//			capture >> frame;
//			step_t--;
//		}
//	}
//	return true;
//}
//
//
//int main(int argc, char** argv)
//{
//	// IO operation
//
//	/*const char* keys =
//	{
//		"{ f  | vidFile      | ex2.avi | filename of video }"
//		"{ x  | xFlowFile    | flow_x | filename of flow x component }"
//		"{ y  | yFlowFile    | flow_y | filename of flow x component }"
//		"{ i  | imgFile      | flow_i | filename of flow image}"
//		"{ b  | bound | 15 | specify the maximum of optical flow}"
//	};
//
//	CommandLineParser cmd(argc, argv, keys);
//	string vidFile = cmd.get<string>("vidFile");
//	string xFlowFile = cmd.get<string>("xFlowFile");
//	string yFlowFile = cmd.get<string>("yFlowFile");
//	string imgFile = cmd.get<string>("imgFile");
//	int bound = cmd.get<int>("bound");*/
//	/*std::string vidFile = "D:\\ucf_hmdb\\UCF101\\UCF101_video\\ApplyEyeMakeup\\v_ApplyEyeMakeup_g01_c01.avi";
//	std::string xFlowFile = "D:\\ucf_hmdb\\UCF101\\flowdata\\flow_x";
//	std::string yFlowFile = "D:\\ucf_hmdb\\UCF101\\flowdata\\flow_y";
//	std::string imgFile = "D:\\ucf_hmdb\\UCF101\\flowdata\\flow_iamge";*/
//	
//	
//	std::ifstream correctoutput1("UCF_correct.txt", ios::in);
//	string line;
//	string last_line;
//	while (getline(correctoutput1, line))
//		last_line = line;
//	cout << last_line << endl << endl;
//	correctoutput1.close();
//	//correctoutput << "sdaaaaaaaaaaaaa" << std::endl;
//	std::ofstream erroroutput("UCF_error.txt", ios::out);
//	std::ofstream correctoutput("UCF_correct.txt", ios::app);
//	bool find = false;
//	path UCF_ROOT("G:\\paper\\UCF101\\UCF101");
//	recursive_directory_iterator end;
//	for (recursive_directory_iterator pos(UCF_ROOT); pos != end; pos++) {
//		//cout << "level" << pos.level() << ":" << *pos << endl;
//		if (!is_directory(*pos)) {
//			//path parentPath = path(*pos).parent_path();
//			//path filename = path(*pos).stem();
//			if (last_line != "") {
//				if (!find) {
//					if (!path(*pos).string().compare(last_line)) {
//						find = true;
//					}
//					continue;
//				}
//			}
//			//correctoutput << path(*pos).string() << std::endl;
//			//break;
//			if (HandleVideo(path(*pos).string(), path(*pos).stem().string())) {
//				correctoutput << path(*pos).string() << std::endl;
//			}
//			else {
//				erroroutput << path(*pos).string() << std::endl;
//			}
//
//			//cout << writePathflow_x << endl << writePathflow_y << endl << writePathiamge << endl;
//			//cout << "============================================================" << endl;
//		}
//	}
//
//	
//	correctoutput.close();
//	erroroutput.close();
//	std::cout << "done!" << std::endl;
//	std::system("pause");
//	return 0;
//}
//
