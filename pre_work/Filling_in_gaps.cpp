#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"
#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\types_c.h>
#include <boost\filesystem.hpp>
#include<boost\algorithm\string.hpp>

using namespace boost::filesystem;
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;
using namespace cv;
using namespace cv::cuda;


int bound = 20;
int type = 1;
int device_id = 0;
int step = 1;
std::string outcatalog = "G:\\paper\\HMDB51\\HMDB_Img\\";

static void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y,
	double lowerBound, double higherBound) {
#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
	for (int i = 0; i < flow_x.rows; ++i) {
		for (int j = 0; j < flow_y.cols; ++j) {
			float x = flow_x.at<float>(i, j);
			float y = flow_y.at<float>(i, j);
			img_x.at<uchar>(i, j) = CAST(x, lowerBound, higherBound);
			img_y.at<uchar>(i, j) = CAST(y, lowerBound, higherBound);
		}
	}
#undef CAST
}

bool HandleVideo(std::string videopath, std::string videotype,std::string videoname) {
	
	std::string outsubcatalog_flowx = outcatalog + videotype + "\\" +videoname + "\\flowx";
	std::string outsubcatalog_flowy = outcatalog + videotype + "\\" + videoname + "\\flowy";
	std::string outsubcatalog_image = outcatalog + videotype + "\\" + videoname + "\\image";
	if (!exists(path(outsubcatalog_flowx))) {
		create_directories(path(outsubcatalog_flowx));
	}
	if (!exists(path(outsubcatalog_flowy))) {
		create_directories(path(outsubcatalog_flowy));
	}
	if (!exists(path(outsubcatalog_image))) {
		create_directories(path(outsubcatalog_image));
	}
	std::string writePathflow_x = outsubcatalog_flowx + "\\" +"flow_x";
	std::string writePathflow_y = outsubcatalog_flowy + "\\" +"flow_y";
	std::string writePathiamge = outsubcatalog_image + "\\" +"iamge";
	
	cout << videopath << endl;
	//cout << writePathflow_x << endl << writePathflow_y << endl << writePathiamge << endl;
	//cout << "============================================================" << endl;
	
	VideoCapture capture(videopath);
	if (!capture.isOpened()) {
		printf("Could not initialize capturing..\n");
		return false;
	}
	
	int frame_num = 0;
	Mat image, prev_image, prev_grey, grey, frame, flow_x, flow_y;

	GpuMat frame_0, frame_1, flow_u, flow_v;

	setDevice(device_id);
	//cv::cuda::FarnebackOpticalFlow alg_farn;
	//Ptr<OpticalFlowDual_TVL1> alg_tvl1 = createOptFlow_DualTVL1();
	Ptr<OpticalFlowDual_TVL1> alg_tvl1 = cv::cuda::OpticalFlowDual_TVL1::create();
	//cv::cuda::OpticalFlowDual_TVL1 alg_tvl1;
	//cv::cuda::BroxOpticalFlow alg_brox(0.197f, 50.0f, 0.8f, 10, 77, 10);

	while (true) {
		capture >> frame;
		if (frame.empty())
			break;
		if (frame_num == 0) {
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_image.create(frame.size(), CV_8UC3);
			prev_grey.create(frame.size(), CV_8UC1);

			frame.copyTo(prev_image);
			cv::cvtColor(prev_image, prev_grey, CV_BGR2GRAY);

			frame_num++;

			int step_t = step;
			while (step_t > 1) {
				capture >> frame;
				step_t--;
			}
			continue;
		}

		frame.copyTo(image);
		cv::cvtColor(image, grey, CV_BGR2GRAY);

		//  Mat prev_grey_, grey_;
		//  resize(prev_grey, prev_grey_, Size(453, 342));
		//  resize(grey, grey_, Size(453, 342));
		frame_0.upload(prev_grey);
		frame_1.upload(grey);


		// GPU optical flow
		switch (type) {
		case 0:
			//alg_farn(frame_0, frame_1, flow_u, flow_v);
			break;
		case 1:
			//alg_tvl1->calc(frame_0, frame_1, flow_u, flow_v);
			alg_tvl1->calc(frame_0, frame_1, flow_u);
			break;
		case 2:
			GpuMat d_frame0f, d_frame1f;
			frame_0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
			frame_1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);
			//alg_brox(d_frame0f, d_frame1f, flow_u, flow_v);
			break;
		}

		flow_u.download(flow_x);
		Mat flows[2];
		split(flow_x, flows);

		//flow_v.download(flow_y);

		//// Output optical flow
		Mat imgX(flows[0].size(), CV_8UC1);
		Mat imgY(flows[0].size(), CV_8UC1);
		convertFlowToImage(flows[0], flows[1], imgX, imgY, -bound, bound);

		//imshow("out", imgX);
		//cv::waitKey(1);
		char tmp[20];
		sprintf(tmp, "_%05d.jpg", int(frame_num));

		imwrite(writePathflow_x + tmp, imgX);
		imwrite(writePathflow_y + tmp, imgY);
		imwrite(writePathiamge + tmp, image);

		std::swap(prev_grey, grey);
		std::swap(prev_image, image);
		frame_num = frame_num + 1;

		int step_t = step;
		while (step_t > 1) {
			capture >> frame;
			step_t--;
		}
	}
	return true;
}

int main() {

	//sigle_test
	//string single_path = "G:/paper/HMDB51/HMDB51_Video/climb_stairs/Braune_Stiefel_brown_boots_hot_high_heels_Treppen_steigen_Down-_and_upstairs_climb_stairs_l_cm_np1_ba_med_0";
	
	//txt test
	std::ifstream empty_classes("G:/paper/HMDB51/Check_HMDB_img_empty.txt", ios::in);
	string single_path;
	int oknum = 0;
	while (getline(empty_classes, single_path)) {
		//std::cout << single_path << endl;
		string::size_type pos = 0;
		while ((pos = single_path.find("\\", pos)) != string::npos) {
			single_path.replace(pos, 1, "/");
		}
		pos = single_path.find("HMDB_Img",0);
		single_path.replace(pos, 8, "");
		single_path.insert(pos, "HMDB51_Video");
		//std::cout << single_path << endl;
		//continue;
		std::vector<std::string> pathsplits;
		boost::split(pathsplits, single_path, boost::is_any_of("/"), boost::token_compress_on);
		std::cout << pathsplits.at(4) << endl;
		
		if (HandleVideo(single_path + ".avi", pathsplits.at(4), pathsplits.at(5))) {
			oknum++;
			cout << single_path << endl;
		}
	}
	empty_classes.close();
	cout << oknum << endl;
	std::cout << "done!" << std::endl;
	system("pause");
	return 0;
}