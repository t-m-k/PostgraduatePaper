//#include "opencv2/video/tracking.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/cudaoptflow.hpp"
//#include <opencv2\opencv.hpp>
//#include <opencv2\imgproc\types_c.h>
//#include <stdio.h>
//#include <iostream>
//#include <string>
//using namespace std;
//using namespace cv;
//using namespace cv::cuda;
//
//
//inline bool isFlowCorrect(Point2f u)
//{
//	return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.y) < 1e9;
//}
//
//static Vec3b computeColor(float fx, float fy)
//{
//	static bool first = true;
//
//	// relative lengths of color transitions:
//	// these are chosen based on perceptual similarity
//	// (e.g. one can distinguish more shades between red and yellow
//	//  than between yellow and green)
//	const int RY = 15;
//	const int YG = 6;
//	const int GC = 4;
//	const int CB = 11;
//	const int BM = 13;
//	const int MR = 6;
//	const int NCOLS = RY + YG + GC + CB + BM + MR;
//	static Vec3i colorWheel[NCOLS];
//
//	if (first) {
//		int k = 0;
//
//		for (int i = 0; i < RY; ++i, ++k)
//			colorWheel[k] = Vec3i(255, 255 * i / RY, 0);
//
//		for (int i = 0; i < YG; ++i, ++k)
//			colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);
//
//		for (int i = 0; i < GC; ++i, ++k)
//			colorWheel[k] = Vec3i(0, 255, 255 * i / GC);
//
//		for (int i = 0; i < CB; ++i, ++k)
//			colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);
//
//		for (int i = 0; i < BM; ++i, ++k)
//			colorWheel[k] = Vec3i(255 * i / BM, 0, 255);
//
//		for (int i = 0; i < MR; ++i, ++k)
//			colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);
//
//		first = false;
//	}
//
//	const float rad = sqrt(fx * fx + fy * fy);
//	const float a = atan2(-fy, -fx) / (float)CV_PI;
//
//	const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
//	const int k0 = static_cast<int>(fk);
//	const int k1 = (k0 + 1) % NCOLS;
//	const float f = fk - k0;
//
//	Vec3b pix;
//
//	for (int b = 0; b < 3; b++)
//	{
//		const float col0 = colorWheel[k0][b] / 255.f;
//		const float col1 = colorWheel[k1][b] / 255.f;
//
//		float col = (1 - f) * col0 + f * col1;
//
//		if (rad <= 1)
//			col = 1 - rad * (1 - col); // increase saturation with radius
//		else
//			col *= .75; // out of range
//
//		pix[2 - b] = static_cast<uchar>(255.f * col);
//	}
//
//	return pix;
//}
//
//static void drawOpticalFlow(const Mat_<Point2f>& flow, Mat& dst, float maxmotion = -1)
//{
//	dst.create(flow.size(), CV_8UC3);
//	dst.setTo(Scalar::all(0));
//
//	// determine motion range:
//	float maxrad = maxmotion;
//
//	if (maxmotion <= 0)
//	{
//		maxrad = 1;
//		for (int y = 0; y < flow.rows; ++y)
//		{
//			for (int x = 0; x < flow.cols; ++x)
//			{
//				Point2f u = flow(y, x);
//
//				if (!isFlowCorrect(u))
//					continue;
//
//				maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
//			}
//		}
//	}
//
//	for (int y = 0; y < flow.rows; ++y)
//	{
//		for (int x = 0; x < flow.cols; ++x)
//		{
//			Point2f u = flow(y, x);
//
//			if (isFlowCorrect(u))
//				dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
//		}
//	}
//}
//
//static void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y,
//	double lowerBound, double higherBound) {
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
//static void convertFlowToImage2(const Mat &flow_x, Mat &img_x, 
//	double lowerBound, double higherBound) {
//#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
//	for (int i = 0; i < flow_x.rows; ++i) {
//		for (int j = 0; j < flow_x.cols; ++j) {
//			float x = flow_x.at<float>(i, j);
//			
//			img_x.at<uchar>(i, j) = CAST(x, lowerBound, higherBound);
//			
//		}
//	}
//#undef CAST
//}
//
//static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, double, const Scalar& color) {
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
//int main() {
//	// IO operation
//	/*const char* keys =
//	{
//	"{ f  | vidFile      | ex2.avi | filename of video }"
//	"{ x  | xFlowFile    | flow_x | filename of flow x component }"
//	"{ y  | yFlowFile    | flow_y | filename of flow x component }"
//	"{ i  | imgFile      | flow_i | filename of flow image}"
//	"{ b  | bound | 15 | specify the maximum of optical flow}"
//	"{ t  | type | 0 | specify the optical flow algorithm }"
//	"{ d  | device_id    | 0  | set gpu id}"
//	"{ s  | step  | 1 | specify the step for frame sampling}"
//	};
//
//	CommandLineParser cmd(argc, argv, keys);
//	string vidFile = cmd.get<string>("vidFile");
//	string xFlowFile = cmd.get<string>("xFlowFile");
//	string yFlowFile = cmd.get<string>("yFlowFile");
//	string imgFile = cmd.get<string>("imgFile");
//	int bound = cmd.get<int>("bound");
//	int type = cmd.get<int>("type");
//	int device_id = cmd.get<int>("device_id");
//	int step = cmd.get<int>("step");*/
//	//-f test.avi -x tmp/flow_x -y tmp/flow_x -i tmp/image -b 20 -t 1 -d 0 -s 1
//
//	string vidFile = "G:\\paper\\HMDB51\\HMDB51_Video\\brush_hair\\April_09_brush_hair_u_nm_np1_ba_goo_0.avi";
//	string xFlowFile = "G:\\paper\\HMDB51\\HMDB_Img\\flow_x";
//	string yFlowFile = "G:\\paper\\HMDB51\\HMDB_Img\\flow_y";
//	string imgFile = "G:\\paper\\HMDB51\\HMDB_Img\\image";
//	int bound = 20;
//	int type = 1;
//	int device_id = 0;
//	int step = 1;
//
//	VideoCapture capture(vidFile);
//	if (!capture.isOpened()) {
//		printf("Could not initialize capturing..\n");
//		return -1;
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
//		//Mat imgX(flow_x.size(), CV_8UC1);
//		//convertFlowToImage2(flow_x,imgX,  -bound, bound);
//		//drawOpticalFlow(flow_x, imgX);
//		imshow("out", imgX);
//		cv::waitKey(1);
//		//char tmp[20];
//		//sprintf(tmp, "_%05d.jpg", int(frame_num));
//
//		
//		// Mat imgX_, imgY_, image_;
//		// resize(imgX,imgX_,cv::Size(340,256));
//		// resize(imgY,imgY_,cv::Size(340,256));
//		// resize(image,image_,cv::Size(340,256));
//
//		/*imwrite(xFlowFile + tmp, imgX);
//		imwrite(yFlowFile + tmp, imgY);
//		imwrite(imgFile + tmp, image);*/
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
//	return 0;
//}