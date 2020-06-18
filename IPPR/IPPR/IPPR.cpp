#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <fstream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/dnn.hpp>


using namespace cv;
using namespace dnn;
using namespace saliency;
using namespace std;

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;        // Width of network's input image
int inpHeight = 416;       // Height of network's input image


void FinalFrame(Mat frame);
vector<string> classes;
vector<Mat> Segmentation(Mat src, Mat rgb);
Mat KMeans(Mat src, int clusterCount);

void FinalOutput(Mat frame) {
    Mat output = frame.clone();
    medianBlur(output, output, 3);

    Ptr<StaticSaliencySpectralResidual> Saliency = StaticSaliencySpectralResidual::create();
    Mat srMask, fgMask;

    Saliency->computeSaliency(output, fgMask);
    fgMask.convertTo(fgMask, CV_8U, 255);

    cvtColor(fgMask, fgMask, COLOR_GRAY2BGR);
    vector<Mat> segment = Segmentation(KMeans(fgMask, 3), frame);

    for (Mat image : segment) {
        FinalFrame(image);
    }

}

Mat KMeans(Mat src, int clusterCount) {
    Mat means(src.rows * src.cols, 3, CV_32F);
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++)
            for (int k = 0; k < 3; k++)
                means.at<float>(i + j * src.rows, k) = src.at<Vec3b>(i, j)[k];

    Mat labels;
    Mat centers;
    int attempts = 5;
    kmeans(means, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

    Mat image = Mat::zeros(src.size(), src.type());
    Vec2i pointVal = { 0, 0 };

    //Get highest intensity
    for (int i = 0; i < centers.rows; i++) {
        int sum = 0;
        for (int j = 0; j < centers.cols; j++) {
            sum += centers.at<float>(i, j);
        }
        if (sum / 3 > pointVal[1]) {
            pointVal[0] = i;
            pointVal[1] = sum / 3;
        }
    }

    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++)
        {
            int cluster = labels.at<int>(i + j * src.rows, 0);
            if (cluster == pointVal[0]) {
                image.at<Vec3b>(i, j)[0] = centers.at<float>(cluster, 0);
                image.at<Vec3b>(i, j)[1] = centers.at<float>(cluster, 1);
                image.at<Vec3b>(i, j)[2] = centers.at<float>(cluster, 2);
            }
        }
    cvtColor(image, image, COLOR_BGR2GRAY);
    return image;
}

vector<Mat> Segmentation(Mat src, Mat original) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<Mat> segment;

    int Dilation = 8;
    dilate(src, src, getStructuringElement(MORPH_RECT, Size(1, Dilation * 2 + 1), Point(0, Dilation)));

    int Erosion = 6;
    erode(src, src, getStructuringElement(MORPH_RECT, Size(Erosion * 2 + 1, 1), Point(Erosion, 0)));

    findContours(src, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    Mat drawing = Mat::zeros(src.size(), CV_8UC3);
    // Original image clone
    RNG rng(12345);

    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        Rect rec = boundingRect(contours.at(i));
        double ratio = rec.width / rec.height;

        if (rec.width < original.cols * 0.06 || rec.height < original.rows * 0.1 || rec.y < original.rows * 0.25 || (rec.x < original.cols * 0.3) || ratio > 3.0) {
            continue;
        }

        segment.push_back(original(rec));
    }
    return segment;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& out);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);


int main() {
    VideoCapture capture("C:\\Users\\scpy2\\OneDrive\\Pictures\\IPPR.mp4");
    int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
    int fps = capture.get(CAP_PROP_FPS) / 10;
    VideoWriter video("output.mp4v", VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, Size(frame_width, frame_height));

    while (true)
    {
        Mat frame;
        for (int i = 0; i < 10; i++)
            capture >> frame;

        if (frame.empty())
            break;

        FinalOutput(frame);
        video.write(frame);
        imshow("YOLO Final Output", frame);
        char c = (char)waitKey(30);
        if (c == 27)
            break;
    }
    capture.release();
    video.release();
}

void FinalFrame(Mat frame) {

    // Load names of classes
    string classesFile = "C:\\Users\\scpy2\\darknet\\build\\darknet\\x64\\data\\coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Give the configuration and weight files for the model
    String modelConfiguration = "C:\\Users\\scpy2\\darknet\\cfg\\yolov3.cfg";
    String modelWeights = "C:\\Users\\scpy2\\darknet\\yolov3.weights";

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Create a 4D blob from a frame.
    Mat blob;
    blobFromImage(frame, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

    //Set the input to the network
    net.setInput(blob);

    // Runs the forward pass to get output of the output layers
    vector<Mat> outs;
    net.forward(outs, getOutputsNames(net));

    // Remove the bounding boxes with low confidence
    postprocess(frame, outs);

}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
            box.x + box.width, box.y + box.height, frame);
    }
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}
