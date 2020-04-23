// inkers.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
//#include <include.h>
//#include <exit.h>

using namespace std;
using namespace cv;


int login()
{
	string username;
	string password;
	int tries = 3;   // number of tries to ques 
	int success = 0;  // false

	while (tries && !success)
	{
		cout << "Username: ";
		cin >> username;
		cout << "Password: ";
		cin >> password;
		if (username == "admin" && password == "inkers")
		{
			cout << "You have got access. \n";
			success = 1;
		}
		else
		{
			tries--;
		}
	}
	return success;
}


int main() {
	int log = login();
	if (log == 0) {
		cout << "Incorrect Credentials";
		return 0;
	}
	
	VideoCapture cap(0);

	if (!cap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return -1;
	}

	int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH)); // width of frames of the video
	int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT)); // height of frames of the video

	Size frame_size(frame_width, frame_height);
	int frames_per_second = 10;

	//Create and initialize the VideoWriter object 
	VideoWriter oVideoWriter("./MyVideo.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'),
		frames_per_second, frame_size, true);

	//If the VideoWriter object is not initialized successfully, exit the program
	if (oVideoWriter.isOpened() == false)
	{
		cout << "Cannot save the video to a file" << endl;
		cin.get(); //wait for any key press
		return -1;
	}


	while (true) {
		Mat frame;
		cap >> frame;
		// If the frame is empty, then break.
		if (frame.empty())
			break;

		Mat planes[3]; // extract red channel
		split(frame, planes);  // planes[2] is the red channel

		Mat final;
		final = planes[2].clone();
		GaussianBlur(planes[2], final, Size(7, 7), 0, 0);

		oVideoWriter.write(final);

		imshow("Frame", final);

		// Press  ESC on keyboard to exit
		char c = (char)waitKey(25);
		if (c == 27)
			break;
	}

	// When everything done, release the video capture object
	cap.release();
	oVideoWriter.release();
	destroyAllWindows();
	
	return 0;

}

