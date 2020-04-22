## Object Detection

Implementation of YOLOv3 : object detector.

The model recognizes 80 different objects.

**Working of YOLO:** 

First, it divides the 416x416 image into a 32×32 grid of cells. Each cell is then responsible for predicting a number of boxes in the image.

For each bounding box, the network predicts the confidence that the bounding box actually encloses an object, and the probability of the enclosed object being a particular class.

Most of these bounding boxes are eliminated because their confidence is low or because they are enclosing the same object as another bounding box with very high confidence score. This technique is called non-maximum suppression.


**Steps:**

   - We already have pre-trained network weights (yolov3.weights file), the yolov3.cfg file (containing the network configuration) and the coco.names file which contains the 80 different class names used in the COCO dataset.

   - The YOLOv3 algorithm generates bounding boxes as the predicted detection outputs. Every predicted box is associated with a confidence score. In the beginning, all the boxes below the confidence threshold parameter are rejected.

   - The rest of the boxes undergo non-maximum suppression which removes redundant overlapping bounding boxes. Non-maximum suppression is controlled by a parameter nmsThreshold. You can try to change these values and see how the number of output predicted boxes changes.

   - Next, the default values for the input width (inpWidth) and height (inpHeight) for the network’s input image are set. We set each of them to 416, so that we can compare our runs to the Darknet’s C code given by YOLOv3’s authors.

   - Next, we load the network which has two parts —
      - yolov3.weights : The pre-trained weights.
      - yolov3.cfg : The configuration file.

   - We set the DNN backend to OpenCV here and the target to CPU.

   - Then, we read the image, video stream or the webcam. In addition, we also open the video writer to save the frames with detected output bounding boxes.

   - The input image to a neural network needs to be in a certain format called a blob.

   - After a frame is read from the input image or video stream, it is passed through the blobFromImage function to convert it to an input blob for the neural network. In this process, it scales the image pixel values to a target range of 0 to 1 using a scale factor of 1/255. It also resizes the image to the given size of (416, 416) without cropping.

   - The output blob is then passed in to the network as its input and a forward pass is run to get a list of predicted bounding boxes as the network’s output. These boxes go through a post-processing step in order to filter out the ones with low confidence scores. We print out the inference time for each frame at the top left. The image with the final bounding boxes is then saved to the disk.

   - The network outputs bounding boxes are each represented by a vector of number of classes + 5 elements. The first 4 elements represent the center_x, center_y, width and height. The fifth element represents the confidence that the bounding box encloses an object.

   - The rest of the elements are the confidence associated with each class. The box is assigned to the class corresponding to the highest score for the box. 

   - The boxes with their confidence equal to or greater than the confidence threshold are then subjected to Non Maximum Suppression. This would reduce the number of overlapping boxes. The Non Maximum Suppression is controlled by the nmsThreshold parameter. 
   
   - Finally, we draw the boxes that were filtered through the non maximum suppression, on the input frame with their assigned class label and confidence scores.
