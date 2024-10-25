package com.example.javasp;

import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;

import java.util.ArrayList;
import java.util.List;

public class SIFTBadgerDetection {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        // Load images

        // Testing Data
        Mat img1 = Imgcodecs.imread("Image-Classification-Java/assets/badger/4e04eaba81.jpg"); // template
        // Image-Classification-Java/assets/badger/4e04eaba81.jpg

        // Training Data
        Mat img2 = Imgcodecs.imread("Image-Classification-Java/assets/badger/4c2236afb5.jpg");  // scene

        Imgproc.resize(img2, img2, new Size(img2.cols() * 0.3, img2.rows() * 0.3));

//        // Gray Scale
//        Mat grayImage1 = new Mat();
//        Mat grayImage2 = new Mat();
//        Imgproc.cvtColor(img1, grayImage1, Imgproc.COLOR_BGR2GRAY);
//        Imgproc.cvtColor(img2, grayImage2, Imgproc.COLOR_BGR2GRAY);
//        // HighGui.imshow("Grayed Image", grayImage1);

//        // Binary Image
//        Mat binary = new Mat();
//        Imgproc.threshold(grayImage1, binary, 100, 255, Imgproc.THRESH_BINARY_INV);

        // Initialize SIFT detector
        SIFT sift = SIFT.create();

        // Detect keypoints and compute descriptors
        MatOfKeyPoint keypoints1 = new MatOfKeyPoint();
        MatOfKeyPoint keypoints2 = new MatOfKeyPoint();
        Mat descriptors1 = new Mat();
        Mat descriptors2 = new Mat();
        sift.detectAndCompute(img1, new Mat(), keypoints1, descriptors1);
        sift.detectAndCompute(img2, new Mat(), keypoints2, descriptors2);

        // Matching descriptor vectors with a FLANN based matcher
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        List<MatOfDMatch> knnMatches = new ArrayList<>();
        matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

        // Filter matches using the Lowe's ratio test
        float ratioThresh = 0.80f;
        List<DMatch> listOfGoodMatches = new ArrayList<>();
        for (MatOfDMatch knnMatch : knnMatches) {
            if (knnMatch.rows() > 1) {
                DMatch[] matches = knnMatch.toArray();
                if (matches[0].distance < ratioThresh * matches[1].distance) {
                    listOfGoodMatches.add(matches[0]);
                }
            }
        }

        // Draw good matches
        MatOfDMatch goodMatches = new MatOfDMatch();
        goodMatches.fromList(listOfGoodMatches);
        Mat imgMatches = new Mat();
//        Features2d.drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, imgMatches, Scalar.all(-1),
//                Scalar.all(-1), new MatOfByte(), Features2d.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS);


        // Calculate the rectangle to be drawn
//        List<Point> objPts = new ArrayList<>();
//        List<Point> scenePts = new ArrayList<>();
//        for (DMatch match : listOfGoodMatches) {
//            objPts.add(keypoints1.toArray()[match.queryIdx].pt);
//            scenePts.add(keypoints2.toArray()[match.trainIdx].pt);
//        }
//        MatOfPoint2f objMatOfPoint2f = new MatOfPoint2f(objPts.toArray(new Point[0]));
//        MatOfPoint2f sceneMatOfPoint2f = new MatOfPoint2f(scenePts.toArray(new Point[0]));
//
//        Mat H = Calib3d.findHomography(objMatOfPoint2f, sceneMatOfPoint2f, Calib3d.RANSAC, 5);


        // I'm setting the condition for it to be a badger to have
        // more than or equal 10 matches
        if (listOfGoodMatches.size() >= 9) {

            // Draw square around detected object

            // This part just makes it so that it centers on where the most matches are
            // Or so it should be lmao
            Point center = new Point();
            for (DMatch match : listOfGoodMatches) {

                // We determine the center of mass of the matched points by averaging their x and y coordinates.
                // This provides a rough estimate of the object's center, which is used to position the bounding box.
                center.x += keypoints2.toArray()[match.trainIdx].pt.x;
                center.y += keypoints2.toArray()[match.trainIdx].pt.y;
            }
            center.x /= listOfGoodMatches.size();
            center.y /= listOfGoodMatches.size();
            // In theory this square placement method should work
            // But the problem is that the classification method is not precise enough
            // Therefore it will not be perfectly centered towards the object

            double sideLength = 125;

            Point topLeft = new Point(center.x  , center.y );

            Imgproc.rectangle(img1, topLeft, new Point(topLeft.x + sideLength, topLeft.y + sideLength), new Scalar(0, 255, 0), 2);
            Imgproc.putText(img1, "Badger", new Point(topLeft.x, topLeft.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0), 2);

            // I changed this so that it's just a square otherwise it would've
            // made some whacky 4 sided shape.

//            Mat objCorners = new Mat(4, 1, CvType.CV_32FC2);
//            Mat sceneCorners = new Mat(4, 1, CvType.CV_32FC2);
//
//            objCorners.put(0, 0, new double[]{0, 0});
//            objCorners.put(1, 0, new double[]{img1.cols(), 0});
//            objCorners.put(2, 0, new double[]{img1.cols(), img1.rows()});
//            objCorners.put(3, 0, new double[]{0, img1.rows()});
//
//            Core.perspectiveTransform(objCorners, sceneCorners, H);
//
//            Imgproc.line(imgMatches, new Point(sceneCorners.get(0, 0)), new Point(sceneCorners.get(1, 0)), new Scalar(0, 255, 0), 4);
//            Imgproc.line(imgMatches, new Point(sceneCorners.get(1, 0)), new Point(sceneCorners.get(2, 0)), new Scalar(0, 255, 0), 4);
//            Imgproc.line(imgMatches, new Point(sceneCorners.get(2, 0)), new Point(sceneCorners.get(3, 0)), new Scalar(0, 255, 0), 4);
//            Imgproc.line(imgMatches, new Point(sceneCorners.get(3, 0)), new Point(sceneCorners.get(0, 0)), new Scalar(0, 255, 0), 4);

            // Display the result
            System.out.println("Picture is a Badger");
            HighGui.imshow("Good Matches & Object detection", img1);
            HighGui.waitKey(0);

        } else {

            // Same thing here but for the case if it's not a badger

            Point center = new Point();
            for (DMatch match : listOfGoodMatches) {


                center.x += keypoints2.toArray()[match.trainIdx].pt.x;
                center.y += keypoints2.toArray()[match.trainIdx].pt.y;
            }
            center.x /= listOfGoodMatches.size();
            center.y /= listOfGoodMatches.size();


            double sideLength = 125;

            Point topLeft = new Point(center.x  , center.y );

            Imgproc.rectangle(img1, topLeft, new Point(topLeft.x + sideLength, topLeft.y + sideLength), new Scalar(0, 255, 0), 2);
            Imgproc.putText(img1, "Not a Badger", new Point(topLeft.x, topLeft.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0), 2);


            HighGui.imshow("Bad Matches & Bad Object detection", img1);
            HighGui.waitKey(0);
            System.out.println("Not a Badger");
        }


    }
}
