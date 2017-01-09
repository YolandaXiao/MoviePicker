#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <algorithm>    // std::nth_element, std::random_shuffle
#include <vector>       // std::vector

using namespace cv;
using namespace std;

Vec3b unpackColor(float f) {
    Vec3b color;
    color[2] = floor(f / 256.0 / 256.0);
    color[1] = floor((f - color[2] * 256.0 * 256.0) / 256.0);
    color[0] = floor(f - color[2] * 256.0 * 256.0 - color[1] * 256.0);
    // now we have a vec3 with the 3 components in range [0..255]. Let's normalize it!
    return color;
}

int main( int argc, char** argv )
{
    //Mat src = imread( argv[1], 1 );
    
    map<string,int> color;
    
    char name[10];
    int i=1;
    while(1)
    {
        sprintf(name,"%d.jpg",i);
        Mat src= imread(name,1);
        if(!src.data ) break;
        
        // use kmeans to find five top colors
        Mat samples(src.rows * src.cols, 3, CV_32F);
        for( int y = 0; y < src.rows; y++ )
            for( int x = 0; x < src.cols; x++ )
                for( int z = 0; z < 3; z++)
                    samples.at<float>(y + x*src.rows, z) = src.at<Vec3b>(y,x)[z];
        
        
        int clusterCount = 5;
        Mat labels;
        int attempts = 5;
        Mat centers;
        kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers );
        
        //color for one image
        //int num = src.rows * src.cols;
        for( int y = 0; y < src.rows; y++ )
        {
            for( int x = 0; x < src.cols; x++ )
            {
                int cluster_idx = labels.at<int>(y + x*src.rows,0);
                
                float colorValue = static_cast<int>(centers.at<float>(cluster_idx, 0))
                + static_cast<int>(centers.at<float>(cluster_idx, 1)) * 256.0
                + static_cast<int>(centers.at<float>(cluster_idx, 2)) * 256.0 * 256.0;
                
                //int colorVal = static_cast<int>(colorValue);
                string temp = to_string(colorValue);
                
                if (color.find(temp) == color.end() ) {
                    color[temp]=1; // not found
                }
                else {
                    color[temp]++; // found
                }
            }
        }
        
        i++;
    }
    
    //sort through the map
    vector<std::pair<std::string, int>> top_values(5);
    partial_sort_copy(color.begin(), color.end(), top_values.begin(), top_values.end(),
                           [](std::pair<const std::string, int> const& l,
                              std::pair<const std::string, int> const& r)
                           {
                               return l.second > r.second;
                           });
    
    //output the top five colors
    for(auto it = top_values.cbegin(); it != top_values.cend(); ++it)
    {
        float color_rgb =strtof((it->first).c_str(),0);
        cout << unpackColor(color_rgb) << " " << it->second << "\n";
    }
    
    //write the image out
    Mat image = Mat::zeros( 400, 400, CV_8UC3 );
    
    // Draw a filled rectangle ( 5th argument is -ve)
    int n=0;
    for(auto it = top_values.cbegin(); it != top_values.cend(); ++it)
    {
        float color_rgb =  strtof((it->first).c_str(),0);
        rectangle( image, Point( 0, n ), Point( 400, n+40), Scalar(unpackColor(color_rgb)), -1, 8 );
        n+=40;
    }
    imshow("ColorAnalysis",image);
    imwrite("ColorAnalysis.jpg",image);
    
    waitKey( 0 );
    
    return 0;
}
