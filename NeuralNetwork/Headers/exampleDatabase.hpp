#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <fstream>
#include <string>

using namespace std;

// tuple(image, layer)
typedef std::pair<Eigen::VectorXf, Eigen::VectorXf> example;

class ExampleDatabase
{
private:

    std::vector< example, Eigen::aligned_allocator<example> > examples;
    int size = 0;

    int reverseInt (int i);
    

public:
    ExampleDatabase();
    ~ExampleDatabase();

    //void readMnist(int desiredSize);
    void readMnistImages(string file_path, int desiredSize);
    void readMnistLabels(string file_path);

    const example& getExample(size_t exampleIndex);
    std::vector<const example*> getAllExamples();

};