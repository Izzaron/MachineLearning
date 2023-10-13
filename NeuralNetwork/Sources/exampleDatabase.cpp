#include "exampleDatabase.hpp"
#include <iostream>

ExampleDatabase::ExampleDatabase()
{
}

ExampleDatabase::~ExampleDatabase()
{
}

int ExampleDatabase::reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void ExampleDatabase::readMnistImages(string file_path, int desiredSize)
{
    ifstream file (file_path.c_str(), ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;

        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);

        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);

        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        

        this->size = min(number_of_images,desiredSize);
        this->examples.resize(this->size);
        
        for(int i=0; i < this->size; ++i)
        {
            //Eigen::Matrix<unsigned char, 28, 28> &mat = examples[i].image;
            Eigen::VectorXf& v = examples[i].first;
            v.resize(n_rows * n_cols);

            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp,sizeof(temp));
                    //mat(r,c) = temp;
                    v(r * n_rows + c) = (float)temp;
                }
            }
        }
    }
    else
    {
        throw runtime_error("Unable to open file `" + file_path + "`!");
    }
}


void ExampleDatabase::readMnistLabels(string file_path)
{

    ifstream file(file_path, ios::binary);

    if(file.is_open())
    {
        int magic_number = 0;
        int number_of_labels = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);

        if(this->size > number_of_labels) throw runtime_error("Size bigger than number of MNIST labels!");

        for(int i = 0; i < this->size; i++) {
            unsigned char temp = 0;
            file.read((char*)&temp, 1);
            
            //examples[i].label = (int)temp;
            
            Eigen::VectorXf& rtn = examples[i].second;
            rtn.resize(10);
            rtn.setZero();
            rtn((int)temp) = 1.f;
        }
    }
    else
    {
        throw runtime_error("Unable to open file `" + file_path + "`!");
    }
}

//void ExampleDatabase::readMnist(int desiredSize)
//{
//    readMnistImages("A:\\minst\\train-images.idx3-ubyte",desiredSize); //This has to move. Make the user find the files via explorer window
//    readMnistLabels("A:\\minst\\train-labels.idx1-ubyte"); //This has to move. Make the user find the files via explorer window
//}

const example& ExampleDatabase::getExample(size_t exampleIndex)
{
    if(exampleIndex < 0 || exampleIndex >= this->size)
        throw runtime_error("Example index in getExample out of bounds");
    return examples[exampleIndex];
}

std::vector<const example*> ExampleDatabase::getAllExamples()
{
    std::vector<const example*> rtn(examples.size());

    for (size_t i = 0; i < examples.size(); i++)
    {
        rtn[i] = &examples[i];
    }
    
    return rtn;
}