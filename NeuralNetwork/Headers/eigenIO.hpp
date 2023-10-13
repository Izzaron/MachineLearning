#pragma once

#include <string>
#include <Eigen/Core>
#include <fstream>
#include <iostream>

namespace EigenIO{

    template<class Derived>
    void write_binary(const std::string &filename, const Eigen::PlainObjectBase<Derived> &matrix){

        typedef typename Derived::Index Index;
        typedef typename Derived::Scalar Scalar;
        
        std::ofstream out(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
        
        Index rows=matrix.rows(), cols=matrix.cols();
        out.write((char*) (&rows), sizeof(Index));
        out.write((char*) (&cols), sizeof(Index));
        
        out.write((char*) matrix.data(), rows*cols*sizeof(Scalar) );
        
        out.close();
    }

    template<class Derived>
    void read_binary(const std::string &filename, Eigen::PlainObjectBase<Derived> &matrix){

        typedef typename Derived::Index Index;
        typedef typename Derived::Scalar Scalar;
        
        std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);

        Index rows=0, cols=0; 
        in.read((char*) (&rows),sizeof(Index));
        in.read((char*) (&cols),sizeof(Index));

        if (rows != matrix.rows() || cols != matrix.cols())
        {
            std::cout << "Warning! Rows and or Cols of read weights dont match the network. Resizing..." << std::endl;
        }
        
        matrix.resize(rows, cols);
        
        in.read( (char *) matrix.data() , rows*cols*sizeof(Scalar) );
        
        in.close();
    }
}