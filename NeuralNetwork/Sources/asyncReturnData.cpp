#include "asyncReturnData.hpp"

AsyncReturnData::AsyncReturnData(int nrOfLayers): layerCount(nrOfLayers), weights(nrOfLayers), biases(nrOfLayers) {};
AsyncReturnData::~AsyncReturnData(){};