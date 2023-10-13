using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace DigitRecognizer
{
    public class NeuralNetwork
    {
        private readonly IntPtr _neuralNetworkPointer;
        private readonly int _firstLayerLength;
        private readonly int _lastLayerLength;

        [DllImport("NeuralNetwork.dll")]
        private static extern IntPtr CreateNeuralNetwork(int[] layerLengths, int nrOfLayers);
        public NeuralNetwork(int[] layerLengths)
        {
            _neuralNetworkPointer = CreateNeuralNetwork(layerLengths, layerLengths.Length);
            _firstLayerLength = layerLengths.First();
            _lastLayerLength = layerLengths.Last();
        }

        [DllImport("NeuralNetwork.dll")]
        private static extern void DeleteNeuralNetwork(IntPtr neuralNetworkPointer);
        ~NeuralNetwork()
        {
            DeleteNeuralNetwork(_neuralNetworkPointer);
        }

        [DllImport("NeuralNetwork.dll")]
        private static extern int LoadNeuralNetworkWeights(IntPtr neuralNetworkPointer);
        public bool LoadWeights()
        {
            Console.WriteLine("Loading weights");
            bool readStatus = LoadNeuralNetworkWeights(_neuralNetworkPointer) != 0;
            if (readStatus)
            {
                Console.WriteLine("Loading successful");
            }
            else
            {
                Console.WriteLine("Loading failed");
            }
            return readStatus;
        }

        [DllImport("NeuralNetwork.dll",CallingConvention = CallingConvention.Cdecl)]
        private static extern int MakeNeuralNetworkPrediction(IntPtr neuralNetwork, float[] input, int inputLength, [In, Out] float[] output, int outputLength);
        public float[] MakePrediction(float[] input)
        {
            float[] output = new float[_lastLayerLength];

            if (MakeNeuralNetworkPrediction(_neuralNetworkPointer, input, input.Length, output, output.Length) == 0)
                throw new Exception("Something went wrong when trying to make a neural network prediction. Does the input and output match the networks first and last layer?");

            return output;
        }

        [DllImport("NeuralNetwork.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int TrainOnMnist(IntPtr neuralNetwork, string images_filepath, string labels_filepath, int nrOfEpochs);
        public bool TrainOnMnist(string images_filepath, string labels_filepath, int nrOfEpochs)
        {
            return TrainOnMnist(_neuralNetworkPointer, images_filepath,labels_filepath,nrOfEpochs) != 0;
        }

        // This is to be removed when backpropagate and forward propagate is more stable
        [DllImport("NeuralNetwork.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void AddWithCuda(IntPtr neuralNetwork);
        public void NNAddWithCuda()
        {
            AddWithCuda(_neuralNetworkPointer);
        }
    }
}
